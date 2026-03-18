"""
LLaVA Vision Encoder — Extract vision tower from a full LLaVA VLM.

Instead of loading a standalone vision model (e.g., google/siglip-so400m-patch14-384),
this loads a full LLaVA model (e.g., lmms-lab/llava-onevision-qwen2-0.5b-si)
and extracts the vision encoder that was trained jointly with the LLM.

The extracted vision tower already has:
    1. Last encoder layer removed
    2. Pooling head replaced with nn.Identity()
    3. Weights that were fine-tuned during VLM training (if applicable)

Usage:
    python extract_video_features.py \
        --encoder llava \
        --model_name_or_path lmms-lab/llava-onevision-qwen2-0.5b-si \
        --llava_model_name llava_qwen \
        ...
"""

import sys
import os
from typing import Optional

import torch
import torch.nn as nn

# Add LLaVA-NeXT root to path
_LLAVA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "LLaVA-NeXT"))
if _LLAVA_ROOT not in sys.path:
    sys.path.insert(0, _LLAVA_ROOT)

from llava.model.builder import load_pretrained_model

from .base import BaseVisionEncoder, VisionEncoderConfig


class LLaVAVisionEncoder(BaseVisionEncoder):
    """
    Vision encoder extracted from a full LLaVA VLM.

    Loads the entire LLaVA model (LLM + vision tower + projector),
    then keeps only the vision tower for feature extraction.
    The LLM and projector weights are discarded after extraction
    to save GPU memory.

    Args:
        model_name_or_path: HuggingFace model ID or local path for the full LLaVA model.
            e.g., "lmms-lab/llava-onevision-qwen2-0.5b-si"
                  "lmms-lab/llava-onevision-qwen2-7b-si"
                  "lmms-lab/llava-onevision-qwen2-72b-si"
        llava_model_name: Model name string for LLaVA builder.
            e.g., "llava_qwen" (for OneVision), "llava_llama", "llava_mistral"
        dtype: torch dtype for inference. Default: torch.float16
        device: Device string. Default: "cuda"
        attn_implementation: Attention implementation. Default: "sdpa"
            Use "flash_attention_2" if available, "sdpa" for PyTorch native,
            or "eager" as fallback.
    """

    def __init__(
        self,
        model_name_or_path: str = "lmms-lab/llava-onevision-qwen2-0.5b-si",
        llava_model_name: str = "llava_qwen",
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        attn_implementation: str = "sdpa",
    ):
        self.model_name_or_path = model_name_or_path
        self.llava_model_name = llava_model_name
        self._dtype = dtype
        self._device = device
        self._attn_implementation = attn_implementation

        self.vision_tower = None  # SigLipVisionTower (or CLIPVisionTower, etc.)
        self.image_processor = None
        self._config: Optional[VisionEncoderConfig] = None

    def load_model(self) -> None:
        """
        Load full LLaVA model, extract vision tower, then discard the rest.

        Steps:
            1. Load entire VLM via load_pretrained_model()
            2. Extract vision_tower = model.get_vision_tower()
            3. Get image_processor from the vision tower
            4. Read config (image_size, patch_size, hidden_size) from vision tower
            5. Delete the full model (LLM + projector) to free memory
            6. Keep only the vision tower on target device
        """
        if self.vision_tower is not None:
            print(f"[LLaVAVisionEncoder] Model already loaded, skipping.")
            return

        print(f"[LLaVAVisionEncoder] Loading full LLaVA model: {self.model_name_or_path}")
        print(f"  - Model name: {self.llava_model_name}")
        print(f"  - This will load the full VLM temporarily...")

        # Step 1: Load the full VLM
        torch_dtype_str = {
            torch.float16: "float16",
            torch.bfloat16: "bfloat16",
            torch.float32: "float32",
        }.get(self._dtype, "float16")

        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=self.model_name_or_path,
            model_base=None,
            model_name=self.llava_model_name,
            device_map=None,  # Avoid accelerate's meta tensor dispatch (causes no-op weight copy)
            torch_dtype=torch_dtype_str,
            attn_implementation=self._attn_implementation,
        )

        # Step 2: Extract vision tower
        self.vision_tower = model.get_vision_tower()
        self.image_processor = image_processor

        # Step 3: Read config from the vision tower
        vt_config = self.vision_tower.config
        image_size = getattr(vt_config, 'image_size', 384)
        patch_size = getattr(vt_config, 'patch_size', 14)
        hidden_size = getattr(vt_config, 'hidden_size', 1152)

        self._config = VisionEncoderConfig(
            model_name_or_path=self.model_name_or_path,
            image_size=image_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            dtype=self._dtype,
            device=self._device,
        )

        # Step 4: Detach vision tower from the full model and move to target device
        # First, ensure vision tower is frozen
        self.vision_tower.requires_grad_(False)
        self.vision_tower.eval()

        # Move vision tower to target device
        self.vision_tower = self.vision_tower.to(device=self._device, dtype=self._dtype)

        # Step 5: Delete the full model to free memory
        del model
        del tokenizer
        torch.cuda.empty_cache()

        num_layers = len(self.vision_tower.vision_tower.vision_model.encoder.layers)
        print(f"[LLaVAVisionEncoder] Vision tower extracted successfully.")
        print(f"  - Source VLM: {self.model_name_or_path}")
        print(f"  - Vision tower type: {type(self.vision_tower).__name__}")
        print(f"  - Encoder layers: {num_layers}")
        print(f"  - Hidden size: {hidden_size}")
        print(f"  - Image size: {image_size}")
        print(f"  - Patch size: {patch_size}")
        print(f"  - Num patches: {self.num_patches} ({self.num_patches_per_side}x{self.num_patches_per_side})")
        print(f"  - Device: {self._device}, Dtype: {self._dtype}")

    def get_image_processor(self):
        """Return the image processor from the loaded vision tower."""
        if self.image_processor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.image_processor

    @torch.no_grad()
    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract visual features using the vision tower from LLaVA.

        The vision tower's forward() already handles:
            - Moving inputs to correct device/dtype
            - Passing through the vision model with output_hidden_states=True
            - Returning hidden_states[-1] (penultimate layer features)

        Args:
            pixel_values: Tensor of shape (B, 3, H, W)

        Returns:
            features: Tensor of shape (B, num_patches, hidden_size)
                      e.g., (B, 729, 1152) for SigLIP-SO400M
        """
        assert self.vision_tower is not None, "Model not loaded. Call load_model() first."

        pixel_values = pixel_values.to(device=self._device, dtype=self._dtype)

        # SigLipVisionTower.forward() returns (B, 729, 1152)
        features = self.vision_tower(pixel_values)

        return features

    @property
    def encoder_config(self) -> VisionEncoderConfig:
        if self._config is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self._config
