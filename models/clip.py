"""
CLIP Vision Encoder — following LLaVA-1.5's approach.

Key differences from SigLIP:
    - Uses HuggingFace CLIPVisionModel + CLIPImageProcessor
    - Has CLS token (index 0) → we remove it, return patch tokens only
    - LLaVA-1.5 uses select_layer=-2 (penultimate hidden state)
    - Output shape: (B, 576, 1024) for clip-vit-large-patch14-336
      → 576 = (336/14)^2 = 24*24 patches (CLS removed)
"""

from typing import Optional

import torch
from transformers import CLIPVisionModel, CLIPImageProcessor

from .base import BaseVisionEncoder, VisionEncoderConfig


class CLIPEncoder(BaseVisionEncoder):
    """
    CLIP vision encoder for feature extraction.

    Uses the same loading logic as LLaVA-1.5:
        - Load CLIPVisionModel from pretrained
        - Extract hidden_states[select_layer] (default: -2, penultimate)
        - Remove CLS token, return patch tokens only

    Args:
        model_name_or_path: HuggingFace model ID or local path.
            e.g., "openai/clip-vit-large-patch14-336"
        select_layer: Which hidden state layer to extract. Default: -2 (LLaVA-1.5 default)
        dtype: torch dtype for inference. Default: torch.float16
        device: Device string. Default: "cuda"
    """

    # Default config for clip-vit-large-patch14-336
    IMAGE_SIZE = 336
    PATCH_SIZE = 14
    HIDDEN_SIZE = 1024

    def __init__(
        self,
        model_name_or_path: str = "openai/clip-vit-large-patch14-336",
        select_layer: int = -2,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        self.model_name_or_path = model_name_or_path
        self.select_layer = select_layer
        self._dtype = dtype
        self._device = device

        self.model: Optional[CLIPVisionModel] = None
        self.image_processor: Optional[CLIPImageProcessor] = None

        self._config = VisionEncoderConfig(
            model_name_or_path=model_name_or_path,
            image_size=self.IMAGE_SIZE,
            patch_size=self.PATCH_SIZE,
            hidden_size=self.HIDDEN_SIZE,
            dtype=dtype,
            device=device,
        )

    def load_model(self) -> None:
        """
        Load CLIP model:
            1. Load pretrained CLIPVisionModel
            2. Freeze all parameters
            3. Move to target device and dtype
        """
        if self.model is not None:
            print(f"[CLIPEncoder] Model already loaded, skipping.")
            return

        print(f"[CLIPEncoder] Loading model from: {self.model_name_or_path}")

        self.model = CLIPVisionModel.from_pretrained(self.model_name_or_path)
        self.model.requires_grad_(False)
        self.model.eval()
        self.model = self.model.to(device=self._device, dtype=self._dtype)

        self.image_processor = CLIPImageProcessor.from_pretrained(self.model_name_or_path)

        num_layers = self.model.config.num_hidden_layers
        print(f"[CLIPEncoder] Loaded successfully.")
        print(f"  - Encoder layers: {num_layers}")
        print(f"  - Select layer: {self.select_layer}")
        print(f"  - Hidden size: {self.HIDDEN_SIZE}")
        print(f"  - Image size: {self.IMAGE_SIZE}")
        print(f"  - Patch size: {self.PATCH_SIZE}")
        print(f"  - Num patches: {self.num_patches} ({self.num_patches_per_side}x{self.num_patches_per_side})")
        print(f"  - Device: {self._device}, Dtype: {self._dtype}")

    def get_image_processor(self) -> CLIPImageProcessor:
        if self.image_processor is None:
            self.image_processor = CLIPImageProcessor.from_pretrained(self.model_name_or_path)
        return self.image_processor

    @torch.no_grad()
    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract visual features from preprocessed images.

        Follows LLaVA-1.5's approach:
            - Pass through vision model with output_hidden_states=True
            - Select hidden_states[select_layer]
            - Separate CLS token (index 0) and patch tokens
            - Cache CLS tokens for temporal representation experiments

        Args:
            pixel_values: (B, 3, 336, 336)

        Returns:
            features: (B, 576, 1024) — patch tokens only (CLS removed)

        Side effect:
            self._last_cls_tokens is set to (B, 1, 1024) — CLS tokens
            Retrieve via get_last_cls_tokens()
        """
        assert self.model is not None, "Model not loaded. Call load_model() first."

        pixel_values = pixel_values.to(device=self._device, dtype=self._dtype)

        outputs = self.model(pixel_values, output_hidden_states=True)

        # Select penultimate layer (same as LLaVA-1.5)
        hidden = outputs.hidden_states[self.select_layer]

        # Separate CLS token and patch tokens
        self._last_cls_tokens = hidden[:, 0:1]  # (B, 1, D) — cache for retrieval
        patch_tokens = hidden[:, 1:]              # (B, 576, D)

        return patch_tokens

    def get_last_cls_tokens(self) -> torch.Tensor:
        """
        Return CLS tokens from the most recent encode_images() call.

        Returns:
            cls_tokens: (B, 1, D) where D = hidden_size (1024 for CLIP-L/14)

        Usage for temporal representation:
            After extracting T frames, collect T CLS tokens → (T, D)
            → concat as temporal representation for linear probing
        """
        if not hasattr(self, '_last_cls_tokens') or self._last_cls_tokens is None:
            raise RuntimeError("No CLS tokens available. Call encode_images() first.")
        return self._last_cls_tokens

    def get_debug_info(self) -> dict:
        info = super().get_debug_info()
        if self.model is not None:
            total_layers = self.model.config.num_hidden_layers
        else:
            total_layers = 24  # default for CLIP-L/14

        # select_layer=-2 means penultimate; positive means absolute index
        if self.select_layer < 0:
            abs_index = total_layers + self.select_layer
        else:
            abs_index = self.select_layer

        info.update({
            "total_num_layers": total_layers,
            "select_layer": self.select_layer,
            "extract_layer_absolute_index": abs_index,
            "extract_method": f"hidden_states[{self.select_layer}] (0-indexed: layer {abs_index})",
            "extract_layer_description": (
                f"Layer {abs_index}/{total_layers} "
                f"(hidden_states[{self.select_layer}]). "
                f"CLS token at index 0 is removed; only patch tokens returned."
            ),
            "cls_token": "Separated and cached via get_last_cls_tokens()",
        })
        return info

    @property
    def encoder_config(self) -> VisionEncoderConfig:
        return self._config
