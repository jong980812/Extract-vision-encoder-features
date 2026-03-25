"""
Qwen2-VL / Qwen2.5-VL Vision Encoder — Extract vision encoder from Qwen VLMs.

Loads the full VLM, extracts the vision transformer (model.model.visual),
and provides a compatible interface for feature extraction.

Key architecture details:
    - Custom ViT with Conv3d patch embedding (temporal_patch_size=2)
    - 2D Rotary position embeddings
    - PatchMerger: merges 2x2 spatial patches via MLP
    - Qwen2.5-VL additionally uses windowed attention
    - Output is post-merger features (what the LLM actually sees)

Supports both Qwen2-VL and Qwen2.5-VL (auto-detected from config).

Usage:
    python extract_video_features.py \
        --encoder qwen2_vl \
        --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
        ...
"""

import os
import numpy as np
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image

from .base import BaseVisionEncoder, VisionEncoderConfig


OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


class QwenSimpleImageProcessor:
    """
    Fixed-resolution image processor for Qwen vision encoders.

    Compatible with the existing collate_fn which calls:
        processor.preprocess(frames, return_tensors="pt") -> {"pixel_values": (N, C, H, W)}
    """

    def __init__(self, image_size):
        self.image_size = image_size

    def preprocess(self, images, return_tensors="pt"):
        if isinstance(images, np.ndarray) and images.ndim == 4:
            images = [images[i] for i in range(images.shape[0])]

        mean = np.array(OPENAI_CLIP_MEAN, dtype=np.float32)
        std = np.array(OPENAI_CLIP_STD, dtype=np.float32)

        processed = []
        for img in images:
            if isinstance(img, np.ndarray):
                pil_img = Image.fromarray(img.astype(np.uint8))
            else:
                pil_img = img
            pil_img = pil_img.convert("RGB")
            pil_img = pil_img.resize((self.image_size, self.image_size), Image.BICUBIC)

            arr = np.array(pil_img, dtype=np.float32) / 255.0
            arr = (arr - mean) / std
            arr = arr.transpose(2, 0, 1)  # HWC -> CHW
            processed.append(arr)

        pixel_values = torch.from_numpy(np.stack(processed))
        return {"pixel_values": pixel_values}


class Qwen2VLVisionEncoder(BaseVisionEncoder):
    """
    Vision encoder extracted from Qwen2-VL / Qwen2.5-VL.

    Loads the full model, extracts model.model.visual, discards the LLM.
    Returns post-merger features: (B, num_merged_patches, out_hidden_size).

    For a 392x392 image with patch_size=14, merge_size=2:
        - Grid: 28x28 spatial patches
        - After merger: 14x14 = 196 tokens, dim=3584

    Args:
        model_name_or_path: HuggingFace model ID.
            e.g., "Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen2.5-VL-3B-Instruct"
        image_size: Fixed input resolution. Must be divisible by (patch_size * merge_size).
            Default: 392 (= 28 * 14). Other options: 364, 420, 448, ...
        dtype: torch dtype. Default: torch.float16
        device: Device string. Default: "cuda"
    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        image_size: int = 392,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        self.model_name_or_path = model_name_or_path
        self.image_size = image_size
        self._dtype = dtype
        self._device = device

        self.vision_model = None
        self.image_processor = None
        self._config: Optional[VisionEncoderConfig] = None

        # Set after loading model config
        self._patch_size = None
        self._temporal_patch_size = None
        self._merge_size = None
        self._embed_dim = None
        self._out_hidden_size = None
        self._depth = None
        self._model_type = None

    def load_model(self) -> None:
        if self.vision_model is not None:
            print(f"[Qwen2VLVisionEncoder] Model already loaded, skipping.")
            return

        print(f"[Qwen2VLVisionEncoder] Loading full model: {self.model_name_or_path}")

        from transformers import AutoModelForVision2Seq, AutoConfig

        # Read config to get vision parameters
        config = AutoConfig.from_pretrained(
            self.model_name_or_path,
            cache_dir=os.getenv('HF_HOME', None),
        )
        vis_config = config.vision_config
        self._model_type = config.model_type

        self._patch_size = vis_config.patch_size
        self._temporal_patch_size = vis_config.temporal_patch_size
        self._merge_size = vis_config.spatial_merge_size
        self._depth = vis_config.depth

        # Config field names differ between Qwen2-VL and Qwen2.5-VL
        if config.model_type == "qwen2_vl":
            # Qwen2-VL: embed_dim=1280 (ViT), hidden_size=3584 (LLM projection)
            self._embed_dim = vis_config.embed_dim
            self._out_hidden_size = vis_config.hidden_size
        else:
            # Qwen2.5-VL: hidden_size=1280 (ViT), out_hidden_size=3584 (LLM projection)
            self._embed_dim = vis_config.hidden_size
            self._out_hidden_size = vis_config.out_hidden_size

        factor = self._patch_size * self._merge_size
        assert self.image_size % factor == 0, \
            f"image_size ({self.image_size}) must be divisible by patch_size * merge_size = {factor}"

        # Load the full model
        model = AutoModelForVision2Seq.from_pretrained(
            self.model_name_or_path,
            torch_dtype=self._dtype,
            device_map=None,
            cache_dir=os.getenv('HF_HOME', None),
        )

        # Extract vision encoder
        self.vision_model = model.model.visual
        self.vision_model.requires_grad_(False)
        self.vision_model.eval()
        self.vision_model = self.vision_model.to(device=self._device, dtype=self._dtype)

        # Config for BaseVisionEncoder interface
        effective_patch_size = self._patch_size * self._merge_size
        self._config = VisionEncoderConfig(
            model_name_or_path=self.model_name_or_path,
            image_size=self.image_size,
            patch_size=effective_patch_size,
            hidden_size=self._out_hidden_size,
            dtype=self._dtype,
            device=self._device,
        )

        self.image_processor = QwenSimpleImageProcessor(self.image_size)

        # Delete the rest
        del model
        torch.cuda.empty_cache()

        print(f"[Qwen2VLVisionEncoder] Vision encoder extracted successfully.")
        print(f"  - Source VLM: {self.model_name_or_path}")
        print(f"  - Model type: {self._model_type}")
        print(f"  - Vision depth: {self._depth}")
        print(f"  - ViT embed dim: {self._embed_dim}")
        print(f"  - Output hidden size (post-merger): {self._out_hidden_size}")
        print(f"  - Patch size: {self._patch_size}, Merge size: {self._merge_size}")
        print(f"  - Image size: {self.image_size}")
        print(f"  - Num patches (post-merger): {self.num_patches} ({self.num_patches_per_side}x{self.num_patches_per_side})")
        print(f"  - Device: {self._device}, Dtype: {self._dtype}")

    def get_image_processor(self):
        if self.image_processor is None:
            self.image_processor = QwenSimpleImageProcessor(self.image_size)
        return self.image_processor

    def _pixels_to_patches(self, pixel_values: torch.Tensor):
        """
        Convert (B, C, H, W) images to Qwen's flat patch format.

        Each single-frame image is repeated temporally (temporal_patch_size times)
        and reshaped following Qwen's spatial merge ordering so that
        2x2 adjacent patches are consecutive (required by PatchMerger).

        Returns:
            hidden_states: (total_patches, C * temporal_patch_size * patch_size^2)
            grid_thw: (B, 3) LongTensor with [t=1, grid_h, grid_w] per image
        """
        B, C, H, W = pixel_values.shape
        p = self._patch_size
        t_p = self._temporal_patch_size
        m = self._merge_size

        grid_h = H // p
        grid_w = W // p

        # Repeat each frame temporally: (B, C, H, W) -> (B, t_p, C, H, W)
        frames = pixel_values.unsqueeze(1).expand(-1, t_p, -1, -1, -1)

        # Reshape following Qwen's patch ordering for PatchMerger
        # dims: (B, t_p, C, grid_h//m, m, p, grid_w//m, m, p)
        frames = frames.reshape(B, t_p, C, grid_h // m, m, p, grid_w // m, m, p)
        # Transpose to: (B, grid_h//m, grid_w//m, m, m, C, t_p, p, p)
        frames = frames.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).contiguous()

        patch_dim = C * t_p * p * p
        patches_per_image = grid_h * grid_w

        hidden_states = frames.reshape(B * patches_per_image, patch_dim)

        grid_thw = torch.tensor(
            [[1, grid_h, grid_w]] * B,
            dtype=torch.long,
            device=pixel_values.device,
        )

        return hidden_states, grid_thw

    @torch.no_grad()
    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract post-merger visual features.

        Args:
            pixel_values: (B, 3, H, W) preprocessed images

        Returns:
            features: (B, num_merged_patches, out_hidden_size)
                e.g., (B, 196, 3584) for 392x392 with patch_size=14, merge=2
        """
        assert self.vision_model is not None, "Model not loaded. Call load_model() first."

        pixel_values = pixel_values.to(device=self._device, dtype=self._dtype)
        B = pixel_values.shape[0]

        hidden_states, grid_thw = self._pixels_to_patches(pixel_values)

        # Full forward (includes merger)
        output = self.vision_model(hidden_states, grid_thw)

        # Qwen3-VL returns tuple; Qwen2/2.5-VL returns tensor
        if isinstance(output, tuple):
            output = output[0]

        patches_per_image = self.num_patches
        features = output.reshape(B, patches_per_image, -1)

        return features

    def get_debug_info(self) -> dict:
        info = super().get_debug_info()
        info.update({
            "source_vlm": self.model_name_or_path,
            "model_type": self._model_type,
            "actual_patch_size": self._patch_size,
            "temporal_patch_size": self._temporal_patch_size,
            "merge_size": self._merge_size,
            "vision_depth": self._depth,
            "vit_embed_dim": self._embed_dim,
            "post_merger_hidden_size": self._out_hidden_size,
            "extract_method": "Full vision encoder forward (post-merger)",
            "extract_layer_description": (
                f"All {self._depth} ViT blocks + PatchMerger. "
                f"2x2 spatial patches merged via MLP: {self._embed_dim}d -> {self._out_hidden_size}d."
            ),
        })
        return info

    @property
    def encoder_config(self) -> VisionEncoderConfig:
        if self._config is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self._config
