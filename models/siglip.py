"""
SigLIP Vision Encoder — following LLaVA-OneVision's approach.

Key design choices (from LLaVA-NeXT/llava/model/multimodal_encoder/siglip_encoder.py):
    1. Load SigLipVisionModel from pretrained weights
    2. Remove the last encoder layer
    3. Replace pooling head with nn.Identity()
    4. Extract hidden_states[-1] as features (penultimate layer output)
    5. Output shape: (B, 729, 1152) for siglip-so400m-patch14-384
"""

import sys
import os
from typing import Optional

import torch
import torch.nn as nn

# Add LLaVA-NeXT root to path so we can import its siglip implementation
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from llava.model.multimodal_encoder.siglip_encoder import (
    SigLipVisionModel,
    SigLipVisionConfig,
    SigLipImageProcessor,
)

from .base import BaseVisionEncoder, VisionEncoderConfig


class SigLIPEncoder(BaseVisionEncoder):
    """
    SigLIP vision encoder for feature extraction.

    Uses the same loading logic as LLaVA-OneVision:
        - Removes last encoder layer
        - Replaces pooling head with Identity
        - Extracts penultimate hidden states

    Args:
        model_name_or_path: HuggingFace model ID or local checkpoint path.
            e.g., "google/siglip-so400m-patch14-384"
        dtype: torch dtype for inference. Default: torch.float16
        device: Device string. Default: "cuda"
    """

    # Default config values for siglip-so400m-patch14-384
    IMAGE_SIZE = 384
    PATCH_SIZE = 14
    HIDDEN_SIZE = 1152

    def __init__(
        self,
        model_name_or_path: str = "google/siglip-so400m-patch14-384",
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        self.model_name_or_path = model_name_or_path
        self._dtype = dtype
        self._device = device

        self.model: Optional[SigLipVisionModel] = None
        self.image_processor: Optional[SigLipImageProcessor] = None

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
        Load SigLIP model following LLaVA-OneVision's approach:
            1. Load pretrained SigLipVisionModel
            2. Delete last encoder layer
            3. Replace pooling head with Identity
            4. Freeze all parameters
            5. Move to target device and dtype
        """
        if self.model is not None:
            print(f"[SigLIPEncoder] Model already loaded, skipping.")
            return

        print(f"[SigLIPEncoder] Loading model from: {self.model_name_or_path}")

        # Step 1: Load pretrained
        self.model = SigLipVisionModel.from_pretrained(self.model_name_or_path)

        # Step 2: Remove last encoder layer (same as LLaVA-NeXT)
        del self.model.vision_model.encoder.layers[-1:]

        # Step 3: Replace pooling head with Identity
        self.model.vision_model.head = nn.Identity()

        # Step 4: Freeze all parameters
        self.model.requires_grad_(False)
        self.model.eval()

        # Step 5: Move to target device and dtype
        self.model = self.model.to(device=self._device, dtype=self._dtype)

        # Initialize image processor
        self.image_processor = SigLipImageProcessor()

        num_layers = len(self.model.vision_model.encoder.layers)
        print(f"[SigLIPEncoder] Loaded successfully.")
        print(f"  - Encoder layers (after removing last): {num_layers}")
        print(f"  - Hidden size: {self.HIDDEN_SIZE}")
        print(f"  - Image size: {self.IMAGE_SIZE}")
        print(f"  - Patch size: {self.PATCH_SIZE}")
        print(f"  - Num patches: {self.num_patches} ({self.num_patches_per_side}x{self.num_patches_per_side})")
        print(f"  - Device: {self._device}, Dtype: {self._dtype}")

    def get_image_processor(self) -> SigLipImageProcessor:
        """Return the SigLIP image preprocessor."""
        if self.image_processor is None:
            self.image_processor = SigLipImageProcessor()
        return self.image_processor

    @torch.no_grad()
    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract visual features from preprocessed images.

        Follows LLaVA-NeXT's SigLipVisionTower.forward():
            - Pass through vision model with output_hidden_states=True
            - Return hidden_states[-1] (penultimate layer features)

        Args:
            pixel_values: Tensor of shape (B, 3, 384, 384)

        Returns:
            features: Tensor of shape (B, 729, 1152)
        """
        assert self.model is not None, "Model not loaded. Call load_model() first."

        pixel_values = pixel_values.to(device=self._device, dtype=self._dtype)

        outputs = self.model(
            pixel_values,
            output_hidden_states=True,
        )

        # Extract penultimate layer hidden states (same as LLaVA-NeXT)
        features = outputs.hidden_states[-1]

        return features

    @property
    def encoder_config(self) -> VisionEncoderConfig:
        return self._config
