# extract_features/models/clip.py

"""
CLIP Vision Encoder (OpenAI ViT-L/14-336)

Key differences from SigLIP:
    - Uses HuggingFace CLIPVisionModel (not custom impl)
    - Uses CLIPImageProcessor from transformers
    - select_layer로 중간 layer feature 추출 가능
    - Output shape: (B, 577, 1024) for ViT-L/14-336
      → 577 = 1 CLS token + 576 patches (24x24)
"""

import torch
import torch.nn as nn
from typing import Optional
from transformers import CLIPVisionModel, CLIPImageProcessor

from .base import BaseVisionEncoder, VisionEncoderConfig


class CLIPEncoder(BaseVisionEncoder):

    # Default config for openai/clip-vit-large-patch14-336
    IMAGE_SIZE = 336
    PATCH_SIZE = 14
    HIDDEN_SIZE = 1024

    def __init__(
        self,
        model_name_or_path: str = "openai/clip-vit-large-patch14-336",
        select_layer: int = -2,        # LLaVA 기본값: 두번째 마지막 layer
        select_feature: str = "patch",  # "patch" = CLS 제외, "cls_patch" = CLS 포함
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        self.model_name_or_path = model_name_or_path
        self.select_layer = select_layer
        self.select_feature = select_feature
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
        if self.model is not None:
            print(f"[CLIPEncoder] Already loaded, skipping.")
            return

        print(f"[CLIPEncoder] Loading from: {self.model_name_or_path}")

        self.model = CLIPVisionModel.from_pretrained(self.model_name_or_path)
        self.model.requires_grad_(False)
        self.model.eval()
        self.model = self.model.to(device=self._device, dtype=self._dtype)

        self.image_processor = CLIPImageProcessor.from_pretrained(self.model_name_or_path)

        num_layers = self.model.config.num_hidden_layers
        print(f"[CLIPEncoder] Loaded successfully.")
        print(f"  - Encoder layers: {num_layers}")
        print(f"  - Select layer: {self.select_layer}")
        print(f"  - Select feature: {self.select_feature}")
        print(f"  - Hidden size: {self.HIDDEN_SIZE}")
        print(f"  - Device: {self._device}, Dtype: {self._dtype}")

    def get_image_processor(self) -> CLIPImageProcessor:
        if self.image_processor is None:
            self.image_processor = CLIPImageProcessor.from_pretrained(
                self.model_name_or_path
            )
        return self.image_processor

    @torch.no_grad()
    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, 3, 336, 336)
        Returns:
            features: (B, num_patches, 1024)
                - select_feature="patch"     → CLS 제외, (B, 576, 1024)
                - select_feature="cls_patch" → CLS 포함, (B, 577, 1024)
        """
        assert self.model is not None, "Call load_model() first."

        pixel_values = pixel_values.to(device=self._device, dtype=self._dtype)

        outputs = self.model(pixel_values, output_hidden_states=True)

        # select_layer로 원하는 중간 layer의 hidden state 추출
        features = outputs.hidden_states[self.select_layer]

        # CLS token 처리
        if self.select_feature == "patch":
            features = features[:, 1:]  # CLS 제외
        # "cls_patch"이면 그대로 유지

        return features

    @property
    def encoder_config(self) -> VisionEncoderConfig:
        return self._config