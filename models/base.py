"""
Base class for vision encoders.
All vision encoder implementations should inherit from this class.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class VisionEncoderConfig:
    """Common config for all vision encoders."""
    model_name_or_path: str
    image_size: int
    patch_size: int
    hidden_size: int
    dtype: torch.dtype = torch.float16
    device: str = "cuda"


class BaseVisionEncoder(ABC):
    """
    Abstract base class for vision encoders.

    Subclasses must implement:
        - load_model(): Load pretrained weights
        - get_image_processor(): Return the image preprocessor
        - encode_images(pixel_values) -> torch.Tensor: Extract features from images
        - config property: Return VisionEncoderConfig
    """

    @abstractmethod
    def load_model(self) -> None:
        """Load pretrained model weights."""
        ...

    @abstractmethod
    def get_image_processor(self):
        """Return the image preprocessor (transforms, tokenizer, etc.)."""
        ...

    @abstractmethod
    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract visual features from preprocessed pixel values.

        Args:
            pixel_values: Tensor of shape (B, C, H, W)

        Returns:
            features: Tensor of shape (B, num_patches, hidden_size)
        """
        ...

    @property
    @abstractmethod
    def encoder_config(self) -> VisionEncoderConfig:
        """Return the encoder configuration."""
        ...

    def get_debug_info(self) -> dict:
        """
        Return debug information about the encoder.
        Subclasses should override to add encoder-specific details.

        Returns:
            dict with keys like:
                - encoder_type, model_name_or_path
                - total_layers, extract_layer_index, extract_layer_desc
                - image_size, patch_size, hidden_size, num_patches
                - dtype, device
        """
        cfg = self.encoder_config
        return {
            "encoder_type": self.__class__.__name__,
            "model_name_or_path": cfg.model_name_or_path,
            "image_size": cfg.image_size,
            "patch_size": cfg.patch_size,
            "hidden_size": cfg.hidden_size,
            "num_patches": self.num_patches,
            "num_patches_per_side": self.num_patches_per_side,
            "dtype": str(cfg.dtype),
            "device": cfg.device,
        }

    @property
    def num_patches(self) -> int:
        cfg = self.encoder_config
        return (cfg.image_size // cfg.patch_size) ** 2

    @property
    def num_patches_per_side(self) -> int:
        cfg = self.encoder_config
        return cfg.image_size // cfg.patch_size
