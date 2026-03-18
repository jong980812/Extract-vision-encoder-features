from .base import BaseVisionEncoder, VisionEncoderConfig
from .siglip import SigLIPEncoder
from .clip import CLIPEncoder       # ← 추가
from .llava import LLaVAVisionEncoder

ENCODER_REGISTRY = {
    "siglip": SigLIPEncoder,
    "clip": CLIPEncoder,             # ← 추가
    "llava": LLaVAVisionEncoder,
}

def build_vision_encoder(encoder_name: str, model_name_or_path: str, **kwargs) -> BaseVisionEncoder:
    """
    Factory function to build a vision encoder by name.

    Args:
        encoder_name: Key in ENCODER_REGISTRY (e.g., "siglip", "clip")
        model_name_or_path: HuggingFace model ID or local path
        **kwargs: Additional arguments passed to the encoder constructor

    Returns:
        An initialized BaseVisionEncoder instance with model loaded.

    Example:
        encoder = build_vision_encoder("siglip", "google/siglip-so400m-patch14-384")
        features = encoder.encode_images(pixel_values)
    """
    if encoder_name not in ENCODER_REGISTRY:
        available = ", ".join(ENCODER_REGISTRY.keys())
        raise ValueError(
            f"Unknown encoder: '{encoder_name}'. Available: [{available}]"
        )

    encoder_cls = ENCODER_REGISTRY[encoder_name]
    encoder = encoder_cls(model_name_or_path=model_name_or_path, **kwargs)
    encoder.load_model()
    return encoder
