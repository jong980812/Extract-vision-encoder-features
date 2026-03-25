from .base import BaseVisionEncoder, VisionEncoderConfig


def _get_encoder_class(encoder_name: str):
    """Lazy import to avoid loading incompatible dependencies."""
    if encoder_name == "siglip":
        from .siglip import SigLIPEncoder
        return SigLIPEncoder
    elif encoder_name == "clip":
        from .clip import CLIPEncoder
        return CLIPEncoder
    elif encoder_name == "llava":
        from .llava import LLaVAVisionEncoder
        return LLaVAVisionEncoder
    elif encoder_name == "qwen2_vl":
        from .qwen2_vl import Qwen2VLVisionEncoder
        return Qwen2VLVisionEncoder
    elif encoder_name == "qwen3_vl":
        from .qwen3_vl import Qwen3VLVisionEncoder
        return Qwen3VLVisionEncoder
    else:
        return None


ENCODER_NAMES = ["siglip", "clip", "llava", "qwen2_vl", "qwen3_vl"]


def build_vision_encoder(encoder_name: str, model_name_or_path: str, **kwargs) -> BaseVisionEncoder:
    """
    Factory function to build a vision encoder by name.

    Args:
        encoder_name: Key in ENCODER_NAMES (e.g., "siglip", "clip", "qwen2_vl")
        model_name_or_path: HuggingFace model ID or local path
        **kwargs: Additional arguments passed to the encoder constructor

    Returns:
        An initialized BaseVisionEncoder instance with model loaded.

    Example:
        encoder = build_vision_encoder("siglip", "google/siglip-so400m-patch14-384")
        features = encoder.encode_images(pixel_values)
    """
    encoder_cls = _get_encoder_class(encoder_name)
    if encoder_cls is None:
        available = ", ".join(ENCODER_NAMES)
        raise ValueError(
            f"Unknown encoder: '{encoder_name}'. Available: [{available}]"
        )

    encoder = encoder_cls(model_name_or_path=model_name_or_path, **kwargs)
    encoder.load_model()
    return encoder
