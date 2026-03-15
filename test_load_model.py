"""
Quick sanity check: load SigLIP encoder and run a dummy forward pass.

Usage:
    python extract_features/test_load_model.py
"""

import torch
from models import build_vision_encoder


def main():
    # --- 1. Build encoder via factory ---
    encoder = build_vision_encoder(
        encoder_name="siglip",
        model_name_or_path="google/siglip-so400m-patch14-384",
        dtype=torch.float16,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # --- 2. Print summary ---
    cfg = encoder.encoder_config
    print(f"\n=== Encoder Summary ===")
    print(f"  Model: {cfg.model_name_or_path}")
    print(f"  Image size: {cfg.image_size}")
    print(f"  Patch size: {cfg.patch_size}")
    print(f"  Hidden size: {cfg.hidden_size}")
    print(f"  Num patches: {encoder.num_patches} ({encoder.num_patches_per_side}x{encoder.num_patches_per_side})")

    # --- 3. Dummy forward pass ---
    B = 2
    dummy_input = torch.randn(B, 3, cfg.image_size, cfg.image_size)
    features = encoder.encode_images(dummy_input)

    print(f"\n=== Forward Pass ===")
    print(f"  Input shape:  {dummy_input.shape}")
    print(f"  Output shape: {features.shape}")
    print(f"  Expected:     ({B}, {encoder.num_patches}, {cfg.hidden_size})")
    assert features.shape == (B, encoder.num_patches, cfg.hidden_size), "Shape mismatch!"
    print(f"  ✓ Shape verified!")

    # --- 4. Test image processor ---
    from PIL import Image
    processor = encoder.get_image_processor()
    dummy_pil = Image.new("RGB", (640, 480), color=(128, 128, 128))
    processed = processor.preprocess([dummy_pil], return_tensors="pt")
    print(f"\n=== Image Processor ===")
    print(f"  Input: PIL Image (640x480)")
    print(f"  Output pixel_values shape: {processed['pixel_values'].shape}")

    features_from_pil = encoder.encode_images(processed["pixel_values"])
    print(f"  Encoded features shape: {features_from_pil.shape}")
    print(f"  ✓ End-to-end pipeline works!")


if __name__ == "__main__":
    main()
