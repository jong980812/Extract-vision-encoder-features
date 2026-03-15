"""
Quick sanity check: load SigLIP encoder and run a dummy forward pass.

Usage:
    python test_load_model.py [--debug]

"""

import torch
import argparse
from models import build_vision_encoder
from data import make_dataloader

def main(args):

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

    loader = make_dataloader(
        json_path="annotations.json",
        video_root_dir="/data/videos",
        image_processor=encoder.get_image_processor(),
        num_frames=8,
        batch_size=4,
    )

    for batch in loader:
        features = encoder.encode_images(batch["pixel_values"])  # (total_frames, 729, 1152)
        per_video = torch.split(features, batch["frame_counts"]) # list of (8, 729, 1152)
        ids = batch["ids"]          # [0, 1, 2, 3]
        meta = batch["metadata"]  
    if args.debug:
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
    parser = argparse.ArgumentParser(description="Test load vision encoder")
    parser.add_argument("--debug", action="store_true", help="Run debug forward pass")
    args = parser.parse_args()
    main(args)

