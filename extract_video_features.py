"""
Extract vision encoder features from videos and save as .pt files.

Usage:
    cd extract_features
    python extract_video_features.py \
        --encoder siglip \
        --model_name_or_path google/siglip-so400m-patch14-384 \
        --json_path /path/to/annotations.json \
        --video_root_dir /path/to/videos \
        --output_dir /path/to/output \
        --num_frames 8 \
        --batch_size 4 \
        --spatial_pool_stride 2 \
        --spatial_pool_mode average

Output structure:
    output_dir/
    ├── 0.pt        # {"features": (num_frames, 196, 1152), "id": 0, "video": "...", ...}
    ├── 1.pt
    ├── ...
    └── config.json  # extraction config for reproducibility
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from models import build_vision_encoder
from data import make_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="Extract video features using vision encoders")

    # Encoder
    parser.add_argument("--encoder", type=str, default="siglip",
                        help="Vision encoder name (e.g., siglip, clip)")
    parser.add_argument("--model_name_or_path", type=str, default="google/siglip-so400m-patch14-384",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"],
                        help="Model dtype")

    # Data
    parser.add_argument("--json_path", type=str, required=True,
                        help="Path to JSON annotation file")
    parser.add_argument("--video_root_dir", type=str, required=True,
                        help="Root directory for video files")
    parser.add_argument("--video_key", type=str, default="video",
                        help="JSON key for video relative path")
    parser.add_argument("--id_key", type=str, default="id",
                        help="JSON key for sample ID")
    parser.add_argument("--num_frames", type=int, default=8,
                        help="Number of frames to sample per video")
    parser.add_argument("--force_sample", action="store_true",
                        help="Always use uniform sampling")

    # DataLoader
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Number of videos per batch")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers")

    # Spatial pooling (same as LLaVA-NeXT get_2dPool)
    parser.add_argument("--spatial_pool_stride", type=int, default=1,
                        help="Spatial pooling stride. 1=no pooling, 2=729→196, 4=729→49")
    parser.add_argument("--spatial_pool_mode", type=str, default="bilinear",
                        choices=["average", "max", "bilinear"],
                        help="Spatial pooling mode (following LLaVA-NeXT config). LLaVA-OneVision uses bilinear.")

    # Output
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save extracted features")
    parser.add_argument("--save_format", type=str, default="per_video", choices=["per_video", "single"],
                        help="per_video: one .pt per video, single: all features in one file")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip videos whose .pt files already exist")

    return parser.parse_args()


def get_dtype(dtype_str: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype_str]


def spatial_pool_2d(features: torch.Tensor, num_patches_per_side: int,
                    stride: int, mode: str) -> torch.Tensor:
    """
    Spatial pooling on patch features — same as LLaVA-NeXT's get_2dPool.

    Args:
        features: (num_frames, num_patches, hidden_size)
                  e.g. (8, 729, 1152)
        num_patches_per_side: sqrt(num_patches), e.g. 27 for SigLIP
        stride: pooling stride. 1=no-op, 2=729→196, 4=729→49
        mode: "average", "max", or "bilinear"

    Returns:
        pooled: (num_frames, pooled_patches, hidden_size)
                e.g. stride=2 → (8, 196, 1152)
    """
    if stride <= 1:
        return features

    num_frames, num_tokens, num_dim = features.shape
    h = w = num_patches_per_side

    # (N, 729, 1152) → (N, 27, 27, 1152) → (N, 1152, 27, 27)
    features = features.view(num_frames, h, w, num_dim)
    features = features.permute(0, 3, 1, 2).contiguous()

    if mode == "average":
        features = F.avg_pool2d(features, stride)
    elif mode == "max":
        features = F.max_pool2d(features, stride)
    elif mode == "bilinear":
        scaled_h = math.ceil(h / stride)
        scaled_w = math.ceil(w / stride)
        features = F.interpolate(features, size=(scaled_h, scaled_w), mode="bilinear")
    else:
        raise ValueError(f"Unexpected spatial_pool_mode: {mode}")

    # (N, 1152, 14, 14) → (N, 14, 14, 1152) → (N, 196, 1152)
    features = features.permute(0, 2, 3, 1)
    features = features.view(num_frames, -1, num_dim)

    return features


def main():
    args = parse_args()

    # --- 1. Setup output directory ---
    os.makedirs(args.output_dir, exist_ok=True)

    # --- 2. Build vision encoder ---
    dtype = get_dtype(args.dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = build_vision_encoder(
        encoder_name=args.encoder,
        model_name_or_path=args.model_name_or_path,
        dtype=dtype,
        device=device,
    )

    # --- 3. Build dataloader ---
    loader = make_dataloader(
        json_path=args.json_path,
        video_root_dir=args.video_root_dir,
        image_processor=encoder.get_image_processor(),
        num_frames=args.num_frames,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        video_key=args.video_key,
        id_key=args.id_key,
        force_sample=args.force_sample,
    )

    # --- 4. Compute output feature shape ---
    num_patches_per_side = encoder.num_patches_per_side  # 27 for SigLIP
    if args.spatial_pool_stride > 1:
        pooled_per_side = math.ceil(num_patches_per_side / args.spatial_pool_stride)
        pooled_patches = pooled_per_side * pooled_per_side
    else:
        pooled_patches = encoder.num_patches

    # --- 5. Save extraction config ---
    config = {
        "encoder": args.encoder,
        "model_name_or_path": args.model_name_or_path,
        "dtype": args.dtype,
        "num_frames": args.num_frames,
        "json_path": args.json_path,
        "video_root_dir": args.video_root_dir,
        "batch_size": args.batch_size,
        "save_format": args.save_format,
        "spatial_pool_stride": args.spatial_pool_stride,
        "spatial_pool_mode": args.spatial_pool_mode,
        "encoder_config": {
            "image_size": encoder.encoder_config.image_size,
            "patch_size": encoder.encoder_config.patch_size,
            "hidden_size": encoder.encoder_config.hidden_size,
            "num_patches_raw": encoder.num_patches,
            "num_patches_pooled": pooled_patches,
        },
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"\n[Config] Saved to {args.output_dir}/config.json")

    # --- 6. Extract features ---
    print(f"\n[Extract] Starting feature extraction...")
    print(f"  - Total videos: {len(loader.dataset)}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Num frames per video: {args.num_frames}")
    print(f"  - Spatial pool: stride={args.spatial_pool_stride}, mode={args.spatial_pool_mode}")
    print(f"  - Patches per frame: {encoder.num_patches} → {pooled_patches}")
    print(f"  - Output: {args.output_dir}")

    all_results = []  # for single file save mode
    total_videos = 0
    start_time = time.time()

    for batch in tqdm(loader, desc="Extracting features"):
        pixel_values = batch["pixel_values"]     # (total_frames, C, H, W)
        frame_counts = batch["frame_counts"]      # [num_frames_vid1, num_frames_vid2, ...]
        ids = batch["ids"]
        video_paths = batch["video_paths"]
        metadata = batch["metadata"]

        # Encode all frames in batch at once
        features = encoder.encode_images(pixel_values)  # (total_frames, num_patches, hidden_size)

        # Apply spatial pooling per video (each video may have different frame counts)
        if args.spatial_pool_stride > 1:
            features = spatial_pool_2d(
                features, num_patches_per_side,
                stride=args.spatial_pool_stride,
                mode=args.spatial_pool_mode,
            )
            # (total_frames, 729, 1152) → (total_frames, 196, 1152) with stride=2

        # Split features back to per-video
        per_video_features = torch.split(features, frame_counts, dim=0)

        # Save per video
        for i, (vid_id, vid_features, vid_path, meta) in enumerate(
            zip(ids, per_video_features, video_paths, metadata)
        ):
            result = {
                "features": vid_features.cpu(),   # (num_frames, num_patches, hidden_size)
                "id": vid_id,
                "video": vid_path,
                "num_frames": vid_features.shape[0],
                "metadata": meta,
            }

            if args.save_format == "per_video":
                save_path = os.path.join(args.output_dir, f"{vid_id}.pt")
                if args.skip_existing and os.path.exists(save_path):
                    continue
                torch.save(result, save_path)
            else:
                all_results.append(result)

            total_videos += 1

    # Save all at once if single mode
    if args.save_format == "single":
        save_path = os.path.join(args.output_dir, "all_features.pt")
        torch.save(all_results, save_path)
        print(f"\n[Save] All features saved to {save_path}")

    elapsed = time.time() - start_time
    print(f"\n[Done] Extracted features for {total_videos} videos in {elapsed:.1f}s")
    print(f"  - Speed: {total_videos / elapsed:.1f} videos/s")
    print(f"  - Feature shape per video: ({args.num_frames}, {pooled_patches}, {encoder.encoder_config.hidden_size})")
    print(f"  - Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
