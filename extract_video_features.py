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
                        help="Vision encoder name (e.g., siglip, clip, llava)")
    parser.add_argument("--model_name_or_path", type=str, default="google/siglip-so400m-patch14-384",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"],
                        help="Model dtype")

    # LLaVA-specific
    parser.add_argument("--llava_model_name", type=str, default="llava_qwen",
                        help="LLaVA model name for builder (e.g., llava_qwen, llava_llama, llava_mistral). "
                             "Only used when --encoder=llava")
    parser.add_argument("--attn_implementation", type=str, default="sdpa",
                        choices=["sdpa", "flash_attention_2", "eager"],
                        help="Attention implementation for LLaVA model loading. Only used when --encoder=llava")

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

    # Debug
    parser.add_argument("--debug", action="store_true",
                        help="Print encoder spec, extraction layer info, and run a dummy forward pass to verify")

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
    os.makedirs(os.path.join(args.output_dir,'features'), exist_ok=True)

    # --- 2. Build vision encoder ---
    dtype = get_dtype(args.dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prepare encoder kwargs
    encoder_kwargs = dict(dtype=dtype, device=device)
    if args.encoder == "llava":
        encoder_kwargs["llava_model_name"] = args.llava_model_name
        encoder_kwargs["attn_implementation"] = args.attn_implementation

    encoder = build_vision_encoder(
        encoder_name=args.encoder,
        model_name_or_path=args.model_name_or_path,
        **encoder_kwargs,
    )

    # --- 2.1 Debug: print encoder spec & verify extraction layer ---
    if args.debug:
        print("\n" + "=" * 60)
        print(" [DEBUG] Vision Encoder Spec")
        print("=" * 60)
        debug_info = encoder.get_debug_info()
        for k, v in debug_info.items():
            print(f"  {k}: {v}")

        # Dummy forward pass with hooks to verify which layer activations are used
        print("\n" + "-" * 60)
        print(" [DEBUG] Dummy forward pass — verifying extraction layer")
        print("-" * 60)

        img_size = encoder.encoder_config.image_size
        dummy_input = torch.randn(8, 3, img_size, img_size, device=device, dtype=dtype)

        hook_records = []

        def _get_hook(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    out = output[0]
                else:
                    out = output
                hook_records.append({
                    "layer_idx": layer_idx,
                    "output_shape": tuple(out.shape),
                    "output_norm": out.float().norm().item(),
                })
            return hook_fn

        # Register hooks on encoder layers
        handles = []
        if args.encoder == "clip":
            layers = encoder.model.vision_model.encoder.layers
        elif args.encoder == "siglip":
            layers = encoder.model.vision_model.encoder.layers
        elif args.encoder == "llava":
            layers = encoder.vision_tower.vision_tower.vision_model.encoder.layers
        else:
            layers = []

        for idx, layer in enumerate(layers):
            h = layer.register_forward_hook(_get_hook(idx))
            handles.append(h)

        # Run dummy forward
        with torch.no_grad():
            dummy_out = encoder.encode_images(dummy_input)

        # Remove hooks
        for h in handles:
            h.remove()

        # Print layer activation summary
        total_hooked = len(hook_records)
        print(f"  Encoder layers fired: {total_hooked}")
        print(f"  Output feature shape: {tuple(dummy_out.shape)}")
        print(f"  Output feature norm:  {dummy_out.float().norm().item():.4f}")
        print()

        # Show last few layers to highlight which one is extracted
        show_n = min(5, total_hooked)
        print(f"  Last {show_n} layer activations:")
        for rec in hook_records[-show_n:]:
            marker = " ← extracted" if rec["layer_idx"] == total_hooked - 1 else ""
            print(f"    layer {rec['layer_idx']:3d} | shape: {rec['output_shape']} | norm: {rec['output_norm']:.4f}{marker}")

        # Verify output matches last hooked layer
        if hook_records:
            last_norm = hook_records[-1]["output_norm"]
            out_norm = dummy_out.float().norm().item()
            # For CLIP, norms differ because CLS is removed; skip exact match check
            if args.encoder != "clip":
                match_str = "MATCH" if abs(last_norm - out_norm) < 1.0 else "DIFFER (projector/head may be applied)"
                print(f"\n  Last layer norm vs output norm: {match_str}")

        print("=" * 60 + "\n")

        del dummy_input, dummy_out, hook_records

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

        # For CLIP: also retrieve CLS tokens (cached during encode_images)
        cls_tokens = None
        if hasattr(encoder, 'get_last_cls_tokens'):
            cls_tokens = encoder.get_last_cls_tokens()  # (total_frames, 1, D)
            cls_tokens = cls_tokens.squeeze(1)            # (total_frames, D)

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

        # Split CLS tokens per-video (only for CLIP)
        if cls_tokens is not None:
            per_video_cls = torch.split(cls_tokens, frame_counts, dim=0)
        else:
            per_video_cls = [None] * len(ids)

        # Save per video
        for i, (vid_id, vid_features, vid_path, meta, vid_cls) in enumerate(
            zip(ids, per_video_features, video_paths, metadata, per_video_cls)
        ):
            result = {
                "features": vid_features.cpu(),   # (num_frames, num_patches, hidden_size)
                "id": vid_id,
                "video": vid_path,
                "num_frames": vid_features.shape[0],
                "metadata": meta,
            }
            # Save CLS tokens separately for temporal representation experiments
            if vid_cls is not None:
                result["cls_tokens"] = vid_cls.cpu()  # (num_frames, hidden_size)
            if args.save_format == "per_video":
                save_path = os.path.join(args.output_dir,'features', f"{vid_id}.pt")
                if args.skip_existing and os.path.exists(save_path):
                    continue
                torch.save(result, save_path)
            else:
                all_results.append(result)

            total_videos += 1

    # Save all at once if single mode
    if args.save_format == "single":
        save_path = os.path.join(args.output_dir, 'features',"all_features.pt")
        torch.save(all_results, save_path)
        print(f"\n[Save] All features saved to {save_path}")

    elapsed = time.time() - start_time
    print(f"\n[Done] Extracted features for {total_videos} videos in {elapsed:.1f}s")
    print(f"  - Speed: {total_videos / elapsed:.1f} videos/s")
    print(f"  - Feature shape per video: ({args.num_frames}, {pooled_patches}, {encoder.encoder_config.hidden_size})")
    print(f"  - Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
