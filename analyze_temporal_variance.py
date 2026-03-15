"""
Analyze temporal variance of extracted video features.

Measures how much per-frame features change across time within each video,
then aggregates statistics across the dataset.

Key insight: If a vision encoder "recognizes" real-world objects, it may encode
them similarly across frames (high cosine similarity, low temporal variance),
making it harder to distinguish temporal dynamics like direction of motion.

Metrics (all computed on spatially-pooled features: (T, D)):
    1. cosine_sim_consecutive: mean cosine similarity between adjacent frames
       → High = frames are similar = low temporal sensitivity
    2. cosine_sim_first_last: cosine similarity between first and last frame
       → High = start/end look the same to the encoder
    3. feature_std_normalized: std of L2-normalized features across time
       → Scale-invariant, comparable across datasets
    4. feature_std_raw: raw std across time (for reference, not for cross-dataset comparison)

Usage:
    cd extract_features
    python analyze_temporal_variance.py \
        --feature_dir /path/to/features_simple \
        --json_path /path/to/annotations.json \
        --dataset_name E2E_VP

    # Compare two datasets
    python analyze_temporal_variance.py \
        --feature_dir /path/to/features_simple \
        --feature_dir_2 /path/to/features_real \
        --json_path /path/to/annotations_simple.json \
        --json_path_2 /path/to/annotations_real.json \
        --dataset_name E2E_VP \
        --dataset_name_2 E2E_real_VP
        
        

"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def compute_temporal_metrics(features: torch.Tensor) -> dict:
    """
    Compute temporal variance metrics for a single video.

    Args:
        features: (num_frames, num_patches, hidden_size)

    Returns:
        dict of metrics
    """
    # Spatial mean pool: (T, num_patches, D) → (T, D)
    feat = features.float().mean(dim=1)  # (T, D)
    T, D = feat.shape

    if T < 2:
        return None

    # --- 1. L2-normalize for scale-invariant metrics ---
    feat_norm = F.normalize(feat, dim=-1)  # (T, D), unit vectors

    # --- 2. Cosine similarity between consecutive frames ---
    cos_consecutive = F.cosine_similarity(feat_norm[:-1], feat_norm[1:], dim=-1)  # (T-1,)
    mean_cos_consecutive = cos_consecutive.mean().item()

    # --- 3. Cosine similarity between first and last frame ---
    cos_first_last = F.cosine_similarity(
        feat_norm[0].unsqueeze(0), feat_norm[-1].unsqueeze(0), dim=-1
    ).item()

    # --- 4. Feature std (L2-normalized) across time ---
    #    std of unit vectors → purely directional variance
    std_normalized = feat_norm.std(dim=0).mean().item()

    # --- 5. Feature std (raw) across time ---
    std_raw = feat.std(dim=0).mean().item()

    # --- 6. Mean pairwise cosine distance across all frame pairs ---
    #    More robust than just consecutive
    cos_matrix = feat_norm @ feat_norm.T  # (T, T)
    # Upper triangle (excluding diagonal)
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    mean_cos_all_pairs = cos_matrix[mask].mean().item()

    # --- 7. Temporal gradient magnitude ---
    #    How much does the feature vector change per frame step
    diffs = feat_norm[1:] - feat_norm[:-1]  # (T-1, D)
    grad_magnitude = diffs.norm(dim=-1).mean().item()  # mean L2 distance per step

    return {
        "cos_sim_consecutive": mean_cos_consecutive,
        "cos_sim_first_last": cos_first_last,
        "cos_sim_all_pairs": mean_cos_all_pairs,
        "std_normalized": std_normalized,
        "std_raw": std_raw,
        "temporal_grad": grad_magnitude,
    }


def analyze_dataset(feature_dir: str, json_path: str, id_key: str = "id",
                    label_key: str = None) -> dict:
    """
    Analyze temporal variance across all videos in a dataset.

    Returns:
        dict with per-metric statistics (mean, std, per-class breakdown if label_key given)
    """
    with open(json_path, "r") as f:
        annotations = json.load(f)

    all_metrics = []
    per_class_metrics = defaultdict(list)
    skipped = 0

    for ann in tqdm(annotations, desc="Analyzing"):
        sample_id = ann[id_key]
        pt_path = os.path.join(feature_dir, f"{sample_id}.pt")
        if not os.path.exists(pt_path):
            skipped += 1
            continue

        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        features = data["features"]  # (T, num_patches, D)

        metrics = compute_temporal_metrics(features)
        if metrics is None:
            skipped += 1
            continue

        all_metrics.append(metrics)

        if label_key and label_key in ann:
            per_class_metrics[ann[label_key]].append(metrics)

    if skipped > 0:
        print(f"  Skipped {skipped} samples (missing .pt or T<2)")

    # Aggregate
    metric_keys = all_metrics[0].keys()
    summary = {}
    for key in metric_keys:
        values = [m[key] for m in all_metrics]
        summary[key] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
        }

    # Per-class breakdown
    per_class_summary = {}
    for cls, cls_metrics in per_class_metrics.items():
        per_class_summary[cls] = {}
        for key in metric_keys:
            values = [m[key] for m in cls_metrics]
            per_class_summary[cls][key] = {
                "mean": np.mean(values),
                "std": np.std(values),
            }

    return {
        "n_samples": len(all_metrics),
        "summary": summary,
        "per_class": per_class_summary,
    }


def print_report(name: str, result: dict):
    """Pretty print analysis results."""
    print(f"\n{'='*70}")
    print(f"  Dataset: {name} ({result['n_samples']} samples)")
    print(f"{'='*70}")

    summary = result["summary"]
    print(f"\n  {'Metric':<25} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print(f"  {'-'*65}")
    for key, stats in summary.items():
        print(f"  {key:<25} {stats['mean']:>10.4f} {stats['std']:>10.4f} "
              f"{stats['min']:>10.4f} {stats['max']:>10.4f}")

    if result["per_class"]:
        print(f"\n  Per-class breakdown (mean):")
        print(f"  {'-'*65}")
        classes = sorted(result["per_class"].keys())
        header = f"  {'Class':<15}"
        for key in summary.keys():
            short_key = key.replace("cos_sim_", "cos_").replace("consecutive", "consec")
            header += f" {short_key:>10}"
        print(header)

        for cls in classes:
            row = f"  {str(cls):<15}"
            for key in summary.keys():
                row += f" {result['per_class'][cls][key]['mean']:>10.4f}"
            print(row)


def print_comparison(name1: str, result1: dict, name2: str, result2: dict):
    """Print side-by-side comparison of two datasets."""
    print(f"\n{'='*70}")
    print(f"  Comparison: {name1} vs {name2}")
    print(f"{'='*70}")

    s1, s2 = result1["summary"], result2["summary"]
    print(f"\n  {'Metric':<25} {name1:>12} {name2:>12} {'Δ (2-1)':>12}")
    print(f"  {'-'*65}")
    for key in s1.keys():
        m1, m2 = s1[key]["mean"], s2[key]["mean"]
        delta = m2 - m1
        print(f"  {key:<25} {m1:>12.4f} {m2:>12.4f} {delta:>+12.4f}")

    print(f"\n  Interpretation:")
    cos_diff = s2["cos_sim_consecutive"]["mean"] - s1["cos_sim_consecutive"]["mean"]
    if cos_diff > 0.01:
        print(f"  → {name2} has HIGHER frame similarity (Δcos={cos_diff:+.4f})")
        print(f"    Features change LESS across frames → lower temporal sensitivity")
    elif cos_diff < -0.01:
        print(f"  → {name2} has LOWER frame similarity (Δcos={cos_diff:+.4f})")
        print(f"    Features change MORE across frames → higher temporal sensitivity")
    else:
        print(f"  → Similar frame similarity between datasets (Δcos={cos_diff:+.4f})")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze temporal variance of extracted features")

    # Dataset 1 (required)
    parser.add_argument("--feature_dir", type=str, required=True)
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="Dataset1")

    # Dataset 2 (optional, for comparison)
    parser.add_argument("--feature_dir_2", type=str, default=None)
    parser.add_argument("--json_path_2", type=str, default=None)
    parser.add_argument("--dataset_name_2", type=str, default="Dataset2")

    # Options
    parser.add_argument("--id_key", type=str, default="id")
    parser.add_argument("--label_key", type=str, default=None,
                        help="JSON key for class label (for per-class breakdown)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Save results as JSON. Defaults to feature_dir/temporal_analysis/")

    return parser.parse_args()


def main():
    args = parse_args()

    # --- Analyze dataset 1 ---
    print(f"\n[Analyzing] {args.dataset_name}...")
    result1 = analyze_dataset(
        args.feature_dir, args.json_path,
        id_key=args.id_key, label_key=args.label_key,
    )
    print_report(args.dataset_name, result1)

    # --- Analyze dataset 2 (if provided) ---
    result2 = None
    if args.feature_dir_2 and args.json_path_2:
        print(f"\n[Analyzing] {args.dataset_name_2}...")
        result2 = analyze_dataset(
            args.feature_dir_2, args.json_path_2,
            id_key=args.id_key, label_key=args.label_key,
        )
        print_report(args.dataset_name_2, result2)
        print_comparison(args.dataset_name, result1, args.dataset_name_2, result2)

    # --- Save ---
    output_dir = args.output_dir or os.path.join(args.feature_dir, "temporal_analysis")
    os.makedirs(output_dir, exist_ok=True)

    # Convert numpy to python floats for JSON serialization
    def to_serializable(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        return obj

    save_data = {
        args.dataset_name: to_serializable(result1),
    }
    if result2:
        save_data[args.dataset_name_2] = to_serializable(result2)

    save_path = os.path.join(output_dir, "temporal_variance.json")
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n[Save] Results → {save_path}")


if __name__ == "__main__":
    main()
