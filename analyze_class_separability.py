"""
Analyze class separability of extracted video features.

Measures how well-separated the classes are in feature space.
Helps diagnose WHY linear probing accuracy differs between datasets.

Metrics:
    1. intra_class_cos_sim: avg cosine similarity WITHIN each class
       → High = class is tight/consistent, Low = class is spread out
    2. inter_class_cos_sim: avg cosine similarity BETWEEN classes
       → High = classes overlap, Low = classes are separated
    3. separability_gap: intra - inter
       → Higher = easier to classify
    4. fisher_ratio: (inter-class variance) / (intra-class variance)
       → Higher = more separable (classic Fisher criterion)
    5. centroid_distance: cosine distance between class centroids
       → Higher = centroids are far apart

Usage:
    cd extract_features

    # Single dataset
    python analyze_class_separability.py \
        --feature_dir /path/to/features \
        --json_path /path/to/annotations.json \
        --label_key answer \
        --pool_mode concat \
        --dataset_name E2E_VP

    # Compare two datasets
    python analyze_class_separability.py \
        --feature_dir /path/to/features_simple \
        --json_path /path/to/annotations_simple.json \
        --feature_dir_2 /path/to/features_real \
        --json_path_2 /path/to/annotations_real.json \
        --label_key answer \
        --pool_mode concat \
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


def load_features_by_class(feature_dir: str, json_path: str,
                           label_key: str, id_key: str,
                           pool_mode: str) -> dict:
    """
    Load all features grouped by class label.

    Returns:
        {class_label: Tensor of shape (N_class, feature_dim)}
    """
    with open(json_path, "r") as f:
        annotations = json.load(f)

    class_features = defaultdict(list)

    for ann in tqdm(annotations, desc="Loading features"):
        sample_id = ann[id_key]
        pt_path = os.path.join(feature_dir, f"{sample_id}.pt")
        if not os.path.exists(pt_path):
            continue

        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        feat = data["features"].float()  # (T, num_patches, D)

        # Spatial mean pool: (T, num_patches, D) → (T, D)
        feat = feat.mean(dim=1)

        if pool_mode == "mean":
            pooled = feat.mean(dim=0)  # (D,)
        elif pool_mode == "concat":
            pooled = feat.flatten()  # (T*D,)
        else:
            raise ValueError(f"Unknown pool_mode: {pool_mode}")

        label = ann[label_key]
        class_features[label].append(pooled)

    # Stack into tensors
    for label in class_features:
        class_features[label] = torch.stack(class_features[label])

    return dict(class_features)


def compute_separability(class_features: dict) -> dict:
    """
    Compute class separability metrics.

    Args:
        class_features: {label: Tensor (N, D)}

    Returns:
        dict of metrics
    """
    labels = sorted(class_features.keys())
    num_classes = len(labels)

    # L2-normalize all features
    normalized = {}
    for label in labels:
        normalized[label] = F.normalize(class_features[label], dim=-1)

    # --- 1. Intra-class cosine similarity ---
    intra_cos = {}
    for label in labels:
        feats = normalized[label]  # (N, D)
        N = feats.shape[0]
        if N < 2:
            intra_cos[label] = 1.0
            continue
        cos_matrix = feats @ feats.T  # (N, N)
        mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
        intra_cos[label] = cos_matrix[mask].mean().item()

    mean_intra = np.mean(list(intra_cos.values()))

    # --- 2. Inter-class cosine similarity ---
    inter_cos_pairs = {}
    for i, l1 in enumerate(labels):
        for j, l2 in enumerate(labels):
            if j <= i:
                continue
            cross = normalized[l1] @ normalized[l2].T  # (N1, N2)
            inter_cos_pairs[f"{l1}_vs_{l2}"] = cross.mean().item()

    mean_inter = np.mean(list(inter_cos_pairs.values()))

    # --- 3. Separability gap ---
    separability_gap = mean_intra - mean_inter

    # --- 4. Centroid distance ---
    centroids = {}
    for label in labels:
        centroids[label] = F.normalize(class_features[label].mean(dim=0), dim=0)

    centroid_cos_pairs = {}
    for i, l1 in enumerate(labels):
        for j, l2 in enumerate(labels):
            if j <= i:
                continue
            cos = F.cosine_similarity(
                centroids[l1].unsqueeze(0), centroids[l2].unsqueeze(0)
            ).item()
            centroid_cos_pairs[f"{l1}_vs_{l2}"] = cos

    mean_centroid_cos = np.mean(list(centroid_cos_pairs.values()))
    centroid_distance = 1.0 - mean_centroid_cos  # cosine distance

    # --- 5. Fisher ratio (in feature space) ---
    # inter-class variance / intra-class variance
    all_feats = torch.cat([class_features[l] for l in labels], dim=0)
    global_mean = all_feats.mean(dim=0)

    # Between-class scatter (weighted by class size)
    between_var = 0.0
    total_n = all_feats.shape[0]
    for label in labels:
        n_k = class_features[label].shape[0]
        class_mean = class_features[label].mean(dim=0)
        between_var += (n_k / total_n) * ((class_mean - global_mean) ** 2).sum().item()

    # Within-class scatter (weighted by class size)
    within_var = 0.0
    for label in labels:
        n_k = class_features[label].shape[0]
        class_mean = class_features[label].mean(dim=0)
        diffs = class_features[label] - class_mean.unsqueeze(0)
        within_var += (n_k / total_n) * (diffs ** 2).sum(dim=-1).mean().item()

    fisher_ratio = between_var / (within_var + 1e-8)

    # --- 6. Per-class stats ---
    per_class = {}
    for label in labels:
        feats = class_features[label]
        per_class[label] = {
            "n_samples": feats.shape[0],
            "intra_cos_sim": intra_cos[label],
            "feature_norm_mean": feats.norm(dim=-1).mean().item(),
            "feature_norm_std": feats.norm(dim=-1).std().item(),
        }

    return {
        "num_classes": num_classes,
        "class_labels": labels,
        "mean_intra_cos_sim": mean_intra,
        "mean_inter_cos_sim": mean_inter,
        "separability_gap": separability_gap,
        "centroid_distance": centroid_distance,
        "centroid_cos_sim": mean_centroid_cos,
        "fisher_ratio": fisher_ratio,
        "between_class_var": between_var,
        "within_class_var": within_var,
        "inter_cos_pairs": inter_cos_pairs,
        "centroid_cos_pairs": centroid_cos_pairs,
        "per_class": per_class,
    }


def print_report(name: str, result: dict):
    print(f"\n{'='*70}")
    print(f"  Dataset: {name}")
    print(f"{'='*70}")

    print(f"\n  Classes: {result['class_labels']}")
    for label, stats in result["per_class"].items():
        print(f"    {label}: {stats['n_samples']} samples, "
              f"intra_cos={stats['intra_cos_sim']:.4f}, "
              f"norm={stats['feature_norm_mean']:.2f}±{stats['feature_norm_std']:.2f}")

    print(f"\n  {'Metric':<30} {'Value':>12}")
    print(f"  {'-'*45}")
    print(f"  {'Intra-class cos sim':<30} {result['mean_intra_cos_sim']:>12.4f}")
    print(f"  {'Inter-class cos sim':<30} {result['mean_inter_cos_sim']:>12.4f}")
    print(f"  {'Separability gap (intra-inter)':<30} {result['separability_gap']:>12.4f}")
    print(f"  {'Centroid cos sim':<30} {result['centroid_cos_sim']:>12.4f}")
    print(f"  {'Centroid distance (1-cos)':<30} {result['centroid_distance']:>12.4f}")
    print(f"  {'Fisher ratio (between/within)':<30} {result['fisher_ratio']:>12.4f}")
    print(f"  {'Between-class variance':<30} {result['between_class_var']:>12.4f}")
    print(f"  {'Within-class variance':<30} {result['within_class_var']:>12.4f}")


def print_comparison(name1: str, r1: dict, name2: str, r2: dict):
    print(f"\n{'='*70}")
    print(f"  Comparison: {name1} vs {name2}")
    print(f"{'='*70}")

    metrics = [
        ("Intra-class cos sim", "mean_intra_cos_sim"),
        ("Inter-class cos sim", "mean_inter_cos_sim"),
        ("Separability gap", "separability_gap"),
        ("Centroid distance", "centroid_distance"),
        ("Fisher ratio", "fisher_ratio"),
        ("Between-class var", "between_class_var"),
        ("Within-class var", "within_class_var"),
    ]

    print(f"\n  {'Metric':<30} {name1:>12} {name2:>12} {'Δ (2-1)':>12}")
    print(f"  {'-'*70}")
    for label, key in metrics:
        v1, v2 = r1[key], r2[key]
        print(f"  {label:<30} {v1:>12.4f} {v2:>12.4f} {v2 - v1:>+12.4f}")

    print(f"\n  Interpretation:")
    gap_diff = r2["separability_gap"] - r1["separability_gap"]
    fisher_diff = r2["fisher_ratio"] - r1["fisher_ratio"]

    if gap_diff < 0:
        print(f"  → {name2} has LOWER separability gap ({gap_diff:+.4f})")
        print(f"    Classes are harder to separate → explains lower LP accuracy")
    else:
        print(f"  → {name2} has HIGHER separability gap ({gap_diff:+.4f})")

    intra_diff = r2["within_class_var"] - r1["within_class_var"]
    if intra_diff > 0:
        print(f"  → {name2} has HIGHER within-class variance ({intra_diff:+.4f})")
        print(f"    Samples within same class are more spread out (diverse objects?)")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze class separability of extracted features")

    parser.add_argument("--feature_dir", type=str, required=True)
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="Dataset1")

    parser.add_argument("--feature_dir_2", type=str, default=None)
    parser.add_argument("--json_path_2", type=str, default=None)
    parser.add_argument("--dataset_name_2", type=str, default="Dataset2")

    parser.add_argument("--label_key", type=str, default="answer")
    parser.add_argument("--id_key", type=str, default="id")
    parser.add_argument("--pool_mode", type=str, default="concat",
                        choices=["mean", "concat"],
                        help="Feature pooling before analysis")
    parser.add_argument("--output_dir", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    # --- Dataset 1 ---
    print(f"\n[Loading] {args.dataset_name}...")
    cf1 = load_features_by_class(
        args.feature_dir, args.json_path,
        args.label_key, args.id_key, args.pool_mode,
    )
    r1 = compute_separability(cf1)
    print_report(args.dataset_name, r1)

    # --- Dataset 2 ---
    r2 = None
    if args.feature_dir_2 and args.json_path_2:
        print(f"\n[Loading] {args.dataset_name_2}...")
        cf2 = load_features_by_class(
            args.feature_dir_2, args.json_path_2,
            args.label_key, args.id_key, args.pool_mode,
        )
        r2 = compute_separability(cf2)
        print_report(args.dataset_name_2, r2)
        print_comparison(args.dataset_name, r1, args.dataset_name_2, r2)

    # --- Save ---
    output_dir = args.output_dir or os.path.join(args.feature_dir, "separability_analysis")
    os.makedirs(output_dir, exist_ok=True)

    def to_serializable(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        return obj

    save_data = {args.dataset_name: to_serializable(r1)}
    if r2:
        save_data[args.dataset_name_2] = to_serializable(r2)

    save_path = os.path.join(output_dir, "class_separability.json")
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n[Save] Results → {save_path}")


if __name__ == "__main__":
    main()
