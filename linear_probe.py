"""
Linear probing on pre-extracted vision encoder features.

Usage (random split - single JSON):
    python linear_probe.py \
        --feature_dir /path/to/extracted_features \
        --json_path /path/to/annotations.json \
        --label_key answer \
        --pool_mode mean \
        --train_ratio 0.8 \
        --seed 42

Usage (pre-split - separate train/val JSONs):
    python linear_probe.py \
        --feature_dir /path/to/extracted_features \
        --train_json /path/to/train.json \
        --val_json /path/to/val.json \
        --label_key answer \
        --pool_mode mean

Pooling modes:
    mean:   (num_frames, num_patches, D) → mean over frames & patches → (D,)
    concat: (num_frames, num_patches, D) → mean over patches, concat frames → (num_frames * D,)
             ※ preserves temporal order but feature dim gets large
    cls:    CLS tokens (num_frames, D) → concat → (num_frames * D,)
             ※ CLIP only — uses CLS token as per-frame representation for temporal concat
    concat_full: (num_frames, num_patches, D) → flatten → (num_frames * num_patches * D,)
             ※ spatial mean 없이 전체 patch를 temporal concat. dim이 크므로 spatial_pool_stride 필요
"""

import argparse
import json
import os
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm


# ============================================================
# Dataset
# ============================================================

class FeatureDataset(Dataset):
    """
    Loads pre-extracted .pt features and maps labels from JSON annotations.

    Each .pt file contains:
        {"features": (num_frames, num_patches, hidden), "id": ..., "metadata": {...}}
    """

    def __init__(
        self,
        feature_dir: str,
        annotations: list,
        label_to_idx: dict,
        label_key: str = "answer",
        pool_mode: str = "mean",
        id_key: str = "id",
    ):
        self.feature_dir = feature_dir
        self.annotations = annotations
        self.label_to_idx = label_to_idx
        self.label_key = label_key
        self.pool_mode = pool_mode
        self.id_key = id_key

    def __len__(self):
        return len(self.annotations)

    def _pool_features(self, features: torch.Tensor, cls_tokens: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            features: (num_frames, num_patches, hidden_size)
            cls_tokens: (num_frames, hidden_size) — only for pool_mode="cls" (CLIP only)

        Returns:
            pooled: (feature_dim,)
                mean:   (hidden_size,)
                concat: (num_frames * hidden_size,)
                cls:    (num_frames * hidden_size,)
        """
        if self.pool_mode == "mean":
            # (num_frames, num_patches, D) → (D,)
            return features.mean(dim=0).mean(dim=0)

        elif self.pool_mode == "concat":
            # (num_frames, num_patches, D) → spatial mean → (num_frames, D) → flatten → (num_frames * D,)
            return features.mean(dim=1).flatten()

        elif self.pool_mode == "cls":
            # CLS token concat: (num_frames, D) → flatten → (num_frames * D,)
            assert cls_tokens is not None, (
                "pool_mode='cls' requires 'cls_tokens' in .pt file. "
                "This is only available for CLIP features."
            )
            return cls_tokens.flatten()

        elif self.pool_mode == "concat_full":
            # (num_frames, num_patches, D) → flatten → (num_frames * num_patches * D,)
            return features.flatten()

        else:
            raise ValueError(f"Unknown pool_mode: {self.pool_mode}")

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        sample_id = ann[self.id_key]

        # Load .pt
        pt_path = os.path.join(self.feature_dir, f"{sample_id}.pt")
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        features = data["features"]  # (num_frames, num_patches, hidden)

        # CLS tokens (only present for CLIP features)
        cls_tokens = data.get("cls_tokens", None)  # (num_frames, D) or None
        if cls_tokens is not None:
            cls_tokens = cls_tokens.float()

        # Pool
        pooled = self._pool_features(features.float(), cls_tokens=cls_tokens)

        # Label
        label = self.label_to_idx[ann[self.label_key]]

        return pooled, label


# ============================================================
# Train / Eval
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        logits = model(features)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        logits = model(features)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    total = len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / total, acc, all_preds, all_labels


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Linear probing on extracted features")

    # Data
    parser.add_argument("--feature_dir", type=str, required=True,
                        help="Directory containing per-video .pt files")
    parser.add_argument("--json_path", type=str, default=None,
                        help="Path to single JSON annotation file (random split mode)")
    parser.add_argument("--train_json", type=str, default=None,
                        help="Path to train JSON annotation file (pre-split mode)")
    parser.add_argument("--val_json", type=str, default=None,
                        help="Path to val JSON annotation file (pre-split mode)")
    parser.add_argument("--label_key", type=str, default="answer",
                        help="JSON key for class label (e.g., 'answer', 'direction', 'answer_text')")
    parser.add_argument("--id_key", type=str, default="id",
                        help="JSON key for sample ID")

    # Pooling
    parser.add_argument("--pool_mode", type=str, default="mean",
                        choices=["mean", "concat", "cls", "concat_full"],
                        help="mean: temporal+spatial mean → (D,). "
                             "concat: spatial mean then concat frames → (num_frames*D,). "
                             "cls: concat CLS tokens across frames → (num_frames*D,) (CLIP only). "
                             "concat_full: no spatial mean, full flatten → (num_frames*num_patches*D,)")

    # Split
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Train split ratio. Rest goes to val.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    # Training
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")

    # Output
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save results. Defaults to feature_dir/probe_results/")

    return parser.parse_args()


def main():
    args = parse_args()

    # --- Seed ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device if torch.cuda.is_available() else "cpu"

    # --- Load annotations ---
    def filter_annotations(annotations, feature_dir, id_key):
        """Filter annotations to only keep those with corresponding .pt files."""
        valid = []
        for ann in annotations:
            pt_path = os.path.join(feature_dir, f"{ann[id_key]}.pt")
            if os.path.exists(pt_path):
                valid.append(ann)
        return valid

    if args.train_json and args.val_json:
        # Pre-split mode: train/val JSON files provided separately
        with open(args.train_json, "r") as f:
            train_raw = json.load(f)
        with open(args.val_json, "r") as f:
            val_raw = json.load(f)

        train_anns = filter_annotations(train_raw, args.feature_dir, args.id_key)
        val_anns = filter_annotations(val_raw, args.feature_dir, args.id_key)
        valid_annotations = train_anns + val_anns

        print(f"[Data] Train: {len(train_anns)}/{len(train_raw)}, "
              f"Val: {len(val_anns)}/{len(val_raw)} samples have extracted features")
        print(f"[Split] Pre-split mode (--train_json / --val_json)")

    elif args.json_path:
        # Random split mode: single JSON + train_ratio
        with open(args.json_path, "r") as f:
            annotations = json.load(f)

        valid_annotations = filter_annotations(annotations, args.feature_dir, args.id_key)
        print(f"[Data] {len(valid_annotations)}/{len(annotations)} samples have extracted features")

        indices = list(range(len(valid_annotations)))
        random.shuffle(indices)
        split_idx = int(len(indices) * args.train_ratio)
        train_anns = [valid_annotations[i] for i in indices[:split_idx]]
        val_anns = [valid_annotations[i] for i in indices[split_idx:]]
        print(f"[Split] Random split (train_ratio={args.train_ratio})")

    else:
        raise ValueError("Provide either --json_path (random split) or --train_json + --val_json (pre-split)")

    # --- Build label mapping ---
    all_labels = [ann[args.label_key] for ann in valid_annotations]
    label_set = sorted(set(all_labels))
    label_to_idx = {label: idx for idx, label in enumerate(label_set)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    num_classes = len(label_set)

    print(f"[Labels] {num_classes} classes: {label_set}")
    print(f"[Labels] Distribution: {dict(Counter(all_labels))}")
    print(f"[Split] Train: {len(train_anns)}, Val: {len(val_anns)}")

    # --- Determine feature dim from first sample ---
    sample_pt = torch.load(
        os.path.join(args.feature_dir, f"{valid_annotations[0][args.id_key]}.pt"),
        map_location="cpu", weights_only=False,
    )
    sample_feat = sample_pt["features"]  # (num_frames, num_patches, hidden)
    num_frames, num_patches, hidden_size = sample_feat.shape

    if args.pool_mode == "mean":
        feature_dim = hidden_size
    elif args.pool_mode in ("concat", "cls"):
        feature_dim = num_frames * hidden_size
    elif args.pool_mode == "concat_full":
        feature_dim = num_frames * num_patches * hidden_size
    else:
        raise ValueError(f"Unknown pool_mode: {args.pool_mode}")

    print(f"\n[Model] Feature shape from .pt: ({num_frames}, {num_patches}, {hidden_size})")
    print(f"[Model] Pool mode: {args.pool_mode} → feature_dim: {feature_dim}")
    print(f"[Model] Linear: {feature_dim} → {num_classes}")

    # --- Datasets & Loaders ---
    train_dataset = FeatureDataset(
        args.feature_dir, train_anns, label_to_idx,
        label_key=args.label_key, pool_mode=args.pool_mode, id_key=args.id_key,
    )
    val_dataset = FeatureDataset(
        args.feature_dir, val_anns, label_to_idx,
        label_key=args.label_key, pool_mode=args.pool_mode, id_key=args.id_key,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # --- Linear probe model ---
    model = nn.Linear(feature_dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # --- Training loop ---
    print(f"\n[Train] Starting linear probing (epochs={args.epochs}, lr={args.lr})")
    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion, device)

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = model.state_dict().copy()
            best_preds = val_preds
            best_labels = val_labels

        if epoch % 10 == 0 or epoch == 1 or is_best:
            marker = " *" if is_best else ""
            print(f"  Epoch {epoch:3d} | "
                  f"Train loss: {train_loss:.4f} acc: {train_acc:.4f} | "
                  f"Val loss: {val_loss:.4f} acc: {val_acc:.4f}{marker}")

    # --- Last epoch results ---
    last_val_loss, last_val_acc, last_val_preds, last_val_labels = evaluate(
        model, val_loader, criterion, device
    )
    last_state = model.state_dict().copy()

    # --- Final report ---
    target_names = [idx_to_label[i] for i in range(num_classes)]

    print(f"\n{'='*60}")
    print(f"[Result] Last val acc: {last_val_acc:.4f} (epoch {args.epochs})")
    print(f"[Result] Best val acc: {best_val_acc:.4f} (epoch {best_epoch})")
    print(f"{'='*60}")

    last_report_str = classification_report(last_val_labels, last_val_preds, target_names=target_names)
    last_report_dict = classification_report(last_val_labels, last_val_preds, target_names=target_names, output_dict=True)
    best_report_str = classification_report(best_labels, best_preds, target_names=target_names)
    best_report_dict = classification_report(best_labels, best_preds, target_names=target_names, output_dict=True)

    print(f"\n--- Last epoch (epoch {args.epochs}) ---")
    print(last_report_str)
    print(f"--- Best epoch (epoch {best_epoch}) ---")
    print(best_report_str)

    # --- Save results ---
    output_dir = args.output_dir or os.path.join(os.path.dirname(args.feature_dir), "probe_results",args.pool_mode)
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "last_val_acc": last_val_acc,
        "last_epoch": args.epochs,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "num_classes": num_classes,
        "label_mapping": label_to_idx,
        "last_classification_report": last_report_dict,
        "best_classification_report": best_report_dict,
        "pool_mode": args.pool_mode,
        "feature_dim": feature_dim,
        "feature_shape": [num_frames, num_patches, hidden_size],
        "train_size": len(train_anns),
        "val_size": len(val_anns),
        "split_mode": "pre-split" if args.train_json else "random",
        "train_ratio": args.train_ratio if args.json_path else None,
        "lr": args.lr,
        "epochs": args.epochs,
        "seed": args.seed,
        "args": vars(args),
    }

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    torch.save(last_state, os.path.join(output_dir, "last_linear.pt"))
    torch.save(best_state, os.path.join(output_dir, "best_linear.pt"))

    print(f"\n[Save] Results     → {output_dir}/results.json")
    print(f"[Save] Last model  → {output_dir}/last_linear.pt")
    print(f"[Save] Best model  → {output_dir}/best_linear.pt")


if __name__ == "__main__":
    main()
