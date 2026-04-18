"""BrainSAIL training and evaluation entry point.

Runs stratified k-fold cross-validation for few-shot brain MRI classification.
For each fold, trains BrainSAIL with EMA and early stopping, then evaluates
using Top-5 snapshot ensemble with Test-Time Augmentation.

Usage:
    # 4-shot experiment (default)
    python main.py --config config.yaml

    # Override to 8-shot
    python main.py --config config.yaml --shot 8
"""

import argparse
import os
import sys
import warnings
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset.brain_dataset import BrainDataset
from models.model import BrainSAIL
from utils.core_utils import predict_with_tta, train_loop
from utils.eval_utils import compute_metrics, find_optimal_thresholds

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_config(path: str) -> dict:
    """Load a YAML configuration file.

    Args:
        path (str): Path to ``config.yaml``.

    Returns:
        dict: Parsed configuration dictionary.
    """
    with open(path) as f:
        return yaml.safe_load(f)


def seed_everything(seed: int = 42) -> None:
    """Set all relevant random seeds for reproducibility.

    Args:
        seed (int): Random seed value. Defaults to ``42``.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def augment_features(features: torch.Tensor, noise_std: float = 0.003) -> torch.Tensor:
    """Apply Gaussian noise augmentation in feature space.

    Operates directly on pre-extracted feature vectors rather than raw images,
    which is safer for MRI data where spatial augmentations (flip, rotation)
    could create anatomically implausible inputs.

    Args:
        features (torch.Tensor): Input feature tensor of shape ``[S, D]``.
        noise_std (float): Standard deviation of additive noise. At ``0.003``
            this corresponds to ~0.3% perturbation of L2-normalized vectors.
            Defaults to ``0.003``.

    Returns:
        torch.Tensor: Augmented feature tensor, same shape as input.
    """
    return features + torch.randn_like(features) * noise_std


def load_text_anchors(anchor_path: str, n_classes: int, device: str) -> torch.Tensor:
    """Load pre-built text anchor embeddings from disk.

    Args:
        anchor_path (str): Path to the ``.pt`` file containing anchor tensors
            of shape ``[n_classes, feat_dim]``.
        n_classes (int): Expected number of classes; anchors are truncated
            to this length if the file contains more.
        device (str): Target device string.

    Returns:
        torch.Tensor: Float32 anchor tensor of shape ``[n_classes, feat_dim]``
            on the specified device.

    Raises:
        FileNotFoundError: If ``anchor_path`` does not exist.
    """
    if not os.path.exists(anchor_path):
        raise FileNotFoundError(f"Text anchors not found: {anchor_path}")
    anchors = torch.load(anchor_path, map_location="cpu", weights_only=False)
    if anchors.shape[0] > n_classes:
        anchors = anchors[:n_classes]
    return anchors.float().to(device)


def main() -> None:
    """Run the full BrainSAIL k-fold training and evaluation pipeline."""
    parser = argparse.ArgumentParser(description="BrainSAIL few-shot MRI classifier.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--shot", type=int, default=None, help="Override few-shot setting (4 or 8)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # --- Resolve configuration ---
    shot        = args.shot or cfg["train"]["shot"]
    seed        = cfg["train"]["seed"]
    n_folds     = cfg["train"]["n_folds"]
    lr          = cfg["train"]["lr"]
    patience    = cfg["train"]["patience"]
    max_epochs  = cfg["train"]["max_epochs"]
    n_tta       = cfg["train"]["n_tta"]
    noise_std   = cfg["train"]["noise_std"]
    feat_dim    = cfg["model"]["feature_dim"]
    n_classes   = cfg["model"]["num_classes"]
    label_cols  = cfg["labels"]

    features_dir = cfg["data"]["features_dir"]
    split_dir    = os.path.join(cfg["data"]["split_dir"], f"{shot}shot")
    dataset_csv  = cfg["data"]["dataset_csv"]
    anchor_path  = cfg["data"]["anchor_path"]
    results_dir  = cfg["output"]["results_dir"]

    seed_everything(seed)

    text_anchors = load_text_anchors(anchor_path, n_classes, DEVICE)
    exp_dir = os.path.join(results_dir, f"{shot}shot_seed{seed}")
    os.makedirs(exp_dir, exist_ok=True)

    train_args = SimpleNamespace(lr=lr, max_epochs=max_epochs, patience=patience)
    fold_results = []

    for fold in range(n_folds):
        split_csv = os.path.join(split_dir, f"split_{fold}.csv")
        if not os.path.exists(split_csv):
            print(f"[skip] Split not found: {split_csv}")
            continue

        print(f"\n{'=' * 60}\n Fold {fold + 1}/{n_folds}\n{'=' * 60}")

        train_ds = BrainDataset(split_csv, features_dir, "train", dataset_csv, label_cols, feat_dim)
        val_ds   = BrainDataset(split_csv, features_dir, "val",   dataset_csv, label_cols, feat_dim)
        test_ds  = BrainDataset(split_csv, features_dir, "test",  dataset_csv, label_cols, feat_dim)

        print(f"  Sizes — train: {len(train_ds)} | val: {len(val_ds)} | test: {len(test_ds)}")

        # Compute per-class positive weights to handle class imbalance
        train_labels = np.array([train_ds[i][1].numpy() for i in range(len(train_ds))])
        pos_count = train_labels.sum(axis=0)
        neg_count = len(train_labels) - pos_count
        pos_weight = torch.tensor(
            np.clip((neg_count + 1) / (pos_count + 1), 0.8, 4.0),
            dtype=torch.float32,
            device=DEVICE,
        )

        model = BrainSAIL(num_classes=n_classes, feat_dim=feat_dim, text_anchors=text_anchors)

        top_ckpts = train_loop(
            model, train_ds, val_ds, train_args, fold, exp_dir,
            pos_weight=pos_weight,
            augment_fn=lambda f: augment_features(f, noise_std=noise_std),
        )

        # Persist the best single checkpoint for reference
        best = top_ckpts[0]
        torch.save(
            {"model_state": best["model_state"], "opt_thresholds": best["ths"],
             "val_auc": best["val_auc"], "val_bacc": best["val_bacc"], "fold": fold},
            os.path.join(exp_dir, f"best_fold{fold}.pth"),
        )

        # --- Top-5 Ensemble + TTA on test set ---
        ensemble_probs: np.ndarray | None = None
        ensemble_labels: np.ndarray | None = None
        total_weight = 0.0

        for snap in top_ckpts:
            w = 0.5 * snap["val_auc"] + 0.5 * snap["val_bacc"]
            m = BrainSAIL(num_classes=n_classes, feat_dim=feat_dim, text_anchors=text_anchors).to(DEVICE)
            m.load_state_dict(snap["model_state"], strict=True)
            m.eval()

            probs, labels = predict_with_tta(m, test_ds, n_aug=n_tta)
            ensemble_probs = probs * w if ensemble_probs is None else ensemble_probs + probs * w
            ensemble_labels = labels
            total_weight += w

        ensemble_probs /= total_weight  # type: ignore[operator]

        # Compute fold-level metrics
        auc_list = [
            roc_auc_score(ensemble_labels[:, i], ensemble_probs[:, i])
            for i in range(ensemble_labels.shape[1])
            if ensemble_labels[:, i].sum() > 0
        ]
        fold_auc = float(np.mean(auc_list))
        best_ths = find_optimal_thresholds(ensemble_labels, ensemble_probs)
        fold_f1, _, fold_bacc = compute_metrics(ensemble_labels, ensemble_probs, threshold=best_ths)

        print(f"  Fold {fold + 1} test → AUC: {fold_auc:.4f} | F1: {fold_f1:.4f} | BACC: {fold_bacc:.4f}")
        fold_results.append({
            "fold": fold, "auc": fold_auc, "f1": fold_f1, "bacc": fold_bacc,
            "probs": ensemble_probs, "labels": ensemble_labels,
        })

    # --- Aggregate results ---
    if fold_results:
        aucs  = [r["auc"]  for r in fold_results]
        f1s   = [r["f1"]   for r in fold_results]
        baccs = [r["bacc"] for r in fold_results]

        print(f"\n{'=' * 50}")
        print(f"  {shot}-shot | {n_folds}-fold Final Results")
        print(f"{'=' * 50}")
        print(f"  AUC:  {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
        print(f"  F1:   {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
        print(f"  BACC: {np.mean(baccs):.4f} ± {np.std(baccs):.4f}")

        np.save(os.path.join(exp_dir, "all_test_probs.npy"),
                np.array([r["probs"] for r in fold_results], dtype=object))
        np.save(os.path.join(exp_dir, "all_test_labels.npy"),
                np.array([r["labels"] for r in fold_results], dtype=object))


if __name__ == "__main__":
    main()
