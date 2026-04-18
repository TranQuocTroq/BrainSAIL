"""Training loop, EMA, validation, and TTA inference utilities.

This module provides the core training infrastructure for BrainSAIL,
including:

- ``EMA``          — exponential moving average of model weights
- ``train_loop``   — full training loop with early stopping and snapshot saving
- ``validate``     — evaluation on a validation split
- ``predict_with_tta`` — test-time augmentation inference

Example:
    >>> top_ckpts = train_loop(model, train_ds, val_ds, args, fold=0, exp_dir="results/")
    >>> probs, labels = predict_with_tta(model, test_ds, n_aug=5)
"""

import heapq
import os

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingLR

from .eval_utils import compute_metrics, find_optimal_thresholds

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class EMA:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy of trainable weights updated as:

        shadow = decay * shadow + (1 - decay) * current

    Shadow weights typically generalize better than the last gradient step,
    especially with small datasets.

    Args:
        model (nn.Module): Model whose parameters are tracked.
        decay (float): EMA decay factor. Higher values mean slower adaptation.
            Defaults to ``0.99``.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.99) -> None:
        self.model = model
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {
            name: val.clone().float()
            for name, val in model.state_dict().items()
        }

    def update(self) -> None:
        """Update shadow weights from current trainable parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    (1.0 - self.decay) * param.data.float()
                    + self.decay * self.shadow[name]
                )

    def apply_shadow(self) -> None:
        """Copy shadow weights into the model for evaluation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].float())


def train_loop(
    model: torch.nn.Module,
    train_ds,
    val_ds,
    args,
    fold: int,
    exp_dir: str,
    pos_weight: torch.Tensor | None = None,
    augment_fn=None,
) -> list[dict]:
    """Run one complete training fold and return the top-5 snapshots.

    Training uses cosine-annealing LR, gradient clipping, EMA evaluation,
    and early stopping. The top-5 checkpoints (by ``0.5*AUC + 0.5*BACC``)
    are kept on disk; others are deleted to save space.

    Args:
        model (nn.Module): BrainSAIL model instance.
        train_ds: Training dataset (iterable of ``(features, label, slide_id)``).
        val_ds: Validation dataset (same format as ``train_ds``).
        args: Namespace with attributes ``lr``, ``max_epochs``, ``patience``.
        fold (int): Current fold index (used for checkpoint naming).
        exp_dir (str): Directory to save checkpoint files.
        pos_weight (torch.Tensor, optional): Per-class positive weights for
            BCE loss, shape ``[num_classes]``. Defaults to None.
        augment_fn (callable, optional): Feature augmentation applied per
            batch; signature ``augment_fn(features) -> features``.

    Returns:
        list[dict]: Top-5 snapshot dictionaries, each containing:
            ``epoch``, ``model_state``, ``val_auc``, ``val_bacc``,
            ``val_f1``, and ``ths`` (optimal thresholds).
    """
    model = model.to(DEVICE).float()
    optimizer = (
        model.build_optimizer(lr_base=args.lr, weight_decay=0.01)
        if hasattr(model, "build_optimizer")
        else optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    )

    max_epochs = min(getattr(args, "max_epochs", 60), 60)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    ema = EMA(model, decay=0.99)

    TOP_K = 5
    ckpt_heap: list = []
    best_score = -1e9
    patience = getattr(args, "patience", 10)
    no_improve = 0
    all_snapshots: list[dict] = []

    print(f"\n[Fold {fold}] max_epochs={max_epochs} | patience={patience} | lr={args.lr:.0e}")

    for epoch in range(max_epochs):
        # ---- Training step ----
        model.train()
        epoch_loss = 0.0

        for features, label, _ in train_ds:
            features = features.to(DEVICE).float()
            label = label.to(DEVICE).float()

            if augment_fn is not None:
                features = augment_fn(features)

            logits, loss = model(features, label, pos_weight=pos_weight, epoch=epoch)

            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            ema.update()
            epoch_loss += loss.item()

        scheduler.step()

        # ---- Validation with EMA weights ----
        orig = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        ema.apply_shadow()
        _, v_auc, _, v_f1, v_bacc, val_probs, val_labels, _ = validate(model, val_ds)
        for name, param in model.named_parameters():
            if name in orig:
                param.data.copy_(orig[name])

        current_score = 0.5 * v_auc + 0.5 * v_bacc
        best_th = find_optimal_thresholds(val_labels, val_probs)
        ckpt_path = os.path.join(exp_dir, f"fold{fold}_ep{epoch + 1:02d}.pt")

        # Keep top-K checkpoints on disk
        if len(ckpt_heap) < TOP_K or current_score > ckpt_heap[0][0]:
            torch.save({"model_state": ema.shadow, "opt_thresholds": best_th}, ckpt_path)
            heapq.heappush(ckpt_heap, (current_score, epoch, ckpt_path))
            if len(ckpt_heap) > TOP_K:
                _, _, old_path = heapq.heappop(ckpt_heap)
                if os.path.exists(old_path):
                    os.remove(old_path)

        all_snapshots.append({
            "epoch":       epoch + 1,
            "model_state": {k: v.cpu().clone() for k, v in ema.shadow.items()},
            "val_auc":     v_auc,
            "val_bacc":    v_bacc,
            "val_f1":      v_f1,
            "ths":         best_th,
        })

        if current_score > best_score:
            best_score, no_improve = current_score, 0
            status = f"* best={best_score:.4f}"
        else:
            no_improve += 1
            status = f"no_improve={no_improve}/{patience}"

        avg_loss = epoch_loss / max(len(train_ds), 1)
        current_lr = optimizer.param_groups[-1]["lr"]
        print(
            f"  [{epoch + 1:02d}/{max_epochs}] loss={avg_loss:.3f} "
            f"auc={v_auc:.4f} f1={v_f1:.4f} bacc={v_bacc:.4f} "
            f"lr={current_lr:.1e} {status}"
        )

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch + 1}.")
            break

    # Return top-5 snapshots sorted by combined score
    all_snapshots.sort(
        key=lambda s: 0.5 * s["val_auc"] + 0.5 * s["val_bacc"], reverse=True
    )
    top_5 = all_snapshots[:TOP_K]
    print(f"  Top-{len(top_5)} snapshots selected for ensemble.")
    return top_5


def validate(model: torch.nn.Module, dataset) -> tuple:
    """Evaluate the model on a dataset split.

    Args:
        model (nn.Module): Model to evaluate (EMA weights should be applied
            before calling this function if desired).
        dataset: Iterable of ``(features, label, slide_id)`` tuples.

    Returns:
        tuple: ``(loss, macro_auc, macro_prauc, macro_f1, macro_bacc,
                  all_probs, all_labels, prob_range)``
    """
    model.eval()
    all_labels, all_probs = [], []
    total_loss = 0.0

    with torch.no_grad():
        for features, label, _ in dataset:
            features = features.to(DEVICE).float()
            label = label.to(DEVICE).float()
            logits, loss = model(features, label, epoch=0)
            total_loss += loss.item()
            all_probs.append(torch.sigmoid(logits).cpu().float().numpy()[0])
            all_labels.append(label.cpu().float().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    if all_probs.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, None, None, 0.0

    auc_list, prauc_list = [], []
    for i in range(all_labels.shape[1]):
        if all_labels[:, i].sum() > 0 and (1 - all_labels[:, i]).sum() > 0:
            auc_list.append(roc_auc_score(all_labels[:, i], all_probs[:, i]))
            prauc_list.append(average_precision_score(all_labels[:, i], all_probs[:, i]))

    macro_auc = float(np.mean(auc_list)) if auc_list else 0.5
    macro_prauc = float(np.mean(prauc_list)) if prauc_list else 0.0

    best_ths = find_optimal_thresholds(all_labels, all_probs)
    macro_f1, _, macro_bacc = compute_metrics(all_labels, all_probs, threshold=best_ths)
    prob_range = float(all_probs.max() - all_probs.min())

    return (
        total_loss / len(dataset),
        macro_auc, macro_prauc, macro_f1, macro_bacc,
        all_probs, all_labels, prob_range,
    )


def predict_with_tta(
    model: torch.nn.Module,
    dataset,
    n_aug: int = 5,
    noise_std: float = 0.005,
    device: str = DEVICE,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference with Test-Time Augmentation (TTA).

    For each patient, runs ``n_aug`` forward passes with independent
    Gaussian noise added to the feature bag, then averages the resulting
    sigmoid probabilities. This reduces prediction variance without
    requiring additional labeled data.

    Args:
        model (nn.Module): Trained BrainSAIL model.
        dataset: Iterable of ``(features, label, slide_id)`` tuples.
        n_aug (int): Number of augmented passes per patient. Defaults to ``5``.
        noise_std (float): Standard deviation of additive Gaussian noise.
            Defaults to ``0.005``.
        device (str): Target device string. Defaults to ``DEVICE``.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - **probs** — averaged probabilities, shape ``[N, C]``.
            - **labels** — ground-truth labels, shape ``[N, C]``.
    """
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for features, label, _ in dataset:
            features = features.to(device).float()
            runs = []
            for _ in range(n_aug):
                noisy = features + torch.randn_like(features) * noise_std
                logits, _ = model(noisy, epoch=0)
                runs.append(torch.sigmoid(logits).cpu().float().numpy()[0])
            all_probs.append(np.mean(runs, axis=0))
            all_labels.append(label.numpy())

    return np.array(all_probs), np.array(all_labels)
