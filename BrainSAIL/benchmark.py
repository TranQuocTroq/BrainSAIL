"""Benchmark script for BrainSAIL few-shot evaluation.

Runs the full training and evaluation pipeline for both 4-shot and 8-shot
settings by invoking ``main.py`` as a subprocess, then aggregates the
results across folds and prints a summary table.

Usage:
    python benchmark.py --config config.yaml
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.eval_utils import compute_metrics, find_optimal_thresholds


def load_config(path: str) -> dict:
    """Load a YAML configuration file.

    Args:
        path (str): Path to ``config.yaml``.

    Returns:
        dict: Parsed configuration dictionary.
    """
    with open(path) as f:
        return yaml.safe_load(f)


def run_experiment(shot: int, config_path: str) -> dict | None:
    """Run one few-shot experiment via subprocess and load the saved results.

    Calls ``main.py --config {config_path} --shot {shot}`` and waits for
    completion. If successful, loads the ``.npy`` result arrays written by
    ``main.py`` and computes final metrics.

    Args:
        shot (int): Few-shot setting (4 or 8).
        config_path (str): Path to the configuration file.

    Returns:
        dict | None: Dictionary with keys ``"auc"``, ``"f1"``, ``"bacc"``
            (each a list of per-fold values), or ``None`` if the run failed.
    """
    cfg = load_config(config_path)
    seed        = cfg["train"]["seed"]
    results_dir = cfg["output"]["results_dir"]
    exp_dir     = os.path.join(results_dir, f"{shot}shot_seed{seed}")

    print(f"\n{'=' * 60}")
    print(f"  Running {shot}-shot experiment...")
    print(f"{'=' * 60}")

    cmd = [sys.executable, "main.py", "--config", config_path, "--shot", str(shot)]
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print(f"  [ERROR] main.py returned exit code {result.returncode}")
        return None

    prob_path  = os.path.join(exp_dir, "all_test_probs.npy")
    label_path = os.path.join(exp_dir, "all_test_labels.npy")

    if not (os.path.exists(prob_path) and os.path.exists(label_path)):
        print(f"  [ERROR] Result files not found in {exp_dir}")
        return None

    all_probs  = np.load(prob_path,  allow_pickle=True)
    all_labels = np.load(label_path, allow_pickle=True)

    fold_aucs, fold_f1s, fold_baccs = [], [], []

    for fold_probs, fold_labels in zip(all_probs, all_labels):
        auc_list = [
            roc_auc_score(fold_labels[:, i], fold_probs[:, i])
            for i in range(fold_labels.shape[1])
            if fold_labels[:, i].sum() > 0
        ]
        fold_aucs.append(float(np.mean(auc_list)))

        best_ths = find_optimal_thresholds(fold_labels, fold_probs)
        f1, _, bacc = compute_metrics(fold_labels, fold_probs, threshold=best_ths)
        fold_f1s.append(f1)
        fold_baccs.append(bacc)

    return {"auc": fold_aucs, "f1": fold_f1s, "bacc": fold_baccs}


def print_results_table(results: dict[int, dict | None]) -> None:
    """Print a formatted summary table of benchmark results.

    Args:
        results (dict[int, dict | None]): Mapping from shot count to result
            dict returned by ``run_experiment``.
    """
    header = f"{'Setting':<12} {'AUC':<22} {'F1':<22} {'BACC':<22}"
    print(f"\n{'*' * 70}")
    print("  BrainSAIL Benchmark Results")
    print(f"{'*' * 70}")
    print(header)
    print("-" * 70)

    for shot, res in results.items():
        if res is None:
            print(f"  {shot}-shot       FAILED")
            continue
        auc_str  = f"{np.mean(res['auc']):.4f} ± {np.std(res['auc']):.4f}"
        f1_str   = f"{np.mean(res['f1']):.4f} ± {np.std(res['f1']):.4f}"
        bacc_str = f"{np.mean(res['bacc']):.4f} ± {np.std(res['bacc']):.4f}"
        print(f"  {shot}-shot       {auc_str:<22} {f1_str:<22} {bacc_str:<22}")

    print(f"{'*' * 70}")


def main() -> None:
    """Entry point: run benchmark for 4-shot and 8-shot settings."""
    parser = argparse.ArgumentParser(description="BrainSAIL benchmark runner.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--shots", nargs="+", type=int, default=[4, 8],
        help="Few-shot settings to benchmark (default: 4 8)"
    )
    args = parser.parse_args()

    results: dict[int, dict | None] = {}
    for shot in args.shots:
        results[shot] = run_experiment(shot=shot, config_path=args.config)

    print_results_table(results)


if __name__ == "__main__":
    main()
