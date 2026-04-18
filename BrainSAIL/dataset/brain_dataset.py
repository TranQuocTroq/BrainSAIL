"""PyTorch Dataset for Brain-OHN MIL classification.

Loads pre-extracted patient feature tensors (.pt files) and multi-label
annotations from a CSV, supporting stratified k-fold splits.

Example:
    >>> ds = BrainDataset(
    ...     split_csv="data/splits/4shot/split_0.csv",
    ...     features_dir="data/features",
    ...     split="train",
    ...     dataset_csv="data/brain_dataset.csv",
    ... )
    >>> features, label, slide_id = ds[0]
    >>> features.shape   # [S, 768]
    >>> label.shape      # [5]
"""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

LABEL_COLS = ["Normal", "WMH", "Atrophy", "Old_Lesion", "Special"]


class BrainDataset(Dataset):
    """Multi-label MRI brain pathology dataset for MIL classification.

    Each sample corresponds to one patient, represented as a variable-length
    bag of slice feature vectors pre-extracted by UniMedCLIP ViT-L/14.

    Args:
        split_csv (str): Path to a fold CSV file (e.g., ``split_0.csv``)
            containing columns ``train``, ``val``, and ``test`` with
            patient slide IDs.
        features_dir (str): Directory containing ``.pt`` feature files
            named ``{slide_id}.pt``, each of shape ``[S, feat_dim]``.
        split (str): Which partition to load — ``"train"``, ``"val"``,
            or ``"test"``. Defaults to ``"train"``.
        dataset_csv (str): Path to the master label CSV with columns
            ``slide_id`` and one column per label in ``label_cols``.
        label_cols (list[str], optional): Ordered list of label column names.
            Defaults to ``LABEL_COLS``.
        feature_dim (int): Expected feature dimension; used for zero tensors
            when a file is missing. Defaults to ``768``.

    Raises:
        FileNotFoundError: If ``dataset_csv`` does not exist.
    """

    def __init__(
        self,
        split_csv: str,
        features_dir: str,
        split: str = "train",
        dataset_csv: str | None = None,
        label_cols: list[str] | None = None,
        feature_dim: int = 768,
    ) -> None:
        self.features_dir = features_dir
        self.split = split
        self.label_cols = label_cols or LABEL_COLS
        self.feature_dim = feature_dim

        if dataset_csv is None or not os.path.exists(dataset_csv):
            raise FileNotFoundError(f"Label CSV not found: {dataset_csv}")

        main_df = pd.read_csv(dataset_csv, encoding="utf-8-sig")
        splits_df = pd.read_csv(split_csv, encoding="utf-8-sig")

        if split in splits_df.columns:
            slide_ids = splits_df[split].dropna().astype(str).str.strip().tolist()
            self.data = main_df[
                main_df["slide_id"].astype(str).isin(slide_ids)
            ].reset_index(drop=True)
        else:
            print(f"[WARNING] Column '{split}' not found in {split_csv}; using full dataset.")
            self.data = main_df.reset_index(drop=True)

        self.n_classes = len(self.label_cols)

    def __len__(self) -> int:
        """Return the number of patients in this split."""
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        """Return the feature bag, label vector, and slide ID for one patient.

        Args:
            idx (int): Patient index.

        Returns:
            tuple[torch.Tensor, torch.Tensor, str]:
                - **features** — float32 tensor of shape ``[S, feat_dim]``.
                - **label** — float32 multi-label vector of shape ``[n_classes]``.
                - **slide_id** — patient identifier string.
        """
        row = self.data.iloc[idx]
        slide_id = str(row["slide_id"])
        feat_path = os.path.join(self.features_dir, f"{slide_id}.pt")

        if not os.path.exists(feat_path):
            print(f"[WARNING] Feature file not found: {feat_path}")
            features = torch.zeros(1, self.feature_dim)
        else:
            try:
                raw = torch.load(feat_path, weights_only=True)
                features = raw if isinstance(raw, torch.Tensor) else raw["features"]

                if isinstance(features, np.ndarray):
                    features = torch.from_numpy(features)

                features = features.float()

                if features.numel() == 0:
                    features = torch.zeros(1, self.feature_dim)
                if features.dim() == 1:
                    features = features.unsqueeze(0)

            except Exception as exc:
                print(f"[ERROR] Failed to load {feat_path}: {exc}")
                features = torch.zeros(1, self.feature_dim)

        label = torch.tensor(
            [float(row[col]) for col in self.label_cols],
            dtype=torch.float32,
        )
        return features, label, slide_id
