"""Offline feature extraction using UniMedCLIP ViT-L/14.

Scans a directory of skull-stripped NIfTI files, extracts per-slice
feature vectors, and saves them as ``{patient_id}.pt`` tensors.
This script is run **once** before training; features are then cached
and loaded by ``BrainDataset`` at runtime.

Usage:
    python extract_features.py --config config.yaml

Output:
    One ``.pt`` file per patient in ``config.data.features_dir``,
    each of shape ``[S, 768]`` where S is the number of valid slices.
"""

import gc
import glob
import os
import argparse

import nibabel as nib
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
import yaml
from PIL import Image


def load_config(config_path: str) -> dict:
    """Load a YAML configuration file.

    Args:
        config_path (str): Path to ``config.yaml``.

    Returns:
        dict: Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_model(weights_path: str, backbone: str, device: str):
    """Load UniMedCLIP visual encoder from a checkpoint.

    Handles DataParallel ``module.`` prefixes and verifies that visual
    weights are present in the checkpoint.

    Args:
        weights_path (str): Path to the ``.pt`` checkpoint file.
        backbone (str): open_clip model name (e.g. ``"ViT-L-14-336-quickgelu"``).
        device (str): Target device string (e.g. ``"cuda:0"``).

    Returns:
        tuple: ``(visual_model, preprocess_fn)`` where ``visual_model`` is
            the frozen ViT visual encoder and ``preprocess_fn`` is the
            corresponding image preprocessing transform.

    Raises:
        RuntimeError: If no visual weights are found in the checkpoint.
    """
    print(f"  Loading {backbone} → {device}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        backbone, pretrained=None, device="cpu"
    )
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)

    # Strip DataParallel prefix if present
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    missing, _ = model.load_state_dict(state_dict, strict=False)
    visual_keys = [k for k in state_dict if k.startswith("visual")]

    if not visual_keys:
        raise RuntimeError("No visual weights found in checkpoint.")

    print(f"  Visual keys: {len(visual_keys)} | Missing keys: {len(missing)}")
    visual_model = model.visual.to(device).eval()

    del checkpoint, state_dict, model
    gc.collect()
    torch.cuda.empty_cache()
    return visual_model, preprocess


def preprocess_slice(slice_2d: np.ndarray, preprocess_fn) -> torch.Tensor | None:
    """Convert a single 2-D MRI slice to a model-ready tensor.

    Applies percentile clipping, 8-bit normalization, and the model's
    image preprocessing pipeline. Returns None for blank or near-uniform
    slices that carry no diagnostic information.

    Args:
        slice_2d (np.ndarray): Raw 2-D slice array of shape ``[H, W]``.
        preprocess_fn: open_clip preprocessing transform.

    Returns:
        torch.Tensor | None: Preprocessed tensor, or ``None`` if the
            slice is considered invalid.
    """
    img = slice_2d.astype(np.float32)

    # Reject completely blank slices
    if img.max() - img.min() < 1e-6:
        return None

    # Percentile clip to suppress intensity outliers
    p_low, p_high = np.percentile(img, 1), np.percentile(img, 99)
    img = np.clip(img, p_low, p_high)

    # Normalize to [0, 255] uint8
    img_range = img.max() - img.min()
    img_8bit = ((img - img.min()) / (img_range + 1e-6) * 255).astype(np.uint8)

    # Reject near-uniform slices (e.g., background-only after skull strip)
    if img_8bit.std() < 3:
        return None

    return preprocess_fn(Image.fromarray(img_8bit).convert("RGB"))


def extract_patient(
    nii_path: str,
    model: torch.nn.Module,
    preprocess_fn,
    device: str,
    axis: int = 2,
    batch_size: int = 32,
) -> torch.Tensor | None:
    """Extract L2-normalized feature vectors for all valid slices of a patient.

    Args:
        nii_path (str): Path to a skull-stripped NIfTI file.
        model (nn.Module): Frozen visual encoder.
        preprocess_fn: Preprocessing transform compatible with the encoder.
        device (str): Device for inference.
        axis (int): Volume axis to slice along — 0: sagittal, 1: coronal,
            2: axial. Defaults to ``2``.
        batch_size (int): Number of slices per inference batch. Defaults to ``32``.

    Returns:
        torch.Tensor | None: Feature tensor of shape ``[S, D]``, or ``None``
            if no valid slices were found.
    """
    nii = nib.load(nii_path)
    vol = nii.get_fdata(dtype=np.float32)

    # Handle 4-D fMRI volumes — use the first time point
    if vol.ndim == 4:
        vol = vol[..., 0]

    n_slices = vol.shape[axis]
    valid_tensors = []

    for i in range(n_slices):
        if axis == 0:
            sl = vol[i, :, :]
        elif axis == 1:
            sl = vol[:, i, :]
        else:
            sl = vol[:, :, i]

        sl = np.rot90(sl, k=1)  # Correct NIfTI axis orientation
        tensor = preprocess_slice(sl, preprocess_fn)
        if tensor is not None:
            valid_tensors.append(tensor)

    if not valid_tensors:
        return None

    feats = []
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for i in range(0, len(valid_tensors), batch_size):
            batch = torch.stack(valid_tensors[i : i + batch_size]).to(device)
            out = model(batch)
            if isinstance(out, tuple):
                out = out[0]
            feats.append(F.normalize(out.float(), p=2, dim=-1).cpu())

    return torch.cat(feats, dim=0)  # [S, D]


def scan_patients(nii_dir: str, start_idx: int = 1) -> list[dict]:
    """Scan a directory for NIfTI files and assign sequential patient IDs.

    Args:
        nii_dir (str): Root directory to search recursively.
        start_idx (int): Starting integer for patient ID numbering.
            Defaults to ``1``, producing IDs like ``MRI_001``.

    Returns:
        list[dict]: List of ``{"nii_path": str, "patient_id": str}`` dicts.
    """
    nii_files = sorted(
        glob.glob(os.path.join(nii_dir, "**/*.nii"),    recursive=True)
        + glob.glob(os.path.join(nii_dir, "**/*.nii.gz"), recursive=True)
    )
    return [
        {"nii_path": path, "patient_id": f"MRI_{idx:03d}"}
        for idx, path in enumerate(nii_files, start=start_idx)
    ]


def main() -> None:
    """Entry point for offline feature extraction."""
    parser = argparse.ArgumentParser(description="Extract UniMedCLIP features from NIfTI files.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    weights_path = cfg["model"]["weights_path"]
    backbone     = cfg["model"]["backbone"]
    nii_dir      = cfg["data"]["stripped_dir"]
    output_dir   = cfg["data"]["features_dir"]
    axis         = cfg["preprocess"]["axis"]
    batch_size   = cfg["preprocess"]["batch_size"]
    start_idx    = cfg["preprocess"]["start_idx"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(output_dir, exist_ok=True)

    model, preprocess = load_model(weights_path, backbone, device)
    patients = scan_patients(nii_dir, start_idx=start_idx)
    print(f"\nFound {len(patients)} NIfTI files → output: {output_dir}\n")

    for p in patients:
        save_path = os.path.join(output_dir, f"{p['patient_id']}.pt")

        if os.path.exists(save_path):
            print(f"  [skip]  {p['patient_id']} already exists.")
            continue

        if not os.path.exists(p["nii_path"]):
            print(f"  [warn]  File not found: {p['nii_path']}")
            continue

        try:
            feats = extract_patient(p["nii_path"], model, preprocess, device, axis, batch_size)
            if feats is None:
                print(f"  [fail]  {p['patient_id']}: no valid slices.")
                continue
            torch.save(feats, save_path)
            print(f"  [ok]    {p['patient_id']}: {tuple(feats.shape)} → {save_path}")
        except Exception as exc:
            print(f"  [error] {p['patient_id']}: {exc}")

    print("\nFeature extraction complete.")


if __name__ == "__main__":
    main()
