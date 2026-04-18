"""DICOM to NIfTI conversion and SynthStrip skull stripping.

Converts raw DICOM directories to NIfTI format using ``dcm2niix``, then
runs SynthStrip to remove non-brain structures (skull, scalp, sinuses).
Skull stripping is required before feature extraction so that the visual
encoder attends exclusively to brain parenchyma.

Usage:
    # Process a single patient
    python preprocess/skull_strip.py \\
        --config config.yaml \\
        --dicom_dir data/raw_dicom/MRI_001 \\
        --case_name MRI_001

Dependencies:
    - ``dcm2niix`` must be installed and on PATH.
    - ``mri_synthstrip`` script and ``synthstrip.1.pt`` model must be present
      (download from https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/).
"""

import argparse
import glob
import os
import subprocess

import yaml


def load_config(config_path: str) -> dict:
    """Load a YAML configuration file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Parsed configuration dictionary.
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def dicom_to_nifti(dicom_dir: str, output_dir: str, case_name: str) -> str:
    """Convert a DICOM series to a compressed NIfTI file.

    Uses ``dcm2niix`` for conversion. The output file is named
    ``{case_name}.nii.gz``.

    Args:
        dicom_dir (str): Directory containing the raw DICOM files.
        output_dir (str): Destination directory for the NIfTI output.
        case_name (str): Base name for the output file (e.g. ``"MRI_001"``).

    Returns:
        str: Full path to the generated ``.nii.gz`` file.

    Raises:
        RuntimeError: If ``dcm2niix`` returns a non-zero exit code.
        FileNotFoundError: If no NIfTI file is found after conversion.
    """
    os.makedirs(output_dir, exist_ok=True)
    result = subprocess.run(
        ["dcm2niix", "-z", "y", "-o", output_dir, "-f", case_name, dicom_dir],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"dcm2niix failed:\n{result.stderr}")

    nii_files = glob.glob(os.path.join(output_dir, f"{case_name}*.nii.gz"))
    if not nii_files:
        raise FileNotFoundError(f"No NIfTI output found in {output_dir}")

    return nii_files[0]


def skull_strip(
    input_nii: str,
    output_nii: str,
    synthstrip_script: str,
    model_path: str,
) -> None:
    """Apply SynthStrip skull stripping to a NIfTI volume.

    SynthStrip uses a deep learning model trained on synthetic MRI data,
    making it robust across pulse sequences (T1, T2, FLAIR, DWI, etc.)
    without retraining.

    Args:
        input_nii (str): Path to the input NIfTI file.
        output_nii (str): Path for the skull-stripped output NIfTI.
        synthstrip_script (str): Path to the ``mri_synthstrip`` Python script.
        model_path (str): Path to the ``synthstrip.1.pt`` model weights.

    Raises:
        RuntimeError: If SynthStrip returns a non-zero exit code.
        FileNotFoundError: If the expected output file is not created.
    """
    result = subprocess.run(
        ["python", synthstrip_script, "-i", input_nii, "-o", output_nii, "--model", model_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"SynthStrip failed:\n{result.stderr}")
    if not os.path.exists(output_nii):
        raise FileNotFoundError(f"SynthStrip did not produce output: {output_nii}")


def process_patient(
    dicom_dir: str,
    nii_dir: str,
    stripped_dir: str,
    case_name: str,
    synthstrip_script: str,
    model_path: str,
) -> str:
    """Run the full preprocessing pipeline for a single patient.

    Performs DICOM → NIfTI conversion followed by skull stripping, saving
    the final brain-only volume to ``{stripped_dir}/{case_name}_stripped.nii.gz``.

    Args:
        dicom_dir (str): Raw DICOM input directory.
        nii_dir (str): Intermediate NIfTI output directory.
        stripped_dir (str): Output directory for skull-stripped volumes.
        case_name (str): Patient identifier (e.g. ``"MRI_001"``).
        synthstrip_script (str): Path to the ``mri_synthstrip`` script.
        model_path (str): Path to the SynthStrip model weights.

    Returns:
        str: Path to the skull-stripped NIfTI file.
    """
    print(f"[{case_name}] Converting DICOM → NIfTI...")
    nii_path = dicom_to_nifti(dicom_dir, nii_dir, case_name)
    print(f"[{case_name}] Saved: {nii_path}")

    os.makedirs(stripped_dir, exist_ok=True)
    stripped_path = os.path.join(stripped_dir, f"{case_name}_stripped.nii.gz")

    print(f"[{case_name}] Running SynthStrip...")
    skull_strip(nii_path, stripped_path, synthstrip_script, model_path)
    print(f"[{case_name}] Stripped saved: {stripped_path}")

    return stripped_path


def main() -> None:
    """Entry point for single-patient skull stripping."""
    parser = argparse.ArgumentParser(description="DICOM → NIfTI → Skull strip pipeline.")
    parser.add_argument("--config",            default="config.yaml",   help="Path to config.yaml")
    parser.add_argument("--dicom_dir",         default=None,            help="DICOM input directory")
    parser.add_argument("--case_name",         default="MRI_001",       help="Patient ID (e.g. MRI_001)")
    parser.add_argument("--synthstrip_script", default="mri_synthstrip", help="Path to mri_synthstrip script")
    parser.add_argument("--synthstrip_model",  default="synthstrip.1.pt", help="Path to synthstrip model weights")
    args = parser.parse_args()

    cfg = load_config(args.config)
    nii_dir      = cfg["data"]["nii_dir"]
    stripped_dir = cfg["data"]["stripped_dir"]
    dicom_dir    = args.dicom_dir or cfg["data"]["dicom_dir"]

    process_patient(
        dicom_dir=dicom_dir,
        nii_dir=nii_dir,
        stripped_dir=stripped_dir,
        case_name=args.case_name,
        synthstrip_script=args.synthstrip_script,
        model_path=args.synthstrip_model,
    )


if __name__ == "__main__":
    main()
