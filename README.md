# BrainSAIL 🧠

**Few-shot Brain MRI Classification via Soft Attention Multiple Instance Learning**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

BrainSAIL is a few-shot learning framework for **multi-label brain pathology classification** from MRI scans. It is designed to work with as few as **4–8 labeled patients per class** — a realistic constraint in clinical settings where expert annotations are scarce and expensive.

The system combines a frozen vision-language foundation model ([UniMedCLIP](https://github.com/SuperMedIntel/UniMedCLIP)) with a lightweight classification head trained under few-shot supervision, inspired by [FOCUS (CVPR 2025)](https://github.com/dddavid4real/FOCUS).

---

## Task

Given a bag of MRI slices from a patient, predict the presence of one or more of the following brain pathologies:

| Label | Pathology |
|---|---|
| Normal | Healthy brain, no abnormalities |
| WMH | White matter hyperintensities |
| Atrophy | Brain atrophy, ventricular enlargement |
| Old-Lesion | Old infarct, microbleeds |
| Special | Hemangioma, hematoma, vascular MSA |

**Dataset:** Brain-OHN — 83 patients, ~26 axial slices per patient, DICOM format.

---

## Method

BrainSAIL decouples feature extraction from classification training, allowing the use of a large-scale foundation model (307M parameters) without GPU memory constraints during training.

![BrainSAIL Architecture](assets/architecture.png)

### Pipeline

```
Stage 1 — Feature Extraction (offline, run once)

  MRI (DICOM)
    → Skull Stripping (SynthStrip)
    → Dynamic Crop + Zero-pad + Min-max Normalize
    → UniMedCLIP ViT-L/14 (frozen, 307M params)
    → Patient bag: [S × 768] saved as .pt

  Clinical descriptions (LLM-generated, see Text_Prompt/)
    → UniMedCLIP Text Encoder (frozen)
    → 5 Text Anchors: [5 × 768]

Stage 2 — BrainSAIL Classification (trained, ~2-3M params)

  [S × 768]
    → Visual Adapter       (bottleneck 768 → 192 → 768, residual)
    → Soft Attention MIL   (per-class scorer + learnable temperature)
    → [5 × 768]
    → Cosine Classifier    (similarity to Text Anchors)
    → [5] sigmoid scores   (multi-label output)
```

### Key Design Choices

- **Per-class attention** — each pathology class learns its own attention scorer over slices, so WMH can focus on periventricular regions while Atrophy attends to ventricular slices
- **Cosine Classifier** — consistent with the contrastive pretraining objective of UniMedCLIP; no linear projection needed
- **~2–3M trainable parameters** — keeps the model regularized under few-shot constraints
- **EMA + Label Smoothing + Orthogonality Loss** — prevent overfitting and anchor collapse
- **Top-5 Snapshot Ensemble + TTA** — reduces prediction variance at test time

---

## Results

Evaluated with stratified 3-fold cross-validation on Brain-OHN.
Metrics: Macro Balanced Accuracy (BACC), Macro AUC, Macro F1.

| Setting | BACC | AUC | F1 |
|---|---|---|---|
| 4-shot | 0.74 | 0.71 | 0.68 |
| 8-shot | 0.77 | 0.75 | 0.70 |

---

## Project Structure

```
BrainSAIL/
├── config.yaml               # All paths and hyperparameters
├── main.py                   # Training and evaluation entry point
├── benchmark.py              # Run 4-shot and 8-shot experiments
├── extract_features.py       # Offline feature extraction (run once)
├── requirements.txt
├── assets/
│   └── architecture.png
├── Text_Prompt/              # LLM-generated clinical descriptions (5 LLMs)
├── data/                     # Populated by the user (see config.yaml)
│   ├── raw_dicom/
│   ├── nifti/
│   ├── stripped/
│   ├── features/
│   └── splits/
│       ├── 4shot/
│       └── 8shot/
├── weights/                  # UniMedCLIP checkpoint (download separately)
├── dataset/
│   └── brain_dataset.py      # PyTorch Dataset for MIL bags
├── models/
│   └── model.py              # BrainSAIL architecture
├── preprocess/
│   └── skull_strip.py        # DICOM → NIfTI → SynthStrip
└── utils/
    ├── core_utils.py         # Training loop, EMA, TTA
    └── eval_utils.py         # Metrics, threshold optimization
```

---

## Installation

```bash
git clone https://github.com/TranQuocTroq/BrainSAIL.git
cd BrainSAIL
pip install -r requirements.txt
```

**Requirements:** Python 3.9+, PyTorch 2.0+, CUDA 11.8+

SynthStrip (for skull stripping):
```bash
pip install surfa
wget -O mri_synthstrip https://raw.githubusercontent.com/freesurfer/freesurfer/dev/mri_synthstrip/mri_synthstrip
wget -O synthstrip.1.pt https://huggingface.co/lemuelpuglisi/generics/resolve/main/synthstrip.1.pt
```

---

## Usage

**Step 1 — Edit `config.yaml`**

Set your local paths for data, weights, and splits before running anything.

**Step 2 — Skull stripping (per patient)**
```bash
python preprocess/skull_strip.py \
    --config config.yaml \
    --dicom_dir data/raw_dicom/MRI_001 \
    --case_name MRI_001
```

**Step 3 — Extract features (run once)**
```bash
python extract_features.py --config config.yaml
```

**Step 4 — Train and evaluate**
```bash
# 4-shot
python main.py --config config.yaml --shot 4

# 8-shot
python main.py --config config.yaml --shot 8
```

**Step 5 — Run full benchmark**
```bash
python benchmark.py --config config.yaml
```

---

## Text Prompts

The `Text_Prompt/` directory contains LLM-generated clinical descriptions for each brain pathology class, produced by 5 different language models (ChatGPT, Claude, Gemini, Grok, Deepsite). These are used to build the text anchor embeddings that guide classification.

All variants achieve above **83% Balanced Accuracy** in the 8-shot setting, with ChatGPT prompts yielding the most consistent results across folds.

---

## References

- Guo et al. [FOCUS: Knowledge-enhanced Adaptive Visual Compression for Few-shot Whole Slide Image Classification](https://github.com/dddavid4real/FOCUS). CVPR 2025.
- Hoopes et al. [SynthStrip: Skull-stripping for Any Brain Image](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/). NeuroImage, 2022.
- Ilse et al. [Attention-based Deep Multiple Instance Learning](https://arxiv.org/abs/1802.04712). ICML 2018.
- Zhang et al. [BiomedCLIP: A Multimodal Biomedical Foundation Model](https://arxiv.org/abs/2303.00915). ICLR 2024.

---

## Contact

**Tran Quoc Trong** · tranquoct157@gmail.com  
Water Resources University · Faculty of Information Technology · Class of 2026
