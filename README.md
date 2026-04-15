# OC-Distill

**Learning Clinical Representations through Ontology-Aware Contrastive Pretraining and Cross-Modal Distillation**

This repository contains the implementation of OC-Distill, a two-stage framework for learning vital sign representations that leverages multimodal supervision during training while requiring only vital signs at inference.

## Overview

OC-Distill addresses two key limitations in clinical time-series representation learning:

1. **Standard contrastive learning ignores clinical relationships** - existing methods treat all patients as equally strong negatives, ignoring that patients often share related disease profiles.

2. **Training on vital signs alone ignores rich clinical context** - clinical notes contain reasoning and observations not captured in physiological signals.

Our framework:
- **Stage 1**: Ontology-aware contrastive pretraining that incorporates ICD diagnosis hierarchy to down-weight clinically similar negatives
- **Stage 2**: Knowledge distillation from a multimodal teacher (vitals + notes) to a vitals-only student

## Installation

```bash
conda env create -f environment.yml
conda activate oc-distill
```

## Data Preparation

This project uses the [MIMIC-III](https://physionet.org/content/mimiciii/) clinical database. We include a modified version of [YerevaNN/mimic3-benchmarks](https://github.com/YerevaNN/mimic3-benchmarks) to extract vital signs, clinical notes, diagnosis codes, and the ICD-9 ontology for patient similarity computation.

See [`mimic3_benchmark/README.md`](mimic3_benchmark/README.md) for detailed data preparation instructions.

## Usage

### Full Pipeline

```bash
bash scripts/run_pipeline.sh
```

### Individual Steps

#### 1. Compute Patient Similarity Matrix

Computes pairwise patient similarity using ICD ontology hierarchy:

```bash
python methods/compute_similarity.py --method ontology_aware --time 48h
```

Options:
- `--method`: `ontology_aware` (ours) or `flat_diagnosis` (baseline)
- `--time`: Input horizon T (`48h`, `72h`, `96h`)

#### 2. Stage 1: Ontology-Aware Contrastive Pretraining

Pretrain the vitals encoder with the ontology-weighted NT-Xent loss:

```bash
python methods/stage1_pretrain.py --method ontology_aware --time 48h --gamma 5.0
```

Options:
- `--method`: `ontology_aware`, `flat_diagnosis`, or `simclr`
- `--gamma`: Power transform exponent γ (default: 5.0)

#### 3. Stage 2: Teacher Model Training

Train the multimodal teacher on vitals + clinical notes:

```bash
python methods/stage2_teacher.py --task mortality --time 48h
```

Options:
- `--task`: `mortality` (binary) or `los` (10-class length of stay)
- `--p_summary`: Probability of using LLM-generated note summaries (default: 0.0). To use this, you must first generate your own summaries from MIMIC-III clinical notes using an LLM and save them as `data/notes_summary_train_full_{time}.csv` with columns `id` (ICUSTAY_ID) and `summary`.

#### 4. Stage 2: Student Training with Knowledge Distillation

Distill teacher knowledge into a vitals-only student (uses Stage 1 pretrained encoder):

```bash
python methods/stage2_student.py --task mortality --time 48h --lambda_distill 2.0
```

Options:
- `--lambda_distill`: Weight for distillation loss λ_distill
- `--temperature`: Distillation temperature T

### Contrastive Pretraining Evaluation

These scripts evaluate the **Stage 1 pretrained encoder**:

#### Linear Probe

Evaluates representation quality by training only a linear classifier on frozen encoder embeddings:

```bash
python methods/eval_linear_probe.py --method ontology_aware --gamma 5.0 --task mortality --train_frac 0.1
```

#### Full Fine-tuning

Fine-tunes the entire pretrained encoder along with a classification head:

```bash
python methods/eval_finetune.py --method ontology_aware --gamma 5.0 --task mortality --train_frac 1.0
```


## Citation

```bibtex
[Citation will be added upon publication]
```
