#!/bin/bash
# OC-Distill: Full Experimental Pipeline
# Reference: Algorithm flow from the paper

set -e

TIME="48h"
GAMMA=5.0
TASK="mortality"
METHOD="ontology_aware"

echo "=============================================="
echo "OC-Distill: Ontology-Aware Contrastive Learning"
echo "with Cross-Modal Distillation"
echo "=============================================="

echo ""
echo "=== Step 1: Compute Patient Similarity Matrix ==="
echo "Computing ontology-aware similarity from ICD hierarchy..."
python methods/compute_similarity.py --method ${METHOD} --time ${TIME}

echo ""
echo "=== Step 2: Stage 1 - Ontology-Aware Contrastive Pretraining ==="
echo "Pretraining vitals encoder with L_OW-NTXent loss..."
python methods/stage1_pretrain.py --method ${METHOD} --time ${TIME} --gamma ${GAMMA}

echo ""
echo "=== Step 3: Stage 2 - Train Multimodal Teacher ==="
echo "Training teacher model on vitals + clinical notes..."
python methods/stage2_teacher.py --task ${TASK} --time ${TIME} --lr 5e-5

echo ""
echo "=== Step 4: Stage 2 - Train Student with Knowledge Distillation ==="
echo "Distilling teacher knowledge into vitals-only student..."
python methods/stage2_student.py --task ${TASK} --time ${TIME} --lambda_distill 2.0

echo ""
echo "=== Evaluation: Linear Probe ==="
echo "Evaluating representation quality with frozen encoder..."
for FRAC in 0.01 0.05 0.10; do
    python methods/eval_linear_probe.py --method ${METHOD} --gamma ${GAMMA} --task ${TASK} --time ${TIME} --train_frac ${FRAC}
done

echo ""
echo "=== Evaluation: Full Fine-tuning ==="
echo "Evaluating with end-to-end fine-tuning..."
for FRAC in 0.5 1.0; do
    python methods/eval_finetune.py --method ${METHOD} --gamma ${GAMMA} --task ${TASK} --time ${TIME} --train_frac ${FRAC}
done

echo ""
echo "=============================================="
echo "Pipeline complete! Results saved to results/"
echo "=============================================="
