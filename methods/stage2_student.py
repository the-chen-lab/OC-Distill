import os
import copy
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from pathlib import Path

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.data import PatientDataset, make_collate
from utils.models import VitalTransformer, RiskPredictor
from utils.utils import (
    set_seed, load_cohort, build_vitals_tensors,
    split_by_subject_indices, bootstrap_ci
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "cache"
PRETRAIN_DIR = ROOT / "checkpoints" / "stage1_pretrain"


def main(task, lr, temperature, lambda_distill, train_frac, time):
    """Train student model with knowledge distillation."""
    set_seed(0)
    device = torch.device("cuda")

    print("=== Stage 2: Student Model Training (Knowledge Distillation) ===")
    print(f"Task: {task}, Time: {time}, λ_distill: {lambda_distill}, T: {temperature}")

    # Load data
    df_train = load_cohort(task, split="train", time=time, cache_root=str(CACHE_DIR))
    df_test = load_cohort(task, split="test", time=time, cache_root=str(CACHE_DIR))

    norm_stats_path = ROOT / f"vital_norm_stats_{time}.pt"
    vitals_tensor_tr = build_vitals_tensors(df_train, norm_stats_path=norm_stats_path)
    vitals_tensor_te = build_vitals_tensors(df_test, norm_stats_path=norm_stats_path)
    
    if task == "los":
        labels_tr = torch.tensor(df_train["label"].astype(np.int64).values)
        labels_te = torch.tensor(df_test["label"].astype(np.int64).values)
    else:
        labels_tr = torch.tensor(df_train[["label"]].values.astype(np.float32))
        labels_te = torch.tensor(df_test[["label"]].values.astype(np.float32))
    
    batch_size = 1024

    # Load teacher logits (z_teacher from Stage 2 teacher)
    teacher_logits_path = (
        ROOT / "results" / "teacher_results" / str(time) /
        f"teacher_logits_task_{task}_lr_5e-05_train_frac_{train_frac}_p_summary_0.5.pt"
    )
    teacher_logits = torch.load(teacher_logits_path)
    teacher_logits_tensor = teacher_logits.to(torch.float32)
    
    # Split data
    df_train = df_train.reset_index(drop=True)
    train_idx, val_idx = split_by_subject_indices(df_train, train_frac=0.8, val_frac=0.2)
    test_idx = np.arange(len(df_test))

    if train_frac < 1:
        n = int(round(len(train_idx) * train_frac))
        train_idx = np.random.choice(train_idx, size=n, replace=False)
    
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # Create datasets
    train_ds = PatientDataset(
        train_idx, vitals_tensor_tr, labels_tr, 
        teacher_logits_tensor=teacher_logits_tensor
    )
    val_ds = PatientDataset(
        val_idx, vitals_tensor_tr, labels_tr,
        teacher_logits_tensor=teacher_logits_tensor
    )
    test_ds = PatientDataset(test_idx, vitals_tensor_te, labels_te)

    collate_fn = make_collate()

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
        num_workers=8, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
        num_workers=8, pin_memory=True, persistent_workers=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
        num_workers=8, pin_memory=True, persistent_workers=True
    )

    # Initialize model
    vital_encoder = VitalTransformer(
        input_dim=vitals_tensor_tr[0].shape[1],
        hidden_dim=768, n_heads=4, n_layers=2
    ).to(device)
    predictor = RiskPredictor(
        input_dim=768, num_labels=(10 if task == "los" else 1)
    ).to(device)

    # Load Stage 1 pretrained encoder
    pretrained_path = str(
        PRETRAIN_DIR / f"vitals_encoder_ontology_aware_time_{time}_gamma5.0.pt"
    )
    print(f"Loading pretrained encoder from: {pretrained_path}")
    state_dict = torch.load(pretrained_path, map_location=device)
    vital_encoder.load_state_dict(state_dict)

    params = list(vital_encoder.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr)

    # Hard loss
    if task == "los":
        criterion = nn.CrossEntropyLoss()
    else:
        train_labels_only = labels_tr[train_idx]
        num_pos = train_labels_only.sum(dim=0)
        num_neg = train_labels_only.size(0) - num_pos
        pos_weight = (num_neg / num_pos).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auc = float("-inf")
    best_epoch = 0
    patience = 15
    patience_counter = 0

    # Training loop
    for epoch in range(100):
        train_labels, train_preds = [], []
        vital_encoder.train()
        predictor.train()

        for batch in train_loader:
            vitals = batch["vitals"].to(device)
            label = batch["y"].to(device)
            z_teacher = batch["teacher_logits"].to(device)

            # Student forward pass
            h_student = vital_encoder(vitals)
            z_student = predictor(h_student)
            
            # Hard loss (L_hard): standard classification loss
            loss_hard = criterion(z_student, label)
            
            # Distillation loss (L_distill): match teacher's soft targets
            T = temperature
            if task == "los":
                # Multi-class: KL divergence on temperature-scaled softmax
                with torch.no_grad():
                    p_teacher_T = torch.softmax(z_teacher / T, dim=1)
                log_p_student_T = torch.log_softmax(z_student / T, dim=1)
                loss_distill = F.kl_div(log_p_student_T, p_teacher_T, reduction="batchmean")
                probs = torch.softmax(z_student, dim=1)
            else:
                # Binary: BCE on temperature-scaled sigmoid
                with torch.no_grad():
                    p_teacher_T = torch.sigmoid(z_teacher / T).clamp(1e-4, 1-1e-4)
                loss_distill = F.binary_cross_entropy_with_logits(
                    z_student / T, p_teacher_T, reduction="mean"
                )
                probs = torch.sigmoid(z_student)

            # Total loss
            loss = loss_hard + lambda_distill * loss_distill

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            optimizer.step()

            train_labels.append(label.detach().cpu())
            train_preds.append(probs.detach().cpu())

        # Validation
        vital_encoder.eval()
        predictor.eval()

        val_labels, val_preds = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} - Validation"):
                vitals = batch["vitals"].to(device)
                label = batch["y"].to(device)

                h_student = vital_encoder(vitals)
                z_student = predictor(h_student)
                probs = torch.softmax(z_student, dim=1) if task == "los" else torch.sigmoid(z_student)
                
                val_labels.append(label.detach().cpu())
                val_preds.append(probs.detach().cpu())

        if task == "los":
            train_auc = roc_auc_score(
                torch.cat(train_labels), torch.cat(train_preds),
                multi_class="ovr", average="macro"
            )
            val_auc = roc_auc_score(
                torch.cat(val_labels), torch.cat(val_preds),
                multi_class="ovr", average="macro"
            )
        else:
            train_auc = roc_auc_score(torch.cat(train_labels), torch.cat(train_preds))
            val_auc = roc_auc_score(torch.cat(val_labels), torch.cat(val_preds))

        print(f"Epoch {epoch+1}: Train AUROC = {train_auc:.4f}, Val AUROC = {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch + 1
            best_model_state = {
                "vital_encoder": copy.deepcopy(vital_encoder.state_dict()),
                "predictor": copy.deepcopy(predictor.state_dict())
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"Best AUROC: {best_auc:.4f} at epoch {best_epoch}")
    vital_encoder.load_state_dict(best_model_state["vital_encoder"])
    predictor.load_state_dict(best_model_state["predictor"])
    vital_encoder.eval()
    predictor.eval()

    # Test evaluation (inference uses only vital signs)
    test_labels, test_preds = [], []
    with torch.no_grad():
        for batch in test_loader:
            vitals = batch["vitals"].to(device)
            label = batch["y"].to(device)

            h_student = vital_encoder(vitals)
            z_student = predictor(h_student)
            probs = torch.softmax(z_student, dim=1) if task == "los" else torch.sigmoid(z_student)
            
            test_labels.append(label.detach().cpu())
            test_preds.append(probs.detach().cpu())

    y_true = torch.cat(test_labels).cpu()
    y_prob = torch.cat(test_preds).cpu()

    auc, auc_low, auc_high, prc, prc_low, prc_high = bootstrap_ci(
        y_true, y_prob, B=1000, task_type="multiclass" if task == "los" else "binary"
    )
    print(f"Test AUROC = {auc:.4f} (95% CI: {auc_low:.4f} - {auc_high:.4f}), "
          f"PRC = {prc:.4f} (95% CI: {prc_low:.4f} - {prc_high:.4f})")

    # Save results
    results_df = pd.DataFrame({
        "task": [task], "lr": [lr], "train_frac": [train_frac],
        "temperature": [temperature], "lambda_distill": [lambda_distill],
        "time": [time], "epoch": [best_epoch], "val_auc": [best_auc],
        "test_auc": [auc], "test_auc_ci_low": [auc_low], "test_auc_ci_high": [auc_high],
        "test_prc": [prc], "test_prc_ci_low": [prc_low], "test_prc_ci_high": [prc_high],
    })

    out_dir = ROOT / "results" / "student_results" / str(time)
    out_dir.mkdir(parents=True, exist_ok=True)

    fname = (f"results_task_{task}_lr_{lr}_T_{temperature}_"
             f"lambda_{lambda_distill}_train_frac_{train_frac}.csv")
    results_df.to_csv(out_dir / fname, index=False)
    print(f"Results saved to: {out_dir / fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 2: Train student model with knowledge distillation"
    )
    parser.add_argument("--task", type=str, default="mortality",
                        choices=["mortality", "los"])
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Distillation temperature T")
    parser.add_argument("--lambda_distill", type=float, default=2.0,
                        help="Weight for distillation loss λ_distill")
    parser.add_argument("--train_frac", type=float, default=1)
    parser.add_argument("--time", type=str, default="48h",
                        choices=["48h", "72h", "96h"])

    args = parser.parse_args()
    print(
        f"Training student: task={args.task}, lr={args.lr}, T={args.temperature}, "
        f"λ_distill={args.lambda_distill}, train_frac={args.train_frac}, time={args.time}"
    )

    main(args.task, args.lr, args.temperature, args.lambda_distill,
         args.train_frac, args.time)
