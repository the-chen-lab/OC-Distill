import os
import copy
import math
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from pathlib import Path

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.data import PatientDataset, make_collate
from utils.models import BioClinicalBERT, VitalTransformer, RiskPredictor
from utils.utils import (
    set_seed, load_cohort, build_vitals_tensors,
    split_by_subject_indices, bootstrap_ci
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "cache"


def main(task, lr, train_frac, p_summary, time):
    """
    Train multimodal teacher model. The teacher uses both vital signs and clinical notes with element-wise fusion."""
    set_seed(0)
    device = torch.device("cuda")

    print("=== Stage 2: Teacher Model Training ===")
    print(f"Task: {task}, Time: {time}, p_summary: {p_summary}")

    # Load data
    df_train = load_cohort(task, split="train", time=time, cache_root=str(CACHE_DIR))
    df_test = load_cohort(task, split="test", time=time, cache_root=str(CACHE_DIR))

    notes_tr = df_train["TEXT"].tolist()
    notes_te = df_test["TEXT"].tolist()

    norm_stats_path = ROOT / f"vital_norm_stats_{time}.pt"
    vitals_tensor_tr = build_vitals_tensors(df_train, norm_stats_path=norm_stats_path)
    vitals_tensor_te = build_vitals_tensors(df_test, norm_stats_path=norm_stats_path)

    if task == "los":
        labels_tr = torch.tensor(df_train["label"].astype(np.int64).values)
        labels_te = torch.tensor(df_test["label"].astype(np.int64).values)
    else:
        labels_tr = torch.tensor(df_train[["label"]].values.astype(np.float32))
        labels_te = torch.tensor(df_test[["label"]].values.astype(np.float32))
    
    batch_size = 8 if time in ["96h", "72h"] else 16

    # Split data
    df_train = df_train.reset_index(drop=True)
    train_idx, val_idx = split_by_subject_indices(df_train, train_frac=0.8, val_frac=0.2)
    test_idx = np.arange(len(df_test))

    if train_frac < 1:
        n = int(round(len(train_idx) * train_frac))
        train_idx = np.random.choice(train_idx, size=n, replace=False)

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # Load LLM-generated summaries if available
    summaries_tr = None
    summary_path = ROOT / f"notes_summary_train_full_{time}.csv"
    if summary_path.exists() and p_summary > 0:
        notes_summary = pd.read_csv(summary_path)
        notes_summary["summary"] = notes_summary["summary"].fillna("")
        sum_map = dict(zip(notes_summary["id"].astype(int).tolist(), 
                          notes_summary["summary"].tolist()))
        summaries_tr = (
            df_train["ICUSTAY_ID"].astype(int).map(sum_map).fillna("").tolist()
        )

    # Create datasets
    train_ds = PatientDataset(
        train_idx, vitals_tensor_tr, labels_tr,
        notes=notes_tr, summaries=summaries_tr, p_summary=p_summary
    )
    val_ds = PatientDataset(val_idx, vitals_tensor_tr, labels_tr, notes=notes_tr)
    test_ds = PatientDataset(test_idx, vitals_tensor_te, labels_te, notes=notes_te)
    all_train_ds = PatientDataset(
        np.arange(len(df_train)), vitals_tensor_tr, labels_tr, notes=notes_tr
    )

    model_path = "emilyalsentzer/Bio_ClinicalBERT"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    collate_fn = make_collate(tokenizer=tokenizer)

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
    all_train_loader = DataLoader(
        all_train_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
        num_workers=8, pin_memory=True, persistent_workers=True
    )

    # Initialize models
    note_encoder = BioClinicalBERT(model_path).to(device)
    vital_encoder = VitalTransformer(
        input_dim=vitals_tensor_tr[0].shape[1],
        hidden_dim=768, n_heads=4, n_layers=2
    ).to(device)
    predictor = RiskPredictor(
        input_dim=768, num_labels=(10 if task == "los" else 1)
    ).to(device)

    params = (list(vital_encoder.parameters())
              + list(note_encoder.parameters())
              + list(predictor.parameters()))
    optimizer = torch.optim.AdamW(params, lr=lr)

    # Loss function
    if task == "los":
        criterion = nn.CrossEntropyLoss()
    else:
        train_labels_only = labels_tr[train_idx]
        num_pos = train_labels_only.sum(dim=0)
        num_neg = train_labels_only.size(0) - num_pos
        pos_weight = (num_neg / num_pos).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Training config
    grad_accum_steps = 5 if train_frac < 0.05 else 8
    epochs = 30 if train_frac < 0.05 else 15
    evals_per_epoch = 1 if train_frac < 0.5 else 2

    best_auc = float("-inf")
    best_epoch = 0
    best_model_state = None
    global_step = 0

    def run_validation(epoch, global_step):
        note_encoder.eval()
        vital_encoder.eval()
        predictor.eval()

        val_labels, val_preds = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} - Validation"):
                vitals = batch["vitals"].to(device)
                input_ids, attention_mask, owners = batch["text_pack"]
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                owners = owners.to(device)
                label = batch["y"].to(device)

                # Element-wise fusion: h_fuse = h_vitals + h_notes
                h_vitals = vital_encoder(vitals)
                h_notes = note_encoder(input_ids, attention_mask, owners)
                h_fuse = h_vitals + h_notes

                pred = predictor(h_fuse)
                probs = torch.softmax(pred, dim=1) if task == "los" else torch.sigmoid(pred)

                val_labels.append(label.detach().cpu())
                val_preds.append(probs.detach().cpu())

        if task == "los":
            val_auc = roc_auc_score(
                torch.cat(val_labels), torch.cat(val_preds),
                multi_class="ovr", average="macro"
            )
        else:
            val_auc = roc_auc_score(torch.cat(val_labels), torch.cat(val_preds))

        print(f"Epoch {epoch+1}: Val AUROC = {val_auc:.4f}")
        return val_auc

    # Training loop
    for epoch in range(epochs):
        note_encoder.train()
        vital_encoder.train()
        predictor.train()

        epoch_start_global_step = global_step
        num_batches = 0
        accum_unscaled_loss = 0.0
        accum_count = 0

        updates_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
        if evals_per_epoch == 1:
            eval_update_points = {updates_per_epoch}
        else:
            eval_update_points = {updates_per_epoch // 2, updates_per_epoch}
            eval_update_points = {p for p in eval_update_points if p >= 1}

        for mb_idx, batch in enumerate(train_loader, start=1):
            vitals = batch["vitals"].to(device)
            input_ids, attention_mask, owners = batch["text_pack"]
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            owners = owners.to(device)
            label = batch["y"].to(device)

            h_vitals = vital_encoder(vitals)
            h_notes = note_encoder(input_ids, attention_mask, owners)
            h_fuse = h_vitals + h_notes

            pred = predictor(h_fuse)
            loss = criterion(pred, label)
            num_batches += 1

            accum_unscaled_loss += loss.detach().item()
            accum_count += 1

            (loss / grad_accum_steps).backward()

            if mb_idx % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                accum_unscaled_loss = 0.0
                accum_count = 0

                update_idx_in_epoch = global_step - epoch_start_global_step
                if update_idx_in_epoch in eval_update_points:
                    val_auc = run_validation(epoch, global_step)
                    if val_auc > best_auc:
                        best_auc = val_auc
                        best_epoch = epoch + 1
                        best_model_state = {
                            "vital_encoder": copy.deepcopy(vital_encoder.state_dict()),
                            "note_encoder": copy.deepcopy(note_encoder.state_dict()),
                            "predictor": copy.deepcopy(predictor.state_dict())
                        }

                note_encoder.train()
                vital_encoder.train()
                predictor.train()

        # Handle remaining gradients
        if (mb_idx % grad_accum_steps) != 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            update_idx_in_epoch = global_step - epoch_start_global_step
            if update_idx_in_epoch in eval_update_points:
                val_auc = run_validation(epoch, global_step)
                if val_auc > best_auc:
                    best_auc = val_auc
                    best_epoch = epoch + 1
                    best_model_state = {
                        "vital_encoder": copy.deepcopy(vital_encoder.state_dict()),
                        "note_encoder": copy.deepcopy(note_encoder.state_dict()),
                        "predictor": copy.deepcopy(predictor.state_dict())
                    }

    print(f"Best AUROC: {best_auc:.4f} at epoch {best_epoch}")

    # Load best model
    note_encoder.load_state_dict(best_model_state["note_encoder"])
    vital_encoder.load_state_dict(best_model_state["vital_encoder"])
    predictor.load_state_dict(best_model_state["predictor"])
    note_encoder.eval()
    vital_encoder.eval()
    predictor.eval()

    # Test evaluation
    test_labels, test_preds = [], []
    with torch.no_grad():
        for batch in test_loader:
            vitals = batch["vitals"].to(device)
            input_ids, attention_mask, owners = batch["text_pack"]
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            owners = owners.to(device)
            label = batch["y"].to(device)

            h_vitals = vital_encoder(vitals)
            h_notes = note_encoder(input_ids, attention_mask, owners)
            h_fuse = h_vitals + h_notes

            pred = predictor(h_fuse)
            probs = torch.softmax(pred, dim=1) if task == "los" else torch.sigmoid(pred)

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
        "p_summary": [p_summary], "time": [time], "epoch": [best_epoch],
        "val_auc": [best_auc], "test_auc": [auc],
        "test_auc_ci_low": [auc_low], "test_auc_ci_high": [auc_high],
        "test_prc": [prc], "test_prc_ci_low": [prc_low], "test_prc_ci_high": [prc_high],
    })

    # Save teacher logits for distillation
    teacher_logits = []
    with torch.no_grad():
        for batch in all_train_loader:
            vitals = batch["vitals"].to(device)
            input_ids, attention_mask, owners = batch["text_pack"]
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            owners = owners.to(device)

            h_vitals = vital_encoder(vitals)
            h_notes = note_encoder(input_ids, attention_mask, owners)
            h_fuse = h_vitals + h_notes

            pred = predictor(h_fuse)
            teacher_logits.append(pred.detach().cpu())

    teacher_logits = torch.cat(teacher_logits, dim=0)

    out_dir = ROOT / "results" / "teacher_results" / str(time)
    out_dir.mkdir(parents=True, exist_ok=True)

    fname = f"results_task_{task}_lr_{lr}_train_frac_{train_frac}_p_summary_{p_summary}.csv"
    results_df.to_csv(out_dir / fname, index=False)

    logits_fname = f"teacher_logits_task_{task}_lr_{lr}_train_frac_{train_frac}_p_summary_{p_summary}.pt"
    torch.save(teacher_logits, out_dir / logits_fname)
    print(f"Saved teacher logits to: {out_dir / logits_fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: Train multimodal teacher model")
    parser.add_argument("--task", type=str, default="mortality",
                        choices=["mortality", "los"])
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--train_frac", type=float, default=1)
    parser.add_argument("--p_summary", type=float, default=0.5,
                        help="Probability of using LLM summary (default: 0.5)")
    parser.add_argument("--time", type=str, default="48h",
                        choices=["48h", "72h", "96h"])
    args = parser.parse_args()

    print(f"Training teacher: task={args.task}, lr={args.lr}, "
          f"train_frac={args.train_frac}, p_summary={args.p_summary}, time={args.time}")
    main(args.task, args.lr, args.train_frac, args.p_summary, args.time)
