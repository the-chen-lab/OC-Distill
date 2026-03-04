import os
import copy
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from pathlib import Path

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.data import PatientDataset, make_collate
from utils.models import VitalTransformer
from utils.utils import (
    set_seed, load_cohort, build_vitals_tensors,
    split_by_subject_indices, bootstrap_ci
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "cache"
PRETRAIN_DIR = ROOT / "checkpoints" / "stage1_pretrain"


class LinearHead(nn.Module):
    """Simple linear classification head."""
    def __init__(self, input_dim=768, num_labels=1):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_labels)

    def forward(self, x):
        return self.fc(x)


def extract_embeddings(encoder, data_loader, device):
    """Extract embeddings from frozen encoder."""
    encoder.eval()
    X_list, y_list = [], []

    with torch.no_grad():
        for batch in data_loader:
            vitals = batch["vitals"].to(device)
            label = batch["y"].to(device)

            emb = encoder(vitals)
            X_list.append(emb.cpu())
            y_list.append(label.view(-1).cpu().long())

    X = torch.cat(X_list, dim=0).numpy()
    y = torch.cat(y_list, dim=0).numpy()
    return X, y


def main(method, time, gamma, task, train_frac):
    """Run linear probe evaluation."""
    set_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== Linear Probe Evaluation ===")
    print(f"Method: {method}, Time: {time}, Gamma: {gamma}")
    print(f"Task: {task}, Train fraction: {train_frac}")

    head_lr = 0.005
    batch_size = 4096

    pretrained_path = str(
        PRETRAIN_DIR / f"vitals_encoder_{method}_time_{time}_gamma{gamma}.pt"
    )

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

    # Split data
    df_train = df_train.reset_index(drop=True)
    train_idx, val_idx = split_by_subject_indices(df_train, train_frac=0.8, val_frac=0.2)
    test_idx = np.arange(len(df_test))

    if train_frac < 1:
        n = int(round(len(train_idx) * train_frac))
        train_idx = np.random.choice(train_idx, size=n, replace=False)
    
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # Create datasets
    train_ds = PatientDataset(train_idx, vitals_tensor_tr, labels_tr)
    val_ds = PatientDataset(val_idx, vitals_tensor_tr, labels_tr)
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

    # Load pretrained encoder and freeze
    vital_encoder = VitalTransformer(
        input_dim=vitals_tensor_tr[0].shape[1],
        hidden_dim=768, n_heads=4, n_layers=2
    ).to(device)
    
    print(f"Loading pretrained encoder from: {pretrained_path}")
    state_dict = torch.load(pretrained_path, map_location=device)
    vital_encoder.load_state_dict(state_dict)

    for p in vital_encoder.parameters():
        p.requires_grad = False
    vital_encoder.eval()

    # Extract embeddings
    print("Extracting embeddings...")
    X_train_np, y_train_np = extract_embeddings(vital_encoder, train_loader, device)
    X_val_np, y_val_np = extract_embeddings(vital_encoder, val_loader, device)
    X_test_np, y_test_np = extract_embeddings(vital_encoder, test_loader, device)

    X_train = torch.from_numpy(X_train_np).to(device)
    X_val = torch.from_numpy(X_val_np).to(device)
    X_test = torch.from_numpy(X_test_np).to(device)

    if task == "los":
        y_train = torch.from_numpy(y_train_np).to(device).long()
        y_val = torch.from_numpy(y_val_np).to(device).long()
        y_test = torch.from_numpy(y_test_np).to(device).long()
        num_labels = 10
    else:
        y_train = torch.from_numpy(y_train_np).float().unsqueeze(1).to(device)
        y_val = torch.from_numpy(y_val_np).float().unsqueeze(1).to(device)
        y_test = torch.from_numpy(y_test_np).float().unsqueeze(1).to(device)
        num_labels = 1

    # Train linear head
    head = LinearHead(input_dim=768, num_labels=num_labels).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=head_lr)

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
    max_epochs = 2000

    for epoch in range(max_epochs):
        head.train()
        logits_tr = head(X_train)
        
        if task == "los":
            loss = criterion(logits_tr, y_train)
            probs_tr = torch.softmax(logits_tr, dim=1)
        else:
            loss = criterion(logits_tr, y_train)
            probs_tr = torch.sigmoid(logits_tr)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
        optimizer.step()

        # Validation
        head.eval()
        with torch.no_grad():
            if task == "los":
                train_auc = roc_auc_score(
                    y_train.detach().cpu().numpy(),
                    probs_tr.detach().cpu().numpy(),
                    multi_class="ovr", average="macro"
                )
            else:
                train_auc = roc_auc_score(
                    y_train.detach().cpu().numpy(),
                    probs_tr.detach().cpu().numpy()
                )

            logits_val = head(X_val)
            if task == "los":
                probs_val = torch.softmax(logits_val, dim=1)
                val_auc = roc_auc_score(
                    y_val.detach().cpu().numpy(),
                    probs_val.detach().cpu().numpy(),
                    multi_class="ovr", average="macro"
                )
            else:
                probs_val = torch.sigmoid(logits_val)
                val_auc = roc_auc_score(
                    y_val.detach().cpu().numpy(),
                    probs_val.detach().cpu().numpy()
                )

        if epoch % 100 == 0:
            print(f"Epoch {epoch+1}: Train AUROC = {train_auc:.4f}, Val AUROC = {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch + 1
            best_head_state = copy.deepcopy(head.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"Best Val AUROC: {best_auc:.4f} at epoch {best_epoch}")
    head.load_state_dict(best_head_state)

    # Test evaluation
    head.eval()
    with torch.no_grad():
        logits_te = head(X_test)
        probs_te = torch.softmax(logits_te, dim=1) if task == "los" else torch.sigmoid(logits_te)

    y_true = y_test.detach().cpu()
    y_prob = probs_te.detach().cpu()

    auc, auc_low, auc_high, prc, prc_low, prc_high = bootstrap_ci(
        y_true, y_prob, B=1000, task_type="multiclass" if task == "los" else "binary"
    )

    print(
        f"Test AUROC = {auc:.4f} (95% CI: {auc_low:.4f} - {auc_high:.4f}), "
        f"PRC = {prc:.4f} (95% CI: {prc_low:.4f} - {prc_high:.4f})"
    )

    # Save results
    results_df = pd.DataFrame({
        "method": [method], "time": [time], "gamma": [gamma],
        "task": [task], "train_frac": [train_frac], "best_epoch": [best_epoch],
        "val_auc": [best_auc], "test_auc": [auc],
        "test_auc_ci_low": [auc_low], "test_auc_ci_high": [auc_high],
        "test_prc": [prc], "test_prc_ci_low": [prc_low], "test_prc_ci_high": [prc_high],
    })

    out_dir = ROOT / "results" / "linear_probe" / str(time)
    out_dir.mkdir(parents=True, exist_ok=True)

    fname = f"linear_probe_{method}_gamma_{gamma}_task_{task}_train_frac_{train_frac}.csv"
    results_df.to_csv(out_dir / fname, index=False)
    print(f"Results saved to: {out_dir / fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear probe evaluation")
    parser.add_argument("--method", type=str, default="ontology_aware",
                        choices=["ontology_aware", "flat_diagnosis", "simclr"],
                        help="Pretraining method")
    parser.add_argument("--time", type=str, default="48h",
                        choices=["48h", "72h", "96h"])
    parser.add_argument("--gamma", type=float, default=5.0,
                        help="Gamma parameter from pretraining")
    parser.add_argument("--task", type=str, default="mortality",
                        choices=["mortality", "los"])
    parser.add_argument("--train_frac", type=float, default=0.1,
                        help="Fraction of training data (e.g., 0.01, 0.05, 0.1)")

    args = parser.parse_args()
    print(
        f"Linear probe: method={args.method}, time={args.time}, "
        f"gamma={args.gamma}, task={args.task}, train_frac={args.train_frac}"
    )

    main(args.method, args.time, args.gamma, args.task, args.train_frac)
