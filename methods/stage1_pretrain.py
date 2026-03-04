import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.data import ContrastiveVitalsDataset
from utils.models import VitalTransformer, ProjectionHead
from utils.utils import set_seed, cohort_creation_no_notes, build_vitals_tensors

os.environ["TOKENIZERS_PARALLELISM"] = "false"

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "cache"
SIM_DIR_BASE = ROOT / "precomputed_similarity"
SAVE_DIR = ROOT / "checkpoints" / "stage1_pretrain"


def batch_similarity_from_precomputed(sim_all, idx_batch, device):
    """Extract batch similarity submatrix from precomputed similarity."""
    idx_np = idx_batch.cpu().numpy()
    sub = sim_all[np.ix_(idx_np, idx_np)]
    return torch.from_numpy(sub).to(device)


def ontology_weighted_nt_xent(z1, z2, tau=1.0, sim_batch=None, gamma=1.0):
    """Ontology-Weighted NT-Xent Loss (L_OW-NTXent)."""
    device = z1.device
    B = z1.size(0)

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)

    N = 2 * B
    idx = torch.arange(N, device=device)

    logits = torch.matmul(z, z.T) / tau
    p_idx = torch.arange(N, device=device) % B

    if sim_batch is not None:
        S = sim_batch.to(device)
        # Phi(s) = (1 - s)^gamma (power transform)
        W_pat = (1.0 - S).pow(gamma)
        Pi = p_idx.view(N, 1).expand(N, N)
        Pj = p_idx.view(1, N).expand(N, N)
        W = W_pat[Pi, Pj]
    else:
        W = 1

    pos_idx = torch.cat([idx[B:], idx[:B]])
    eye = torch.eye(N, dtype=torch.bool, device=device)
    neg_mask = ~eye
    neg_mask[idx, pos_idx] = False

    pos_logits = logits[idx, pos_idx]
    exp_logits = torch.exp(logits)
    neg_exp = exp_logits * W * neg_mask.float()
    neg_sum = neg_exp.sum(dim=1)

    denom = torch.exp(pos_logits) + neg_sum
    loss = -(pos_logits - torch.log(denom))

    return loss.mean()


def main(method, time, gamma):
    """Main training loop for ontology-aware contrastive pre-training."""
    set_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== Stage 1: Ontology-Aware Contrastive Pretraining ===")
    print(f"Method: {method}, Time: {time}, Gamma: {gamma}")
    print(f"Device: {device}")

    save_name = f"vitals_encoder_{method}_time_{time}_gamma{gamma}.pt"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    save_path = SAVE_DIR / save_name

    # Load data
    df_all = cohort_creation_no_notes(
        task='mortality', split='train', time=time, cache_root=str(CACHE_DIR)
    )

    norm_stats_path = ROOT / f"vital_norm_stats_{time}.pt"
    vitals_tensor_all = build_vitals_tensors(df_all, norm_stats_path=norm_stats_path)
    N = len(vitals_tensor_all)

    # Load precomputed similarity matrix (skip for SimCLR baseline)
    if method != "simclr":
        sim_dir = SIM_DIR_BASE / f"{method}_{time}"
        sim_path = sim_dir / "icd_similarity_memmap.float32.npy"
        meta_path = sim_dir / "icd_similarity_meta.npz"
        sim_all = np.memmap(str(sim_path), dtype="float32", mode="r")
        sim_all = sim_all.reshape(N, N)
        
        meta = np.load(meta_path)
        icu_ids_pre = meta["icustay_ids"]
        icu_ids_df = df_all["ICUSTAY_ID"].values
        assert np.array_equal(icu_ids_pre, icu_ids_df), "ICUSTAY_ID ordering mismatch!"
    else:
        sim_all = None

    # Training config
    batch_size = 4096 if time == "48h" else 2048
    lr = 1e-4
    tau = 1.0  # Temperature
    num_epochs = 1000

    ds = ContrastiveVitalsDataset(vitals_tensor_all)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )

    vital_encoder = VitalTransformer(
        input_dim=vitals_tensor_all[0].shape[1],
        hidden_dim=768, n_heads=4, n_layers=2
    ).to(device)
    projector = ProjectionHead(input_dim=768, proj_dim=128, hidden_dim=2048).to(device)
    
    params = list(vital_encoder.parameters()) + list(projector.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr)

    for epoch in range(num_epochs):
        vital_encoder.train()
        projector.train()

        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for v1, v2, idx_batch in pbar:
            v1 = v1.to(device)
            v2 = v2.to(device)
            idx_batch = idx_batch.to(device)

            z1_enc = vital_encoder(v1)
            z2_enc = vital_encoder(v2)
            z1 = projector(z1_enc)
            z2 = projector(z2_enc)

            sim_batch = None
            if sim_all is not None:
                sim_batch = batch_similarity_from_precomputed(sim_all, idx_batch, device)

            loss = ontology_weighted_nt_xent(
                z1, z2, tau=tau, sim_batch=sim_batch, gamma=gamma
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": epoch_loss / num_batches})

        print(f"Epoch {epoch+1}: avg loss = {epoch_loss / num_batches:.4f}")

    torch.save(vital_encoder.state_dict(), save_path)
    print(f"Saved encoder to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 1: Ontology-Aware Contrastive Pretraining"
    )
    parser.add_argument("--method", type=str, default="ontology_aware",
                        choices=["ontology_aware", "flat_diagnosis", "simclr"],
                        help="Similarity method: ontology_aware (ours), flat_diagnosis, or simclr")
    parser.add_argument("--time", type=str, default="48h",
                        choices=["48h", "72h", "96h"],
                        help="Input horizon T")
    parser.add_argument("--gamma", type=float, default=5.0,
                        help="Power transform exponent γ (default: 5)")
    args = parser.parse_args()
    
    print(f"Config: method={args.method}, time={args.time}, gamma={args.gamma}")
    main(args.method, args.time, args.gamma)
