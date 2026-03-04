import numpy as np
import torch
from torch.utils.data import Dataset
import random


class PatientDataset(Dataset):
    """Dataset for patient vitals, labels, and clinical notes."""
    
    def __init__(
        self,
        indices,
        vitals_tensor,
        labels_tensor,
        notes=None,
        summaries=None,
        p_summary=0.5,
        teacher_logits_tensor=None,
    ):
        self.indices = np.array(indices, dtype=int)
        self.vitals = vitals_tensor
        self.labels = labels_tensor
        self.notes = notes
        self.summaries = summaries
        self.p_summary = float(p_summary)
        self.teacher_logits = teacher_logits_tensor

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        sample = {
            "x_vitals": self.vitals[idx],
            "y": self.labels[idx],
        }

        # Text field (notes or summaries)
        if self.notes is not None or self.summaries is not None:
            use_summary = (
                (self.summaries is not None)
                and (random.random() < self.p_summary)
            )
            if use_summary:
                x_text = self.summaries[idx]
            else:
                x_text = self.notes[idx]
            sample["x_text"] = x_text

        # Knowledge distillation
        if self.teacher_logits is not None:
            sample["teacher_logits"] = self.teacher_logits[idx]

        return sample


def make_collate(tokenizer=None):
    """Create a collate function for DataLoader."""
    def collate(batch):
        x_vitals = torch.stack([b["x_vitals"] for b in batch], dim=0)

        out = {
            "vitals": x_vitals,
            "y": torch.stack([b["y"] for b in batch], dim=0),
        }

        # Text tokenization
        if tokenizer is not None:
            texts = [b["x_text"] if isinstance(b.get("x_text"), str) else " " for b in batch]
            owners, input_ids_all, attn_mask_all = [], [], []
            for bidx, text in enumerate(texts):
                tok = tokenizer(
                    text,
                    return_overflowing_tokens=True,
                    truncation=True,
                    padding="max_length",
                    max_length=512,
                    stride=0,
                    return_attention_mask=True,
                    add_special_tokens=True,
                    return_tensors="pt",
                )
                ids, am = tok["input_ids"], tok["attention_mask"]
                input_ids_all.append(ids)
                attn_mask_all.append(am)
                owners.append(torch.full((ids.size(0),), bidx, dtype=torch.long))

            out["text_pack"] = (
                torch.cat(input_ids_all, dim=0),
                torch.cat(attn_mask_all, dim=0),
                torch.cat(owners, dim=0),
            )

        # Knowledge distillation logits
        if "teacher_logits" in batch[0]:
            out["teacher_logits"] = torch.stack([b["teacher_logits"] for b in batch], dim=0)

        return out
    return collate


class ContrastiveVitalsDataset(Dataset):
    """Dataset for self-supervised contrastive learning on vital signs."""
    def __init__(self, vitals_list):
        self.vitals_list = vitals_list
        self.val_idx = torch.arange(0, 12, dtype=torch.long)
        self.mask_idx = torch.arange(12, 24, dtype=torch.long)

    def __len__(self):
        return len(self.vitals_list)

    def __getitem__(self, idx):
        ts = self.vitals_list[idx]
        v1 = self.augment_ts(ts)
        v2 = self.augment_ts(ts)
        return v1, v2, idx

    def augment_ts(self, x):
        """Apply time series augmentations."""
        x_aug = x.clone()
        val_idx = self.val_idx.to(x_aug.device)
        mask_idx = self.mask_idx.to(x_aug.device)

        x_aug = jitter_by_index(x_aug, val_idx, mask_idx, sigma=0.01)
        x_aug = random_time_mask(x_aug, p=0.2)
        x_aug = random_feature_mask_by_index(x_aug, val_idx, mask_idx, p=0.2)

        return x_aug


def jitter_by_index(x, val_idx, mask_idx, sigma=0.01):
    """
    Add Gaussian noise to continuous value columns.
    Only applies noise where observation mask is 1.
    """
    x = x.clone()
    vals = x[:, val_idx]
    obs = x[:, mask_idx]

    noise = sigma * torch.randn_like(vals)
    vals = vals + noise * obs

    x[:, val_idx] = vals
    return x


def random_feature_mask_by_index(x, val_idx, mask_idx, p=0.2):
    """
    Randomly drop entire features across all timesteps.
    Sets both values and observation masks to 0.
    """
    x = x.clone()
    K = val_idx.numel()
    feat_mask = (torch.rand(K, device=x.device) < p)
    if feat_mask.any():
        drop_vals = val_idx[feat_mask]
        drop_mask = mask_idx[feat_mask]
        x[:, drop_vals] = 0.0
        x[:, drop_mask] = 0.0
    return x


def random_time_mask(x, p=0.05):
    """Randomly mask entire timesteps with probability p."""
    T = x.size(0)
    m = (torch.rand(T, device=x.device) < p)
    x = x.clone()
    x[m] = 0.0
    return x
