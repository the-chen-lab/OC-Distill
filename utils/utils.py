import os
import re
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize
from pathlib import Path

# Path configuration
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"
DEMO_CSV = DATA_DIR / "demographic_table.csv"
NOTES_CSV = DATA_DIR / "notes_table.csv"
BENCHMARK_DATA_DIR = ROOT / "mimic3_benchmark_data"

TASK_FILENAME = {
    "mortality": "mortality_{split}_{time}.npz",
    "los": "los_{split}_{time}.npz",
}


def set_seed(seed=0):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def preprocess_notes(df):
    """Clean and preprocess clinical notes text."""
    df = df.copy()
    df['TEXT'] = df['TEXT'].fillna(' ')
    df['TEXT'] = df['TEXT'].str.replace('\n', ' ', regex=False)
    df['TEXT'] = df['TEXT'].str.lower().str.strip()

    def clean_text(x):
        x = re.sub(r'\[(.*?)\]', '', x)
        x = re.sub(r'[-_=]{2,}', '', x)
        x = re.sub(r'\s+', ' ', x)
        x = re.sub(r'\b\d{1,2}:\d{2}\s?(am|pm)\b', '', x)
        return x.strip()

    df['TEXT'] = df['TEXT'].apply(clean_text)
    return df


def build_vitals_tensors(df, norm_stats_path):
    """Build normalized vital sign tensors from DataFrame."""
    norm_stats = torch.load(norm_stats_path)
    mean = torch.tensor(norm_stats["mean"], dtype=torch.float32)
    std = torch.tensor(norm_stats["std"], dtype=torch.float32)
    cont_cols = torch.tensor(range(12), dtype=torch.long)

    out = []
    for arr in df["ts"].tolist():
        ts = torch.as_tensor(arr, dtype=torch.float32)
        ts_out = ts.clone()
        vitals = ts_out[:, cont_cols]
        vitals = (vitals - mean) / std
        ts_out[:, cont_cols] = vitals
        out.append(ts_out)
    return out


def cohort_creation_no_notes(task, split, time, cache_root):
    """Create cohort DataFrame without clinical notes."""
    cache_path = CACHE_DIR / task / time / f"{split}_cohort_no_notes.pkl"
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    # Load from NPZ file
    fname = TASK_FILENAME[task].format(split=split, time=time)
    data = np.load(BENCHMARK_DATA_DIR / fname)
    
    stay_ids = data["stay_id"]
    ts_list = data["X"]
    labels = data["y"]

    base = pd.DataFrame({"ICUSTAY_ID": stay_ids, "label": labels})
    ts_per_stay = [ts_list[i] for i in range(len(stay_ids))]
    base["ts"] = pd.Series(ts_per_stay, dtype=object)
    base["t_len"] = base["ts"].apply(lambda a: int(np.asarray(a).shape[0]))
    base["n_feat"] = base["ts"].apply(
        lambda a: int(np.asarray(a).shape[-1]) if np.asarray(a).ndim >= 2 else int(np.asarray(a).shape[0])
    )

    demo = pd.read_csv(DEMO_CSV)
    df = base.merge(demo, on="ICUSTAY_ID", how="inner")

    pd.to_pickle(df, cache_path)
    return df


def attach_notes(df_all, task, split, time, cache_root):
    """Attach clinical notes to cohort DataFrame."""
    cache_path = os.path.join(cache_root, task, time, f"{split}_cohort.pkl")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    notes = pd.read_csv(NOTES_CSV)

    stay_meta = df_all[["ICUSTAY_ID", "HADM_ID", "INTIME", "OUTTIME"]].drop_duplicates()
    notes_w_stay = notes.merge(stay_meta, on="HADM_ID", how="inner")

    notes_w_stay["INTIME"] = pd.to_datetime(notes_w_stay["INTIME"], errors="coerce")
    notes_w_stay["CHARTTIME"] = pd.to_datetime(notes_w_stay["CHARTTIME"], errors="coerce")

    mask = (notes_w_stay["CHARTTIME"] >= notes_w_stay["INTIME"]) & \
           (notes_w_stay["CHARTTIME"] <= notes_w_stay["INTIME"] + pd.Timedelta(hours=int(time.split('h')[0])))
    notes_w_stay = notes_w_stay[mask].copy()

    notes_w_stay = notes_w_stay.sort_values(["ICUSTAY_ID", "CHARTTIME"])
    notes_agg = (notes_w_stay.groupby("ICUSTAY_ID")["TEXT"]
                 .apply(lambda s: " ".join(s.dropna().astype(str)))
                 .reset_index())

    notes_table = preprocess_notes(notes_agg)
    df_all = df_all.merge(notes_table, on="ICUSTAY_ID", how="inner")
    pd.to_pickle(df_all, cache_path)
    return df_all


def load_cohort(task, split, time, cache_root):
    """Load or create cohort with clinical notes."""
    cache_path = os.path.join(cache_root, task, time, f"{split}_cohort.pkl")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.exists(cache_path):
        return pd.read_pickle(cache_path)
    df = cohort_creation_no_notes(task, split, time, cache_root)
    df = attach_notes(df, task, split, time, cache_root)
    pd.to_pickle(df, cache_path)
    return df


def split_by_subject_indices(df, train_frac=0.8, val_frac=0.2):
    """Split data by subject ID to prevent data leakage."""
    subjects = df["SUBJECT_ID"].unique()
    np.random.shuffle(subjects)

    n_total = len(subjects)
    n_train = int(round(train_frac * n_total))
    n_val = int(round(val_frac * n_total))
    train_groups = set(subjects[:n_train])
    val_groups = set(subjects[n_train:n_train+n_val])

    groups = df["SUBJECT_ID"].to_numpy()
    train_idx = np.where(np.isin(groups, list(train_groups)))[0]
    val_idx = np.where(np.isin(groups, list(val_groups)))[0]

    return train_idx, val_idx


def bootstrap_ci(y_true, y_score, B=1000, task_type="binary"):
    """Compute bootstrap confidence intervals for AUROC and AUPRC."""
    if not isinstance(y_true, np.ndarray):
        y_true = y_true.numpy()
        y_score = y_score.numpy()
    rng = np.random.default_rng(0)

    if task_type == "binary":
        base_auc = roc_auc_score(y_true, y_score)
        base_prc = average_precision_score(y_true, y_score)

        aucs, prcs = [], []
        while len(aucs) < B:
            idx = rng.integers(0, y_true.shape[0], size=y_true.shape[0])
            yt = y_true[idx]
            ys = y_score[idx]
            aucs.append(roc_auc_score(yt, ys))
            prcs.append(average_precision_score(yt, ys))

    elif task_type == "multiclass":
        num_classes = y_score.shape[1]
        y_true_oh = label_binarize(y_true, classes=np.arange(num_classes))

        base_auc = roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
        base_prc = average_precision_score(y_true_oh, y_score, average="macro")

        aucs, prcs = [], []
        while len(aucs) < B:
            idx = rng.integers(0, y_true.shape[0], size=y_true.shape[0])
            yt = y_true[idx]
            ys = y_score[idx]
            yt_oh = y_true_oh[idx]

            aucs.append(roc_auc_score(yt, ys, multi_class="ovr", average="macro"))
            prcs.append(average_precision_score(yt_oh, ys, average="macro"))

    lower_auc = np.percentile(aucs, 2.5)
    upper_auc = np.percentile(aucs, 97.5)
    lower_prc = np.percentile(prcs, 2.5)
    upper_prc = np.percentile(prcs, 97.5)
    
    return base_auc, lower_auc, upper_auc, base_prc, lower_prc, upper_prc
