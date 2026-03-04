import os
import ast
import re
import argparse
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from rdflib import Graph, Namespace
from rdflib.namespace import RDFS

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.utils import cohort_creation_no_notes

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"
OUT_DIR_BASE = ROOT / "precomputed_similarity"

CACHE_ROOT = str(CACHE_DIR)
DIAGNOSIS_PATH = DATA_DIR / "diagnosis_table.csv"
TTL_PATH = ROOT / "ICD9CM.ttl"
ROOT_CODE = "001-999.99"

SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")


def uri_to_code(uri: str) -> str:
    """Convert ICD9 URI to code string."""
    return re.sub(r".*/ICD9CM/", "", str(uri))


def build_parent_map(ttl_path=TTL_PATH):
    """Build parent-child mapping from ICD-9 ontology."""
    g = Graph()
    g.parse(str(ttl_path), format="turtle")

    edges_all = []
    for s, o in g.subject_objects(RDFS.subClassOf):
        s, o = str(s), str(o)
        edges_all.append((uri_to_code(o), uri_to_code(s)))

    children_map = defaultdict(list)
    for p, c in edges_all:
        children_map[p].append(c)

    parent_map = {}
    for p, children in children_map.items():
        for c in children:
            parent_map[c] = p

    return parent_map


def get_path_to_root(code, parent_map, root_code=ROOT_CODE):
    """Get path from code to root in ICD hierarchy."""
    path = [code]
    while code in parent_map and code != root_code:
        code = parent_map[code]
        path.append(code)
    if root_code not in path:
        path.append(root_code)
    path.reverse()
    return path


_path_cache = {}

def get_path_cached(code, parent_map, root_code=ROOT_CODE):
    """Cached version of get_path_to_root."""
    if code not in _path_cache:
        _path_cache[code] = get_path_to_root(code, parent_map, root_code)
    return _path_cache[code]


def code_similarity(a, b, parent_map, root_code=ROOT_CODE, count_root=False):
    """
    Compute similarity s(a, b) between two ICD codes.
    
    s(a, b) = Shared(a, b) / (|Path(a) ∪ Path(b)| - 1)
    
    where Shared(a, b) = |Path(a) ∩ Path(b)| - 1 (excluding root)
    """
    path_a = get_path_cached(a, parent_map, root_code)
    path_b = get_path_cached(b, parent_map, root_code)

    if not count_root:
        path_a = path_a[1:] if len(path_a) > 0 else path_a
        path_b = path_b[1:] if len(path_b) > 0 else path_b

    set_a = set(path_a)
    set_b = set(path_b)
    union = len(set_a | set_b) or 1
    return len(set_a & set_b) / union


def patient_similarity_ontology(codes_A, codes_B, parent_map, root_code=ROOT_CODE):
    """
    Compute patient-level similarity Sim(A, B) using best-match averaging.
    
    For each code in A, find best matching code in B (and vice versa),
    then average both directions.
    
    Sim(A, B) = 0.5 * (avgA + avgB)
    """
    if not codes_A or not codes_B:
        return 0.0

    best_A = []
    for a in codes_A:
        sims = [code_similarity(a, b, parent_map, root_code) for b in codes_B]
        best_A.append(max(sims))

    best_B = []
    for b in codes_B:
        sims = [code_similarity(a, b, parent_map, root_code) for a in codes_A]
        best_B.append(max(sims))

    avg_A = sum(best_A) / len(best_A)
    avg_B = sum(best_B) / len(best_B)
    return 0.5 * (avg_A + avg_B)


def patient_similarity_flat(set_A, set_B):
    """
    Flat diagnosis matching: Jaccard similarity |A ∩ B| / |A ∪ B|
    
    This baseline ignores the ICD hierarchy and only counts exact code matches.
    """
    if not set_A or not set_B:
        return 0.0
    return len(set_A & set_B) / max(1, len(set_A | set_B))


# Multiprocessing globals
_CODES_LIST_ALL = None
_CODES_SETS_ALL = None
_PARENT_MAP = None
_METHOD = None


def _init_worker(codes_list_all, codes_sets_all, parent_map, method):
    """Initialize worker process globals."""
    global _CODES_LIST_ALL, _CODES_SETS_ALL, _PARENT_MAP, _METHOD
    _CODES_LIST_ALL = codes_list_all
    _CODES_SETS_ALL = codes_sets_all
    _PARENT_MAP = parent_map
    _METHOD = method


def _compute_row_upper(i: int):
    """Compute upper triangle values for row i."""
    N = len(_CODES_LIST_ALL)
    out = np.zeros((N - i - 1,), dtype=np.float32)

    if _METHOD == "flat_diagnosis":
        set_i = _CODES_SETS_ALL[i]
        for off, j in enumerate(range(i + 1, N)):
            out[off] = patient_similarity_flat(set_i, _CODES_SETS_ALL[j])
    else:
        codes_i = _CODES_LIST_ALL[i]
        for off, j in enumerate(range(i + 1, N)):
            out[off] = patient_similarity_ontology(
                codes_i, _CODES_LIST_ALL[j], 
                parent_map=_PARENT_MAP, root_code=ROOT_CODE
            )

    return i, out


def main(method="ontology_aware", time="48h", workers=None, chunksize=4):
    """Precompute patient similarity matrix."""
    OUT_DIR = OUT_DIR_BASE / f"{method}_{time}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    OUT_SIM_PATH = OUT_DIR / "icd_similarity_memmap.float32.npy"
    OUT_META_PATH = OUT_DIR / "icd_similarity_meta.npz"

    print(f"=== Computing Patient Similarity Matrix ===")
    print(f"Method: {method}, Time: {time}")
    
    print("Loading cohort with diagnosis codes...")
    df_all = cohort_creation_no_notes(
        task="mortality", split="train", time=time, cache_root=CACHE_ROOT
    )

    diag_df = pd.read_csv(DIAGNOSIS_PATH)
    diag_df["codes_list"] = diag_df["codes"].apply(
        lambda s: ast.literal_eval(s) if isinstance(s, str) else []
    )

    df_all = df_all.merge(
        diag_df[["SUBJECT_ID", "HADM_ID", "codes_list"]],
        on=["SUBJECT_ID", "HADM_ID"],
        how="left",
    )
    df_all["codes_list"] = df_all["codes_list"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    codes_list_all = df_all["codes_list"].tolist()
    icu_ids = df_all["ICUSTAY_ID"].values
    N = len(codes_list_all)

    print(f"N ICU stays: {N}")
    print(f"Similarity matrix size (float32): ~{(N*N*4)/1e9:.2f} GB")

    print("Building ICD parent map from ontology...")
    parent_map = build_parent_map()

    codes_sets_all = None
    if method == "flat_diagnosis":
        print("Precomputing code sets for flat diagnosis matching...")
        codes_sets_all = [set(x) for x in codes_list_all]

    print("Allocating memmap for similarity matrix...")
    sim = np.memmap(
        str(OUT_SIM_PATH),
        dtype="float32",
        mode="w+",
        shape=(N, N),
    )
    np.fill_diagonal(sim, 0.0)

    if workers is None:
        workers = max(1, os.cpu_count() or 1)

    print(f"Computing upper triangle in parallel with {workers} workers...")

    with mp.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(codes_list_all, codes_sets_all, parent_map, method),
    ) as pool:
        for i, upper_vals in tqdm(
            pool.imap_unordered(_compute_row_upper, range(N), chunksize=chunksize),
            total=N,
            desc="Rows (upper triangle)",
        ):
            sim[i, i+1:] = upper_vals

    print("Mirroring upper triangle to lower triangle...")
    for i in tqdm(range(N), desc="Mirror"):
        sim[i+1:, i] = sim[i, i+1:]

    print("Flushing memmap to disk...")
    sim.flush()

    print("Saving metadata...")
    np.savez(str(OUT_META_PATH), icustay_ids=icu_ids)

    print("Done.")
    print(f"Similarity memmap saved to: {OUT_SIM_PATH}")
    print(f"Metadata saved to: {OUT_META_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Precompute patient similarity matrix from ICD ontology"
    )
    parser.add_argument("--method", type=str, default="ontology_aware",
                        choices=["ontology_aware", "flat_diagnosis"],
                        help="Similarity method")
    parser.add_argument("--time", type=str, default="48h",
                        choices=["48h", "72h", "96h"],
                        help="Input horizon T")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers")
    parser.add_argument("--chunksize", type=int, default=4)
    args = parser.parse_args()

    main(
        method=args.method,
        time=args.time,
        workers=args.workers,
        chunksize=args.chunksize
    )
