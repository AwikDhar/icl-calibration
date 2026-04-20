import argparse
import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
import torch

from data_utils import set_prompt_params, IclDataset, IclDatasetSplit, load_dataset_with_embeddings
from utils.gen_utils import ROOT_DIR

def get_label_dict(dataset: str) -> Dict[int, str]:
    """
    Returns {label_idx: label_string} for a dataset via set_prompt_params.
    Uses a dummy model string since we only need label_dict, not tokenizer output.
    """
    params = {"dataset": dataset, "model": "meta-llama/Llama-3.1-8B-Instruct"}
    set_prompt_params(params)
 
    label_dict = params.get("label_dict", {})
    result = {}
    for idx, v in label_dict.items():
        if isinstance(v, dict):         # set_label_tokens format: {"label":..., "tokens":...}
            result[int(idx)] = v["label"]
        else:
            result[int(idx)] = str(v)
    return result
 
 
def build_dataset_arrays(
    dataset: str,
    label_dict: Dict[int, str],
    label_model: SentenceTransformer,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        input_embs  : (N, d_input)  L2-normalised sentence embeddings
        label_embs  : (N, d_label)  L2-normalised label embeddings, one per instance
    
    Label embeddings are computed by embedding each unique label string once,
    then mapping instances to their label embedding via their integer label index.
    """
    # load pre-computed sentence embeddings and integer labels from both splits
    train_sents, train_labels, train_input_embs = load_dataset_with_embeddings(dataset, "train")
    test_sents,  test_labels,  test_input_embs  = load_dataset_with_embeddings(dataset, "test")
 
    all_labels      = np.array(train_labels + test_labels, dtype=np.int64)    # (N,)
    all_input_embs  = np.concatenate([train_input_embs, test_input_embs], axis=0)  # (N, d)
 
    # L2-normalise input embeddings
    norms = np.linalg.norm(all_input_embs, axis=1, keepdims=True)
    all_input_embs = all_input_embs / (norms + 1e-9)
 
    # embed unique label strings once
    unique_idxs    = sorted(label_dict.keys())
    unique_strings = [label_dict[i] for i in unique_idxs]
 
    with torch.inference_mode():
        unique_label_embs = label_model.encode(
            unique_strings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )                                                                       # (n_labels, d_label)
 
    # L2-normalise label embeddings
    norms = np.linalg.norm(unique_label_embs, axis=1, keepdims=True)
    unique_label_embs = unique_label_embs / (norms + 1e-9)
 
    # map each instance's integer label → its embedding
    idx_to_row = {label_idx: row for row, label_idx in enumerate(unique_idxs)}
    label_emb_rows = np.array([idx_to_row[l] for l in all_labels], dtype=np.int32)
    all_label_embs = unique_label_embs[label_emb_rows]                         # (N, d_label)
 
    print(f"  [{dataset}] {len(all_labels)} instances, "
          f"{len(unique_idxs)} unique labels: {unique_strings}")
 
    return all_input_embs, all_label_embs
 
 
# ── kernel & sampling ─────────────────────────────────────────────────────────
 
def cosine_kernel_normalised(a: np.ndarray, b: np.ndarray) -> float:
    """(cosine_similarity + 1) / 2 for two already-L2-normalised vectors → [0, 1]."""
    cos = float(np.dot(a, b))
    assert abs(cos) <= 1.01, f"Cosine value {cos} is suspiciously out of range [-1, 1]"
    
    return (cos + 1.0) / 2.0
 
def rbf_kernel(a: np.ndarray, b: np.ndarray, gamma=0.5) -> float:
    np.exp(-gamma*np.abs(a-b)**2)
    
def sample_pairs(
    n_a: int,
    n_b: int,
    n_pairs: int,
    same_dataset: bool,
    rng: random.Random,
) -> List[Tuple[int, int]]:
    """Sample n_pairs (i, j) index pairs; if same_dataset, enforce i != j."""
    pairs = set()
    max_attempts = n_pairs * 20
    attempts = 0
    while len(pairs) < n_pairs and attempts < max_attempts:
        i = rng.randrange(n_a)
        j = rng.randrange(n_b)
        if same_dataset and i == j:
            attempts += 1
            continue
        pairs.add((i, j))
        attempts += 1
 
    if len(pairs) < n_pairs:
        print(f"  Warning: only sampled {len(pairs)}/{n_pairs} unique pairs "
              f"(dataset may be too small).")
    return list(pairs)
 
 
def compute_similarity(
    input_embs_a: np.ndarray,
    label_embs_a: np.ndarray,
    input_embs_b: np.ndarray,
    label_embs_b: np.ndarray,
    n_pairs: int,
    same_dataset: bool,
    rng: random.Random,
) -> float:
    """
    Mean over n_pairs sampled pairs of:
        K(i, j) = K_input(input_i, input_j) * K_label(label_i, label_j)
 
    where K_input and K_label are both (cosine + 1) / 2.
    """
    pairs = sample_pairs(len(input_embs_a), len(input_embs_b), n_pairs, same_dataset, rng)
 
    kernel_vals = []
    for i, j in pairs:
        k_input = cosine_kernel_normalised(input_embs_a[i], input_embs_b[j])
        k_label = cosine_kernel_normalised(label_embs_a[i], label_embs_b[j])
        kernel_vals.append(k_input * k_label)
 
    return float(np.mean(kernel_vals))
 
 
# ── main ──────────────────────────────────────────────────────────────────────
 
def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
 
    train_datasets  = [s.strip() for s in args.train_datasets.split(",")]
    unseen_datasets = [s.strip() for s in args.unseen_datasets.split(",")]
    all_datasets    = list(dict.fromkeys(train_datasets + unseen_datasets))
 
    rng = random.Random(args.seed)
 
    # ── 1. label embedding model ──────────────────────────────────────────────
    print(f"Loading label embedding model: {args.embedding_model}\n")
    HF_HOME = os.environ.get('HF_HOME')     
    cache_dir = os.environ.get('HF_HUB_CACHE', HF_HOME)
    
    label_model = SentenceTransformer(
        args.embedding_model,
        device="cuda:0",
        cache_folder=cache_dir,
    )
 
    # ── 2. pre-compute per-dataset arrays ─────────────────────────────────────
    print("Pre-computing input + label embeddings per dataset...")
    input_embs: Dict[str, np.ndarray] = {}
    label_embs: Dict[str, np.ndarray] = {}
 
    for ds in all_datasets:
        label_dict = get_label_dict(ds)
        if not label_dict:
            raise ValueError(
                f"Could not extract label strings for '{ds}'. "
                "Check set_prompt_params covers this dataset."
            )
        input_embs[ds], label_embs[ds] = build_dataset_arrays(ds, label_dict, label_model)
 
    # ── 3. pairwise kernel similarities ──────────────────────────────────────
    # each train dataset vs itself (intra) + all unseen datasets (inter)
    print("\nComputing pairwise kernel similarities...")
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
 
    for train_ds in train_datasets:
        results[train_ds] = {}
        comparisons = [train_ds] + unseen_datasets      # self-entry first
 
        for other_ds in comparisons:
            same = (train_ds == other_ds)
            sim  = compute_similarity(
                input_embs[train_ds], label_embs[train_ds],
                input_embs[other_ds], label_embs[other_ds],
                n_pairs=args.n_pairs,
                same_dataset=same,
                rng=rng,
            )
            tag = " (intra)" if same else ""
            print(f"  {train_ds} vs {other_ds}{tag}: "
                  f"similarity={sim:.4f}  shift={1-sim:.4f}")
 
            results[train_ds][other_ds] = {
                "similarity":   round(sim,       6),
                "domain_shift": round(1.0 - sim, 6),
            }
 
    # ── 4. save ───────────────────────────────────────────────────────────────
    save_path = ROOT_DIR/"data"/"domain_shift.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {save_path}")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Measure domain shift between ICL datasets via per-instance cosine kernel."
    )
    parser.add_argument("--train_datasets", required=True,
                        help="Comma-separated training dataset names, e.g. agnews,sst2,trec")
    parser.add_argument("--unseen_datasets", required=True,
                        help="Comma-separated unseen dataset names, e.g. yelp_reviews,banking77")
    parser.add_argument("--embedding_model", required=False, default="google/embeddinggemma-300m",
                        help="SentenceTransformer model for label string embeddings")
    parser.add_argument("--n_pairs", type=int, default=1000,
                        help="Pairs to sample per (A, B) comparison (default: 1000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", type=int, default=0)
 
    args = parser.parse_args()

    main(args)