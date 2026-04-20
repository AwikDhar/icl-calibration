#!/usr/bin/env python3
"""
MRL (Matryoshka Representation Learning) Truncation Benchmark
=============================================================
Evaluates an embedding model at multiple truncated dimensions to show
how quality degrades as you shrink embedding size.
 
Tested with:
  - Qwen/Qwen3-Embedding-0.6B / 4B / 8B
  - google/embeddinggemma-300m  (needs transformers preview branch, see below)
  - nomic-ai/nomic-embed-text-v1.5
  - mixedbread-ai/mxbai-embed-large-v1
 
Usage:
    python benchmark_mrl.py --model Qwen/Qwen3-Embedding-0.6B
    python benchmark_mrl.py --model Qwen/Qwen3-Embedding-4B --dims 128 256 512 1024 2048
    python benchmark_mrl.py --model google/embeddinggemma-300m --dims 128 256 512
    python benchmark_mrl.py --model Qwen/Qwen3-Embedding-8B --tasks all --output results.json
 
Requirements:
    pip install "mteb>=1.5" "sentence-transformers>=5.0" torch numpy tabulate
 
    For EmbeddingGemma only (needs unreleased transformers architecture):
    pip install git+https://github.com/huggingface/transformers@v4.56.0-Embedding-Gemma-preview
"""
 
from __future__ import annotations
 
import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional
 
import numpy as np
import torch
 
# ---------------------------------------------------------------------------
# Task presets
# ---------------------------------------------------------------------------
TASK_PRESETS: dict[str, list[str]] = {
    "retrieval": [
        "BrightBiologyRetrieval",          # reasoning-intensive; released 2024, low training overlap
        "LegalBenchConsumerContractsQA",  # domain-specific legal QA, niche enough to be clean
    ],
    "sts": [
        "STS22.v2",                 # v2 revision, less overlap than STSBenchmark
        "STSBenchmark",             # keep as baseline reference point
    ],
    "classification": [
        "TweetSentimentExtractionClassification",  # social media domain, less clean-room data
        "EmotionClassification",    # reasonably clean; less represented than Banking77
    ],
    "clustering": [
        "ArXivHierarchicalClustering",  # newer, hierarchical; harder to train against
    ],
    "reranking": [
        "MindSmallReranking",       # still fine; news domain not heavily in embedding training
    ],
}
 
ALL_TASKS: list[str] = [t for tasks in TASK_PRESETS.values() for t in tasks]
 
 
# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
 
def load_model(model_name: str, cache_dir: str, device: str):
    """
    Load a SentenceTransformer, passing cache_folder explicitly.
 
    SentenceTransformer does NOT read HF_HUB_CACHE from the environment
    automatically — cache_folder must be passed as a constructor argument.
 
    Model-specific notes:
      EmbeddingGemma: requires trust_remote_code=True for Gemma3TextModel.
                      Also needs the transformers preview branch (see docstring).
    """
    from sentence_transformers import SentenceTransformer
 
    base_kwargs: dict = dict(device=device, cache_folder=cache_dir)
    model_kwargs={'dtype':torch.bfloat16, 'attn_implementation':"kernels-community/flash-attn2"}
    
    if "embeddinggemma" in model_name.lower():
        print("  [EmbeddingGemma] using trust_remote_code=True")
        model = SentenceTransformer(model_name, trust_remote_code=True, **base_kwargs, model_kwargs=model_kwargs)
    else:
        model = SentenceTransformer(model_name, **base_kwargs, model_kwargs=model_kwargs)
 
    return model
 
 
# ---------------------------------------------------------------------------
# Score extraction
# ---------------------------------------------------------------------------
 
def extract_score(result) -> Optional[float]:
    """Pull the primary scalar metric from an MTEB TaskResult."""
    for split in ("test", "dev", "validation"):
        if split not in result.scores:
            continue
        s = result.scores[split]
        if isinstance(s, list):
            s = s[0]
        for key in (
            "ndcg_at_10",        # retrieval
            "cos_sim_spearman",  # STS
            "spearman",
            "accuracy",          # classification
            "v_measure",         # clustering
            "map",               # reranking
            "main_score",
        ):
            if key in s and s[key] is not None:
                return float(s[key])
    return None
 
 
def task_type_of(task_name: str) -> str:
    for ttype, names in TASK_PRESETS.items():
        if task_name in names:
            return ttype
    return "unknown"
 
 
# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------
 
def run_benchmark(
    model_name: str,
    dims: list[Optional[int]],
    task_names: list[str],
    output_path: Optional[Path],
    batch_size: int,
    device: str,
    cache_dir: str,
) -> dict:
    import mteb
 
    print(f"\n{'='*70}")
    print(f"  Model  : {model_name}")
    print(f"  Cache  : {cache_dir}")
    print(f"  Tasks  : {', '.join(task_names)}")
    print(f"  Dims   : {[str(d) if d else 'full' for d in dims]}")
    print(f"  Device : {device}")
    print(f"{'='*70}\n")
 
    print("Loading model ...")
    t0 = time.time()
    model = load_model(model_name, cache_dir=cache_dir, device=device)
    full_dim = model.get_sentence_embedding_dimension()
    original_truncate_dim = model.truncate_dim  # save to restore after each run
    print(f"  Loaded in {time.time()-t0:.1f}s | full_dim={full_dim}")
    print(f"  Registered prompts: {list(model.prompts.keys()) if model.prompts else 'none'}\n")
 
    # Clamp any dim >= full_dim down to None (= use full dim)
    resolved: list[Optional[int]] = [
        None if (d is None or d >= full_dim) else d for d in dims
    ]
 
    print("Loading MTEB task metadata ...")
    tasks = mteb.get_tasks(tasks=task_names, languages=["eng"])
 
    all_results: dict = {}
 
    for dim in resolved:
        dim_label = "full" if dim is None else str(dim)
        print(f"\n{'─'*60}")
        print(f"  dim = {dim_label}")
        print(f"{'─'*60}")
 
        # KEY FIX: mutate truncate_dim directly on the SentenceTransformer.
        #
        # We do NOT wrap the model in a custom class. MTEB's RetrievalEvaluator
        # (and all other evaluators in recent mteb versions) perform an isinstance
        # check against registered protocol classes (Encoder, SearchInterface,
        # CrossEncoder). A plain wrapper class fails this check even if it has
        # the right methods, because isinstance checks the MRO, not duck typing.
        #
        # SentenceTransformer already satisfies the Encoder protocol natively.
        # Its truncate_dim attribute is used inside encode() to slice + renormalise
        # embeddings (ST >= 2.7). Setting it to None restores full-dim behaviour.
        model.truncate_dim = dim
 
        t_start = time.time()
        evaluation = mteb.MTEB(tasks=tasks)
        results = evaluation.run(
            model,
            verbosity=1,
            encode_kwargs={"batch_size": batch_size},
            output_folder=f"/tmp/mteb_mrl/{model_name.replace('/', '__')}/dim_{dim_label}",
            overwrite_results=True,
        )
        elapsed = time.time() - t_start
 
        dim_scores: dict[str, Optional[float]] = {r.task_name: extract_score(r) for r in results}
        all_results[dim_label] = dim_scores
        print(f"  Done in {elapsed:.1f}s | {dim_scores}")
 
    # Restore original truncate_dim
    model.truncate_dim = original_truncate_dim
 
    all_results["_meta"] = {
        "model": model_name,
        "full_dim": full_dim,
        "dims_evaluated": [str(d) if d else "full" for d in resolved],
        "tasks": task_names,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
 
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(all_results, indent=2))
        print(f"\nResults saved -> {output_path}")
 
    return all_results
 
 
# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------
 
def print_table(results: dict) -> None:
    try:
        from tabulate import tabulate
        use_tabulate = True
    except ImportError:
        use_tabulate = False
 
    meta = results.get("_meta", {})
    dim_labels = [k for k in results if not k.startswith("_")]
    if not dim_labels:
        return
 
    task_names = list(results[dim_labels[0]].keys())
 
    print(f"\n{'='*70}")
    print(f"  SCORES (x100)  |  {meta.get('model','')}  |  full_dim={meta.get('full_dim','?')}")
    print(f"{'='*70}")
 
    headers = ["Task", "Type"] + [f"dim={d}" for d in dim_labels]
    rows = []
    for t in task_names:
        row = [t, task_type_of(t)]
        for d in dim_labels:
            v = results[d].get(t)
            row.append(f"{v*100:.2f}" if v is not None else "N/A")
        rows.append(row)
 
    avg = ["AVG", "—"]
    for d in dim_labels:
        vals = [results[d][t] for t in task_names if results[d].get(t) is not None]
        avg.append(f"{np.mean(vals)*100:.2f}" if vals else "N/A")
    rows.append(avg)
 
    if use_tabulate:
        print(tabulate(rows, headers=headers, tablefmt="github"))
    else:
        w = 30
        print("  ".join(h.ljust(w) for h in headers))
        print("-" * (w * len(headers)))
        for row in rows:
            print("  ".join(str(c).ljust(w) for c in row))
 
    # Retention vs full dim
    full_label = dim_labels[-1]
    ret_headers = ["Task"] + [f"dim={d}" for d in dim_labels[:-1]]
    ret_rows = []
    for t in task_names:
        fv = results[full_label].get(t)
        if not fv:
            continue
        row = [t]
        for d in dim_labels[:-1]:
            v = results[d].get(t)
            row.append(f"{v/fv*100:.1f}%" if v is not None else "N/A")
        ret_rows.append(row)
 
    avg_ret = ["AVG"]
    for d in dim_labels[:-1]:
        pairs = [
            (results[d].get(t), results[full_label].get(t))
            for t in task_names
            if results[d].get(t) is not None and results[full_label].get(t)
        ]
        pcts = [v / fv * 100 for v, fv in pairs]
        avg_ret.append(f"{np.mean(pcts):.1f}%" if pcts else "N/A")
    ret_rows.append(avg_ret)
 
    print(f"\n{'='*70}")
    print("  SCORE RETENTION  (% of full-dim score)")
    print(f"{'='*70}")
    if use_tabulate:
        print(tabulate(ret_rows, headers=ret_headers, tablefmt="github"))
    else:
        w = 20
        print("  ".join(h.ljust(w) for h in ret_headers))
        print("-" * (w * len(ret_headers)))
        for row in ret_rows:
            print("  ".join(str(c).ljust(w) for c in row))

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark MRL embedding model at multiple truncation dims.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", "-m", required=True,
                   help="HuggingFace model ID, e.g. Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--dims", "-d", nargs="+", type=int,
                   default=[64, 128, 256, 512],
                   help="Truncation dims to test. Full dim is always appended. "
                        "Default: 64 128 256 512")
    p.add_argument("--tasks", "-t", nargs="+",
                   choices=list(TASK_PRESETS.keys()) + ["all"],
                   default=["retrieval", "sts"],
                   help="Task groups (default: retrieval sts). 'all' for everything.")
    p.add_argument("--task-names", nargs="+", default=None,
                   help="Override with specific MTEB task names.")
    p.add_argument("--output", "-o", type=Path, default=None,
                   help="Save JSON results to this path.")
    p.add_argument("--batch-size", "-b", type=int, default=12,
                   help="Encoding batch size (default: 32).")
    p.add_argument("--device", default=None,
                   help="cpu / cuda / mps. Auto-detected if unset.")
    p.add_argument("--cache-dir", default=None,
                   help="HF model cache dir. Defaults to $HF_HUB_CACHE or "
                        "~/.cache/huggingface/hub")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Cache dir — explicit flag > env var > HF default
    cache_dir: str = (
        args.cache_dir
        or os.environ.get("HF_HUB_CACHE")
        or os.path.expanduser("~/.cache/huggingface/hub")
    )

    # Tasks
    if args.task_names:
        task_names = args.task_names
    elif "all" in args.tasks:
        task_names = ALL_TASKS
    else:
        task_names = [t for g in args.tasks for t in TASK_PRESETS[g]]

    # Dims: sorted user list + None sentinel for full (must be last)
    dims: list[Optional[int]] = sorted(set(args.dims)) + [None]

    results = run_benchmark(
        model_name=args.model,
        dims=dims,
        task_names=task_names,
        output_path=args.output,
        batch_size=args.batch_size,
        device=device,
        cache_dir=cache_dir,
    )

    print_table(results)


if __name__ == "__main__":
    main()