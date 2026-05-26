"""
Visualise calibrator training coverage via UMAP.

For each unseen (LLM, dataset) combo, produces a figure with two subplots
(ENTROPY | SIMILARITY sampling), where:
  - background scatter = training trajectories (coloured by LLM×dataset combo)
  - foreground scatter = unseen combo projected onto the FROZEN training UMAP

Save paths:
  --coverage per_combo : analysis/coverage/<unseen_llm>/<unseen_dataset>/map.png
  --coverage all       : analysis/coverage/<unseen_llm>/<STRATEGY>/all_datasets_map.png
"""

import argparse
import os
from pathlib import Path
import random
import warnings
from itertools import product
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn.functional as F
from umap import UMAP
import msgspec
from tqdm import tqdm

# warnings.filterwarnings("ignore", category=FutureWarning)

# ─── feature indices (after recalculate_features, final input layout) ──────
# [0] gt_prob_mse   [1] pred_prob (confidence)   [2] shifted_correctness
# [3] shifted_gt_prob   [4] second_highest_prob
CONF_IDX = 1        # pred_prob  == confidence at this shot

SAMPLING_STRATEGIES = ("ENTROPY", "SIMILARITY")
# SAMPLING_STRATEGIES = ("ENTROPY",)

def update_plt_rcparams(hide_train_legends: bool):
    if hide_train_legends:
        plt.rcParams.update({
            'font.size': 23,        
            'lines.linewidth': 2,
            'grid.linewidth': 1.2,
            'xtick.labelsize': 23,    
            'ytick.labelsize': 23,   
            'legend.fontsize': 25,    
            'axes.labelsize': 25      
        })
    else:
        plt.rcParams.update({
            'font.size': 20,        
            'lines.linewidth': 2,
            'grid.linewidth': 1.2,
            'xtick.labelsize': 20,    
            'ytick.labelsize': 20,    
            'legend.fontsize': 25,    
            'axes.labelsize': 20      
        })

# ═══════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_raw(
    llm: str,
    dataset: str,
    split: str,
    shots_start: int = None,
    shots_end: int = None,
) -> Dict[str, List]:

    path = f"calibration/datasets/{llm.replace('/','_')}/{dataset}/{split}.json"
    with open(path) as f:
        raw = msgspec.json.decode(f.read())

    data = {sampling_strategy: [] for sampling_strategy in SAMPLING_STRATEGIES}
    for item in raw:
        strat = item["sampling_strategy"].upper()
        if strat not in data:
            continue
        data[strat].append({
            'inputs': torch.tensor(
                item["inputs"][shots_start:shots_end], dtype=torch.float32),   # (T, F)
            'logits': torch.tensor(
                item["logits"][shots_start:shots_end], dtype=torch.float32),   # (T, C)
            'labels': torch.tensor(
                item["labels"][shots_start:shots_end], dtype=torch.int64),     # (T,)
            'sampling_strategy': item["sampling_strategy"].upper(),
        })
    return data


# ═══════════════════════════════════════════════════════════════════════════
# 2.  AUGMENTATION  (ported from training utils, CPU torch)
# ═══════════════════════════════════════════════════════════════════════════

def apply_temp_augmentation(item: Dict) -> None:
    """In-place temperature scaling on item['logits']."""
    item['inputs'][0][2] = 0.5
    mean_conf = item['inputs'][:, 0].mean()
    beta = 4 if mean_conf < 0.95 else 5
    max_temp = 1 + torch.clamp(torch.tanh(beta * (mean_conf - 0.5)), min=0)
    temp = random.uniform(1, max_temp.item())
    item['logits'] = item['logits'] / temp


def apply_label_augmentation(item: Dict) -> None:
    """In-place label resampling on item['labels']."""
    mean_conf = item['inputs'][:, 0].mean()
    simulate_calibrated = random.random() < mean_conf/2
    
    if simulate_calibrated:
        beta = 1 if mean_conf < 0.95 else 2
        max_temp = 1 + torch.clamp(torch.tanh(beta * (mean_conf - 0.5)), min=0, max=0.3)
        temp = random.uniform(1, max_temp.item())
    else:
        beta = 1 if mean_conf < 0.95 else 2
        max_temp = 2 + 3*torch.clamp(torch.tanh(beta * (mean_conf - 0.4)), min=0.5)
        temp = random.uniform(2, max_temp.item())        
        
    logits = item['logits'] / temp
    probs = F.softmax(logits, dim=-1)
    preds = torch.multinomial(probs, 1).squeeze(-1)
    item['labels'] = preds

def pick_augmentation(temp_augment: bool, label_augment: bool) -> str | None:
    p_temp  = 0.5 if temp_augment  else 0.0
    p_label = 0.3 if label_augment else 0.0
    p_none  = 1.0 - p_temp - p_label

    augmentation = random.choices(
        ['temp', 'label', None],
        weights=[p_temp, p_label, p_none],
        k=1,
    )[0]
    
    return augmentation

def augment_item(item: Dict, augmentation: str | None) -> Dict:
    item_clone = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in item.items()}

    if augmentation == 'temp':
        apply_temp_augmentation(item_clone)
        
        probs = F.softmax(item_clone['logits'], dim=-1)
        pred_probs = probs.max(dim=-1).values
        
        item_clone['inputs'][:,0] = pred_probs 
        
    elif augmentation == 'label':
        apply_label_augmentation(item_clone)

    return item_clone


def build_feature_matrix(
    items: List[Dict],
    n_batches: int,
    batch_size: int,
    rng: np.random.Generator,
    temp_augment: bool = False,
    label_augment: bool = False,
) -> np.ndarray:
    """
    Sample n_batches of batch_size trajectories each, optionally augmenting.
    Each batch → one (2T,) feature vector  [mean_conf_per_shot || mean_acc_per_shot].
    Returns (n_batches, 2T)  as numpy float32.
    """
    n = len(items)
    batch_size = min(batch_size, n)
    vectors = []

    for _ in range(n_batches):
        idxs = rng.choice(n, batch_size, replace=batch_size > n)

        if temp_augment or label_augment:
            batch_items = []
            for i in idxs:
                batch_items.append(items[i])
                augmentation = pick_augmentation(temp_augment, label_augment)
                if augmentation:
                    augmented_item = augment_item(items[i], augmentation)
                    batch_items.append(augmented_item)
                    
            # batch_items = random.sample(batch_items, batch_size)
            
        else:
            batch_items = [items[i] for i in idxs]

        # stack to (batch_size, T, F/C)
        batch_inputs = torch.stack([it['inputs'] for it in batch_items])   # (B, T, F)
        batch_logits = torch.stack([it['logits'] for it in batch_items])   # (B, T, C)
        batch_labels = torch.stack([it['labels'] for it in batch_items])   # (B, T)

        mean_conf = batch_inputs[:, :, CONF_IDX].mean(dim=0)               # (T,)
        preds     = batch_logits.argmax(dim=-1)                             # (B, T)
        mean_acc  = (preds == batch_labels).float().mean(dim=0)            # (T,)

        vec = torch.cat([mean_conf, mean_acc]).numpy()                     # (2T,)
        vectors.append(vec)

    return np.stack(vectors)   # (n_batches, 2T)


# ═══════════════════════════════════════════════════════════════════════════
# 4.  DIMENSIONALITY REDUCTION
# ═══════════════════════════════════════════════════════════════════════════

def fit_umap(X_train: np.ndarray, seed: int = 42) -> UMAP:
    """Fit a UMAP reducer on training features and return it."""
    reducer = UMAP(n_components=2, random_state=seed, n_neighbors=5, min_dist=0.2)
    reducer.fit(X_train)
    return reducer


def project_umap(reducer: UMAP, X: np.ndarray) -> np.ndarray:
    return reducer.transform(X)


def fit_transform_umap(X: np.ndarray, seed: int = 42) -> Tuple[UMAP, np.ndarray]:
    """Fit UMAP on X and return both the reducer and the embeddings."""
    reducer = UMAP(n_components=2, random_state=seed, n_neighbors=5, min_dist=0.2)
    Z = reducer.fit_transform(X)
    return reducer, Z


def fit_transform_tsne(X: np.ndarray, seed: int = 42) -> np.ndarray:
    """t-SNE has no transform – fit on the combined set and return all embeddings."""
    from sklearn.manifold import TSNE
    reducer = TSNE(n_components=2, random_state=seed, perplexity=min(30, len(X) - 1))
    return reducer.fit_transform(X)


# ═══════════════════════════════════════════════════════════════════════════
# 5.  COLOUR PALETTE
# ═══════════════════════════════════════════════════════════════════════════

def build_palette(combos: List[Tuple[str, str]]) -> Dict[Tuple[str, str], str]:
    cmap = plt.cm.get_cmap("tab20", max(len(combos), 1))
    return {combo: matplotlib.colors.to_hex(cmap(i)) for i, combo in enumerate(combos)}


def build_unseen_palette(combos: List[Tuple[str, str]]) -> Dict[Tuple[str, str], str]:
    """Distinct palette for unseen combos using Dark2."""
    cmap = plt.cm.get_cmap("Dark2", max(len(combos), 3))
    return {combo: matplotlib.colors.to_hex(cmap(i)) for i, combo in enumerate(combos)}


UNSEEN_MARKERS = ["*", "^", "D", "P", "X", "v", "s", "o", "h", ">"]


# ═══════════════════════════════════════════════════════════════════════════
# 6.  FIT REDUCERS  (per_combo mode: frozen fit on training only)
# ═══════════════════════════════════════════════════════════════════════════

def fit_reducers(
    train_feats_orig: Dict[Tuple, Dict[str, np.ndarray]],
    reduction: str,
) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    """
    Fit one reducer per sampling strategy on the ORIGINAL (non-augmented)
    training features only. Used for per_combo mode only.

    Returns:
        reducers       : {strat: fitted_reducer}  (empty if tsne)
        train_all_feats: {strat: (N_total, 2T)}
    """
    reducers: Dict[str, object] = {}
    train_all_feats: Dict[str, np.ndarray] = {}

    for strat in SAMPLING_STRATEGIES:
        parts = [
            train_feats_orig[combo][strat]
            for combo in train_feats_orig
            if strat in train_feats_orig[combo]
        ]
        if not parts:
            continue
        X_all = np.concatenate(parts, axis=0)
        train_all_feats[strat] = X_all

        if reduction == "umap":
            print(f"  Fitting UMAP for {strat} on {len(X_all)} points …")
            reducers[strat] = fit_umap(X_all)

    return reducers, train_all_feats


# ═══════════════════════════════════════════════════════════════════════════
# 7.  EMBED
#     per_combo: unchanged — frozen fit on train, transform unseen
#     all mode:  joint fit on (train + unseen), transform train for plotting
# ═══════════════════════════════════════════════════════════════════════════

def embed_all(
    train_feats_orig: Dict[Tuple, Dict[str, np.ndarray]],
    train_feats_aug:  Dict[Tuple, Dict[str, np.ndarray]],
    unseen_feats_map: Dict[Tuple, Dict[str, np.ndarray]],
    reducers: Dict[str, object],
    train_all_feats: Dict[str, np.ndarray],
    reduction: str,
) -> Tuple[
    Dict[Tuple, Dict[str, np.ndarray]],   # train_emb_orig
    Dict[Tuple, Dict[str, np.ndarray]],   # train_emb_aug
    Dict[Tuple, Dict[str, np.ndarray]],   # unseen_emb_map
]:
    """
    per_combo mode: unchanged — project onto frozen training manifold.

    all mode (UMAP): for each (unseen_llm, strategy):
      - fit UMAP jointly on (original train + unseen)  → embed original train
      - fit UMAP jointly on (augmented train + unseen) → embed augmented train
      - transform unseen onto both (same unseen coords used for both panels,
        since unseen features don't change)
    """
    train_emb_orig:  Dict[Tuple, Dict[str, np.ndarray]] = {c: {} for c in train_feats_orig}
    train_emb_aug:   Dict[Tuple, Dict[str, np.ndarray]] = {c: {} for c in train_feats_aug}
    unseen_emb_map:  Dict[Tuple, Dict[str, np.ndarray]] = {c: {} for c in unseen_feats_map}

    if reduction == "umap":
        # ── per_combo: frozen manifold (unchanged) ───────────────────────
        for combo, strat_map in train_feats_orig.items():
            for strat, X in strat_map.items():
                if strat in reducers:
                    train_emb_orig[combo][strat] = project_umap(reducers[strat], X)

        for combo, strat_map in train_feats_aug.items():
            for strat, X in strat_map.items():
                if strat in reducers:
                    train_emb_aug[combo][strat] = project_umap(reducers[strat], X)

        for combo, strat_map in unseen_feats_map.items():
            for strat, X in strat_map.items():
                if strat in reducers:
                    unseen_emb_map[combo][strat] = project_umap(reducers[strat], X)

    else:  # t-SNE: joint fit per strategy over everything (unchanged)
        for strat in SAMPLING_STRATEGIES:
            parts       = []
            sizes       = []
            combo_order = []

            for src_feats, kind in (
                (train_feats_orig, 'orig'),
                (train_feats_aug,  'aug'),
                (unseen_feats_map, 'unseen'),
            ):
                for combo in src_feats:
                    if strat in src_feats[combo]:
                        parts.append(src_feats[combo][strat])
                        sizes.append(len(src_feats[combo][strat]))
                        combo_order.append((kind, combo))

            if not parts:
                continue

            Z_joint = fit_transform_tsne(np.concatenate(parts, axis=0))

            cursor = 0
            for (kind, combo), n in zip(combo_order, sizes):
                z = Z_joint[cursor: cursor + n]
                if kind == 'orig':
                    train_emb_orig[combo][strat] = z
                elif kind == 'aug':
                    train_emb_aug[combo][strat] = z
                else:
                    unseen_emb_map[combo][strat] = z
                cursor += n

    return train_emb_orig, train_emb_aug, unseen_emb_map


def embed_all_joint(
    train_feats_orig: Dict[Tuple, Dict[str, np.ndarray]],
    train_feats_aug:  Dict[Tuple, Dict[str, np.ndarray]],
    unseen_feats_map: Dict[Tuple, Dict[str, np.ndarray]],
) -> Tuple[
    Dict[Tuple, Dict[str, np.ndarray]],   # train_emb_orig  — per (unseen_llm, strat)
    Dict[Tuple, Dict[str, np.ndarray]],   # train_emb_aug   — per (unseen_llm, strat)
    Dict[Tuple, Dict[str, np.ndarray]],   # unseen_emb_map
]:
    """
    Joint UMAP fit for the `all` coverage mode.

    For each (unseen_llm, strategy):
      row 0: fit on (orig_train + unseen)  → embed orig_train, transform unseen
      row 1: fit on (aug_train  + unseen)  → embed aug_train,  transform unseen

    Embeddings are keyed by (unseen_llm, strat) for training dicts and by
    (unseen_llm, unseen_ds) for unseen dict — same keys as before so the
    plotting functions need no changes.
    """
    # group unseen combos by LLM
    unseen_by_llm: Dict[str, List[Tuple]] = {}
    for (u_llm, u_ds) in unseen_feats_map:
        unseen_by_llm.setdefault(u_llm, []).append((u_llm, u_ds))

    # output dicts — same key structure as embed_all
    train_emb_orig: Dict[Tuple, Dict[str, np.ndarray]] = {c: {} for c in train_feats_orig}
    train_emb_aug:  Dict[Tuple, Dict[str, np.ndarray]] = {c: {} for c in train_feats_aug}
    unseen_emb_map: Dict[Tuple, Dict[str, np.ndarray]] = {c: {} for c in unseen_feats_map}

    train_combos = list(train_feats_orig.keys())

    for u_llm, combos_for_llm in unseen_by_llm.items():
        for strat in SAMPLING_STRATEGIES:

            # ── collect unseen features for this (llm, strat) ────────────
            unseen_parts  = []
            unseen_sizes  = []
            for u_combo in combos_for_llm:
                if strat in unseen_feats_map[u_combo]:
                    unseen_parts.append(unseen_feats_map[u_combo][strat])
                    unseen_sizes.append(len(unseen_feats_map[u_combo][strat]))

            if not unseen_parts:
                continue

            X_unseen = np.concatenate(unseen_parts, axis=0)   # (N_unseen, 2T)

            # ── collect original training features ────────────────────────
            orig_parts = []
            orig_sizes = []
            for combo in train_combos:
                if strat in train_feats_orig[combo]:
                    orig_parts.append(train_feats_orig[combo][strat])
                    orig_sizes.append(len(train_feats_orig[combo][strat]))

            # ── row 0: joint fit on orig_train + unseen ───────────────────
            if orig_parts:
                X_orig_all = np.concatenate(orig_parts, axis=0)
                X_joint    = np.concatenate([X_orig_all, X_unseen], axis=0)

                print(f"  Joint UMAP fit (orig+unseen) for {u_llm.split('/')[-1]} × {strat}"
                      f" on {len(X_joint)} points …")
                reducer_orig, Z_joint = fit_transform_umap(X_joint)

                # split back: first len(X_orig_all) rows → training
                Z_orig_all = Z_joint[:len(X_orig_all)]
                Z_unseen_orig = Z_joint[len(X_orig_all):]

                # distribute training embeddings back to per-combo dicts
                cursor = 0
                for combo, n in zip(train_combos, orig_sizes):
                    if strat in train_feats_orig[combo]:
                        train_emb_orig[combo][strat] = Z_orig_all[cursor: cursor + n]
                        cursor += n

                # distribute unseen embeddings
                cursor = 0
                for u_combo, n in zip(combos_for_llm, unseen_sizes):
                    if strat in unseen_feats_map[u_combo]:
                        # use orig-fit coords for unseen (consistent with row 0)
                        unseen_emb_map[u_combo][strat] = Z_unseen_orig[cursor: cursor + n]
                        cursor += n

            # ── row 1: joint fit on aug_train + unseen ────────────────────
            aug_parts = []
            aug_sizes = []
            for combo in train_combos:
                if strat in train_feats_aug[combo]:
                    aug_parts.append(train_feats_aug[combo][strat])
                    aug_sizes.append(len(train_feats_aug[combo][strat]))

            if aug_parts:
                X_aug_all = np.concatenate(aug_parts, axis=0)
                X_joint   = np.concatenate([X_aug_all, X_unseen], axis=0)

                print(f"  Joint UMAP fit (aug+unseen)  for {u_llm.split('/')[-1]} × {strat}"
                      f" on {len(X_joint)} points …")
                reducer_aug, Z_joint = fit_transform_umap(X_joint)

                Z_aug_all    = Z_joint[:len(X_aug_all)]
                Z_unseen_aug = Z_joint[len(X_aug_all):]

                cursor = 0
                for combo, n in zip(train_combos, aug_sizes):
                    if strat in train_feats_aug[combo]:
                        train_emb_aug[combo][strat] = Z_aug_all[cursor: cursor + n]
                        cursor += n

                # unseen coords from aug fit used for row 1 overlay
                # we store these temporarily; plot_all_unseen reads unseen_emb_map
                # which currently holds orig-fit coords. We need per-row coords.
                # Solution: store aug unseen under a sentinel key suffix handled below.
                cursor = 0
                for u_combo, n in zip(combos_for_llm, unseen_sizes):
                    if strat in unseen_feats_map[u_combo]:
                        aug_key = (u_combo[0] + "__aug__", u_combo[1])
                        if aug_key not in unseen_emb_map:
                            unseen_emb_map[aug_key] = {}
                        unseen_emb_map[aug_key][strat] = Z_unseen_aug[cursor: cursor + n]
                        cursor += n

    return train_emb_orig, train_emb_aug, unseen_emb_map


# ═══════════════════════════════════════════════════════════════════════════
# 8.  SHARED PLOT HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _draw_training_background(
    ax,
    strat: str,
    train_embeddings: Dict[Tuple, Dict[str, np.ndarray]],
    palette: Dict[Tuple, str],
    legend_handles: list,
    add_to_legend: bool,
) -> None:
    for combo, strat_map in train_embeddings.items():
        if strat not in strat_map:
            continue
        pts       = strat_map[strat]
        llm_short = combo[0].split("/")[-1]
        ds_short  = combo[1]
        color     = palette[combo]
        ax.scatter(pts[:, 0], pts[:, 1], c=color, s=8, alpha=0.45, linewidths=0)
        if add_to_legend:
            legend_handles.append(
                mpatches.Patch(color=color, label=f"{llm_short} × {ds_short}")
            )


def _finalise_figure(fig, legend_handles: list, save_path: str) -> None:
    if len(legend_handles)>8:
        n_cols = 6
        fontsize = 12
    else:
        n_cols = 4
        fontsize = None
    # n_cols = min(6, len(legend_handles)) 
    
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=n_cols,
        fontsize=fontsize,
        framealpha=0.7,
        bbox_to_anchor=(0.5, 0.0),
        borderaxespad=0.2,
    )
    plt.tight_layout(rect=[0, 0.15, 1, 1])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# 9.  PLOT MODE 1 — per_combo
# ═══════════════════════════════════════════════════════════════════════════

def plot_per_combo(
    train_emb_orig: Dict[Tuple, Dict[str, np.ndarray]],
    unseen_emb_map: Dict[Tuple, Dict[str, np.ndarray]],
    palette: Dict[Tuple, str],
    base_save_dir: str,
) -> None:
    for (u_llm, u_ds), unseen_emb in unseen_emb_map.items():
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        fig.suptitle(
            f"Coverage map  |  unseen: {u_llm.split('/')[-1]} × {u_ds}",
        )

        legend_handles = []

        for ax_idx, (ax, strat) in enumerate(zip(axes, SAMPLING_STRATEGIES)):
            strategy_label = 'RANDOM' if strat == 'ENTROPY' else strat
            ax.set_title(strategy_label)
            # ax.set_xlabel("dim 1")
            # ax.set_ylabel("dim 2")
            ax.grid(True, alpha=0.3, linestyle='--')

            _draw_training_background(
                ax, strat, train_emb_orig, palette, legend_handles,
                add_to_legend=(ax_idx == 0),
            )

            if strat in unseen_emb:
                pts_u = unseen_emb[strat]
                ax.scatter(
                    pts_u[:, 0], pts_u[:, 1],
                    c="black", s=40, alpha=0.9, marker="*",
                    zorder=5, linewidths=0.3, edgecolors="white",
                )

        legend_handles.append(
            mpatches.Patch(
                color="black",
                label=f"UNSEEN: {u_llm.split('/')[-1]} × {u_ds}",
            )
        )

        save_path = os.path.join(
            base_save_dir,
            u_llm.replace("/", "_"),
            u_ds,
            "map.png",
        )
        _finalise_figure(fig, legend_handles, save_path)


# ═══════════════════════════════════════════════════════════════════════════
# 10. PLOT MODE 2 — all
#     Joint-fit coords: training is transformed onto the (train+unseen) manifold.
#     Row 0: orig_train background  +  unseen (orig-fit coords)
#     Row 1: aug_train  background  +  unseen (aug-fit  coords)
# ═══════════════════════════════════════════════════════════════════════════

def plot_all_unseen(
    train_emb_orig: Dict[Tuple, Dict[str, np.ndarray]],
    train_emb_aug:  Dict[Tuple, Dict[str, np.ndarray]],
    unseen_emb_map: Dict[Tuple, Dict[str, np.ndarray]],
    palette: Dict[Tuple, str],
    unseen_palette: Dict[Tuple, str],
    base_save_dir: str,
    has_augmentation: bool,
    hide_train_legends: bool,
    partial_coverage: bool
) -> None:
    # group unseen combos by LLM (ignore __aug__ sentinel keys)
    unseen_by_llm: Dict[str, List[Tuple]] = {}
    for (u_llm, u_ds) in unseen_emb_map:
        if "__aug__" in u_llm:
            continue
        unseen_by_llm.setdefault(u_llm, []).append((u_llm, u_ds))

    n_rows = 2 if has_augmentation else 1
    row_configs = [
        ("original training", train_emb_orig, False),
        ("augmented training", train_emb_aug,  True),
    ] if has_augmentation else [
        ("training", train_emb_orig, False),
    ]

    for u_llm, combos_for_llm in unseen_by_llm.items():
        llm_unseen_palette = {combo: unseen_palette[combo] for combo in combos_for_llm}

        for strat in SAMPLING_STRATEGIES:
            strategy_label = 'RANDOM' if strat == 'ENTROPY' else strat

            fig, axes = plt.subplots(n_rows, 2, figsize=(18, 7 * n_rows))
            if n_rows == 1:
                axes = axes[np.newaxis, :]

            if not hide_train_legends:
                fig.suptitle(
                    f"Coverage map  |  unseen LLM: {u_llm.split('/')[-1]}"
                    f"  |  sampling: {strategy_label}",
                )

            legend_handles = []

            for row_idx, (row_label, train_emb, use_aug_unseen) in enumerate(row_configs):
                ax_ref   = axes[row_idx][0]
                ax_cover = axes[row_idx][1]

                ax_ref.set_title(f"{row_label}")
                ax_cover.set_title(f"{row_label}  + all unseen datasets")
                for ax in (ax_ref, ax_cover):
                    # ax.set_xlabel("dim 1")
                    # ax.set_ylabel("dim 2")
                    ax.grid(True, alpha=0.3, linestyle='--')

                _draw_training_background(
                    ax_ref, strat, train_emb, palette, legend_handles,
                    add_to_legend=(row_idx == 0) and not hide_train_legends,
                )
                _draw_training_background(
                    ax_cover, strat, train_emb, palette, [],
                    add_to_legend=False,
                )

                for u_idx, u_combo in enumerate(combos_for_llm):
                    # row 1 uses aug-fit unseen coords stored under sentinel key
                    if use_aug_unseen:
                        aug_key = (u_combo[0] + "__aug__", u_combo[1])
                        emb_source = unseen_emb_map.get(aug_key, unseen_emb_map[u_combo])
                    else:
                        emb_source = unseen_emb_map[u_combo]

                    if strat not in emb_source:
                        continue
                    _, u_ds = u_combo
                    pts_u  = emb_source[strat]
                    color  = llm_unseen_palette[u_combo]
                    marker = UNSEEN_MARKERS[u_idx % len(UNSEEN_MARKERS)]
                    ax_cover.scatter(
                        pts_u[:, 0], pts_u[:, 1],
                        c=color, s=50, alpha=0.9, marker=marker,
                        zorder=5, linewidths=0.3, edgecolors="white",
                    )
                    if row_idx == 0:
                        label = u_ds if hide_train_legends else f"UNSEEN: {u_ds}"
                        legend_handles.append(
                            mpatches.Patch(color=color, label=label)
                        )

            save_path = Path(base_save_dir)
            
            if hide_train_legends:
                save_path = save_path/'simple'
                
            save_path = save_path/u_llm.replace("/", "_")
            
            if partial_coverage:
                save_path = save_path/'partial_coverage'          
            
            save_path = save_path/strat
            
            file_name = "all_datasets_map"
            if has_augmentation:
                file_name += '_aug'
            file_name += ".png"
            
            save_path = save_path/file_name
            
            _finalise_figure(fig, legend_handles, save_path)


# ═══════════════════════════════════════════════════════════════════════════
# 11. MAIN
# ═══════════════════════════════════════════════════════════════════════════

def _load_combo(llm, ds, shots_start, shots_end):
    """Try train split first, fall back to test."""
    for split in ('train', 'test'):
        try:
            return load_raw(llm, ds, split, shots_start, shots_end)
        except FileNotFoundError:
            pass
    raise FileNotFoundError(f"No train or test split found for {llm} × {ds}")


def main(args):
    rng = np.random.default_rng(42)
    has_augmentation = args.temp_augment or args.label_augment

    train_combos  = list(product(args.train_llms,  args.train_datasets))
    unseen_combos = list(product(args.unseen_llms, args.unseen_datasets))
    base_save_dir = os.path.join("analysis", "coverage")

    # ── load training data ───────────────────────────────────────────────
    print("Loading training data …")
    train_raw: Dict[Tuple, Dict[str, List]] = {}
    for (llm, ds) in tqdm(train_combos, desc="train combos"):
        try:
            train_raw[(llm, ds)] = _load_combo(llm, ds, args.shots_start, args.shots_end)
        except FileNotFoundError as e:
            print(f"  [WARN] skipping {llm} × {ds}: {e}")

    # ── build ORIGINAL training feature matrices ─────────────────────────
    print("\nBuilding original training features …")
    train_feats_orig: Dict[Tuple, Dict[str, np.ndarray]] = {}
    for combo, strat_map in train_raw.items():
        train_feats_orig[combo] = {}
        for strat, items in strat_map.items():
            if items:
                train_feats_orig[combo][strat] = build_feature_matrix(
                    items, args.n_batches, args.batch_size, rng,
                    temp_augment=False, label_augment=False,
                )

    # ── build AUGMENTED training feature matrices ────────────────────────
    if has_augmentation:
        random.seed(42)
        torch.manual_seed(42)
        print("\nBuilding augmented training features …")
        train_feats_aug: Dict[Tuple, Dict[str, np.ndarray]] = {}
        for combo, strat_map in train_raw.items():
            train_feats_aug[combo] = {}
            for strat, items in strat_map.items():
                if items:
                    train_feats_aug[combo][strat] = build_feature_matrix(
                        items, args.n_batches, args.batch_size, rng,
                        temp_augment=args.temp_augment,
                        label_augment=args.label_augment,
                    )
    else:
        train_feats_aug = train_feats_orig

    # ── load all unseen data ─────────────────────────────────────────────
    print("\nLoading unseen data …")
    unseen_feats_map: Dict[Tuple, Dict[str, np.ndarray]] = {}
    for (u_llm, u_ds) in tqdm(unseen_combos, desc="unseen combos"):
        try:
            u_raw = _load_combo(u_llm, u_ds, args.shots_start, args.shots_end)
        except FileNotFoundError as e:
            print(f"  [WARN] skipping {u_llm} × {u_ds}: {e}")
            continue

        u_feats: Dict[str, np.ndarray] = {}
        for strat, items in u_raw.items():
            if items:
                u_feats[strat] = build_feature_matrix(
                    items, args.n_batches, args.batch_size, rng,
                    temp_augment=False, label_augment=False,
                )
        unseen_feats_map[(u_llm, u_ds)] = u_feats

    palette = build_palette(list(train_feats_orig.keys()))

    # ── embed ────────────────────────────────────────────────────────────
    if args.coverage == "per_combo":
        print(f"\nFitting {args.reduction.upper()} on original training features …")
        reducers, train_all_feats = fit_reducers(train_feats_orig, args.reduction)

        print("\nProjecting into 2D (per_combo: frozen manifold) …")
        train_emb_orig, train_emb_aug, unseen_emb_map_emb = embed_all(
            train_feats_orig, train_feats_aug, unseen_feats_map,
            reducers, train_all_feats, args.reduction,
        )

        print("\nPlotting per-combo coverage maps …")
        plot_per_combo(
            train_emb_orig, unseen_emb_map_emb, palette, args.reduction, base_save_dir
        )

    elif args.coverage == "all":
        print("\nProjecting into 2D (all: joint fit per unseen_llm × strategy) …")
        train_emb_orig, train_emb_aug, unseen_emb_map_emb = embed_all_joint(
            train_feats_orig, train_feats_aug, unseen_feats_map,
        )

        print("\nPlotting combined coverage maps …")
        unseen_palette = build_unseen_palette(list(unseen_feats_map.keys()))
        plot_all_unseen(
            train_emb_orig, train_emb_aug, unseen_emb_map_emb,
            palette, unseen_palette, base_save_dir,
            has_augmentation, args.hide_train_legends, args.partial_coverage
        )

    print("\nDone.")

def args_check(args):
    assert len(args.unseen_llms)==1, "Current code overwrites projection learnt from the last unseen llm based combos and uses it for other llms"

# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    def csv(s):
        return [x.strip() for x in s.split(",")]

    parser = argparse.ArgumentParser(
        description="Coverage UMAP/t-SNE for calibrator trajectories"
    )
    parser.add_argument("--train_llms",      type=csv, required=True,  help="Comma-separated training LLMs")
    parser.add_argument("--train_datasets",  type=csv, required=True,  help="Comma-separated training datasets")
    parser.add_argument("--unseen_llms",     type=csv, required=True,  help="Comma-separated unseen LLMs")
    parser.add_argument("--unseen_datasets", type=csv, required=True,  help="Comma-separated unseen datasets")
    parser.add_argument("--shots_start", type=int, default=None,  help="Which shot index to start from")
    parser.add_argument("--shots_end",   type=int, default=None,  help="Which shot index to end at")
    parser.add_argument("--reduction",   type=str, default="umap",      choices=("umap", "tsne"))
    parser.add_argument("--coverage",    type=str, default="all",       choices=("per_combo", "all"))
    parser.add_argument("--partial_coverage",    action="store_true", default=False, 
                        help="Whether to append 'partial' to filename, for partial coverage depiction")
    parser.add_argument("--hide_train_legends",    action="store_true", default=False,
                        help="Only show the unseen combos in the legend")
    
    parser.add_argument("--n_batches",   type=int, default=75,   help="Batches sampled per combo")
    parser.add_argument("--batch_size",  type=int, default=100,  help="Trajectories per batch")
    # parser.add_argument("--group_size",  type=int, default=3,    help="Group size for accuracy slope calculation")
    parser.add_argument("--temp_augment",  action="store_true", default=False,
                        help="Apply temperature augmentation to training data")
    parser.add_argument("--label_augment", action="store_true", default=False,
                        help="Apply label augmentation to training data")

    args = parser.parse_args()
    args_check(args)
    
    update_plt_rcparams(args.hide_train_legends)
    main(args)