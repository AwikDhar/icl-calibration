"""
Correlation analysis: does embedding cosine similarity between in-context examples
predict similarity in LLM correctness?

v2 changes:
  - extract_pairs now also returns pred-agreement mask, enabling:
      * sim_same_pred  = sim * (pred_i == pred_j)
      * sim_diff_pred  = sim * (pred_i != pred_j)
  - analyse_strategy computes correlations for all three sim variants
  - plot_combo gains a third panel: binned P(agree) split by same-pred / diff-pred
    subgroups, so the conditional signal is visible even when overall sim is flat.

Layout of inputs (pre-recalculate_features, T=21 fixed):
  [0]        pred_prob
  [1]        shifted_correctness
  [2]        shifted_gt_prob
  [3]        normalized_entropy
  [4:4+T]    pred_similarity_vectors  (T=21, so indices 4:25)
  [4+T:4+2T] input_similarity_vectors (indices 25:46)
  [4+2T:]    lower_dim_embeddings
"""

import os
import argparse
from itertools import product
import random

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
import msgspec
from tqdm import tqdm

from utils.sampling_utils import get_similarities

T = 21
INPUT_SIM_START = 4 + T   # 25
INPUT_SIM_END   = 4 + 2*T # 46

SAMPLING_STRATEGIES = ('ENTROPY', 'SIMILARITY')

# EMBEDDING_TYPE = 'embedding_model'
EMBEDDING_TYPE = 'hidden_features'


# ── data loading ──────────────────────────────────────────────────────────────

def load_raw_data(path: str, device='cuda'):
    with open(path) as f:
        split_data = msgspec.json.decode(f.read())

    items_by_strategy = {s: [] for s in SAMPLING_STRATEGIES}

    for item in split_data:
        if item.get('corrupt', False):
            continue
        strategy = item.get('sampling_strategy', '').upper()
        if strategy not in SAMPLING_STRATEGIES:
            continue

        logits     = torch.tensor(item['logits'],    dtype=torch.float32, device=device)[:T]
        labels     = torch.tensor(item['labels'],    dtype=torch.long,    device=device)[:T]
        inputs     = torch.tensor(item['inputs'],    dtype=torch.float32, device=device)[:T]
        # embeddings = torch.tensor(item['embeddings'][EMBEDDING_TYPE], dtype=torch.float32, device=device)[:T]
        embeddings = torch.tensor(item['inputs'][:, -128:], dtype=torch.float32, device=device)[:T]

        if torch.isnan(logits).any():
            continue

        items_by_strategy[strategy].append({
            'inputs':     inputs,
            'logits':     logits,
            'labels':     labels,
            'embeddings': embeddings,
        })

    return items_by_strategy


# ── pair extraction ───────────────────────────────────────────────────────────

def extract_pairs(item):
    """
    Extract all causal pairs (j < i) from one sequence.

    Returns:
        sims          : (N_pairs,)  raw cosine similarity between embeddings i and j
        sim_same_pred : (N_pairs,)  sim * (pred_i == pred_j)   — your proposed signal
        sim_diff_pred : (N_pairs,)  sim * (pred_i != pred_j)   — complement
        same_pred_mask: (N_pairs,)  bool, True when predictions match
        correct_agree : (N_pairs,)  1 if both correct or both wrong, else 0
        lifts         : (N_pairs,)  agreement - chance_agreement
        chance_agree  : float       p^2 + (1-p)^2 for this sequence
    """
    correctness = (item['logits'].argmax(-1) == item['labels']).float()   # (T,)
    preds       = item['logits'].argmax(-1)                               # (T,)
    seq_len     = item['logits'].shape[0]

    p_correct    = correctness.mean().item()
    chance_agree = p_correct**2 + (1 - p_correct)**2

    # embeddings   = item['embeddings'].cpu().numpy()
    # similarities = np.tril(get_similarities(embeddings, embeddings))  # (T, T) lower-tri

    sims_list           = []
    same_pred_mask_list = []
    correct_agree_list  = []
    lifts_list          = []

    for i in range(1, seq_len):
        for j in range(i):
            sim   = item['inputs'][i, INPUT_SIM_START + j].item()
            # sim   = float(similarities[i, j])
            agree = 1.0 if correctness[i] == correctness[j] else 0.0
            same  = bool(preds[i] == preds[j])

            sims_list.append(sim)
            same_pred_mask_list.append(same)
            correct_agree_list.append(agree)
            lifts_list.append(agree - chance_agree)

    sims           = np.array(sims_list)
    same_pred_mask = np.array(same_pred_mask_list)
    correct_agree  = np.array(correct_agree_list)
    lifts          = np.array(lifts_list)

    sim_same_pred = sims * same_pred_mask.astype(float)
    sim_diff_pred = sims * (~same_pred_mask).astype(float)

    return sims, sim_same_pred, sim_diff_pred, same_pred_mask, correct_agree, lifts, chance_agree


# ── binned stats helper ───────────────────────────────────────────────────────

def _bin_stats(x, y, n_bins=10):
    """
    Bin x into n_bins equal-width bins; return centres, means, SEs, counts.
    NaN where a bin is empty.
    """
    bin_edges   = np.linspace(np.nanmin(x), np.nanmax(x), n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    means, stds, counts = [], [], []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (x >= lo) & (x < hi)
        n = mask.sum()
        counts.append(n)
        if n == 0:
            means.append(np.nan)
            stds.append(np.nan)
        else:
            means.append(y[mask].mean())
            stds.append(y[mask].std() / np.sqrt(n))

    return np.array(bin_centers), np.array(means), np.array(stds), np.array(counts)


# ── per-strategy analysis ─────────────────────────────────────────────────────

def analyse_strategy(items):
    sample = random.sample(items, min(1000, len(items)))

    all_sims, all_sim_same, all_sim_diff = [], [], []
    all_same_mask = []
    all_agrees, all_lifts = [], []
    all_chance_agree = []

    for item in sample:
        sims, sim_same, sim_diff, same_mask, agrees, lifts, chance = extract_pairs(item)
        all_sims.append(sims)
        all_sim_same.append(sim_same)
        all_sim_diff.append(sim_diff)
        all_same_mask.append(same_mask)
        all_agrees.append(agrees)
        all_lifts.append(lifts)
        all_chance_agree.append(chance)

    if not all_sims:
        return None

    sims      = np.concatenate(all_sims)
    sim_same  = np.concatenate(all_sim_same)
    sim_diff  = np.concatenate(all_sim_diff)
    same_mask = np.concatenate(all_same_mask)
    agrees    = np.concatenate(all_agrees)
    lifts     = np.concatenate(all_lifts)
    mean_chance_agree = float(np.mean(all_chance_agree))

    # ── correlations on lift for three sim variants ───────────────────────────
    def _corr(x, y):
        if len(x) < 5:
            return (np.nan, np.nan, np.nan, np.nan)
        pr, pp   = pearsonr(x, y)
        sr, sp   = spearmanr(x, y)
        return pr, pp, sr, sp

    pr, pp, sr, sp             = _corr(sims,     lifts)
    pr_sp, pp_sp, sr_sp, sp_sp = _corr(sim_same, lifts)
    pr_dp, pp_dp, sr_dp, sp_dp = _corr(sim_diff, lifts[~same_mask])  # diff-pred pairs only

    # ── binned: raw sim → agreement (original) ────────────────────────────────
    bc, bam, bas, bcnt   = _bin_stats(sims,    agrees)
    _, blm, bls, _       = _bin_stats(sims,    lifts)

    # ── binned: same-pred pairs only ──────────────────────────────────────────
    bc_sp, bam_sp, bas_sp, bcnt_sp = _bin_stats(sims[same_mask],  agrees[same_mask])
    # ── binned: diff-pred pairs only ──────────────────────────────────────────
    bc_dp, bam_dp, bas_dp, bcnt_dp = _bin_stats(sims[~same_mask], agrees[~same_mask])

    # ── fraction same-pred (sanity) ───────────────────────────────────────────
    frac_same_pred = same_mask.mean()

    return {
        # raw arrays
        'sims':             sims,
        'sim_same':         sim_same,
        'sim_diff':         sim_diff,
        'same_mask':        same_mask,
        'agrees':           agrees,
        'lifts':            lifts,
        # correlations
        'pearson_r':        pr,   'pearson_p':   pp,
        'spearman_r':       sr,   'spearman_p':  sp,
        'pearson_r_sp':     pr_sp, 'pearson_p_sp': pp_sp,
        'spearman_r_sp':    sr_sp, 'spearman_p_sp': sp_sp,
        'pearson_r_dp':     pr_dp, 'pearson_p_dp': pp_dp,
        'spearman_r_dp':    sr_dp, 'spearman_p_dp': sp_dp,
        # binned stats — original
        'bin_centers':      bc,    'bin_agree_means':  bam,
        'bin_agree_stds':   bas,   'bin_counts':       bcnt,
        'bin_lift_means':   blm,   'bin_lift_stds':    bls,
        # binned stats — same-pred / diff-pred
        'bc_sp':    bc_sp,  'bam_sp':   bam_sp,  'bas_sp':   bas_sp,  'bcnt_sp': bcnt_sp,
        'bc_dp':    bc_dp,  'bam_dp':   bam_dp,  'bas_dp':   bas_dp,  'bcnt_dp': bcnt_dp,
        # scalars
        'n_pairs':           len(sims),
        'n_same_pred':       same_mask.sum(),
        'n_diff_pred':       (~same_mask).sum(),
        'frac_same_pred':    frac_same_pred,
        'mean_agree':        agrees.mean(),
        'mean_lift':         lifts.mean(),
        'mean_chance_agree': mean_chance_agree,
    }


def analyse_combo(llm, dataset, data_dir, splits=('train',), device='cpu'):
    items_by_strategy = {s: [] for s in SAMPLING_STRATEGIES}

    for split in splits:
        path = os.path.join(data_dir, llm.replace('/', '_'), dataset, f'{split}.json')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Not found: {path}")
        loaded = load_raw_data(path, device=device)
        for s in SAMPLING_STRATEGIES:
            items_by_strategy[s].extend(loaded[s])

    return {s: analyse_strategy(items_by_strategy[s]) for s in SAMPLING_STRATEGIES}


# ── plotting ──────────────────────────────────────────────────────────────────

def _plot_binned_line(ax, centers, means, stds, color, label, linestyle='-'):
    v = ~np.isnan(means)
    ax.plot(centers[v], means[v], 'o' + linestyle, color=color, linewidth=2, label=label)
    ax.fill_between(centers[v], means[v] - stds[v], means[v] + stds[v],
                    alpha=0.2, color=color)


def plot_combo(llm, dataset, strategy, res, out_dir):
    """
    Three-panel plot for one LLM × dataset × strategy.

    Panel 1 (original): binned P(agree) vs raw cosine sim, chance reference line.
    Panel 2 (original): hexbin scatter + correlation stats on lift.
    Panel 3 (NEW):      binned P(agree) vs raw cosine sim, split by same-pred /
                        diff-pred subgroups. Correlation stats for sim_same on lift.
    """
    combo_dir = os.path.join(out_dir, EMBEDDING_TYPE, llm.replace('/', '_'), dataset)
    os.makedirs(combo_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    fig.suptitle(f'{llm}  ×  {dataset}  ×  {strategy}', fontsize=11, fontweight='bold')

    # ── Panel 1: raw binned agreement (unchanged) ─────────────────────────────
    ax = axes[0]
    _plot_binned_line(ax, res['bin_centers'], res['bin_agree_means'],
                      res['bin_agree_stds'], 'steelblue', 'P(agree) — all pairs')
    ax.axhline(res['mean_chance_agree'], linestyle='--', color='tomato', linewidth=1.5,
               label=f'chance = {res["mean_chance_agree"]:.3f}  (lift={res["mean_lift"]:+.3f})')
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('P(correctness agreement)')
    ax.set_title('All pairs: similarity → agreement')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)

    # ── Panel 2: hexbin scatter (unchanged) ───────────────────────────────────
    ax = axes[1]
    hb = ax.hexbin(res['sims'], res['agrees'], gridsize=40, cmap='YlOrRd', mincnt=1)
    fig.colorbar(hb, ax=ax, label='count')
    ax.axhline(res['mean_chance_agree'], linestyle='--', color='tomato', linewidth=1,
               label=f'chance {res["mean_chance_agree"]:.3f}')
    ax.legend(fontsize=8)
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Correctness Agreement (0/1)')
    ax.set_title(
        f"Pearson r={res['pearson_r']:.3f} (p={res['pearson_p']:.2e})\n"
        f"Spearman ρ={res['spearman_r']:.3f} (p={res['spearman_p']:.2e})  [on lift]"
    )

    # ── Panel 3: same-pred vs diff-pred conditional curves (NEW) ─────────────
    ax = axes[2]

    _plot_binned_line(ax, res['bc_sp'], res['bam_sp'], res['bas_sp'],
                      color='mediumseagreen',
                      label=f"same pred  (n={res['n_same_pred']:,}, "
                            f"ρ={res['spearman_r_sp']:.3f}, p={res['spearman_p_sp']:.2e})")

    _plot_binned_line(ax, res['bc_dp'], res['bam_dp'], res['bas_dp'],
                      color='darkorange',
                      label=f"diff pred  (n={res['n_diff_pred']:,}, "
                            f"ρ={res['spearman_r_dp']:.3f}, p={res['spearman_p_dp']:.2e})",
                      linestyle='--')

    ax.axhline(res['mean_chance_agree'], linestyle=':', color='tomato', linewidth=1.5,
               label=f'chance = {res["mean_chance_agree"]:.3f}')

    # annotate fraction of same-pred pairs
    ax.text(0.02, 0.04,
            f"{res['frac_same_pred']*100:.1f}% of pairs share same prediction",
            transform=ax.transAxes, fontsize=7.5, color='gray')

    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('P(correctness agreement)')
    ax.set_title('Conditional on prediction match')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7.5)

    fig.tight_layout()
    out_path = os.path.join(combo_dir, f'{strategy}.png')
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved -> {out_path}")


def plot_summary_heatmap(results, out_dir, strategy):
    """Three-row heatmap: raw Spearman rho, same-pred rho, mean lift."""
    llms     = sorted({llm for (llm, _) in results})
    datasets = sorted({ds  for (_, ds)  in results})

    rho_matrix    = np.full((len(llms), len(datasets)), np.nan)
    rho_sp_matrix = np.full((len(llms), len(datasets)), np.nan)
    lift_matrix   = np.full((len(llms), len(datasets)), np.nan)
    agree_matrix  = np.full((len(llms), len(datasets)), np.nan)

    for i, llm in enumerate(llms):
        for j, ds in enumerate(datasets):
            res = results.get((llm, ds), {}).get(strategy)
            if res is not None:
                rho_matrix[i, j]    = res['spearman_r']
                rho_sp_matrix[i, j] = res['spearman_r_sp']
                lift_matrix[i, j]   = res['mean_lift']
                agree_matrix[i, j]  = res['mean_agree']

    fig, axes = plt.subplots(1, 4, figsize=(max(16, len(datasets) * 4), max(3, len(llms) * 0.9)))
    fig.suptitle(f'[{strategy}] Summary', fontsize=11, fontweight='bold')

    panels = [
        (axes[0], rho_matrix,    'Spearman ρ — all pairs (lift)',      -0.3, 0.3, 'RdYlGn', 0.2),
        (axes[1], rho_sp_matrix, 'Spearman ρ — same-pred only (lift)', -0.3, 0.3, 'RdYlGn', 0.2),
        (axes[2], lift_matrix,   'Mean lift (agree - chance)',          -0.1, 0.1, 'RdYlGn', 0.05),
        (axes[3], agree_matrix,  'Mean raw P(agree)',                    0.5, 1.0, 'YlGn',   0.1),
    ]

    for ax, matrix, title, vmin, vmax, cmap, wt in panels:
        im = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap=cmap, aspect='auto')
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels(datasets, rotation=30, ha='right', fontsize=8)
        ax.set_yticks(range(len(llms)))
        ax.set_yticklabels([l.split('/')[-1] for l in llms], fontsize=8)
        ax.set_title(title, fontsize=9)
        for i, j in product(range(len(llms)), range(len(datasets))):
            if not np.isnan(matrix[i, j]):
                val = matrix[i, j]
                col = 'white' if abs(val - (vmin+vmax)/2) > wt*(vmax-vmin) else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7, color=col)

    fig.tight_layout()
    path = os.path.join(out_dir, f'summary_heatmap_{strategy}.png')
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved heatmap -> {path}")


# ── summary table ─────────────────────────────────────────────────────────────

def print_summary_table(results):
    col_w = 28
    header = (
        f"{'LLM':<{col_w}} {'Dataset':<16} {'Strategy':<12} "
        f"{'N pairs':>10} {'P(agree)':>10} {'Chance':>8} {'Lift':>8} "
        f"{'ρ (all)':>9} {'ρ (same-pred)':>14} {'ρ (diff-pred)':>14}"
    )
    sep = "=" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")

    for (llm, ds) in sorted(results):
        for strategy in SAMPLING_STRATEGIES:
            res = results[(llm, ds)].get(strategy)
            if res is None:
                print(f"{llm.split('/')[-1]:<{col_w}} {ds:<16} {strategy:<12} {'N/A':>10}")
                continue

            def _sig(p):
                return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

            print(
                f"{llm.split('/')[-1]:<{col_w}} {ds:<16} {strategy:<12} "
                f"{res['n_pairs']:>10,} {res['mean_agree']:>10.3f} "
                f"{res['mean_chance_agree']:>8.3f} {res['mean_lift']:>+8.3f} "
                f"{res['spearman_r']:>8.3f}{_sig(res['spearman_p']):<3} "
                f"{res['spearman_r_sp']:>13.3f}{_sig(res['spearman_p_sp']):<3} "
                f"{res['spearman_r_dp']:>13.3f}{_sig(res['spearman_p_dp']):<3}"
            )

    print(sep)
    print("ρ computed on lift = P(agree) - chance.  * p<0.05  ** p<0.01  *** p<0.001")
    print("same-pred: pairs where argmax(logits) matches; diff-pred: complement.\n")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Correlation: input similarity vs correctness agreement")
    parser.add_argument('--llms',     required=True,  help='Comma-separated LLM names')
    parser.add_argument('--datasets', required=True,  help='Comma-separated dataset names')
    parser.add_argument('--data_dir', default='calibration/datasets2')
    parser.add_argument('--out_dir',  default='analysis/feat_correlation/tmp')
    parser.add_argument('--splits',   default='train')
    parser.add_argument('--device',   default='cuda')
    args = parser.parse_args()

    llms     = [x.strip() for x in args.llms.split(',')]
    datasets = [x.strip() for x in args.datasets.split(',')]
    splits   = [x.strip() for x in args.splits.split(',')]

    results = {}
    for llm, ds in tqdm(list(product(llms, datasets)), desc='Analysing combos'):
        print(f"\n-> {llm} x {ds}")
        results[(llm, ds)] = analyse_combo(llm, ds, args.data_dir, splits=splits, device=args.device)

    print_summary_table(results)

    for (llm, ds), strat_results in results.items():
        for strategy, res in strat_results.items():
            if res is not None:
                plot_combo(llm, ds, strategy, res, args.out_dir)

    # for strategy in SAMPLING_STRATEGIES:
    #     plot_summary_heatmap(results, args.out_dir, strategy)