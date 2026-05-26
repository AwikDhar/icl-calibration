"""
Correlation analysis: does embedding cosine similarity between in-context examples
predict similarity in LLM correctness?

Layout of inputs (pre-recalculate_features, T=21 fixed):
  [0]        pred_prob
  [1]        shifted_correctness
  [2]        shifted_gt_prob
  [3]        normalized_entropy
  [4:4+T]    pred_similarity_vectors  (T=21, so indices 4:25)
  [4+T:4+2T] input_similarity_vectors (indices 25:46)
  [4+2T:]    lower_dim_embeddings

Plots show raw correctness agreement P(agree) vs similarity, with chance agreement
(p^2 + (1-p)^2, averaged across sequences) drawn as a reference line.
Correlations are computed on lift = agreement - chance_agreement so that a high-accuracy
model's trivially-high agreement doesn't inflate the reported r/rho.

v2: also splits pairs by same-pred vs diff-pred, computing sim_same_pred = sim * (pred_i == pred_j)
and plotting conditional P(agree) curves in a third panel. Hypothesis: the signal is
concentrated in same-pred pairs, which gets washed out when pooling all pairs.
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

plt.rcParams.update({
    'font.size': 12,        
    'lines.linewidth': 1.5,
    'grid.linewidth': 1.5,
    'xtick.labelsize': 15,    
    'ytick.labelsize': 15,    
    'legend.fontsize': 20,    
    'axes.labelsize': 15      
})

T = 21
INPUT_SIM_START = 4 + T   # 25
INPUT_SIM_END   = 4 + 2*T # 46

SAMPLING_STRATEGIES = ('ENTROPY', 'SIMILARITY')

EMBEDDING_TYPE = 'embedding_model'
# EMBEDDING_TYPE = 'hidden_features'

# All embedding configs to compare in the multi-config plot
EMBEDDING_CONFIGS = [
    ('Qwen/Qwen3-Embedding-4B', 128),
    ('Qwen/Qwen3-Embedding-4B', 256),
    ('google/embeddinggemma-300m', 128),
    ('google/embeddinggemma-300m', 256),
]

# Short labels for plot legends
def _cfg_label(model, dim):
    short = model.split('/')[-1]
    return f"{short} d={dim}"


# ── data loading ──────────────────────────────────────────────────────────────

def load_raw_data(path: str, embedding_model: str, embedding_dim: int, device='cpu'):
    """
    Load raw JSON. Returns a dict keyed by sampling strategy,
    each value a list of {inputs, logits, labels, embeddings} dicts.
    embedding_model and embedding_dim select which embedding to load.
    """
    with open(path) as f:
        split_data = msgspec.json.decode(f.read())

    items_by_strategy = {s: [] for s in SAMPLING_STRATEGIES}

    for item in split_data:
        if item.get('corrupt', False):
            continue
        strategy = item.get('sampling_strategy', '').upper()
        if strategy not in SAMPLING_STRATEGIES:
            continue

        logits     = torch.tensor(item['logits'], dtype=torch.float32, device=device)
        labels     = torch.tensor(item['labels'], dtype=torch.long,    device=device)
        inputs     = torch.tensor(item['inputs'], dtype=torch.float32, device=device)
        # embeddings = torch.tensor(
        #     item['embeddings'][EMBEDDING_TYPE][embedding_model][str(embedding_dim)],
        #     dtype=torch.float32, device=device
        # )
        embeddings = torch.tensor(
            item['inputs'][:, -128:],
            dtype=torch.float32, device=device
        )
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
        input_sims    : (N_pairs,)  cosine sim between embeddings i and j
        correct_agree : (N_pairs,)  1 if both correct or both wrong, 0 otherwise
        lift          : (N_pairs,)  agreement - chance_agreement for this sequence
        chance_agree  : float       p^2 + (1-p)^2 for this sequence
        same_pred_mask: (N_pairs,)  bool, True when argmax predictions match
    """
    correctness = (item['logits'].argmax(-1) == item['labels']).float()
    preds       = item['logits'].argmax(-1)
    seq_len     = item['embeddings'].shape[0]

    p_correct    = correctness.mean().item()
    chance_agree = p_correct**2 + (1 - p_correct)**2

    # embeddings   = item['embeddings'].cpu().numpy()
    # similarities = np.tril(get_similarities(embeddings, embeddings))

    input_sims     = []
    correct_agree  = []
    lifts          = []
    same_pred_mask = []

    for i in range(1, seq_len):
        for j in range(i):
            sim   = item['inputs'][i, INPUT_SIM_START + j].item()
            # sim   = similarities[i, j]
            agree = 1.0 if correctness[i] == correctness[j] else 0.0
            input_sims.append(sim)
            correct_agree.append(agree)
            lifts.append(agree - chance_agree)
            same_pred_mask.append(bool(preds[i] == preds[j]))

    return (
        np.array(input_sims),
        np.array(correct_agree),
        np.array(lifts),
        chance_agree,
        np.array(same_pred_mask),
    )


# ── per-strategy analysis ─────────────────────────────────────────────────────

def analyse_strategy(items):
    """Aggregate pairs across a sample of sequences for one strategy."""
    sample = random.sample(items, min(1500, len(items)))

    all_sims         = []
    all_agrees       = []
    all_lifts        = []
    all_chance_agree = []
    all_same_mask    = []

    for item in sample:
        sims, agrees, lifts, chance, same_mask = extract_pairs(item)
        all_sims.append(sims)
        all_agrees.append(agrees)
        all_lifts.append(lifts)
        all_chance_agree.append(chance)
        all_same_mask.append(same_mask)

    if not all_sims:
        return None

    sims      = np.concatenate(all_sims)
    agrees    = np.concatenate(all_agrees)
    lifts     = np.concatenate(all_lifts)
    same_mask = np.concatenate(all_same_mask)

    mean_chance_agree = float(np.mean(all_chance_agree))

    pearson_r,  pearson_p  = pearsonr(sims, lifts)
    spearman_r, spearman_p = spearmanr(sims, lifts)

    pearson_r_sp,  pearson_p_sp  = (pearsonr(sims[same_mask],   lifts[same_mask])   if same_mask.any()   else (np.nan, np.nan))
    spearman_r_sp, spearman_p_sp = (spearmanr(sims[same_mask],  lifts[same_mask])   if same_mask.any()   else (np.nan, np.nan))
    pearson_r_dp,  pearson_p_dp  = (pearsonr(sims[~same_mask],  lifts[~same_mask])  if (~same_mask).any() else (np.nan, np.nan))
    spearman_r_dp, spearman_p_dp = (spearmanr(sims[~same_mask], lifts[~same_mask])  if (~same_mask).any() else (np.nan, np.nan))

    n_bins      = 10
    bin_edges   = np.linspace(sims.min(), sims.max(), n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    bin_agree_means = []
    bin_agree_stds  = []
    bin_lift_means  = []
    bin_lift_stds   = []
    bin_counts      = []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (sims >= lo) & (sims < hi)
        n    = mask.sum()
        bin_counts.append(n)
        if n == 0:
            bin_agree_means.append(np.nan)
            bin_agree_stds.append(np.nan)
            bin_lift_means.append(np.nan)
            bin_lift_stds.append(np.nan)
        else:
            bin_agree_means.append(agrees[mask].mean())
            bin_agree_stds.append(agrees[mask].std() / np.sqrt(n))
            bin_lift_means.append(lifts[mask].mean())
            bin_lift_stds.append(lifts[mask].std() / np.sqrt(n))

    def _bin_subgroup(sub_sims, sub_agrees, sub_lifts):
        if len(sub_sims) == 0:
            nans = [np.nan] * n_bins
            return np.array(bin_centers), np.array(nans), np.array(nans), np.zeros(n_bins, dtype=int)
        edges = np.linspace(sub_sims.min(), sub_sims.max(), n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        means, stds, counts = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (sub_sims >= lo) & (sub_sims < hi)
            n = mask.sum()
            counts.append(n)
            if n == 0:
                means.append(np.nan); stds.append(np.nan)
            else:
                means.append(sub_agrees[mask].mean())
                stds.append(sub_agrees[mask].std() / np.sqrt(n))
        return centers, np.array(means), np.array(stds), np.array(counts)

    bc_sp, bam_sp, bas_sp, bcnt_sp = _bin_subgroup(sims[same_mask],  agrees[same_mask],  lifts[same_mask])
    bc_dp, bam_dp, bas_dp, bcnt_dp = _bin_subgroup(sims[~same_mask], agrees[~same_mask], lifts[~same_mask])

    return {
        'sims':              sims,
        'agrees':            agrees,
        'lifts':             lifts,
        'same_mask':         same_mask,
        'pearson_r':         pearson_r,
        'pearson_p':         pearson_p,
        'spearman_r':        spearman_r,
        'spearman_p':        spearman_p,
        'pearson_r_sp':      pearson_r_sp,
        'pearson_p_sp':      pearson_p_sp,
        'spearman_r_sp':     spearman_r_sp,
        'spearman_p_sp':     spearman_p_sp,
        'pearson_r_dp':      pearson_r_dp,
        'pearson_p_dp':      pearson_p_dp,
        'spearman_r_dp':     spearman_r_dp,
        'spearman_p_dp':     spearman_p_dp,
        'bin_centers':       np.array(bin_centers),
        'bin_agree_means':   np.array(bin_agree_means),
        'bin_agree_stds':    np.array(bin_agree_stds),
        'bin_lift_means':    np.array(bin_lift_means),
        'bin_lift_stds':     np.array(bin_lift_stds),
        'bin_counts':        np.array(bin_counts),
        'bc_sp':    bc_sp,  'bam_sp':   bam_sp,  'bas_sp':   bas_sp,  'bcnt_sp': bcnt_sp,
        'bc_dp':    bc_dp,  'bam_dp':   bam_dp,  'bas_dp':   bas_dp,  'bcnt_dp': bcnt_dp,
        'n_pairs':           len(sims),
        'n_same_pred':       same_mask.sum(),
        'n_diff_pred':       (~same_mask).sum(),
        'frac_same_pred':    same_mask.mean(),
        'mean_agree':        agrees.mean(),
        'mean_lift':         lifts.mean(),
        'mean_chance_agree': mean_chance_agree,
    }


def analyse_combo(llm, dataset, data_dir, embedding_model, embedding_dim, splits=('train',), device='cpu'):
    """Returns dict keyed by sampling strategy, each value the analysis result dict."""
    items_by_strategy = {s: [] for s in SAMPLING_STRATEGIES}

    for split in splits:
        path = os.path.join(data_dir, llm.replace('/', '_'), dataset, f'{split}.json')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Not found: {path}")
        loaded = load_raw_data(path, embedding_model, embedding_dim, device=device)
        for s in SAMPLING_STRATEGIES:
            items_by_strategy[s].extend(loaded[s])

    return {s: analyse_strategy(items_by_strategy[s]) for s in SAMPLING_STRATEGIES}


# ── plotting ──────────────────────────────────────────────────────────────────

def _draw_panels(fig, axes, res, row_label=None):
    """Draw the original 3 panels into the provided axes (one row of subplots)."""
    ax  = axes[0]
    v   = ~np.isnan(res['bin_agree_means'])
    xs  = res['bin_centers'][v]
    ys  = res['bin_agree_means'][v]
    err = res['bin_agree_stds'][v]

    ax.plot(xs, ys, 'o-', color='steelblue', linewidth=2, label='P(agree)')
    ax.fill_between(xs, ys - err, ys + err, alpha=0.25, color='steelblue')
    ax.axhline(
        res['mean_chance_agree'], linestyle='--', color='tomato', linewidth=1.5,
        label=f'chance agree = {res["mean_chance_agree"]:.3f}  (lift = {res["mean_lift"]:+.3f})'
    )
    ax.set_xlabel('Input Cosine Similarity')
    ax.set_ylabel('P(correctness agreement)')
    title0 = 'Binned: similarity -> correctness agreement'
    ax.set_title(f'{row_label}\n{title0}' if row_label else title0)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)

    ax = axes[1]
    hb = ax.hexbin(res['sims'], res['agrees'], gridsize=40, cmap='YlOrRd', mincnt=1)
    fig.colorbar(hb, ax=ax, label='count')
    ax.axhline(res['mean_chance_agree'], linestyle='--', color='tomato', linewidth=1,
               label=f'chance {res["mean_chance_agree"]:.3f}')
    ax.legend(fontsize=8)
    ax.set_xlabel('Input Cosine Similarity')
    ax.set_ylabel('Correctness Agreement (0/1)')
    ax.set_title(
        f"Pearson r={res['pearson_r']:.3f} (p={res['pearson_p']:.2e})\n"
        f"Spearman rho={res['spearman_r']:.3f} (p={res['spearman_p']:.2e})  [on lift]"
    )

    ax = axes[2]

    def _plot_subgroup(centers, means, stds, color, label, linestyle='-'):
        v = ~np.isnan(means)
        if v.any():
            ax.plot(centers[v], means[v], 'o' + linestyle, color=color, linewidth=2, label=label)
            ax.fill_between(centers[v], means[v] - stds[v], means[v] + stds[v], alpha=0.2, color=color)

    _plot_subgroup(res['bc_sp'], res['bam_sp'], res['bas_sp'],
                   color='mediumseagreen',
                   label=f"same pred  (n={res['n_same_pred']:,}, "
                         f"ρ={res['spearman_r_sp']:.3f}, p={res['spearman_p_sp']:.2e})")
    _plot_subgroup(res['bc_dp'], res['bam_dp'], res['bas_dp'],
                   color='darkorange', linestyle='--',
                   label=f"diff pred  (n={res['n_diff_pred']:,}, "
                         f"ρ={res['spearman_r_dp']:.3f}, p={res['spearman_p_dp']:.2e})")

    ax.axhline(res['mean_chance_agree'], linestyle=':', color='tomato', linewidth=1.5,
               label=f'chance = {res["mean_chance_agree"]:.3f}')
    ax.text(0.02, 0.04,
            f"{res['frac_same_pred']*100:.1f}% of pairs share same prediction",
            transform=ax.transAxes, fontsize=7.5, color='gray')
    ax.set_xlabel('Input Cosine Similarity')
    ax.set_ylabel('P(correctness agreement)')
    ax.set_title('Conditional on prediction match')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7.5)


def plot_all_configs(llm, dataset, strategy, cfg_results, out_dir):
    """One figure with one row per embedding config, 3 panels each (the original panels)."""
    configs = [(m, d) for (m, d) in EMBEDDING_CONFIGS
               if cfg_results.get((m, d)) is not None]
    if not configs:
        return

    n = len(configs)
    fig, axes = plt.subplots(n, 3, figsize=(18, 4 * n), squeeze=False)
    fig.suptitle(f'{llm}  x  {dataset}  x  {strategy}', fontsize=11, fontweight='bold')

    for row, (emb_model, emb_dim) in enumerate(configs):
        res = cfg_results[(emb_model, emb_dim)]
        _draw_panels(fig, axes[row], res, row_label=_cfg_label(emb_model, emb_dim))

    fig.tight_layout()
    combo_dir = os.path.join(out_dir, EMBEDDING_TYPE, llm.replace('/', '_'), dataset)
    os.makedirs(combo_dir, exist_ok=True)
    fig.savefig(os.path.join(combo_dir, f'{strategy}.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)



def plot_summary_heatmap(results, out_dir, strategy):
    """Two-panel heatmap: Spearman rho and mean lift, for one strategy."""
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
        (axes[0], rho_matrix,    'Spearman rho (on lift) — all pairs',      'Spearman rho',      -0.3,  0.3,  'RdYlGn', 0.2),
        (axes[1], rho_sp_matrix, 'Spearman rho (on lift) — same-pred only', 'Spearman rho',      -0.3,  0.3,  'RdYlGn', 0.2),
        (axes[2], lift_matrix,   'Mean lift (agree - chance)',               'Mean lift',         -0.1,  0.1,  'RdYlGn', 0.05),
        (axes[3], agree_matrix,  'Mean raw agreement P(agree)',              'P(agree)',           0.5,  1.0,  'YlGn',   0.1),
    ]

    for ax, matrix, title, label, vmin, vmax, cmap, white_thresh in panels:
        im = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap=cmap, aspect='auto')
        fig.colorbar(im, ax=ax, label=label)
        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels(datasets, rotation=30, ha='right', fontsize=8)
        ax.set_yticks(range(len(llms)))
        ax.set_yticklabels([l.split('/')[-1] for l in llms], fontsize=8)
        ax.set_title(title)
        for i, j in product(range(len(llms)), range(len(datasets))):
            if not np.isnan(matrix[i, j]):
                val = matrix[i, j]
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7,
                        color='white' if abs(val - (vmin + vmax) / 2) > white_thresh * (vmax - vmin) else 'black')

    fig.tight_layout()
    path = os.path.join(out_dir, f'summary_heatmap_{strategy}.png')
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved heatmap -> {path}")


# ── summary table ─────────────────────────────────────────────────────────────

def print_summary_table(results):
    col_w  = 28
    header = (
        f"{'LLM':<{col_w}} {'Dataset':<16} {'Strategy':<12} "
        f"{'N pairs':>10} {'P(agree)':>10} {'Chance':>8} {'Lift':>8} "
        f"{'Pearson r':>10} {'Spearman rho':>13} {'p (Sp)':>10} "
        f"{'rho (same-pred)':>16} {'rho (diff-pred)':>16}"
    )
    sep = "=" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")

    for (llm, ds) in sorted(results):
        for strategy in SAMPLING_STRATEGIES:
            res = results[(llm, ds)].get(strategy)
            if res is None:
                print(f"{llm.split('/')[-1]:<{col_w}} {ds:<16} {strategy:<12} {'N/A':>10}")
                continue
            sig = ("***" if res['spearman_p'] < 0.001 else
                   "**"  if res['spearman_p'] < 0.01  else
                   "*"   if res['spearman_p'] < 0.05  else "")
            sig_sp = ("***" if res['spearman_p_sp'] < 0.001 else
                      "**"  if res['spearman_p_sp'] < 0.01  else
                      "*"   if res['spearman_p_sp'] < 0.05  else "")
            sig_dp = ("***" if res['spearman_p_dp'] < 0.001 else
                      "**"  if res['spearman_p_dp'] < 0.01  else
                      "*"   if res['spearman_p_dp'] < 0.05  else "")
            print(
                f"{llm.split('/')[-1]:<{col_w}} {ds:<16} {strategy:<12} "
                f"{res['n_pairs']:>10,} {res['mean_agree']:>10.3f} "
                f"{res['mean_chance_agree']:>8.3f} {res['mean_lift']:>+8.3f} "
                f"{res['pearson_r']:>10.3f} {res['spearman_r']:>12.3f}{sig:<3}  "
                f"{res['spearman_p']:>10.2e} "
                f"{res['spearman_r_sp']:>15.3f}{sig_sp:<3} "
                f"{res['spearman_r_dp']:>15.3f}{sig_dp:<3}"
            )

    print(sep)
    print("Significance: * p<0.05  ** p<0.01  *** p<0.001")
    print("Lift = P(agree) - chance, where chance = p^2 + (1-p)^2 averaged across sequences.")
    print("Correlations (Pearson r, Spearman rho) are computed on lift, not raw agreement.")
    print("same-pred: pairs where argmax predictions match; diff-pred: complement.\n")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Correlation: input similarity vs correctness agreement")
    parser.add_argument('--llms',     required=True,                           help='Comma-separated LLM names')
    parser.add_argument('--datasets', required=True,                           help='Comma-separated dataset names')
    parser.add_argument('--data_dir', default='calibration/datasets2',         help='Root data directory')
    parser.add_argument('--out_dir',  default='analysis/feat_correlation/simvec', help='Output directory for plots')
    parser.add_argument('--splits',   default='train',                         help='Comma-separated splits to include')
    parser.add_argument('--device',   default='cpu')
    args = parser.parse_args()

    llms     = [x.strip() for x in args.llms.split(',')]
    datasets = [x.strip() for x in args.datasets.split(',')]
    splits   = [x.strip() for x in args.splits.split(',')]

    # ── run per-config analysis ───────────────────────────────────────────────
    # results[(llm, ds)][(emb_model, emb_dim)][strategy] = result dict
    results_by_config = {}

    for llm, ds in tqdm(list(product(llms, datasets)), desc='LLM x dataset combos'):
        results_by_config[(llm, ds)] = {}
        for emb_model, emb_dim in tqdm(EMBEDDING_CONFIGS, desc=f'  embedding configs for {llm}/{ds}', leave=False):
            cfg_key = (emb_model, emb_dim)
            try:
                results_by_config[(llm, ds)][cfg_key] = analyse_combo(
                    llm, ds, args.data_dir, emb_model, emb_dim,
                    splits=splits, device=args.device
                )
            except (FileNotFoundError, KeyError) as e:
                print(f"  Skipping {cfg_key}: {e}")
                results_by_config[(llm, ds)][cfg_key] = {s: None for s in SAMPLING_STRATEGIES}

    # ── summary table (first config) ─────────────────────────────────────────
    first_cfg = EMBEDDING_CONFIGS[0]
    print_summary_table({(llm, ds): results_by_config[(llm, ds)].get(first_cfg, {})
                         for llm, ds in results_by_config})

    # ── one figure per (llm, ds, strategy), rows = embedding configs ─────────
    for (llm, ds), cfg_results in results_by_config.items():
        for strategy in SAMPLING_STRATEGIES:
            strat_by_cfg = {cfg: cfg_results[cfg][strategy]
                            for cfg in EMBEDDING_CONFIGS
                            if cfg_results.get(cfg, {}).get(strategy) is not None}
            plot_all_configs(llm, ds, strategy, strat_by_cfg, args.out_dir)