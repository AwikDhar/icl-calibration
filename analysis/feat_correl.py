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

Base rate correction:
  Raw correctness agreement is inflated when model accuracy p is high/low,
  since chance agreement = p^2 + (1-p)^2. We report lift = agreement - chance_agreement,
  computed per-sequence using that sequence's observed accuracy, then averaged.
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

T = 21
INPUT_SIM_START = 4 + T   # 25
INPUT_SIM_END   = 4 + 2*T # 46

SAMPLING_STRATEGIES = ('ENTROPY', 'SIMILARITY')


# ── data loading ──────────────────────────────────────────────────────────────

def load_raw_data(path: str, device='cpu'):
    """
    Load raw JSON. Returns a dict keyed by sampling strategy,
    each value a list of {inputs, logits, labels} dicts.
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

        logits = torch.tensor(item['logits'], dtype=torch.float32, device=device)[:T]
        labels = torch.tensor(item['labels'], dtype=torch.long,    device=device)[:T]
        inputs = torch.tensor(item['inputs'], dtype=torch.float32, device=device)[:T]

        if torch.isnan(logits).any():
            continue

        items_by_strategy[strategy].append({
            'inputs': inputs,
            'logits': logits,
            'labels': labels,
        })

    return items_by_strategy


# ── pair extraction ───────────────────────────────────────────────────────────

def extract_pairs(item):
    """
    Extract all causal pairs (j < i) from one sequence.

    Returns:
        input_sims    : (N_pairs,)  cosine sim between inputs i and j
        correct_agree : (N_pairs,)  1 if both correct or both wrong, 0 otherwise
        lift          : (N_pairs,)  agreement - chance_agreement for this sequence
                        chance_agreement = p^2 + (1-p)^2 where p = sequence accuracy
    """
    inputs      = item['inputs']                                              # (T, D)
    correctness = (item['logits'].argmax(-1) == item['labels']).float()       # (T,)
    seq_len     = inputs.shape[0]

    # per-sequence chance level
    p_correct    = correctness.mean().item()
    chance_agree = p_correct**2 + (1 - p_correct)**2

    input_sims    = []
    correct_agree = []
    lifts         = []

    for i in range(1, seq_len):
        for j in range(i):                                # causal: j < i
            sim   = inputs[i, INPUT_SIM_START + j].item()
            agree = 1.0 if correctness[i] == correctness[j] else 0.0
            input_sims.append(sim)
            correct_agree.append(agree)
            lifts.append(agree - chance_agree)

    return (
        np.array(input_sims),
        np.array(correct_agree),
        np.array(lifts),
    )


# ── per-strategy analysis ─────────────────────────────────────────────────────

def analyse_strategy(items):
    """Aggregate pairs across all sequences for one strategy."""
    all_sims   = []
    all_agrees = []
    all_lifts  = []

    for item in random.sample(items, 1000):
        sims, agrees, lifts = extract_pairs(item)
        all_sims.append(sims)
        all_agrees.append(agrees)
        all_lifts.append(lifts)

    if not all_sims:
        return None

    sims   = np.concatenate(all_sims)
    agrees = np.concatenate(all_agrees)
    lifts  = np.concatenate(all_lifts)

    pearson_r,  pearson_p  = pearsonr(sims, lifts)
    spearman_r, spearman_p = spearmanr(sims, lifts)

    # binned stats
    n_bins      = 10
    bin_edges   = np.linspace(sims.min(), sims.max(), n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_agree_means = []
    bin_lift_means  = []
    bin_lift_stds   = []
    bin_counts      = []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (sims >= lo) & (sims < hi)
        n    = mask.sum()
        bin_counts.append(n)
        if n == 0:
            bin_agree_means.append(np.nan)
            bin_lift_means.append(np.nan)
            bin_lift_stds.append(np.nan)
        else:
            bin_agree_means.append(agrees[mask].mean())
            bin_lift_means.append(lifts[mask].mean())
            bin_lift_stds.append(lifts[mask].std() / np.sqrt(n))

    return {
        'sims':             sims,
        'agrees':           agrees,
        'lifts':            lifts,
        'pearson_r':        pearson_r,
        'pearson_p':        pearson_p,
        'spearman_r':       spearman_r,
        'spearman_p':       spearman_p,
        'bin_centers':      np.array(bin_centers),
        'bin_agree_means':  np.array(bin_agree_means),
        'bin_lift_means':   np.array(bin_lift_means),
        'bin_lift_stds':    np.array(bin_lift_stds),
        'bin_counts':       np.array(bin_counts),
        'n_pairs':          len(sims),
        'base_rate':        agrees.mean(),
        'mean_lift':        lifts.mean(),
    }


def analyse_combo(llm, dataset, data_dir, splits=('train',), device='cpu'):
    """
    Returns dict keyed by sampling strategy, each value the analysis result dict.
    """
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

def plot_combo(llm, dataset, strategy, res, out_dir):
    """Two-panel plot for one LLM x dataset x strategy."""
    combo_dir = os.path.join(out_dir, llm.replace('/', '_'), dataset)
    os.makedirs(combo_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(f'{llm}  x  {dataset}  x  {strategy}', fontsize=11, fontweight='bold')

    # ── left: binned lift vs similarity, raw agreement on twin axis ──
    ax  = axes[0]
    v   = ~np.isnan(res['bin_lift_means'])
    xs  = res['bin_centers'][v]
    ys  = res['bin_lift_means'][v]
    err = res['bin_lift_stds'][v]

    ax.plot(xs, ys, 'o-', color='steelblue', linewidth=2, label='lift = agree - chance')
    ax.fill_between(xs, ys - err, ys + err, alpha=0.25, color='steelblue')
    ax.axhline(0,               linestyle='--', color='grey',   linewidth=1, label='chance (lift=0)')
    ax.axhline(res['mean_lift'], linestyle=':',  color='tomato', linewidth=1,
               label=f'mean lift {res["mean_lift"]:+.3f}')
    ax.set_xlabel('Input Cosine Similarity')
    ax.set_ylabel('Correctness Agreement Lift')
    ax.set_title('Binned: similarity -> lift over chance agreement')
    ax.legend(fontsize=8)

    # raw agreement on secondary axis for context
    ax2 = ax.twinx()
    v2  = ~np.isnan(res['bin_agree_means'])
    ax2.plot(res['bin_centers'][v2], res['bin_agree_means'][v2],
             's--', color='darkorange', linewidth=1, alpha=0.6, markersize=4, label='raw agree')
    ax2.set_ylabel('Raw Agreement Rate', color='darkorange')
    ax2.tick_params(axis='y', labelcolor='darkorange')
    ax2.set_ylim(0, 1)

    # ── right: hexbin of sim vs lift ──
    ax = axes[1]
    hb = ax.hexbin(res['sims'], res['lifts'], gridsize=40, cmap='YlOrRd', mincnt=1)
    fig.colorbar(hb, ax=ax, label='count')
    ax.axhline(0, linestyle='--', color='grey', linewidth=1)
    ax.set_xlabel('Input Cosine Similarity')
    ax.set_ylabel('Lift (agreement - chance)')
    ax.set_title(
        f"Pearson r={res['pearson_r']:.3f} (p={res['pearson_p']:.2e})\n"
        f"Spearman rho={res['spearman_r']:.3f} (p={res['spearman_p']:.2e})"
    )

    fig.tight_layout()
    fig.savefig(os.path.join(combo_dir, f'{strategy}.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_summary_heatmap(results, out_dir, strategy):
    """Spearman rho heatmap for one strategy across all LLM x dataset combos."""
    llms     = sorted({llm for (llm, _) in results})
    datasets = sorted({ds  for (_, ds)  in results})

    rho_matrix  = np.full((len(llms), len(datasets)), np.nan)
    lift_matrix = np.full((len(llms), len(datasets)), np.nan)

    for i, llm in enumerate(llms):
        for j, ds in enumerate(datasets):
            res = results.get((llm, ds), {}).get(strategy)
            if res is not None:
                rho_matrix[i, j]  = res['spearman_r']
                lift_matrix[i, j] = res['mean_lift']

    fig, axes = plt.subplots(1, 2, figsize=(max(8, len(datasets) * 3), max(3, len(llms) * 0.9)))
    fig.suptitle(f'[{strategy}] Summary', fontsize=11, fontweight='bold')

    for ax, matrix, title, label, vmin, vmax, cmap in [
        (axes[0], rho_matrix,  'Spearman rho (sim vs lift)',    'Spearman rho',  -0.3, 0.3, 'RdYlGn'),
        (axes[1], lift_matrix, 'Mean lift (agreement - chance)', 'Mean lift',    -0.1, 0.1, 'RdYlGn'),
    ]:
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
                        color='white' if abs(val) > (0.2 if vmax == 0.3 else 0.06) else 'black')

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
        f"{'N pairs':>10} {'Base rate':>10} {'Mean lift':>10} "
        f"{'Pearson r':>10} {'Spearman rho':>13} {'p (Sp)':>10}"
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
            print(
                f"{llm.split('/')[-1]:<{col_w}} {ds:<16} {strategy:<12} "
                f"{res['n_pairs']:>10,} {res['base_rate']:>10.3f} {res['mean_lift']:>+10.3f} "
                f"{res['pearson_r']:>10.3f} {res['spearman_r']:>12.3f}{sig:<3}  "
                f"{res['spearman_p']:>10.2e}"
            )

    print(sep)
    print("Significance: * p<0.05  ** p<0.01  *** p<0.001")
    print("Lift = raw agreement - (p^2 + (1-p)^2) per sequence, where p = sequence accuracy.\n")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Correlation: input similarity vs correctness agreement lift")
    parser.add_argument('--llms',     required=True,                       help='Comma-separated LLM names')
    parser.add_argument('--datasets', required=True,                       help='Comma-separated dataset names')
    parser.add_argument('--data_dir', default='calibration/datasets',      help='Root data directory')
    parser.add_argument('--out_dir',  default='analysis/feat_correlation', help='Output directory for plots')
    parser.add_argument('--splits',   default='train',                     help='Comma-separated splits to include')
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

    for strategy in SAMPLING_STRATEGIES:
        plot_summary_heatmap(results, args.out_dir, strategy)