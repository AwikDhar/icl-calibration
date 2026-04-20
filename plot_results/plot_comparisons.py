import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from calibration_methods import CalibrationMethods
from plot_results.results_utils import get_saved_results, cvt_to_sampling_type, get_metric_display_name
from plot_results.results_utils import SAVE_DIR_TMP, ROOT_DIR, METHOD_NAME_MAP, METHOD_COLOUR_MAP

PLOT_DIR = ROOT_DIR / "plot_results" / "comparisons"

plt.rcParams.update({
    'font.size': 25,        
    'lines.linewidth': 2.5,
    'grid.linewidth': 1.5,
    'xtick.labelsize': 20,    
    'ytick.labelsize': 20,    
    'legend.fontsize': 20,    
    'axes.labelsize': 25      
})

def convert_to_list(items, cvt_func=None):
    if cvt_func:
        return [cvt_func(s.strip()) for s in items.split(",")]
    else:
        return [s.strip() for s in items.split(",")]

def plot_metrics_row(
    results_dict,
    model,
    dataset,
    sampling_strategy,
    all_shots,
    calibration_methods,
    metrics,
    save_dir=PLOT_DIR
):
    """
    Plot comparison of calibration methods for multiple metrics in a row.
    
    Args:
        results_dict: Nested dictionary from get_saved_results
        model: Model name
        dataset: Dataset name
        sampling_strategy: SamplingStrategy enum
        all_shots: List of shot counts
        calibration_methods: List of CalibrationMethods to compare
        metrics: List of metrics to plot
        save_dir: Directory to save plots
    """
    n_metrics = len(metrics)
    
    # Create figure with subplots in a row
    # Adjust figure width based on number of metrics
    fig_width = 8 * n_metrics
    fig, axes = plt.subplots(1, n_metrics, figsize=(fig_width, 6))
    
    # Ensure axes is always a list for consistent indexing
    if n_metrics == 1:
        axes = [axes]
    
    # Plot each metric
    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]
        
        # Plot each calibration method
        for idx, calibration_method in enumerate(calibration_methods):
            means = []
            stds = []
            
            for num_shots in all_shots:
                data = results_dict[dataset][model][sampling_strategy][num_shots][calibration_method]
                means.append(data[metric]['mean'])
                stds.append(data[metric]['std'])
            
            means = np.array(means)
            stds = np.array(stds)
            
            # Create method label
            if calibration_method == CalibrationMethods.UNCALIBRATED:
                method_name = 'Uncalibrated'
                marker = 'o'
            else:
                method_name = METHOD_NAME_MAP.get(calibration_method, calibration_method.name.replace('_', ' ').title())
                marker = 's'

            color = METHOD_COLOUR_MAP[calibration_method]
            ax.plot(all_shots, means, marker=marker, label=method_name, 
                    color=color, linewidth=2, markersize=6)
            ax.fill_between(all_shots, means - stds, means + stds, 
                             alpha=0.2, color=color)
        
        # Formatting for this subplot
        ax.set_xlabel('Number of Shots')
        ax.set_ylabel(get_metric_display_name(metric))
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xticks(all_shots)
        
        # Set y-axis to start from 0 for ECE
        bottom = 0 if metric == 'ece' else None
        ax.set_ylim(bottom=bottom)
    
    # Create a single legend below all subplots
    # Get handles and labels from the first subplot (they're all the same)
    handles, labels = axes[0].get_legend_handles_labels()
    
    # Add legend below the subplots
    fig.legend(handles, labels, loc='lower center', ncol=len(calibration_methods),
               bbox_to_anchor=(0.5, -0.05), frameon=True, shadow=True)
    
    # Adjust layout to make room for legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    
    # Save plot
    model_name = model.replace('/', '_').replace('-FP8', '')
    save_path = save_dir / model_name / "datasets/plots" / dataset / sampling_strategy.name
    save_path.mkdir(parents=True, exist_ok=True)
    
    metrics_str = '_'.join(metrics)
    filename = save_path / f"{metrics_str}_comparison.png"
    plt.savefig(filename, dpi=400, bbox_inches='tight')
    
    plt.close()


def plot_all_metrics(
    models,
    datasets,
    num_seeds,
    all_shots,
    sampling_strategies,
    calibration_methods,
    metrics,
    results_dir=None,
    save_dir=PLOT_DIR
):
    """
    Generate plots for all specified metrics, models, datasets, and sampling strategies.
    
    Args:
        models: List of model names
        datasets: List of dataset names
        num_seeds: Number of random seeds
        all_shots: List of shot counts
        sampling_strategies: List of SamplingStrategy enums
        calibration_methods: List of CalibrationMethods
        metrics: List of metrics to plot
        results_dir: Directory containing saved results
        save_dir: Directory to save plots
    """
    # Load all results
    print("Loading saved results...")
    results_dict = get_saved_results(
        models=models,
        datasets=datasets,
        num_seeds=num_seeds,
        all_shots=all_shots,
        sampling_strategies=sampling_strategies,
        calibration_methods=calibration_methods,
        results_dir=results_dir
    )
    
    calibration_methods = [CalibrationMethods.UNCALIBRATED] + calibration_methods
    
    print(f"\nGenerating plots for metrics: {metrics}")

    for model in models:
        for dataset in datasets:
            for sampling_strategy in sampling_strategies:
                plot_metrics_row(
                    results_dict=results_dict,
                    model=model,
                    dataset=dataset,
                    sampling_strategy=sampling_strategy,
                    all_shots=all_shots,
                    calibration_methods=calibration_methods,
                    metrics=metrics,
                    save_dir=save_dir
                )

    print(f"\n✓ All plots generated successfully! Saved to {save_dir}")

def main(args):    
    sampling_strategies = [cvt_to_sampling_type(s) for s in args.sampling_strategies.split(",")]
    calibration_methods = [CalibrationMethods[m.strip()] for m in args.calibration_methods.split(",")]
    
    results_dir = Path(args.results_dir) if args.results_dir else None
    save_dir = Path(args.save_dir)
    
    plot_all_metrics(
        models=args.models,
        datasets=args.datasets,
        num_seeds=args.num_seeds,
        all_shots=args.all_shots,
        sampling_strategies=sampling_strategies,
        calibration_methods=calibration_methods,
        metrics=args.metrics,
        results_dir=results_dir,
        save_dir=save_dir
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot calibration methods comparison')
    
    parser.add_argument('--models', type=str, required=True,
                        help='Comma-separated list of model names')
    parser.add_argument('--datasets', type=str, required=True,
                        help='Comma-separated list of dataset names')
    parser.add_argument('--num_seeds', type=int, default=10,
                        help='Number of random seeds')
    parser.add_argument('--all_shots', type=str, required=True,
                        help='shot counts')
    parser.add_argument('--sampling_strategies', type=str, required=True,
                        help='in-context examples sampling strategies')
    parser.add_argument('--calibration_methods', type=str, required=True,
                        help='ICL calibration methods')
    parser.add_argument('--metrics', type=str, required=True,
                        help='metrics to plot comparisons on')
    parser.add_argument('--results_dir', type=str, default=SAVE_DIR_TMP,
                        help='Directory containing saved results of experiments')
    parser.add_argument('--save_dir', type=str, default=str(PLOT_DIR),
                        help='Directory to save calibration method comparison plots')
    
    args = parser.parse_args()
    
    # Parse arguments
    args.models = convert_to_list(args.models)
    args.datasets = convert_to_list(args.datasets)
    args.all_shots = convert_to_list(args.all_shots, int)
    args.metrics = convert_to_list(args.metrics)
     
    main(args)