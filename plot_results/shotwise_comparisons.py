import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from calibration_methods import CalibrationMethods
from plot_results.results_utils import get_saved_results, cvt_to_sampling_type, SAVE_DIR, ROOT_DIR, METHOD_NAME_MAP, METHOD_COLOUR_MAP

PLOT_DIR = ROOT_DIR / "plot_results" / "comparisons"

plt.rcParams.update({'font.size': 20})

def convert_to_list(items, cvt_func=None):
    if cvt_func:
        return [cvt_func(s.strip()) for s in items.split(",")]
    else:
        return [s.strip() for s in items.split(",")]

def average_metrics_list(metric_list):
    """
    Average a list of metric dictionaries (each with 'mean' and 'std').
    Uses pooled standard deviation: sqrt(mean(variances))
    """
    if not metric_list:
        return {'mean': 0.0, 'std': 0.0}
    
    means = [m['mean'] for m in metric_list]
    stds = [m['std'] for m in metric_list]
    
    avg_mean = np.mean(means)
    variances = [s**2 for s in stds]
    pooled_std = np.sqrt(np.mean(variances))
    
    return {'mean': avg_mean, 'std': pooled_std}

def format_metric(metric_dict, show_std):
    """Format metric as 'mean ± std' with appropriate decimal places."""
    mean = metric_dict['mean']
    std = metric_dict['std']
    
    if mean > 1:  # Not brier loss. A percentage. 1 decimal precision
        return f"{mean:.1f} ± {std:.1f}" if show_std else f"{mean:.1f}"
    
    return f"{mean:.3f} ± {std:.3f}" if show_std else f"{mean:.3f}"

def parse_metric(metric_str):
    """Parse 'mean ± std' string back to floats."""
    if pd.isna(metric_str) or metric_str == '':
        return 0.0, 0.0
    parts = str(metric_str).split(' ± ')
    mean = float(parts[0])
    std = float(parts[1]) if len(parts) > 1 else 0.0
    return mean, std

def get_metric_display_name(metric):
    """Convert metric key to display name."""
    if metric.lower() in ('brier', 'accuracy'):
        return metric.upper()
    return metric.title()

def average_across_datasets(
    results_dict,
    model,
    datasets,
    sampling_strategy,
    all_shots,
    calibration_methods,
    metric
):
    """
    Average results across multiple datasets for each calibration method and shot count.
    
    Returns:
        dict: {calibration_method: {'means': [...], 'stds': [...]}}
    """
    averaged_results = {}
    
    for calibration_method in calibration_methods:
        shot_means = []
        shot_stds = []
        
        for num_shots in all_shots:
            dataset_means = []
            dataset_stds = []
            
            for dataset in datasets:
                # try:
                data = results_dict[dataset][model][sampling_strategy][num_shots][calibration_method]
                dataset_means.append(data[metric]['mean'])
                dataset_stds.append(data[metric]['std'])
                # except (KeyError, TypeError):
                #     # Skip missing data
                #     raise
            
            if dataset_means:
                # Average across datasets
                shot_means.append(np.mean(dataset_means))
                # Propagate uncertainty (average of stds)
                shot_stds.append(np.mean(dataset_stds))
            else:
                shot_means.append(np.nan)
                shot_stds.append(np.nan)
        
        averaged_results[calibration_method] = {
            'means': np.array(shot_means),
            'stds': np.array(shot_stds)
        }
    
    return averaged_results

def plot_dataset_averaged_comparison(
    results_dict,
    model,
    datasets,
    sampling_strategy,
    all_shots,
    calibration_methods,
    metric,
    save_dir=PLOT_DIR
):
    """
    Plot comparison of calibration methods averaged across datasets.
    """
    # Get averaged results
    averaged_results = average_across_datasets(
        results_dict,
        model,
        datasets,
        sampling_strategy,
        all_shots,
        calibration_methods,
        metric
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each calibration method
    for calibration_method in calibration_methods:
        means = averaged_results[calibration_method]['means']
        stds = averaged_results[calibration_method]['stds']
        
        # Create method label
        if calibration_method == CalibrationMethods.UNCALIBRATED:
            method_name = 'Uncalibrated'
            marker = 'o'
        else:
            method_name = METHOD_NAME_MAP.get(calibration_method, calibration_method.name.replace('_', ' ').title())
            marker = 's'
        
        color = METHOD_COLOUR_MAP[calibration_method]
        ax.plot(all_shots, means, marker=marker, label=method_name,
                color=color, linewidth=2, markersize=8)
        ax.fill_between(all_shots, means - stds, means + stds,
                        alpha=0.2, color=color)
    
    # Formatting
    ax.set_xlabel('Number of Shots', fontsize=14, fontweight='bold')
    ax.set_ylabel(get_metric_display_name(metric), fontsize=14, fontweight='bold')
    
    dataset_list = ', '.join(datasets) if len(datasets) <= 3 else f"{len(datasets)} datasets"
    # ax.set_title(f"{get_metric_display_name(metric)} (Averaged Across Datasets)\n{model} on {dataset_list} ({sampling_strategy.name})",
    #              fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', frameon=True, shadow=True, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(all_shots)
    
    # Set y-axis to start from 0 for better comparison
    bottom = 0 if metric in ['ece', 'mce'] else None
    ax.set_ylim(bottom=bottom)
    
    plt.tight_layout()
    
    # Save plot
    model_name = model.replace('/', '_').replace('-FP8', '')
    save_path = save_dir / model_name / "shotwise/plots" / sampling_strategy.name
    save_path.mkdir(parents=True, exist_ok=True)
    
    filename = save_path / f"{metric}_comparison.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {filename}")
    
    plt.close()

def create_csv_table(
    results_dict,
    model,
    datasets,
    sampling_strategy,
    all_shots,
    calibration_methods,
    metrics,
    show_std,
    save_dir=PLOT_DIR
):
    """
    Create a CSV table with multi-level headers similar to the reference format.
    
    Format:
    Row 1: Empty, then "Shot Counts" spanning all shot columns
    Row 2: Empty, then shot numbers repeated for each metric (if multiple metrics)
    Row 3: "Method", then metric names for each shot count
    Data rows: One per calibration method
    Last row: Average across all methods
    """
    
    # Collect data for each calibration method
    data_rows = []
    
    for calibration_method in calibration_methods:
        # Create method name
        if calibration_method == CalibrationMethods.UNCALIBRATED:
            method_name = 'Uncalibrated'
        else:
            method_name = METHOD_NAME_MAP.get(calibration_method, calibration_method.name.replace('_', ' ').title())
        
        row_data = [method_name]
        
        # For each shot count, collect metrics
        for num_shots in all_shots:
            for metric in metrics:
                metric_list = []
                
                # Collect metric across all datasets
                for dataset in datasets:
                    try:
                        method_results = results_dict[dataset][model][sampling_strategy][num_shots][calibration_method]
                        if metric in method_results:
                            metric_list.append(method_results[metric])
                    except (KeyError, TypeError):
                        pass
                
                # Average across datasets
                avg_metric = average_metrics_list(metric_list)
                row_data.append(format_metric(avg_metric, show_std))
        
        data_rows.append(row_data)
    
    # Create column structure for MultiIndex
    col_tuples = [('', '', 'Method')]
    
    for num_shots in all_shots:
        for metric in metrics:
            metric_display = get_metric_display_name(metric)
            col_tuples.append(('Shot Counts', f'{num_shots}-shot', metric_display))
    
    # Create DataFrame with MultiIndex columns
    df = pd.DataFrame(data_rows)
    df.columns = pd.MultiIndex.from_tuples(col_tuples)
    
    # Calculate average row across all methods
    avg_row = ['Average']
    
    for i in range(1, len(data_rows[0])):
        values_mean = []
        values_std = []
        
        for row in data_rows:
            mean_val, std_val = parse_metric(row[i])
            values_mean.append(mean_val)
            values_std.append(std_val)
        
        # Average the means and pool the stds
        avg_mean = np.mean(values_mean)
        variances = [std**2 for std in values_std if std > 0]
        avg_std = np.sqrt(np.mean(variances)) if variances else 0.0
        
        avg_row.append(format_metric({'mean': avg_mean, 'std': avg_std}, show_std))
    
    # Append average row
    avg_df = pd.DataFrame([avg_row])
    avg_df.columns = df.columns
    df = pd.concat([df, avg_df], ignore_index=True)
    
    # Save to CSV
    model_name = model.replace('/', '_').replace('-FP8', '')
    save_path = save_dir / model_name / "shotwise/tables" / sampling_strategy.name
    save_path.mkdir(parents=True, exist_ok=True)
    
    metric_str = '_'.join(metrics)
    filename = save_path / f"{metric_str}_dataset_averaged_table.csv"
    
    save_table_with_formatting(df, filename)
    print(f"Saved CSV table: {filename}")
    
    return df

def save_table_with_formatting(df, output_path):
    """
    Save DataFrame to CSV with proper multi-level header formatting.
    """
    # Create header rows
    if isinstance(df.columns, pd.MultiIndex):
        # Row 1: Top level (Shot Counts span)
        row1 = ['']  # Empty cell for Method column
        current_level0 = None
        span_count = 0
        
        for col in df.columns[1:]:  # Skip first column (Method)
            if col[0] != current_level0:
                if current_level0 is not None:
                    row1.extend([current_level0] + [''] * (span_count - 1))
                current_level0 = col[0]
                span_count = 1
            else:
                span_count += 1
        
        # Add the last span
        if current_level0 is not None:
            row1.extend([current_level0] + [''] * (span_count - 1))
        
        # Row 2: Shot count labels
        row2 = ['']  # Empty cell for Method column
        current_level1 = None
        span_count = 0
        
        for col in df.columns[1:]:
            if col[1] != current_level1:
                if current_level1 is not None:
                    row2.extend([current_level1] + [''] * (span_count - 1))
                current_level1 = col[1]
                span_count = 1
            else:
                span_count += 1
        
        if current_level1 is not None:
            row2.extend([current_level1] + [''] * (span_count - 1))
        
        # Row 3: Metric labels
        row3 = [col[2] for col in df.columns]
        
        # Flatten columns for data export
        df_export = df.copy()
        df_export.columns = row3
        
        # Write to CSV with custom headers
        with open(output_path, 'w') as f:
            # Write header rows
            f.write(','.join(row1) + '\n')
            f.write(','.join(row2) + '\n')
            # Write data with column names
            df_export.to_csv(f, index=False)
    else:
        # Simple columns, just save normally
        df.to_csv(output_path, index=False)

def process_all_combinations(
    models,
    datasets,
    num_seeds,
    all_shots,
    sampling_strategies,
    calibration_methods,
    metrics,
    show_std,
    results_dir=None,
    save_dir=PLOT_DIR
):
    """
    Generate plots and CSV tables for all specified combinations.
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
    
    calibration_methods_with_uncal = [CalibrationMethods.UNCALIBRATED] + calibration_methods
    
    print(f"\nGenerating dataset-averaged plots and tables for metrics: {metrics}")
    
    for model in models:
        for sampling_strategy in sampling_strategies:
            print(f"\nProcessing: {model} | {sampling_strategy.name}")
            
            # Generate plots for each metric individually
            for metric in metrics:
                print(f"  - Plotting {metric}...")
                plot_dataset_averaged_comparison(
                    results_dict=results_dict,
                    model=model,
                    datasets=datasets,
                    sampling_strategy=sampling_strategy,
                    all_shots=all_shots,
                    calibration_methods=calibration_methods_with_uncal,
                    metric=metric,
                    save_dir=save_dir
                )
            
            # Generate CSV table with all metrics together
            print(f"  - Creating table with all metrics...")
            create_csv_table(
                results_dict=results_dict,
                model=model,
                datasets=datasets,
                sampling_strategy=sampling_strategy,
                all_shots=all_shots,
                calibration_methods=calibration_methods_with_uncal,
                metrics=metrics,
                show_std=show_std,
                save_dir=save_dir
            )
    
    print(f"\n✓ All plots and tables generated successfully! Saved to {save_dir}")

def main(args):
    sampling_strategies = [cvt_to_sampling_type(s) for s in args.sampling_strategies.split(",")]
    calibration_methods = [CalibrationMethods[m.strip()] for m in args.calibration_methods.split(",")]
    
    results_dir = Path(args.results_dir) if args.results_dir else None
    save_dir = Path(args.save_dir)
    
    process_all_combinations(
        models=args.models,
        datasets=args.datasets,
        num_seeds=args.num_seeds,
        all_shots=args.all_shots,
        sampling_strategies=sampling_strategies,
        calibration_methods=calibration_methods,
        metrics=args.metrics,
        show_std=args.show_std,
        results_dir=results_dir,
        save_dir=save_dir
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Plot calibration methods comparison averaged across datasets'
    )
    
    parser.add_argument('--models', type=str, required=True,
                        help='Comma-separated list of model names')
    parser.add_argument('--datasets', type=str, required=True,
                        help='Comma-separated list of dataset names (use all available if not specified)')
    parser.add_argument('--num_seeds', type=int, default=10,
                        help='Number of random seeds')
    parser.add_argument('--all_shots', type=str, required=True,
                        help='Comma-separated shot counts (e.g., "0,4,8,12")')
    parser.add_argument('--sampling_strategies', type=str, required=True,
                        help='Comma-separated in-context examples sampling strategies')
    parser.add_argument('--calibration_methods', type=str, required=True,
                        help='Comma-separated ICL calibration methods')
    parser.add_argument('--metrics', type=str, required=True,
                        help='Comma-separated metrics to plot (e.g., "ece,brier,mce")')
    parser.add_argument('--show_std', action='store_true', default=False,
                        help='Whether to show standard deviations in the table (default: False)')
    parser.add_argument('--results_dir', type=str, default=SAVE_DIR,
                        help='Directory containing saved results of experiments')
    parser.add_argument('--save_dir', type=str, default=str(PLOT_DIR),
                        help='Directory to save dataset-averaged plots and tables')
    
    args = parser.parse_args()
    
    # Parse arguments
    args.models = convert_to_list(args.models)
    args.datasets = convert_to_list(args.datasets)
    args.all_shots = convert_to_list(args.all_shots, int)
    args.metrics = convert_to_list(args.metrics)
    
    main(args)