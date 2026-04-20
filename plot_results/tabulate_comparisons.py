import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from plot_results.results_utils import cvt_to_sampling_type, get_saved_results, METHOD_NAME_MAP, ROOT_DIR, SAVE_DIR
from calibration_methods import CalibrationMethods
from utils.gen_utils import convert_to_list

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
    
    if mean > 1:  # Not brier loss. A percentage. 1 decimal precision (fix this heuristic)
        return f"{mean:.1f} ± {std:.1f}" if show_std else  f"{mean:.1f}"
    
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
    if metric.lower() in ('brier', 'accuracy'):
        return metric.upper()
    
    return metric.title()  

def create_comparison_table(results, model, datasets, all_shots, sampling_strategies, 
                           calibration_methods, metrics, show_std):
    """
    Create a comparison table with multi-level headers for a single model.
    
    Format:
    Row 1: "Calibration Methods" spanning all method columns
    Row 2: Method#1, Method#2, Method#3, ...
    Row 3: Metric1, Metric2, ... for each method
    Data rows: One per dataset
    Last row: Average across all datasets
    """
    
    # Collect data for each dataset
    data_rows = []
    
    for dataset in datasets:
        row_data = [dataset]  # First column is dataset name
        
        dataset_results = results[dataset][model]
        
        # For each calibration method, collect all specified metrics
        for method in calibration_methods:
            for metric in metrics:
                metric_list = []
                
                # Collect metrics across all shots and sampling strategies
                for sampling_strategy in sampling_strategies:
                    for num_shots in all_shots:
                        method_results = dataset_results[sampling_strategy][num_shots]
                        if method in method_results:
                            method_metrics = method_results[method]
                            if metric in method_metrics:
                                metric_list.append(method_metrics[metric])
                
                # Average across all configurations
                avg_metric = average_metrics_list(metric_list)
                row_data.append(format_metric(avg_metric, show_std))
        
        data_rows.append(row_data)
    
    # Create column structure for MultiIndex
    # Level 0: Empty for dataset column, "Calibration Methods" spanning all method columns
    # Level 1: Empty for dataset column, Method names (each spanning number of metrics)
    # Level 2: "Datasets", then metric names repeated for each method
    
    col_tuples = [('', '', 'Datasets')]
    
    for method in calibration_methods:
        if method == CalibrationMethods.UNCALIBRATED:
            method_name = 'Uncalibrated'
        else:
            method_name = METHOD_NAME_MAP.get(method, method.name.replace('_', ' ').title())
  
        for metric in metrics:

            metric_display = get_metric_display_name(metric)
            col_tuples.append(('Calibration Methods', method_name, metric_display))
    
    # Create DataFrame with MultiIndex columns
    df = pd.DataFrame(data_rows)
    df.columns = pd.MultiIndex.from_tuples(col_tuples)
    
    # Calculate average row across all datasets
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
    
    return df

def save_table_with_formatting(df, output_path):
    """
    Save DataFrame to CSV with proper multi-level header formatting.
    """
    # Flatten the MultiIndex for CSV export while preserving structure
    df_export = df.copy()
    
    # Create header rows
    header_lines = []
    
    # Get the MultiIndex levels
    if isinstance(df.columns, pd.MultiIndex):
        # Row 1: Top level (Calibration Methods span)
        row1 = ['']  # Empty cell for Datasets column
        current_level0 = None
        span_count = 0
        
        for col in df.columns[1:]:  # Skip first column (Datasets)
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
        
        # Row 2: Method names
        row2 = ['']  # Empty cell for Datasets column
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

def main(args):            
    sampling_strategies = [cvt_to_sampling_type(s) for s in args.sampling_strategies]
    calibration_methods = [CalibrationMethods[m.upper()] for m in args.calibration_methods]
    
    # Set results directory
    results_dir = Path(args.results_dir) if args.results_dir else SAVE_DIR
    
    # Load results
    print("Loading experimental results...")
    results = get_saved_results(
        models=args.models,
        datasets=args.datasets,
        num_seeds=args.num_seeds,
        all_shots=args.all_shots,
        sampling_strategies=sampling_strategies,
        calibration_methods=calibration_methods,
        results_dir=results_dir
    )
    
    calibration_methods = [CalibrationMethods.UNCALIBRATED] + calibration_methods
    
    output_dir = ROOT_DIR / args.output_dir
    
    # Determine CSV filename
    csv_filename = args.csv_name if args.csv_name else 'comparison.csv'
    if not csv_filename.endswith('.csv'):
        csv_filename += '.csv'
    
    for model in args.models:
        print(f"\nGenerating comparison table for {model}...")
        
        save_dir = output_dir / model.replace('/', '_').replace('-FP8', '')
        if len(sampling_strategies) == 1:
            save_dir = save_dir / sampling_strategies[0].name.lower()
    
        save_dir.mkdir(parents=True, exist_ok=True)
        
        df = create_comparison_table(
            results=results,
            model=model,
            datasets=args.datasets,
            all_shots=args.all_shots,
            sampling_strategies=sampling_strategies,
            calibration_methods=calibration_methods,
            metrics=args.metrics,
            show_std=args.show_std
        )
        
        # Save to CSV with proper formatting
        save_path = save_dir / csv_filename
        save_table_with_formatting(df, save_path)
        print(f"Saved table to {save_path}")
        
        # Also print to console (simplified view)
        print("\n" + "="*80)
        print(f"Comparison Table: {model}")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80 + "\n")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Generate comparison tables from experimental results')
    parser.add_argument('--models', dest='models', action='store', required=True, 
                       help='name of model(s), e.g., GPT2-XL')
    parser.add_argument('--datasets', dest='datasets', action='store', required=True, 
                       help='name of dataset(s), e.g., agnews')
    parser.add_argument('--num_seeds', type=int, default=10, 
                       help='Number of random seeds')
    parser.add_argument('--all_shots', dest='all_shots', action='store', required=True, 
                       help='num training examples to use')
    parser.add_argument('--sampling_strategies', action='store', required=True, 
                       help='Sampling strategies to include')
    parser.add_argument('--calibration_methods', action='store', required=True, 
                       help='Calibration methods to compare')
    parser.add_argument('--metrics', action='store', required=True,
                       help='Metrics to include in table (e.g., accuracy,ece,mce,brier)')
    parser.add_argument('--show_std', action='store_true', default=False,
                       help='Whther to tabulate standard deviations along with means')
    parser.add_argument('--results_dir', type=str, default=None,
                       help='Directory containing saved results')
    parser.add_argument('--output_dir', type=str, default='plot_results/comparisons_rebuttal/20_samples/tables',
                       help='Output directory for tables')
    parser.add_argument('--csv_name', type=str, default=None,
                       help='Custom CSV filename (default: comparison.csv)')
    
    args = parser.parse_args()
    
    args.models = convert_to_list(args.models)
    args.datasets = convert_to_list(args.datasets)
    args.all_shots = convert_to_list(args.all_shots, int)
    args.sampling_strategies = convert_to_list(args.sampling_strategies)
    args.calibration_methods = convert_to_list(args.calibration_methods)
    args.metrics = convert_to_list(args.metrics)
    
    main(args)