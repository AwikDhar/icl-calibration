import argparse
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from plot_results.results_utils import cvt_to_sampling_type, get_saved_results, get_metric_display_name
from calibration_methods import CalibrationMethods
from sampling_strategies import SamplingStrategy
from utils.gen_utils import convert_to_list

plt.rcParams.update({
    'font.size': 25,        
    'lines.linewidth': 2.5,
    'grid.linewidth': 1.5,
    'xtick.labelsize': 20,    
    'ytick.labelsize': 20,    
    'legend.fontsize': 20,    
    'axes.labelsize': 25      
})

def main(models, datasets, num_seeds, all_shots, sampling_strategies, metric='ece'):
    results = get_saved_results(models, datasets, num_seeds, all_shots, sampling_strategies, 
                               calibration_methods=[CalibrationMethods.TF])
    
    cmap = plt.get_cmap('viridis', len(datasets))  

    for model in models:
        model_save_name = model.replace('/', '_').replace('-FP8', '')
        model_display_name = model.split('/')[1].replace('-FP8', '')
        
        for sampling_strategy in sampling_strategies:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            figtitle = f"{model_display_name} performance across datasets ({sampling_strategy.name.title()} sampled examples)"
            # fig.suptitle(figtitle, fontweight='bold')

            accuracy_means = [[] for _ in range(len(datasets))]
            accuracy_stds  = [[] for _ in range(len(datasets))]
            metric_means   = [[] for _ in range(len(datasets))]
            metric_stds    = [[] for _ in range(len(datasets))]

            for i, dataset in enumerate(datasets):
                for num_shots in all_shots:
                    # Access the metrics through dataset -> model -> sampling_strategy -> num_shots
                    uncalibrated_metrics = results[dataset][model][sampling_strategy][num_shots][CalibrationMethods.UNCALIBRATED]
                    accuracy_means[i].append(uncalibrated_metrics['accuracy']['mean'])
                    accuracy_stds[i].append(uncalibrated_metrics['accuracy']['std'])
                    metric_means[i].append(uncalibrated_metrics[metric]['mean'])
                    metric_stds[i].append(uncalibrated_metrics[metric]['std'])

                acc_mean = np.array(accuracy_means[i])
                acc_std  = np.array(accuracy_stds[i])
                met_mean = np.array(metric_means[i])
                met_std  = np.array(metric_stds[i])

                ax1.plot(all_shots, acc_mean, marker='o', label=dataset, color=cmap(i))
                ax1.fill_between(all_shots, acc_mean - acc_std, acc_mean + acc_std,
                                color=cmap(i), alpha=0.2)

                ax2.plot(all_shots, met_mean, marker='s', label=dataset, color=cmap(i))
                ax2.fill_between(all_shots, met_mean - met_std, met_mean + met_std,
                                color=cmap(i), alpha=0.2)

            # Axis 1: Accuracy
            ax1.set_xlabel("Shots")
            ax1.set_ylabel("Accuracy")
            ax1.grid(True, alpha=0.3)

            # Axis 2: Specified metric
            metric_display = get_metric_display_name(metric)
            ax2.set_xlabel("Shots")
            ax2.set_ylabel(metric_display)
            ax2.grid(True, alpha=0.3)

            handles, labels = ax1.get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=len(datasets),
                        bbox_to_anchor=(0.5, -0.01), frameon=True, shadow=True)
            
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.28)  
            
            save_dir = Path(f"./plot_results/baseline/{model_save_name}/{sampling_strategy.name}")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save with dynamic filename based on metric
            filename = f"accuracy_{metric}.png"
            fig.savefig(save_dir / filename, dpi=400) 
            print(f"Saved plot: {save_dir / filename}")
            plt.close(fig)

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--models', dest='models', action='store', required=True, 
                       help='name of model(s), e.g., GPT2-XL')
    parser.add_argument('--datasets', dest='datasets', action='store', required=True, 
                       help='name of dataset(s), e.g., agnews')
    parser.add_argument('--num_seeds', dest='num_seeds', action='store', required=True, 
                       help='num seeds for the training set', type=int)
    parser.add_argument('--all_shots', dest='all_shots', action='store', required=True, 
                       help='num training examples to use')
    parser.add_argument('--sampling_strategies', action='store', required=True,
                       help='Sampling strategies to use')
    parser.add_argument('--metric', type=str, default='ece',
                       help='Metric to plot alongside accuracy (default: ece)')
  
    args = parser.parse_args()

    args.models = convert_to_list(args.models)
    args.datasets = convert_to_list(args.datasets)
    args.all_shots = convert_to_list(args.all_shots, int)
    args.sampling_strategies = [cvt_to_sampling_type(s) for s in args.sampling_strategies.split(",")]
    
    print(args)
    main(
        models=args.models,
        datasets=args.datasets,
        num_seeds=args.num_seeds,
        all_shots=args.all_shots,
        sampling_strategies=args.sampling_strategies,
        metric=args.metric
    )