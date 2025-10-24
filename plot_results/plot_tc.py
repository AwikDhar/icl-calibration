import argparse
import os
import pickle
import random
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np

def plot_calibration(
        ece_shots_map: Dict, 
        temp_shots_map: Dict,
        conf_shots_map: Dict,
        accuracies: List,
        model: str, 
        dataset: str,
        feature_type: str = None,
        sampling_strategy: str = None,
        save_path=None
    ):

    shots = sorted(list(ece_shots_map['original'].keys()))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 8))
    title = f"{model} calibration on {dataset}"
    if sampling_strategy is not None:
        title += f" | {sampling_strategy.capitalize()}"
    fig.suptitle(title, fontsize=15, fontweight='bold')

    original_ece = [ece_shots_map['original'][shot] for shot in shots]
    calibrated_ece = [ece_shots_map['calibrated'][shot] for shot in shots]
    
    ax1.plot(shots, original_ece, label='Original', marker='o', color='r')
    ax1.plot(shots, calibrated_ece, label='Calibrated', marker='o', color='g')
    
    ax1.set_xlabel('Shots', fontsize=12)
    ax1.set_ylabel('ECE', fontsize=12)
    ax1.set_title(f'Dynamic context Temperature scaling', fontsize=12)
    ax1.legend()
    
    temps = [temp_shots_map[shot] for shot in shots]
    
    ax2.boxplot(temps, positions=shots)
    
    ax2.set_xlabel('Shots', fontsize=12)
    ax2.set_ylabel('Temperature ranges', fontsize=12)
    ax2.set_title(f'Calibration temperatures', fontsize=12)
    
    original_conf = [conf_shots_map['original'][shot] for shot in shots]
    calibrated_conf = [conf_shots_map['calibrated'][shot] for shot in shots]
    
    ax3.plot(shots, original_conf, label='Original', marker='o', color='r')
    ax3.plot(shots, calibrated_conf, label='Calibrated', marker='o', color='g')
    ax3.plot(shots, accuracies, label='Accuracies', marker='s', color='black')
    
    ax3.set_xlabel('Shots', fontsize=12)
    ax3.set_ylabel('Accuracy/Confidence', fontsize=12)
    ax3.set_title(f'Accuracies and confidence means', fontsize=12)
    ax3.legend()
    
    plt.tight_layout()
    
    if save_path is None:
        dir_name = f"plot_results/calibration/TC/{model.replace('/','_')}/{dataset}"
        if feature_type is not None:
            dir_name += f"/{feature_type}"
        if sampling_strategy is not None:
            dir_name += f"/{sampling_strategy}"
            
        os.makedirs(dir_name, exist_ok=True)
        save_path = f"{dir_name}/tc_ece.png"
        
    plt.savefig(save_path, dpi=600)
    
def main(models, datasets, num_seeds, all_shots, sampling_strategy):
    root_node = dict()
    missing_exprs = []

    settings = ["baseline", "TC"]
    for dataset in datasets:
        root_node[dataset] = dict()
        for model in models:
            root_node[dataset][model] = dict()
            for i, setting in enumerate(settings):
                root_node[dataset][model][setting] = dict()
                for num_shots in all_shots:
                    accuracies = []
                    eces = []
                    for seed in range(num_seeds):
                        file_name = (f"../raw_logits_high_bs/{model.replace('/','_')}/{dataset}/" # In case it's an HF model
                                     f"{sampling_strategy}/{num_shots}_shot/{seed}_seed.pkl")
                        try:
                            with open(file_name, 'rb') as file:
                                data = pickle.load(file)
                                accuracies.append(data['accuracies'][i])
                                eces.append(data['eces'][i])
                        except:
                            missing_exprs.append(expr_name)
                        # if num_shots==0:
                        #     break
                    root_node[dataset][model][setting][num_shots] = {
                        "accuracy_mean" : np.mean(accuracies),
                        "accuracy_std" : np.std(accuracies),
                        "ece_mean" : np.mean(eces), 
                        "ece_std" : np.std(eces)
                    }

    if len(missing_exprs):
        missing_map = {dataset:set() for dataset in datasets}
        print("ERROR: The following experiments are missing: ")
        for expr_name in missing_exprs:
            for dataset in datasets:
                for model in models:
                    if model.replace('/','_') in expr_name:
                        missing_map[dataset].add(model)
            print(expr_name)
        print("Summary : ", missing_map)
        exit()

    cmap = plt.get_cmap('viridis', len(settings))  

    for dataset in datasets:
        for model in models:
            model_name = model.split('/')[1]
            fig, ax2 = plt.subplots(1, 1, figsize=(8, 6))
            fig.suptitle(f"{model_name} performance on {dataset}", fontsize=15, fontweight='bold')

            accuracy_means = [[] for _ in range(len(settings))]
            accuracy_stds  = [[] for _ in range(len(settings))]
            ece_means      = [[] for _ in range(len(settings))]
            ece_stds       = [[] for _ in range(len(settings))]
            for i, setting in enumerate(settings):
                for num_shots in all_shots:
                    entry = root_node[dataset][model][setting][num_shots]
                    accuracy_means[i].append(entry['accuracy_mean'])
                    accuracy_stds[i].append(entry['accuracy_std'])
                    ece_means[i].append(entry['ece_mean'])
                    ece_stds[i].append(entry['ece_std'])

                acc_mean = np.array(accuracy_means[i])
                acc_std  = np.array(accuracy_stds[i])
                ece_mean = np.array(ece_means[i])
                ece_std  = np.array(ece_stds[i])
                
                start_idx=0
                # ax1.plot(all_shots[start_idx:], acc_mean[start_idx:], marker='o', label=setting, color=cmap(i))
                # ax1.fill_between(all_shots[start_idx:], acc_mean[start_idx:] - acc_std[start_idx:], acc_mean[start_idx:] + acc_std[start_idx:],
                #                 color=cmap(i), alpha=0.2)

                ax2.plot(all_shots[start_idx:], ece_mean[start_idx:], marker='s', label=setting, color=cmap(i))
                ax2.fill_between(all_shots[start_idx:], ece_mean[start_idx:] - ece_std[start_idx:], ece_mean[start_idx:] + ece_std[start_idx:],
                                color=cmap(i), alpha=0.2)

            # axes labels, legends, titles
            # ax1.set_title("Accuracy vs k-shots", fontsize=15)
            # ax1.set_xlabel("Shots")
            # ax1.set_ylabel("Accuracy")
            # ax1.legend()

            ax2.set_title("ECE vs k-shots", fontsize=15)
            ax2.set_xlabel("Shots")
            ax2.set_ylabel("ECE")
            # ax2.set_ylim([0,1])
            ax2.legend()

            plt.tight_layout()
            save_path_dir = f"./calibration/TC/{sampling_strategy}/"
            os.makedirs(save_path_dir, exist_ok=True)
            save_path = f"{save_path_dir}/{dataset}_{model.replace('/','_')}_tc_ece.png"
            fig.savefig(save_path, dpi=600) 

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--models', dest='models', action='store', required=True, help='name of model(s), e.g., GPT2-XL')
    parser.add_argument('--datasets', dest='datasets', action='store', required=True, help='name of dataset(s), e.g., agnews')
    parser.add_argument('--num_seeds', dest='num_seeds', action='store', required=True, help='num seeds for the training set', type=int)
    parser.add_argument('--all_shots', dest='all_shots', action='store', required=True, help='num training examples to use')
    parser.add_argument('--sampling_strategy', dest='sampling_strategy', action='store', required=True, help='sampling strategy for ICL prompt')
    args = parser.parse_args()
    args = vars(args)

    def convert_to_list(items, cvt_func=None):
        if cvt_func:
            return [cvt_func(s.strip()) for s in items.split(",")]
        else:
            return [s.strip() for s in items.split(",")]

    args['models'] = convert_to_list(args['models'])
    args['datasets'] = convert_to_list(args['datasets'])
    args['all_shots'] = convert_to_list(args['all_shots'], int)
    
    print(args)
    main(**args)