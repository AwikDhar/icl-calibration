import argparse
import os
import pickle
from calibration_plot_data import CalibrationPlotData
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import relplot

def plot_calibration(
        calibration_data: CalibrationPlotData,
        model: str, 
        dataset: str,
        feature_type: str = None,
        llm_agnostic: bool = False,
        sampling_strategy: str = None,
        save_path=None,
        plot_gt_calibration=False
    ):
    sns.set_style("whitegrid")

    shots = sorted(list(calibration_data.ece_shots_map['original'].keys()))

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 16))
    title = f"{model} calibration on {dataset}"
    if sampling_strategy is not None:
        title += f" | {sampling_strategy.capitalize()}"
    fig.suptitle(title, fontsize=15, fontweight='bold')

    # ECE plot
    original_ece = [calibration_data.ece_shots_map['original'][shot] for shot in shots]
    calibrated_ece = [calibration_data.ece_shots_map['calibrated'][shot] for shot in shots]
    dynamic_temp_calibrated_ece = [calibration_data.ece_shots_map['dynamic_temp_calibrated'][shot] for shot in shots]
    static_temp_calibrated_ece = [calibration_data.ece_shots_map['static_temp_calibrated'][shot] for shot in shots]
    global_temp_calibrated_ece = [calibration_data.ece_shots_map['global_temp_calibrated'][shot] for shot in shots]
    
    ax1.plot(shots, original_ece, label='Original', marker='o', color='r')
    ax1.plot(shots, calibrated_ece, label='Calibrated', marker='o', color='g')
    ax1.plot(shots, dynamic_temp_calibrated_ece, label='DT Calibrated', marker='o', color='c')
    ax1.plot(shots, static_temp_calibrated_ece, label='ST Calibrated', marker='o', color='b')
    ax1.plot(shots, global_temp_calibrated_ece, label='GT Calibrated', marker='o', color='purple') if plot_gt_calibration else None
    
    ax1.set_xlabel('Shots', fontsize=12)
    ax1.set_ylabel('ECE', fontsize=12)
    ax1.set_title(f'Dynamic context Temperature scaling', fontsize=12)
    ax1.legend()
    
    # Brier plot
    original_brier = [calibration_data.brier_shots_map['original'][shot] for shot in shots]
    calibrated_brier = [calibration_data.brier_shots_map['calibrated'][shot] for shot in shots]
    dynamic_temp_calibrated_brier = [calibration_data.brier_shots_map['dynamic_temp_calibrated'][shot] for shot in shots]
    static_temp_calibrated_brier = [calibration_data.brier_shots_map['static_temp_calibrated'][shot] for shot in shots]
    global_temp_calibrated_brier = [calibration_data.brier_shots_map['global_temp_calibrated'][shot] for shot in shots] if plot_gt_calibration else None
    
    ax2.plot(shots, original_brier, label='Original', marker='o', color='r')
    ax2.plot(shots, calibrated_brier, label='Calibrated', marker='o', color='g')
    ax2.plot(shots, dynamic_temp_calibrated_brier, label='DT Calibrated', marker='o', color='c')
    ax2.plot(shots, static_temp_calibrated_brier, label='ST Calibrated', marker='o', color='b')
    if plot_gt_calibration:
        ax2.plot(shots, global_temp_calibrated_brier, label='GT Calibrated', marker='o', color='purple')
    
    ax2.set_xlabel('Shots', fontsize=12)
    ax2.set_ylabel('Brier Score', fontsize=12)
    ax2.set_title(f'Brier Score', fontsize=12)
    ax2.legend()
    
    # Temperature plot
    temps = [calibration_data.temp_shots_map[shot] for shot in shots]
    dynamic_temps = [calibration_data.dynamic_temp_shots_map[shot] for shot in shots]
    static_temps = [calibration_data.static_temp_shots_map[shot] for shot in shots]
    if plot_gt_calibration:
        global_temps = [calibration_data.global_temp_shots_map[shot] for shot in shots]
    
    ax3.boxplot(temps, positions=shots, showfliers=False)
    ax3.plot(shots, dynamic_temps, label='Dynamic Temp', marker='x', color='c', linestyle='--')
    ax3.plot(shots, static_temps, label='Static Temp', marker='x', color='b', linestyle='--')
    ax3.plot(shots, global_temps, label='Global Temp', marker='x', color='purple', linestyle='--') if plot_gt_calibration else None
    
    ax3.set_xlabel('Shots', fontsize=12)
    ax3.set_ylabel('Temperature ranges', fontsize=12)
    ax3.set_title(f'Calibration temperatures', fontsize=12)
    ax3.legend()
    
    # Accuracy/Confidence plot
    original_conf = [calibration_data.conf_shots_map['original'][shot] for shot in shots]
    calibrated_conf = [calibration_data.conf_shots_map['calibrated'][shot] for shot in shots]
    dynamic_temp_calibrated_conf = [calibration_data.conf_shots_map['dynamic_temp_calibrated'][shot] for shot in shots]
    static_temp_calibrated_conf = [calibration_data.conf_shots_map['static_temp_calibrated'][shot] for shot in shots]
    global_temp_calibrated_conf = [calibration_data.conf_shots_map['global_temp_calibrated'][shot] for shot in shots] if plot_gt_calibration else None
    
    ax4.plot(shots, original_conf, label='Original confidence', marker='o', color='r')
    ax4.plot(shots, calibrated_conf, label='Calibrated confidence', marker='o', color='g')
    ax4.plot(shots, dynamic_temp_calibrated_conf, label='DT calib confidence', marker='o', color='c')
    ax4.plot(shots, static_temp_calibrated_conf, label='ST calib confidence', marker='o', color='b')
    ax4.plot(shots, global_temp_calibrated_conf, label='GT calib confidence', marker='o', color='purple') if plot_gt_calibration else None
    ax4.plot(shots, calibration_data.accuracies, label='Accuracies', marker='s', color='black')
    
    ax4.set_xlabel('Shots', fontsize=12)
    ax4.set_ylabel('Accuracy/Confidence', fontsize=12)
    ax4.set_title('Accuracies and confidence means', fontsize=12)
    ax4.legend()
    
    plt.tight_layout()
    
    dir_name = f"plot_results/calibration/TC_paper/{model.replace('/','_')}/{dataset}"    
    if feature_type is not None:
        dir_name += f"/{feature_type}"
    if llm_agnostic:
        dir_name += f"/llm_agnostic"
    if sampling_strategy is not None:
        dir_name += f"/{sampling_strategy}"    
    os.makedirs(dir_name, exist_ok=True)
    
    if save_path is None:
        save_path = f"{dir_name}/tc_ece.png"
        
    plt.savefig(save_path, dpi=400)
    plt.close()
    
    # fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6))
    # fig.suptitle("Reliability Diagram Overall", fontsize=15, fontweight='bold')
    
    # relplot.plot_rel_diagram(calibration_data.overall_reldiag['original'], fig, ax1)
    # relplot.plot_rel_diagram(calibration_data.overall_reldiag['calibrated'], fig, ax2)
    
    # ax1.set_title("Original", fontsize=12)
    # ax2.set_title("Calibrated", fontsize=12)
    # plt.tight_layout()
    
    # save_path = f"{dir_name}/rel_diagram.png"
    # fig.savefig(save_path)
    # plt.close()
    
    # fig, ax = plt.subplots(2, len(shots), figsize=(30,15))
    # fig.suptitle("Reliability Diagram Shot-wise", fontsize=30, fontweight='bold')
    # plt.tight_layout()
    
    # for idx, shot in enumerate(shots):
    #     diagram = calibration_data.reldiag_shots_map['original'][shot]
    #     calibrated_diagram = calibration_data.reldiag_shots_map['calibrated'][shot]
        
    #     relplot.plot_rel_diagram(diagram, fig=fig, ax=ax[0,idx])
    #     relplot.plot_rel_diagram(calibrated_diagram, fig=fig, ax=ax[1,idx])
    
    #     ax[0,idx].set_title(f"Original | {shot} shot", fontsize=20)
    #     ax[1,idx].set_title(f"Calibrated | {shot} shot", fontsize=20)
        
    # save_path = f"{dir_name}/rel_diagram_shotwise.png"
    # fig.savefig(save_path)    
    # plt.close()
    
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

    parser.add_argument('--models', dest='models', action='store', required=True, help='name of LLMs')
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