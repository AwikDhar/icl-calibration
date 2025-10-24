import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle

def main(models, datasets, num_seeds, all_shots):
    root_node = dict()
    missing_exprs = []

    for dataset in datasets:
        root_node[dataset] = dict()
        for model in models:
            root_node[dataset][model] = dict()
            for num_shots in all_shots:
                accuracies = []
                eces = []
                for seed in range(num_seeds):
                    expr_name = f"{dataset}_{model.replace('/','_')}_{num_shots}shot_rand_entropy_level_seed{seed}"
                    try:
                        with open(f"../raw_logits_high_bs/{expr_name}.pkl", 'rb') as file:
                            data = pickle.load(file)
                            accuracies.append(data['accuracies'][0])
                            eces.append(data['eces'][0])
                    except:
                        missing_exprs.append(expr_name)
                    # if num_shots==0:
                    #     break
                root_node[dataset][model][num_shots] = {
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
            # print(expr_name)
        print("Summary : ", missing_map)
        exit()

    cmap = plt.get_cmap('viridis', len(models))  

    for dataset in datasets:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f"{dataset} dataset performance", fontweight='bold')

        accuracy_means = [[] for _ in range(len(models))]
        accuracy_stds  = [[] for _ in range(len(models))]
        ece_means      = [[] for _ in range(len(models))]
        ece_stds       = [[] for _ in range(len(models))]

        for i, model in enumerate(models):
            for num_shots in all_shots:
                entry = root_node[dataset][model][num_shots]
                accuracy_means[i].append(entry['accuracy_mean'])
                accuracy_stds[i].append(entry['accuracy_std'])
                ece_means[i].append(entry['ece_mean'])
                ece_stds[i].append(entry['ece_std'])

            acc_mean = np.array(accuracy_means[i])
            acc_std  = np.array(accuracy_stds[i])
            ece_mean = np.array(ece_means[i])
            ece_std  = np.array(ece_stds[i])

            ax1.plot(all_shots, acc_mean, marker='o', label=model, color=cmap(i))
            ax1.fill_between(all_shots, acc_mean - acc_std, acc_mean + acc_std,
                            color=cmap(i), alpha=0.2)

            ax2.plot(all_shots, ece_mean, marker='s', label=model, color=cmap(i))
            ax2.fill_between(all_shots, ece_mean - ece_std, ece_mean + ece_std,
                            color=cmap(i), alpha=0.2)
        # print("Acc",accuracy_means)
        # print("Std",accuracy_stds)
        # print("ece",ece_means)
        # print("std",ece_stds)
        # exit()
        # axes labels, legends, titles
        ax1.set_title(f"{dataset} - Accuracy")
        ax1.set_xlabel("Shots")
        ax1.set_ylabel("Accuracy")
        ax1.legend()

        ax2.set_title(f"{dataset} - ECE")
        ax2.set_xlabel("Shots")
        ax2.set_ylabel("ECE")
        ax2.legend()

        plt.tight_layout()
        fig.savefig(f"./baseline/bs_{dataset}_baseline_accuracy_ece.png", dpi=600) 
        # plt.show() 

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--models', dest='models', action='store', required=True, help='name of model(s), e.g., GPT2-XL')
    parser.add_argument('--datasets', dest='datasets', action='store', required=True, help='name of dataset(s), e.g., agnews')
    parser.add_argument('--num_seeds', dest='num_seeds', action='store', required=True, help='num seeds for the training set', type=int)
    parser.add_argument('--all_shots', dest='all_shots', action='store', required=True, help='num training examples to use')
  
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