import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle

def main(models, datasets, num_seeds, all_shots, entropy_levels):
    root_node = dict()
    missing_exprs = []

    for dataset in datasets:
        root_node[dataset] = dict()
        for model in models:
            root_node[dataset][model] = dict()
            for entropy_level in entropy_levels:
                root_node[dataset][model][entropy_level] = dict()
                for num_shots in all_shots:
                    accuracies = []
                    eces = []
                    for seed in range(num_seeds):
                        if num_shots==0 and entropy_level!="rand":
                            break
                        # if num_shots==2 and entropy_level in ("labelspike", "labelsuppress")
                        if num_shots==1 and entropy_level=="max":
                            break
                        expr_name = f"{dataset}_{model.replace('/','_')}_{num_shots}shot_{entropy_level}_entropy_level_seed{seed}"
                        try:
                            with open(f"../raw_logits_high_bs/{expr_name}.pkl", 'rb') as file:
                                data = pickle.load(file)
                                accuracies.append(data['accuracies'][0])
                                eces.append(data['eces'][0])
                        except:
                            missing_exprs.append(expr_name)
                        if num_shots==0:
                            break
                    root_node[dataset][model][entropy_level][num_shots] = {
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

    cmap = plt.get_cmap('viridis', len(entropy_levels))  

    for dataset in datasets:
        for model in models:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle(f"{model} performance on {dataset}", fontweight='bold')

            accuracy_means = [[] for _ in range(len(entropy_levels))]
            accuracy_stds  = [[] for _ in range(len(entropy_levels))]
            ece_means      = [[] for _ in range(len(entropy_levels))]
            ece_stds       = [[] for _ in range(len(entropy_levels))]
            for i, entropy_level in enumerate(entropy_levels):
                for num_shots in all_shots:
                    entry = root_node[dataset][model][entropy_level][num_shots]
                    accuracy_means[i].append(entry['accuracy_mean'])
                    accuracy_stds[i].append(entry['accuracy_std'])
                    ece_means[i].append(entry['ece_mean'])
                    ece_stds[i].append(entry['ece_std'])

                acc_mean = np.array(accuracy_means[i])
                acc_std  = np.array(accuracy_stds[i])
                ece_mean = np.array(ece_means[i])
                ece_std  = np.array(ece_stds[i])

                start_idx = 0
                if entropy_level in ("labelspike", "llabel suppress"):
                    start_idx = 1
                if entropy_level=="max":
                    start_idx = 1

                ax1.plot(all_shots[start_idx:], acc_mean[start_idx:], marker='o', label=entropy_level, color=cmap(i))
                ax1.fill_between(all_shots[start_idx:], acc_mean[start_idx:] - acc_std[start_idx:], acc_mean[start_idx:] + acc_std[start_idx:],
                                color=cmap(i), alpha=0.2)

                ax2.plot(all_shots[start_idx:], ece_mean[start_idx:], marker='s', label=entropy_level, color=cmap(i))
                ax2.fill_between(all_shots[start_idx:], ece_mean[start_idx:] - ece_std[start_idx:], ece_mean[start_idx:] + ece_std[start_idx:],
                                color=cmap(i), alpha=0.2)

            # axes labels, legends, titles
            ax1.set_title("Accuracy vs k k-shots")
            ax1.set_xlabel("Shots")
            ax1.set_ylabel("Accuracy")
            ax1.legend()

            ax2.set_title("ECE vs k-shots")
            ax2.set_xlabel("Shots")
            ax2.set_ylabel("ECE")
            ax2.legend()

            plt.tight_layout()
            fig.savefig(f"./entropy/{dataset}_{model.replace('/','_')}_entropy_accuracy_ece.png", dpi=500) 
            plt.show() 

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--models', dest='models', action='store', required=True, help='name of model(s), e.g., GPT2-XL')
    parser.add_argument('--datasets', dest='datasets', action='store', required=True, help='name of dataset(s), e.g., agnews')
    parser.add_argument('--num_seeds', dest='num_seeds', action='store', required=True, help='num seeds for the training set', type=int)
    parser.add_argument('--all_shots', dest='all_shots', action='store', required=True, help='num training examples to use')
    parser.add_argument('--entropy_levels', dest='entropy_levels', action='store', required=False, help='the levels of entropy for sampling ICL examples')

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
    args['entropy_levels'] = convert_to_list(args['entropy_levels'], lambda x : x if x.isalpha() else float(x))

    print(args)
    main(**args)