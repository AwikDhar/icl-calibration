from pathlib import Path
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sampling_strategies import EntropyLevels, SamplingStrategy
from calibration_methods import CalibrationMethods
from utils.gen_utils import SAVE_DIR

ROOT_DIR = Path(__file__).resolve().parent.parent
PLOT_DIR = ROOT_DIR / "plot_results" / "comparisons_emnlp"

METHOD_NAME_MAP = {
    CalibrationMethods.UNCALIBRATED: "Uncalibrated",
    CalibrationMethods.ICC: "ICC",
    CalibrationMethods.PERMUT_AVG: "ICPermutation",
    CalibrationMethods.ICT: "TS-AR",
    CalibrationMethods.FS_ICT: "TS-FS",
    CalibrationMethods.TF: "Calibformer",
}

colours = plt.cm.tab10(np.linspace(0, 1, len(METHOD_NAME_MAP)))
METHOD_COLOUR_MAP = {
    CalibrationMethods.UNCALIBRATED: colours[0],
    CalibrationMethods.ICC: colours[1],
    CalibrationMethods.PERMUT_AVG: colours[2],
    CalibrationMethods.ICT: colours[3],
    CalibrationMethods.FS_ICT: colours[4],
    CalibrationMethods.TF: colours[5],
}

def cvt_to_sampling_type(sampling_strategy: str):
    try:
        return SamplingStrategy[sampling_strategy.upper()]
    except:
        return EntropyLevels[sampling_strategy.upper()]
        
def get_metric_display_name(metric):
    """Convert metric key to display name."""
    return metric.upper() if metric.lower()!='brier' else metric.title()

def get_saved_results(models, datasets, num_seeds, all_shots, sampling_strategies, calibration_methods: List[CalibrationMethods]=[], results_dir=SAVE_DIR):
    root_node = dict()
    missing_exprs = []

    for dataset in datasets:
        root_node[dataset] = dict()
        
        for model in models:
            root_node[dataset][model] = dict()
        
            for sampling_strategy in sampling_strategies:
                root_node[dataset][model][sampling_strategy] = dict()
        
                for num_shots in all_shots:
                    metrics = {method: [] for method in calibration_methods}
            
                    accuracies = []
                    eces = []
                    mces = []
                    briers = []
                    confs = []
                    
                    available_seeds = 0
                    for seed in range(num_seeds):
                        # sampling = 'RANDOM' if sampling_strategy==SamplingStrategy.ENTROPY else SamplingStrategy.SIMILARITY.name
                        sampling = sampling_strategy.name
                        
                        file_name = (f"{results_dir}/{model.replace('/','_').replace('-FP8','')}/{dataset}/" # In case it's an HF model
                                     f"{sampling}/{num_shots}_shot/{seed}_seed.pkl") 
                        
                        try:
                            with open(file_name, 'rb') as file:
                                data = pickle.load(file)
                                seed_metrics = data['metrics'] 
                                
                            for calibration_method in calibration_methods:
                                metrics[calibration_method].append(seed_metrics[calibration_method])  

                            accuracies.append(data['accuracies'][0])
                            eces.append(data['eces'][0])
                            mces.append(data['mces'][0])
                            confs.append(data['confs'][0])

                            available_seeds += 1 
                        except FileNotFoundError:
                            missing_exprs.append(file_name)                      
                        except Exception as e:
                            print(file_name)
                            # print(seed_metrics)
                            print(e)
                            raise e
                        # if num_shots==0:
                        #     break
                    # if num_shots==12 and dataset=='when2call':
                    #     exit()
                    if calibration_methods:
                        calibration_method = calibration_methods[0]
                        # print(calibration_method, available_seeds, metrics[calibration_method])
                        accuracies = [metrics[calibration_method][idx].accuracy for idx in range(available_seeds)]
                        eces = [metrics[calibration_method][idx].calibration_metrics.ece for idx in range(available_seeds)]
                        mces = [metrics[calibration_method][idx].calibration_metrics.mce for idx in range(available_seeds)]
                        briers = [metrics[calibration_method][idx].calibration_metrics.brier_score for idx in range(available_seeds)]
                        confs = [metrics[calibration_method][idx].mean_conf for idx in range(available_seeds)]
                        
                    else:
                        accuracies = [metrics[calibration_method][idx].accuracy for idx in range(available_seeds)]
                        eces = [metrics[calibration_method][idx].calibration_metrics.ece for idx in range(available_seeds)]
                        confs = [metrics[calibration_method][idx].mean_conf for idx in range(available_seeds)]
                        
                    metrics[CalibrationMethods.UNCALIBRATED] = get_metrics_stats_from_seeds(accuracies, eces, mces, briers, confs)
    
                    for calibration_method in calibration_methods:
                        calibrated_accuracies = [metrics[calibration_method][idx].calibrated_accuracy for idx in range(available_seeds)]
                        calibrated_eces = [metrics[calibration_method][idx].calibration_metrics.calibrated_ece for idx in range(available_seeds)]
                        calibrated_mces = [metrics[calibration_method][idx].calibration_metrics.calibrated_mce for idx in range(available_seeds)]
                        calibrated_briers = [metrics[calibration_method][idx].calibration_metrics.calibrated_brier_score for idx in range(available_seeds)]
                        calibrated_confs = [metrics[calibration_method][idx].mean_calibrated_conf for idx in range(available_seeds)]
                        
                        metrics[calibration_method] = get_metrics_stats_from_seeds(calibrated_accuracies, calibrated_eces, calibrated_mces, calibrated_briers, calibrated_confs)
                                             
                    root_node[dataset][model][sampling_strategy][num_shots] = metrics

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
        # time.sleep(5)
        # exit()

    # print(root_node)
    return root_node

def get_metrics_stats_from_seeds(accuracies, eces, mces, briers, confs, percentage=True):
    metrics_stats = {
            'accuracy' : {
                'mean':np.mean(accuracies),
                'std':np.std(accuracies)
            },
            'ece' : {
                'mean':np.mean(eces),
                'std':np.std(eces)
            },
            'mce' : {
                'mean':np.mean(mces),
                'std':np.std(mces)
            },
            'brier' : {
                'mean':np.mean(briers),
                'std':np.std(briers)
            },
            'conf' : {
                'mean':np.mean(confs),
                'std':np.std(confs)
            }
        }  
    
    if percentage:
        for metric, stats in metrics_stats.items():
            if metric!='brier' and stats['mean']<1: # brier can be > 1, no concept of %
                stats['mean'] = stats['mean']*100
                stats['std'] = stats['std']*100
                
    return metrics_stats