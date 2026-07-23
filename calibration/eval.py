import argparse
import json
import os
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# from data_utils import * # bad - bandaid solution for a circular import 
from utils.gen_utils import convert_to_list
from calibration.model import CalibrationTransformer, CalibratorOutputType, PositionEmbeddingType
from metrics import CalibrationMetrics

from calibration_plot_data import CalibrationPlotData
from calibration.train import get_batch, get_calibrated_logits_loss, initialise_calibrator, load_datasets 
from calibration.temperature import get_shotwise_dynamic_temperatures, get_shotwise_static_temperatures, get_equivalent_temp
from calibration.comparison_result import ComparisonResult, CalibrationMethodResult 

from plot_results.plot_tc import plot_calibration

def main(llms, 
         datasets,
         feature_type,
         shots_start,
         shots_end,
         calibrator_type: str,
         calibrator_output_type,
         sampling_strategies = None,
         ablation_method: str = None,
         num_seeds: int = 1,
         gpu_id=0, 
         model_name="calibrator",
         model_path=None,
         llm_agnostic=False,
         save_path=None,
         plot_results=False,
         plot_confidence_band=False,
         plot_gt_calibration=False,
         hide_non_summary=False,
         save_seed_results=False):
    
    device = f'cuda:{gpu_id}'
    
    data = load_datasets(llms, datasets, device, feature_type, shots_end, splits=('test',), sampling_strategies=sampling_strategies, temp_augment=False, label_augment=False)
    
    sample_model, sample_dataset, sample_strategy = llms[0], datasets[0], sampling_strategies[0]
    T, C = data[sample_model][sample_dataset][sample_strategy]['test']['inputs'][0].shape

    missing_seeds = []
    validation_eces = []
    
    comparison_results: List[ComparisonResult] = []
    
    for seed in tqdm(range(num_seeds), desc='Evaluating checkpoints'):
                
        model_dir = f"./calibration/models/"
        if llm_agnostic or len(llms)>1: 
            model_dir += 'llm_agnostic' 
        else:
            model_dir += llms[0].replace('/','_')
            
        if ablation_method:
            model_dir += f"/ablations/{ablation_method}"
        # if len(sampling_strategies)==1:
        #     model_dir += f"/{sampling_strategies[0]}"
        if num_seeds>1:
            model_dir += f"/{seed}_seed"
            
        model_path = f'{model_dir}/{model_name}'  
        if not os.path.exists(model_path):
            missing_seeds.append(seed)
            continue
        
        metrics_path = f"{model_dir}/metrics.json"
        if not os.path.exists(metrics_path):
            missing_seeds.append(seed)
            continue
        
        with open(metrics_path, "r") as file:
            metrics = json.load(file)
            seed_val_ece = metrics['best_eval_calibrated_ece']
            
            validation_eces.append({"seed":seed, "ece":seed_val_ece})
                    
        calibrator = initialise_calibrator(calibrator_type, C, calibrator_output_type, device)
        
        # if seed==0:
        #     print(calibrator)
        #     print(f"{sum(p.numel() for p in calibrator.parameters())/10**6: .2f} M parameters")
        
        state_dict = torch.load(model_path, weights_only=True)
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('_orig_mod.'):
                cleaned_state_dict[key.replace('_orig_mod.', '')] = value
            else:
                cleaned_state_dict[key] = value
                
        calibrator.load_state_dict(cleaned_state_dict)    
        
        comparison_result = eval(calibrator, calibrator_output_type,
                                data, llms, datasets, feature_type, shots_start, llm_agnostic,
                                model_name, save_path=save_path,
                                plot_results=plot_results, plot_confidence_band=plot_confidence_band, plot_gt_calibration=plot_gt_calibration, show_summary=not hide_non_summary)
        
        comparison_results.append(comparison_result)

    # print(f"ECEs: {[comparison_result.calibrated.ece for comparison_result in comparison_results]}")
    # print(f"Briers: {[comparison_result.calibrated.brier for comparison_result in comparison_results]}\n")
    
    anomaly_seeds = []
    anomaly_idxs = []
    
    mean_val_ece = np.mean([validation_ece ['ece'] for validation_ece in validation_eces])
    for idx, seed_val in enumerate(validation_eces):
        seed, seed_val_ece = seed_val['seed'], seed_val['ece']
        if (seed_val_ece - mean_val_ece)/mean_val_ece >0.5:
            anomaly_seeds.append(seed)
            anomaly_idxs.append(idx)
    if anomaly_seeds:
        print(f"Anomaly seeds ({len(anomaly_seeds)} total): ", anomaly_seeds)
    if missing_seeds:
        print(f"Missing seeds ({len(missing_seeds)} total): ", missing_seeds)
        
    comparison_results = [comparison_result for idx, comparison_result in enumerate(comparison_results) if idx not in anomaly_idxs]
    comparison_result_overall = np.mean(comparison_results)
    print("\nComparison Summary:\n")
    print(f"{'Method':<20} {'ECE':<10} {'Brier':<10}")
    print("-" * 40)
    print(f"{'Uncalibrated':<20} {comparison_result_overall.uncalibrated.ece:<10.4f} {comparison_result_overall.uncalibrated.brier:<10.4f}")
    print(f"{'Calibrated':<20} {comparison_result_overall.calibrated.ece:<10.4f} {comparison_result_overall.calibrated.brier:<10.4f}")
    print(f"{'Dynamic Calibrated':<20} {comparison_result_overall.dynamic_calibrated.ece:<10.4f} {comparison_result_overall.dynamic_calibrated.brier:<10.4f}")
    
    eval_dir = Path(model_dir)
    if num_seeds>1:
        eval_dir = eval_dir.parent
    eval_path = eval_dir / "eval.json"
    
    eval_metrics = {}
    if eval_path.exists():
        with open(eval_path, "r") as file:
            eval_metrics = json.load(file)

    llms_key = " | ".join(llms)
    eval_metrics[llms_key] = {
        "uncalibrated": {
            "ece": round(comparison_result_overall.uncalibrated.ece, 4),
            "brier": round(comparison_result_overall.uncalibrated.brier, 4)
        },
        "calibrated": {
            "ece": round(comparison_result_overall.calibrated.ece, 4),
            "brier": round(comparison_result_overall.calibrated.brier, 4)
        },
        "dynamic_calibrated": {
            "ece": round(comparison_result_overall.dynamic_calibrated.ece, 4),
            "brier": round(comparison_result_overall.dynamic_calibrated.brier, 4)
        }
    }

    if save_seed_results:
        eval_metrics[llms_key]['seedwise'] = {
            "eces": [comparison_result.calibrated.ece for comparison_result in comparison_results],
            "briers": [comparison_result.calibrated.brier for comparison_result in comparison_results]
        }
        
    # with open(eval_path, "w") as file:
    #     json.dump(eval_metrics, file, indent=2)
     
def eval(calibrator, 
         calibrator_output_type,
         data, 
         llms,
         datasets,
         feature_type,
         shots_start,
         llm_agnostic,
         model_name,
         save_path=None,
         plot_results=False,
         plot_confidence_band=False,
         plot_gt_calibration=False,
         show_summary=False):
    
    llm_eval_summaries = []
    
    llm_eces,         llm_briers = [], []
    llm_dynamic_eces, llm_dynamic_briers = [], []
    llm_static_eces,  llm_static_briers = [], []
    
    for llm in llms:
        dataset_eces,         dataset_briers = [], []
        dataset_dynamic_eces, dataset_dynamic_briers = [], []
        dataset_static_eces,  dataset_static_briers = [], []
        
        print(f"\n||-----Model: {llm}-----||\n")
        for dataset in datasets:
            # Collect metrics per sampling strategy for this dataset
            sampling_eces,         sampling_briers = [], []
            sampling_dynamic_eces, sampling_dynamic_briers = [], []
            sampling_static_eces,  sampling_static_briers = [], []
            
            for sampling_strategy in data[llm][dataset]:

                calibration_data = eval_llm_dataset(calibrator, calibrator_output_type, data, llm, dataset, sampling_strategy, shots_start, 
                                        plot_results, plot_confidence_band, plot_gt_calibration, verbose=show_summary)
                
                sampling_eces.append(calibration_data.overall_ece['calibrated'])
                sampling_briers.append(calibration_data.overall_brier['calibrated'])
                sampling_dynamic_eces.append(calibration_data.overall_ece['dynamic_temp_calibrated'])
                sampling_dynamic_briers.append(calibration_data.overall_brier['dynamic_temp_calibrated'])
                sampling_static_eces.append(calibration_data.overall_ece['static_temp_calibrated'])
                sampling_static_briers.append(calibration_data.overall_brier['static_temp_calibrated'])
                
                if plot_results:
                    plot_calibration(calibration_data, llm, dataset, 
                                    feature_type, llm_agnostic, sampling_strategy, 
                                    save_path=save_path, plot_gt_calibration=plot_gt_calibration)
            
            # Average across sampling strategies for this dataset
            dataset_eces.append(np.mean(sampling_eces))
            dataset_briers.append(np.mean(sampling_briers))
            dataset_dynamic_eces.append(np.mean(sampling_dynamic_eces))
            dataset_dynamic_briers.append(np.mean(sampling_dynamic_briers))
            dataset_static_eces.append(np.mean(sampling_static_eces))
            dataset_static_briers.append(np.mean(sampling_static_briers))
        
        # Average across datasets for this LLM
        llm_eces.append(np.mean(dataset_eces))
        llm_briers.append(np.mean(dataset_briers))
        llm_dynamic_eces.append(np.mean(dataset_dynamic_eces))
        llm_dynamic_briers.append(np.mean(dataset_dynamic_briers))
        llm_static_eces.append(np.mean(dataset_static_eces))
        llm_static_briers.append(np.mean(dataset_static_briers))
        
        llm_eval_summaries.append("\n" + "="*60)
        llm_eval_summaries.append(f"\n{llm} EVALUATION METRICS SUMMARY\n")
        llm_eval_summaries.append("="*60)
        llm_eval_summaries.append("\n--- Transformer Calibrator ---")
        llm_eval_summaries.append(get_stats_summary("Expected Calibration Error (ECE)", dataset_eces))
        llm_eval_summaries.append(get_stats_summary("Brier Score", dataset_briers))
        llm_eval_summaries.append("\n--- Dynamic Temperature Scaling ---")
        llm_eval_summaries.append(get_stats_summary("Expected Calibration Error (ECE)", dataset_dynamic_eces))
        llm_eval_summaries.append(get_stats_summary("Brier Score", dataset_dynamic_briers))
        llm_eval_summaries.append("\n--- Static Temperature Scaling ---")
        llm_eval_summaries.append(get_stats_summary("Expected Calibration Error (ECE)", dataset_static_eces))
        llm_eval_summaries.append(get_stats_summary("Brier Score", dataset_static_briers))
        llm_eval_summaries.append("\n\n")
    
    if show_summary:
        print_llm_summary(model_name, llm_eval_summaries, llm_eces, llm_briers, llm_dynamic_eces, llm_dynamic_briers, llm_static_eces, llm_static_briers)
    
    comparison_result = ComparisonResult(
        uncalibrated=CalibrationMethodResult(
            ece=np.mean(llm_static_eces), # Assuming static temp 1.0 = uncalibrated
            brier=np.mean(llm_static_briers)
        ),
        calibrated=CalibrationMethodResult(
            ece=np.mean(llm_eces), 
            brier=np.mean(llm_briers)
        ),
        dynamic_calibrated=CalibrationMethodResult(
            ece=np.mean(llm_dynamic_eces),
            brier=np.mean(llm_dynamic_briers)
        )
    )
    
    return comparison_result
            
def eval_llm_dataset(
    model, calibrator_output_type, 
    data, llm, dataset, sampling_strategy, shots_start, 
    plot_results, plot_confidence_band, plot_gt_calibration,
    verbose=False):
    calibration_data = CalibrationPlotData()

    inputs, logits, labels = get_batch(data[llm][dataset][sampling_strategy]['test']) # len(eval),T,C | len(eval),T,num_classes | len(eval),T
    B,T,num_classes = logits.shape
    
    shotwise_dynamic_temps = get_shotwise_dynamic_temperatures(data[llm][dataset][sampling_strategy]['test'], shots_start)
    # shotwise_static_temps = get_shotwise_static_temperatures(data[llm][dataset][sampling_strategy]['train'], shots_start, batch_size=100)
    shotwise_static_temps = {shot:1 for shot in range(shots_start, T)}
    if plot_gt_calibration:
        with open("calibration/trained_temperature.json", "r") as file:
            shotwise_global_temps = json.load(file)
            shotwise_global_temps = {int(shot):temp for  shot, temp in shotwise_global_temps.items()}

    calibrator_metrics = {}
    dynamic_temp_metrics = {}
    static_temp_metrics = {}
    global_temp_metrics = {} if plot_gt_calibration else None

    model.eval()
    with torch.inference_mode():                        
        outputs = model(inputs) # B,T,1
        
        calibrated_logits, loss = get_calibrated_logits_loss(logits, outputs, calibrator_output_type, labels, shots_start, num_iters=100)
        temperatures = calibrated_logits/logits 
        
        probs, preds = F.softmax(logits, dim=-1).max(dim=-1)
        calibrated_probs, calibrated_preds = F.softmax(calibrated_logits, dim=-1).max(dim=-1)
        
        # print(temperatures[:, -1, :].mean().item(), temperatures[:, -1, :].min().item(), temperatures[:, -1, :].max().item())
        # print(calibrated_pred_probs[:, -1, :].mean().item(), calibrated_pred_probs[:, -1, :].min().item(), calibrated_pred_probs[:, -1, :].max().item(), calibrated_pred_probs[:, -1, :].std().item())
        
        # print(calibrated_probs[:, -1].mean().item(), calibrated_probs[:, -1].min().item(), calibrated_probs[:, -1].max().item(), calibrated_probs[:, -1].std().item())
        
        if verbose:
            print(f"|------Dataset: {dataset}------| [{sampling_strategy}]")
        
        for shot in range(shots_start, T):
            calibrator_metrics[shot] = CalibrationMetrics(logits[:, [shot], :], calibrated_logits[:, [shot], :], labels[:, [shot]], 
                                   shots_start=0, prepare_rel_diag=plot_results, plot_confidence_band=plot_confidence_band)

            temps = temperatures[:, shot, :].flatten().cpu().numpy()
            accuracy = (labels == preds).cpu()[:, shot].sum() / len(preds)
            conf_original = mean_valid(probs[:, shot])
            conf_calibrated = mean_valid(calibrated_probs[:, shot])

            dynamic_temp_metrics[shot], dynamic_temp_conf_calibrated = get_calibrated_metrics_and_conf(logits[:, [shot], :], labels[:, [shot]], shotwise_dynamic_temps[shot].view(B, 1, 1))

            static_temp_metrics[shot], static_temp_conf_calibrated = get_calibrated_metrics_and_conf(logits[:, [shot], :], labels[:, [shot]], shotwise_static_temps[shot])
            
            if plot_gt_calibration:
                global_temp_metrics[shot], global_temp_conf_calibrated = get_calibrated_metrics_and_conf(logits[:, [shot], :], labels[:, [shot]], shotwise_global_temps[shot])
            
                calibration_data.add_shot_metrics(shot, 
                    calibrator_metrics[shot], dynamic_temp_metrics[shot], static_temp_metrics[shot], global_temp_metrics[shot],
                    temps, mean_valid(shotwise_dynamic_temps[shot]), shotwise_static_temps[shot], shotwise_global_temps[shot],
                    accuracy, 
                    conf_original, conf_calibrated, dynamic_temp_conf_calibrated, static_temp_conf_calibrated, global_temp_conf_calibrated
                )
            else:
                calibration_data.add_shot_metrics(shot, 
                    calibrator_metrics[shot], dynamic_temp_metrics[shot], static_temp_metrics[shot], None,
                    temps, mean_valid(shotwise_dynamic_temps[shot]), shotwise_static_temps[shot], None,
                    accuracy, 
                    conf_original, conf_calibrated, dynamic_temp_conf_calibrated, static_temp_conf_calibrated, None
                )
            
            shot_summary = f"{shot} shot accuracy {calibration_data.accuracies[shot-shots_start]:.4f}, \
    mean prob {calibration_data.conf_shots_map['original'][shot]:.4f}   \
    mean TF calibrated prob {calibration_data.conf_shots_map['calibrated'][shot]:.4f} \
    mean DT calibrated prob {calibration_data.conf_shots_map['dynamic_temp_calibrated'][shot]:.4f}\
    mean ST calibrated prob {calibration_data.conf_shots_map['static_temp_calibrated'][shot]:.4f}"
    
            if plot_gt_calibration:
                shot_summary += f"\
    mean GT calibrated prob {calibration_data.conf_shots_map['global_temp_calibrated'][shot]:.4f}"
        
        # if verbose:
        #     print(shot_summary)
        
        num_shots = T - shots_start
        
        overall_calibrator_metrics = sum(calibrator_metrics.values(), start=CalibrationMetrics.zeros()) / num_shots
        overall_dynamic_temp_metrics = sum(dynamic_temp_metrics.values(), start=CalibrationMetrics.zeros()) / num_shots
        overall_static_temp_metrics = sum(static_temp_metrics.values(), start=CalibrationMetrics.zeros()) / num_shots
        
        if plot_gt_calibration:
            overall_global_temp_metrics = sum(global_temp_metrics.values(), start=CalibrationMetrics()) / num_shots
            calibration_data.add_overall_metrics(overall_calibrator_metrics, overall_dynamic_temp_metrics, 
                                                overall_static_temp_metrics, overall_global_temp_metrics)
        else:
            calibration_data.add_overall_metrics(overall_calibrator_metrics, overall_dynamic_temp_metrics, 
                                                overall_static_temp_metrics)
        
        return calibration_data

def get_calibrated_metrics_and_conf(logits, labels, temps):
    temp_calibrated_logits = logits * temps
    temp_calibrated_probs, temp_calibrated_preds = F.softmax(temp_calibrated_logits, dim=-1).max(dim=-1)

    temp_eval_metrics = CalibrationMetrics(logits, temp_calibrated_logits, labels, 
                                    shots_start=0, prepare_rel_diag=False)

    temp_conf_calibrated = mean_valid(temp_calibrated_probs)
    
    return temp_eval_metrics, temp_conf_calibrated 

def mean_valid(x):
    return np.ma.masked_invalid(x.cpu()).mean()

def get_stats_summary(message, metrics):
    summary = []
    summary.append(f"\n{message}:")
    summary.append(f"  Mean  : {np.mean(metrics):.4f}")
    summary.append(f"  Std   : {np.std(metrics):.4f}")
    summary.append(f"  Min   : {np.min(metrics):.4f}")
    summary.append(f"  Max   : {np.max(metrics):.4f}")
    summary.append(f"  Median: {np.median(metrics):.4f}")
    
    summary = "\n".join(summary)
    
    return summary

def print_llm_summary(
    model_name, llm_eval_summaries, 
    llm_eces, llm_briers, 
    llm_dynamic_eces, llm_dynamic_briers,
    llm_static_eces, llm_static_briers
    ):
    
    eval_summary = "".join(llm_eval_summaries)
    print(eval_summary)
    print("\n" + "="*60)
    print(f"\n {model_name} OVERALL SUMMARY\n")
    print("="*60)
    print("\n--- Transformer Calibrator ---")
    print(get_stats_summary("Expected Calibration Error (ECE)", llm_eces))
    print(get_stats_summary("Brier Score", llm_briers))
    print("\n--- Dynamic Temperature Scaling ---")
    print(get_stats_summary("Expected Calibration Error (ECE)", llm_dynamic_eces))
    print(get_stats_summary("Brier Score", llm_dynamic_briers))
    print("\n--- Static Temperature Scaling ---")
    print(get_stats_summary("Expected Calibration Error (ECE)", llm_static_eces))
    print(get_stats_summary("Brier Score", llm_static_briers))
    
def args_check(args):
    if args['plot_confidence_band'] is True:
        assert args['plot_results'] is True, "Turn on plotting of results, you have plot_confidence_band as True"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--llms', action='store', required=True, help='name of llms to evaluate the calibrator on')
    parser.add_argument('--datasets', action='store', required=True, help='name of datasets to eval the calibrator on')    
    parser.add_argument('--feature_type', action='store', required=False, default="", help='the type of input features that make up the dataset')
    parser.add_argument('--shots_start', action='store', required=False, type=int, default=8, help='which shot # onwards we will do calibration eval')
    parser.add_argument('--shots_end', action='store', required=False, type=int, default=None, help='Till which shot # we will do calibration eval')
    parser.add_argument('--sampling_strategies', action='store', required=False, default=None, help='what sampling strategy data to select(entropy vs similarity) (default: None - means select all)')
    
    parser.add_argument('--calibrator_type', action='store', required=False, default='transformer',
                        choices=['transformer', 'mlp', 'rnn', 'lstm', 'logistic'], help='which calibrator architecture to train')
    parser.add_argument('--calibrator_output_type', action='store', required=False, default="calibrated_probability", help='What the transformer calibrator outputs(temperature/calibrated_probability)')
    parser.add_argument('--ablation_method', action='store', required=False, default=None, help='Ablation method name, if performing ablation')
    parser.add_argument('--num_seeds', action='store', required=False, default=1, type=int, help='Number of seeds to train calibrators for')
    
    parser.add_argument('--model_name', action='store', required=False, default="calibrator", help='custom name for the model(calibrator), used for saving the checkpoints')    
    parser.add_argument('--model_path', action='store', default=None, required=False, help='Path of the model to be loaded ')
    parser.add_argument('--llm_agnostic', action='store_true', help='whether to use the llm agnostic calibrator')
    parser.add_argument('--save_path', action='store', default=None, required=False, help='What path to save the calibration plot to ')
    parser.add_argument('--gpu_id', action='store', default=0, required=False, help='Which CUDA gpu to run model on', type=int)
    
    parser.add_argument('--plot_results', action='store_true', help='Whether to plot the results or just get eval metrics')
    parser.add_argument('--plot_confidence_band', action='store_true', help='Whether to plot the confidence bands for the reliability plots')
    parser.add_argument('--plot_gt_calibration', action='store_true', help='Whether to plot the confidence bands for the reliability plots')
    parser.add_argument('--hide_non_summary', action='store_true', help='Whether to not to print the intermediate metric results before the final summary')
    parser.add_argument('--save_seed_results', action='store_true', help='Whether to not to save the seedwise results in the eval.json')
    
    args = parser.parse_args()
    args = vars(args)

    args['datasets'] = convert_to_list(args['datasets'])
    args['llms'] = convert_to_list(args['llms'])
    
    args['calibrator_output_type'] = CalibratorOutputType[args['calibrator_output_type'].upper()]
    if args.get('sampling_strategies'): 
        args['sampling_strategies'] = convert_to_list(args['sampling_strategies'], lambda s: s.upper())
    else:
        args['sampling_strategies'] = ['ENTROPY', 'SIMILARITY']
        
        
    args_check(args)
    
    main(**args)