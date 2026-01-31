import argparse
import json

import torch
import torch.nn.functional as F
import numpy as np

# from data_utils import * # bad - bandaid solution for a circular import 
from utils import convert_to_list
from calibration.surprise.model import SurpriseCalibrationTransformer, PositionEmbeddingType
from metrics import CalibrationMetrics
from calibration_plot_data import CalibrationPlotData
from calibration.train import get_batch, load_datasets 
from calibration.temperature import get_shotwise_dynamic_temperatures, get_shotwise_static_temperatures, get_equivalent_temp
from plot_results.plot_tc import plot_calibration

def main(llms, 
         datasets,
         feature_type,
         shots_start,
         shots_end,
         sampling_strategy = None,
         gpu_id=0, 
         model_name="calibrator",
         model_path=None,
         llm_agnostic=False,
         save_path=None,
         plot_results=False,
         plot_confidence_band=False,
         plot_gt_calibration=False):
    
    device = f'cuda:{gpu_id}'
    
    data = load_datasets(llms, datasets, device, feature_type, shots_end, splits=('test','train'), sampling_strategy=sampling_strategy, temp_augment=False, label_augment=False, task='bias_pred')
    
    with open(f"calibration/surprise/models/transformer_config.json", 'r') as file:
        config = json.load(file)
    
    if model_path is None:
        if llm_agnostic or len(llms)>1:
            model_dir = f"./calibration/surprise/models/llm_agnostic"
        else:
            model_dir = f"./calibration/surprise/models/{llms[0].replace('/','_')}"
            
        # if sampling_strategy:
        #     model_dir += f"/{sampling_strategy}"
            
        model_path = f'{model_dir}/{model_name}'

    sample_model, sample_dataset = llms[0], datasets[0]
    T, C = data[sample_model][sample_dataset]['test']['inputs'][0].shape
    calibrator = SurpriseCalibrationTransformer(
        in_features=config['in_features'], 
        context_length=config['context_length'], 
        embedding_dim=config['embedding_dim'], 
        num_heads=config['num_heads'], 
        num_layers=config['num_layers'],
        pos_embedding_type=PositionEmbeddingType.Sinusoidal
    ).to(device)
    # print(calibrator)

    state_dict = torch.load(model_path, weights_only=True)
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('_orig_mod.'):
            cleaned_state_dict[key.replace('_orig_mod.', '')] = value
        else:
            cleaned_state_dict[key] = value

    calibrator.load_state_dict(cleaned_state_dict)

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
            calibration_data = eval(calibrator, data, llm, dataset, shots_start, 
                                    plot_results, plot_confidence_band, plot_gt_calibration)
            
            dataset_eces.append(calibration_data.overall_ece['calibrated'])
            dataset_briers.append(calibration_data.overall_brier['calibrated'])
            dataset_dynamic_eces.append(calibration_data.overall_ece['dynamic_temp_calibrated'])
            dataset_dynamic_briers.append(calibration_data.overall_brier['dynamic_temp_calibrated'])
            dataset_static_eces.append(calibration_data.overall_ece['static_temp_calibrated'])
            dataset_static_briers.append(calibration_data.overall_brier['static_temp_calibrated'])
            
            if plot_results:
                plot_calibration(calibration_data, llm, dataset, 
                                 feature_type, llm_agnostic, sampling_strategy, 
                                 save_path=save_path, plot_gt_calibration=plot_gt_calibration)
    
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
        
def eval(model, data, llm, dataset, shots_start, plot_results, plot_confidence_band, plot_gt_calibration):
    calibration_data = CalibrationPlotData()

    inputs, logits, labels = get_batch(data[llm][dataset]['test']) # len(eval),T,C | len(eval),T,num_classes | len(eval),T
    B,T,num_classes = logits.shape
    
    shotwise_dynamic_temps = get_shotwise_dynamic_temperatures(data[llm][dataset]['test'], shots_start)
    shotwise_static_temps = get_shotwise_static_temperatures(data[llm][dataset]['train'], shots_start, batch_size=100)
    # shotwise_static_temps = {shot:1 for shot in range(shots_start, T)}
    if plot_gt_calibration:
        with open("calibration/trained_temperature.json", "r") as file:
            shotwise_global_temps = json.load(file)
            shotwise_global_temps = {int(shot):temp for  shot, temp in shotwise_global_temps.items()}

    calibrator_metrics = {}
    dynamic_temp_metrics = {}
    static_temp_metrics = {}
    global_temp_metrics = {} if plot_gt_calibration else None

    model.eval()
    with torch.no_grad():                        
        prompt_bias = model(inputs) # B,T,1
        prompt_bias = prompt_bias[:, :, :num_classes]
        
        calibrated_logits = F.log_softmax(logits) - prompt_bias
        
        temperatures = torch.ones((B, T, 1))  # B, T, 1

        # calibrated_logits = logits*temperatures # B,T,num_classes * B,T,1
        
        probs, preds = F.softmax(logits, dim=-1).max(dim=-1)
        calibrated_probs, calibrated_preds = F.softmax(calibrated_logits, dim=-1).max(dim=-1)
        
        # print(temperatures[:, -1, :].mean().item(), temperatures[:, -1, :].min().item(), temperatures[:, -1, :].max().item())
        # print(calibrated_pred_probs[:, -1, :].mean().item(), calibrated_pred_probs[:, -1, :].min().item(), calibrated_pred_probs[:, -1, :].max().item(), calibrated_pred_probs[:, -1, :].std().item())
        
        # print(calibrated_probs[:, -1].mean().item(), calibrated_probs[:, -1].min().item(), calibrated_probs[:, -1].max().item(), calibrated_probs[:, -1].std().item())
        
        print(f"|------Dataset: {dataset}------|")
        for shot in range(shots_start, T):
            calibrator_metrics[shot] = CalibrationMetrics(logits[:, [shot], :], calibrated_logits[:, [shot-1], :], labels[:, [shot]], 
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
        
            print(shot_summary)
        
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

def args_check(args):
    if args['plot_confidence_band'] is True:
        assert args['plot_results'] is True, "Turn on plotting of results, you have plot_confidence_band as True"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--llms', dest='llms', action='store', required=True, help='name of llms to evaluate the calibrator on')
    parser.add_argument('--datasets', dest='datasets', action='store', required=True, help='name of datasets to eval the calibrator on')    
    parser.add_argument('--feature_type', dest='feature_type', action='store', required=False, default="", help='the type of input features that make up the dataset')
    parser.add_argument('--shots_start', dest='shots_start', action='store', required=False, type=int, default=8, help='which shot # onwards we will do calibration eval')
    parser.add_argument('--shots_end', dest='shots_end', action='store', required=False, type=int, default=None, help='Till which shot # we will do calibration eval')
    parser.add_argument('--sampling_strategy', dest='sampling_strategy', action='store', required=False, default=None, help='what sampling strategy data to select(entropy vs similarity) (default: None - means select all)')
    parser.add_argument('--gpu_id', dest='gpu_id', action='store', default=0, required=False, help='Which CUDA gpu to run model on', type=int)
    
    parser.add_argument('--model_name', dest='model_name', action='store', required=False, default="calibrator", help='custom name for the model(calibrator), used for saving the checkpoints')    
    parser.add_argument('--model_path', dest='model_path', action='store', default=None, required=False, help='Path of the model to be loaded ')
    parser.add_argument('--llm_agnostic', dest='llm_agnostic', action='store_const', const=True, default=False, help='whether to use the llm agnostic calibrator')
    parser.add_argument('--save_path', dest='save_path', action='store', default=None, required=False, help='What path to save the calibration plot to ')
    
    parser.add_argument('--plot_results', dest='plot_results', action='store_const', const=True, default=False, required=False, help='Whether to plot the results or just get eval metrics')
    parser.add_argument('--plot_confidence_band', dest='plot_confidence_band', action='store_const', const=True, default=False, required=False, help='Whether to plot the confidence bands for the reliability plots')
    parser.add_argument('--plot_gt_calibration', dest='plot_gt_calibration', action='store_const', const=True, default=False, required=False, help='Whether to plot the confidence bands for the reliability plots')
    
    args = parser.parse_args()
    args = vars(args)

    args['datasets'] = convert_to_list(args['datasets'])
    args['llms'] = convert_to_list(args['llms'])
    
    sampling_strategy = args.get('sampling_strategy')
    if sampling_strategy:
        args['sampling_strategy'] = sampling_strategy
        
    args_check(args)
    
    main(**args)