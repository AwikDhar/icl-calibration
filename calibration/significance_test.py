import argparse
import json

import pandas as pd
import numpy as np
from scipy.stats import shapiro, ttest_rel, wilcoxon

from utils.gen_utils import convert_to_list

def get_seed_metrics(llms, method):
    with open(f"calibration/models/llm_agnostic/ablations/{method}/eval.json", "r") as file:
        metrics = json.load(file)
        llms_key = " | ".join(llms)
        seed_metrics = metrics[llms_key]['seedwise']
        
    return seed_metrics

def main(args):
    baseline_metrics = get_seed_metrics(args.llms, method=args.baseline)
    baseline_eces = np.array(baseline_metrics['eces'])
    baseline_briers = np.array(baseline_metrics['briers'])
    
    results = [('Method', 'ECE', 'Brier', 'ECE_p_value', 'Brier_p_value'),
               ('Baseline', np.mean(baseline_metrics['eces']), np.mean(baseline_metrics['briers']))]

    for ablation in args.ablations:
        ablation_metrics = get_seed_metrics(args.llms, method=ablation)
        ablation_eces = np.array(ablation_metrics['eces'])
        ablation_briers = np.array(ablation_metrics['briers'])
        
        ece_differences = ablation_eces - baseline_eces 
        brier_differences = ablation_briers - baseline_briers 
        
        ece_shapiro = shapiro(ece_differences)
        brier_shapiro = shapiro(brier_differences)
        
        ece_normality_ok = ece_shapiro.pvalue >= 0.05
        brier_normality_ok = brier_shapiro.pvalue >= 0.05
        
        if not ece_normality_ok or not brier_normality_ok:
            print("Shapiro violation for ablation method: ", ablation)
            print("ECE Shapiro: ", ece_shapiro) 
            print("Brier Shapiro: ", brier_shapiro)     

        # alternative = 'two-sided'
        alternative = 'greater'
        if ece_normality_ok:
            _, ece_pval = ttest_rel(ablation_eces, baseline_eces, alternative=alternative)
        else:
            _, ece_pval = wilcoxon(ablation_eces - baseline_eces, alternative=alternative)

        if brier_normality_ok:
            _, brier_pval = ttest_rel(ablation_briers, baseline_briers, alternative=alternative)
        else:
            _, brier_pval = wilcoxon(ablation_briers - baseline_briers, alternative=alternative)
        
        results.append(
            (ablation, np.mean(ablation_eces), np.mean(ablation_briers), ece_pval, brier_pval)
        )
        
    save_path = "calibration/models/llm_agnostic/ablations/significance_test.csv"
    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)
    print("Saved ablations statistical significance results to ")
    print(df)
    
if __name__ == '__main__':  
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--llms', action='store', required=True, type=convert_to_list, help='name of llms to evaluate calibration results on')
    parser.add_argument('--baseline', action='store', required=False, default='main', help='Baseline method to compare ablations against')
    parser.add_argument('--ablations', action='store', required=True, type=convert_to_list, help='Ablation methods to test statistical significance for')
 
    args = parser.parse_args()
    
    main(args)