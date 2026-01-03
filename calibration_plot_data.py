from dataclasses import dataclass, field
from typing import Dict, List
from metrics import CalibrationMetrics

@dataclass
class CalibrationPlotData:
    # Shotwise
    ece_shots_map: Dict = field(default_factory=lambda: {'original': {}, 'calibrated': {}, 'dynamic_temp_calibrated': {}, 'static_temp_calibrated': {}, 'global_temp_calibrated': {}})
    brier_shots_map: Dict = field(default_factory=lambda: {'original': {}, 'calibrated': {}, 'dynamic_temp_calibrated': {}, 'static_temp_calibrated': {}, 'global_temp_calibrated': {}})
    reldiag_shots_map: Dict = field(default_factory=lambda: {'original': {}, 'calibrated': {}})
        
    temp_shots_map: Dict = field(default_factory=dict)
    dynamic_temp_shots_map: Dict = field(default_factory=dict)
    static_temp_shots_map: Dict = field(default_factory=dict)
    global_temp_shots_map: Dict = field(default_factory=dict)
    
    conf_shots_map: Dict = field(default_factory=lambda: {'original': {}, 'calibrated': {}, 'dynamic_temp_calibrated': {}, 'static_temp_calibrated': {}, 'global_temp_calibrated': {}})
    accuracies: List = field(default_factory=list)
    
    # Overall
    overall_ece: Dict = field(default_factory=lambda: {'original': None, 'calibrated': None})
    overall_brier: Dict = field(default_factory=lambda: {'original': None, 'calibrated': None})
    overall_reldiag: Dict = field(default_factory=lambda: {'original': None, 'calibrated': None})
    
    def add_shot_metrics(self, shot: int, 
                         eval_metrics, dynamic_temp_eval_metrics, static_temp_eval_metrics, global_temp_eval_metrics,
                         temps, dynamic_temp, static_temp, global_temp,
                         accuracy, conf_original, 
                         conf_calibrated, dynamic_temp_conf_calibrated, static_temp_conf_calibrated, global_temp_conf_calibrated):
        """Add metrics for a specific shot to the calibration data."""
        # ECE and Brier scores
        self.ece_shots_map['original'][shot] = eval_metrics.ece
        self.ece_shots_map['calibrated'][shot] = eval_metrics.calibrated_ece
        self.ece_shots_map['dynamic_temp_calibrated'][shot] = dynamic_temp_eval_metrics.calibrated_ece
        self.ece_shots_map['static_temp_calibrated'][shot] = static_temp_eval_metrics.calibrated_ece
        self.ece_shots_map['global_temp_calibrated'][shot] = global_temp_eval_metrics.calibrated_ece if global_temp_eval_metrics else None
        
        self.brier_shots_map['original'][shot] = eval_metrics.brier_score
        self.brier_shots_map['calibrated'][shot] = eval_metrics.calibrated_brier_score
        self.brier_shots_map['dynamic_temp_calibrated'][shot] = dynamic_temp_eval_metrics.calibrated_brier_score
        self.brier_shots_map['static_temp_calibrated'][shot] = static_temp_eval_metrics.calibrated_brier_score
        self.brier_shots_map['global_temp_calibrated'][shot] = global_temp_eval_metrics.calibrated_brier_score if global_temp_eval_metrics else None
        
        # Reliability diagrams
        self.reldiag_shots_map['original'][shot] = eval_metrics.rel_diag
        self.reldiag_shots_map['calibrated'][shot] = eval_metrics.calibrated_rel_diag
        
        # diagnostic/comparison stuff
        self.temp_shots_map[shot] = temps
        self.dynamic_temp_shots_map[shot] = dynamic_temp
        self.static_temp_shots_map[shot] = static_temp
        self.global_temp_shots_map[shot] = global_temp
        
        self.accuracies.append(accuracy)
        self.conf_shots_map['original'][shot] = conf_original
        
        self.conf_shots_map['calibrated'][shot] = conf_calibrated
        self.conf_shots_map['dynamic_temp_calibrated'][shot] = dynamic_temp_conf_calibrated
        self.conf_shots_map['static_temp_calibrated'][shot] = static_temp_conf_calibrated
        self.conf_shots_map['global_temp_calibrated'][shot] = global_temp_conf_calibrated
        
    def add_overall_metrics(self, eval_metrics: CalibrationMetrics, dynamic_temp_eval_metrics: CalibrationMetrics = None, static_temp_eval_metrics: CalibrationMetrics = None):
        """
        Args:
            eval_metrics (Metrics): Overall(all-shots) metrics for the calibrator alone
            dynamic_temp_eval_metrics (Metrics): Overall metrics for dynamic temperature scaling
            static_temp_eval_metrics (Metrics): Overall metrics for static temperature scaling
        """        
        self.overall_ece = {
            "original": eval_metrics.ece, 
            "calibrated": eval_metrics.calibrated_ece,
            "dynamic_temp_calibrated": dynamic_temp_eval_metrics.calibrated_ece,
            "static_temp_calibrated": static_temp_eval_metrics.calibrated_ece
        }
        self.overall_brier = {
            "original": eval_metrics.brier_score, 
            "calibrated": eval_metrics.calibrated_brier_score,
            "dynamic_temp_calibrated": dynamic_temp_eval_metrics.calibrated_brier_score,
            "static_temp_calibrated": static_temp_eval_metrics.calibrated_brier_score
        }