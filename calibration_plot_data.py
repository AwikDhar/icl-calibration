from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class CalibrationPlotData:
    ece_shots_map: Dict = field(default_factory=lambda: {'original': {}, 'calibrated': {}, 'static_temp_calibrated':{}})
    brier_shots_map: Dict = field(default_factory=lambda: {'original': {}, 'calibrated': {}, 'static_temp_calibrated':{}})
    reldiag_shots_map: Dict = field(default_factory=lambda: {'calibrated': {}, 'original': {}})
        
    temp_shots_map: Dict = field(default_factory=dict)
    static_temp_shots_map: Dict = field(default_factory=dict)
    
    conf_shots_map: Dict = field(default_factory=lambda: {'original': {}, 'calibrated': {}, 'static_temp_calibrated':{}})
    accuracies: List = field(default_factory=list)
    
    overall_reldiag: Dict = field(default_factory=lambda: {'calibrated': None, 'original': None})
    
    def add_shot_metrics(self, shot: int, 
                         eval_metrics, static_temp_eval_metrics, 
                         temps, static_temp, 
                         accuracy, conf_original, 
                         conf_calibrated, static_temp_conf_calibrated):
        """Add metrics for a specific shot to the calibration data."""
        # ECE and Brier scores
        self.ece_shots_map['original'][shot] = eval_metrics.ece
        self.ece_shots_map['calibrated'][shot] = eval_metrics.calibrated_ece
        self.ece_shots_map['static_temp_calibrated'][shot] = static_temp_eval_metrics.calibrated_ece
        
        self.brier_shots_map['original'][shot] = eval_metrics.brier_score
        self.brier_shots_map['calibrated'][shot] = eval_metrics.calibrated_brier_score
        self.brier_shots_map['static_temp_calibrated'][shot] = static_temp_eval_metrics.calibrated_brier_score
        
        # Reliability diagrams
        self.reldiag_shots_map['original'][shot] = eval_metrics.rel_diag
        self.reldiag_shots_map['calibrated'][shot] = eval_metrics.calibrated_rel_diag
        
        # diagnostic/analysis stuff
        self.temp_shots_map[shot] = temps
        self.static_temp_shots_map[shot] = static_temp
        
        self.accuracies.append(accuracy)
        self.conf_shots_map['original'][shot] = conf_original
        
        self.conf_shots_map['calibrated'][shot] = conf_calibrated
        self.conf_shots_map['static_temp_calibrated'][shot] = static_temp_conf_calibrated
    