from collections import namedtuple
from copy import deepcopy

CalibrationMethodResult = namedtuple('CalibrationMethodResult', ['ece', 'brier'])

class ComparisonResult():
    def __init__(self, 
                 uncalibrated: CalibrationMethodResult,
                 calibrated: CalibrationMethodResult,
                 dynamic_calibrated: CalibrationMethodResult):
        
        self.uncalibrated = uncalibrated
        self.calibrated = calibrated
        self.dynamic_calibrated = dynamic_calibrated
    
    def __add__(self, other):
        return ComparisonResult(
            uncalibrated=CalibrationMethodResult(
                ece=self.uncalibrated.ece + other.uncalibrated.ece,
                brier=self.uncalibrated.brier + other.uncalibrated.brier
            ),
            calibrated=CalibrationMethodResult(
                ece=self.calibrated.ece + other.calibrated.ece,
                brier=self.calibrated.brier + other.calibrated.brier
            ),
            dynamic_calibrated=CalibrationMethodResult(
                ece=self.dynamic_calibrated.ece + other.dynamic_calibrated.ece,
                brier=self.dynamic_calibrated.brier + other.dynamic_calibrated.brier
            )
        )
        
    def __truediv__(self, divisor):
        return ComparisonResult(
            uncalibrated=CalibrationMethodResult(
                ece=self.uncalibrated.ece / divisor,
                brier=self.uncalibrated.brier / divisor
            ),
            calibrated=CalibrationMethodResult(
                ece=self.calibrated.ece / divisor,
                brier=self.calibrated.brier / divisor
            ),
            dynamic_calibrated=CalibrationMethodResult(
                ece=self.dynamic_calibrated.ece / divisor,
                brier=self.dynamic_calibrated.brier / divisor
            )
        )