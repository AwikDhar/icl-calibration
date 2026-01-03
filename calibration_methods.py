from enum import Enum

class CalibrationMethods(Enum):
    
    GC=0
    '''Generative Calibration'''
    BC=1
    '''Batch Calibration'''
    ICC=2
    '''In-Context Calibration'''
    TF=3 
    '''Transformer Calibrator'''
    ST=4
    '''Static Temperature scaling learned for the shot from a batch of predictions'''
    ICT=5
    '''In-Context examples learned Temperature scaling'''
    