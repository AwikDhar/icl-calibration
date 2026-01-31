from enum import Enum

class CalibrationMethods(Enum):
    
    GC=0
    '''Generative Calibration - !!Assumes shared prompt!!'''
    BC=1
    '''Batch Calibration - !!Assumes shared prompt!!'''
    ICC=2
    '''In-Context Calibration'''
    TF=3 
    '''Transformer Calibrator'''
    BTF=4 
    '''Batch Transformer Calibrator - takes the calibration set prrdictions as inputs to learn calibration from - !!Assumes shared prompt!!'''
    PTF=5 
    '''Transformer Calibrator used post/on top of Batch/Generative/etc. calibration(calibrates calbrated logits) - !!Assumes shared prompt!!'''
    ST=6
    '''Static(not query specific) Temperature scaling learned for the shot from a batch of predictions'''
    ICT=7
    '''In-Context examples learned Temperature scaling'''
    PERMUT_AVG=8
    '''Average probs over a given number of permutations of in-context examples'''
    UNCALIBRATED=9
    '''Without intervention'''
    FS_ICT=10
    '''Temperature scaling learned from predictions on the k In-Context examples with random (k-1)/2 shot(Fixed Shot) prompts constructed from the In-Context examples'''
    