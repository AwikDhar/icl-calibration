from enum import Enum

class SamplingStrategy(Enum):
    ENTROPY=0
    SIMILARITY=1

class EntropyLevels(Enum):
    RANDOM=0
    RANDOM_SHARED=4
    LABELSUPPRESS=1
    LABELSPIKE=2
    MAX=3
    
