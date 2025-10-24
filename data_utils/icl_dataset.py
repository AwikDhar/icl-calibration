from typing import List
import numpy as np
from dataclasses import dataclass, field

@dataclass
class IclDatasetSplit():
    sentences : List[str]= field(default_factory=list)
    labels: List[int] = field(default_factory=list)
    embeddings: np.ndarray = field(default_factory=lambda : np.empty((0, 0), dtype=np.float32))

@dataclass
class IclDataset():
    train: IclDatasetSplit
    test: IclDatasetSplit