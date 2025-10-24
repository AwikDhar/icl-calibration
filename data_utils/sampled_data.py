from dataclasses import dataclass, field
from typing import List
import numpy as np

@dataclass
class SampledData():
    sentences: List[str] = field(default_factory=list)
    labels: List[int] = field(default_factory=list)
    embeddings: np.ndarray = field(default_factory=lambda : np.empty((0, 0), dtype=np.float32))
