from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any
import numpy as np

from geometric_utils import Rect
# geometry utilities

@dataclass
class Detection:
    rect: Rect
    class_id: int
    #class_name: str
    confidence: float
    tracker_id: Optional[int] = None

    @classmethod
    def from_results(cls, pred: np.ndarray):#, names: Dict[int, str]) -> List[Detection]:
        result = []
        for x_min, y_min, x_max, y_max, confidence, class_id in pred:
            class_id=int(class_id)
            result.append(Detection(
                rect=Rect(
                    x=float(x_min),
                    y=float(y_min),
                    width=float(x_max - x_min),
                    height=float(y_max - y_min)
                ),
                class_id=class_id,
                #class_name=names[class_id],
                confidence=float(confidence)
            ))
        return result