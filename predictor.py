import numpy as np
from ultralytics.yolo.engine.predictor import BasePredictor


class CustomPredictor(BasePredictor):
    def __init__(self, cfg=..., overrides=None):
        super().__init__(cfg, overrides)

    def postprocess(self, preds, img, orig_img, classes=None):
        return super().postprocess(preds, img, orig_img, classes)

    def preprocess(self, img) -> np.ndarray:
        return img
