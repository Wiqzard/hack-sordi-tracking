import os
import torch
import numpy as np


class Exp_Basic:
    def __init__(self, args) -> None:
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self) -> None:
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device("cuda")
            # os.environ["CUDA_VISIBLE_DEVICES"] = (
            #    self.args.devices if self.args.use_multi_gpu else str(self.args.gpu)
            # )
            # device = torch.device(f"cuda:{self.args.gpu}")
            # print(f"Use GPU: cuda:{self.args.gpu}")
            print(f"Use GPU: cuda:{str(device)}")
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
