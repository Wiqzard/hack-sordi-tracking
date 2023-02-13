from data_provider.data_factoy import SordiAiDataset, SordiAiDatasetEval
from network.pretrained import create_model, get_transform
from exp.exp_basic import Exp_Basic
from utils.tools import (
    transform_label,
    adjust_learning_rate,
    EarlyStopping,
    logger,
    log_train_progress,
    log_train_epoch,
    train_test_split,
    write_to_csv,
    collate_fn,
    store_losses,
)
from utils.constants import CLASSES

# from utils.tools import EarlyStopping, adjust_learning_rate
# from utils.metrics import metric
from typing import Tuple, Optional, List, Union
import time
import numpy as np
import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm


class Exp_Main(Exp_Basic):
    classes = CLASSES

    def __init__(self, args) -> None:
        super().__init__(args)

    def _build_model(self):
        model_dict = {"faster_rcnn": 1}
        model, self.weights = create_model(len(self.classes))
        model = model.float().to(self.device)
        return model

    def _get_data(
        self, flag: str = "train"
    ) -> Tuple[Union[SordiAiDataset, SordiAiDatasetEval], DataLoader]:
        args = self.args
        if flag == "eval":
            shuffle_flag = False
            drop_last = False
            batch_size = 1
            data_set = SordiAiDatasetEval(
                root_path=args.root_path,
                transforms=get_transform(),
            )
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size

            full_dataset = SordiAiDataset(
                root_path=args.root_path,
                transforms=get_transform(),
                partion_single_assets=args.partion_single_assets,
                ignore_redundant=args.ignore_redundant,
            )
            train_dataset, test_dataset = train_test_split(
                dataset=full_dataset, ratio=args.ratio
            )
            data_set = train_dataset if flag == "train" else test_dataset
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=collate_fn,
        )
        return data_set, data_loader

    def _select_optimizer(self) -> None:
        args = self.args
        params = [p for p in self.model.parameters() if p.requires_grad]
        if args.optimizer == "adam":
            # sourcery skip: inline-immediately-returned-variable
            model_optim = optim.Adam(params, lr=args.learning_rate)
        elif self.args.optimizer == "sgd":
            model_optim = optim.SGD(
                params,
                lr=args.learning_rate,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
            )
        return model_optim

    def _select_criterion(self) -> None:
        # sourcery skip: inline-immediately-returned-variable
        criterion = nn.MSELoss()
        return criterion

    def _select_scheduler(self, optimizer) -> None:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer, step_size=3, gamma=0.9
        )
        return lr_scheduler

    def _set_checkpoint(self, setting) -> str:
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def _process_one_batch(self, image, label):
        # sourcery skip: inline-immediately-returned-variable
        images = [image_.to(self.device) for image_ in image]
        targets = [
            {k: v.to(self.device) for k, v in t.items() if k != "image_id"}
            for t in label
        ]
        # images = list(image)
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                loss_dict = self.model(images, targets)
        else:
            loss_dict = self.model(images, targets)
        return loss_dict

    def test(self, test_loader) -> float:
        self.model.train()  # train????
        total_loss = []
        test_losses = {
            "loss_classifier": [],
            "loss_box_reg": [],
            "loss_objectness": [],
            "loss_rpn_box_reg": [],
        }
        with torch.no_grad():
            for image, label in test_loader:
                loss = self._process_one_batch(image=image, label=label)
                total_loss.append(sum(loss.values()).detach().item())
                test_losses = store_losses(test_losses, loss)
            total_loss = np.average(total_loss)
        self.model.train()
        return total_loss, test_losses

    def evaluation(self, setting) -> None:
        self.model.eval()
        idx = 0
        with torch.no_grad():
            _, eval_dataloader = self._get_data(flag="eval")
            for i, (image_name, image_width, image_height, image) in tqdm(
                enumerate(eval_dataloader)
            ):
                label = self.model(image)[0]
                idx = write_to_csv(idx, image_name, image_width, image_height, label)

    def train(self, setting):  # sourcery skip: low-code-quality
        train_data, train_loader = self._get_data(flag="train")
        test_data, test_loader = self._get_data(flag="test")

        path = self._set_checkpoint(setting=setting)
        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        scheduler = self._select_scheduler(optimizer=model_optim)
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            train_loss = []
            train_losses = {
                "loss_classifier": [],
                "loss_box_reg": [],
                "loss_objectness": [],
                "loss_rpn_box_reg": [],
            }

            self.model.train()
            epoch_time = time.time()
            iter_count = 0
            for i, (image, label) in tqdm(
                enumerate(train_loader), total=len(train_loader), position=0, leave=True
            ):
                iter_count += 1
                model_optim.zero_grad()

                loss_dict = self._process_one_batch(image=image, label=label)
                loss = sum(loss_dict.values())
                train_loss.append(loss.detach().item())
                train_losses = store_losses(train_losses, loss_dict)

                # if (i + 1) % 300 == 0:
                #    log_train_progress(
                #        args=self.args,
                #        time_now=time_now,
                #        loss=loss,
                #        epoch=epoch,
                #        train_steps=train_steps,
                #        i=i,
                #        iter_count=iter_count,
                #    )

                #    iter_count = 0
                #    time_now = time.time()
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
            #        pbar.update()

            # logger.info(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time} ")
            train_loss = np.average(train_loss)
            test_loss, test_losses = self.test(test_loader=test_loader)
            log_train_epoch(
                epoch=epoch,
                train_steps=train_steps,
                train_loss=train_loss,
                test_loss=test_loss,
                scheduler=scheduler,
            )

            early_stopping(
                train_losses, test_losses, train_loss, test_loss, self.model, path
            )
            if early_stopping.early_stop:
                logger.info("Early stopping")
                break
            scheduler.step()
        best_model_path = f"{path}/checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model
