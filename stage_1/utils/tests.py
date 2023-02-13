from torch.utils.data import DataLoader
from data_provider.data_factoy import SordiAiDataset, SordiAiDatasetEval
from utils.config import config
from network.pretrained import create_model, get_transform  # , model
from typing import Dict, List

from utils.tools import (
    transform_label,
    train_test_split,
    logger,
    show_tranformed_image,
    dotdict,
)
from utils.constants import CLASSES
from exp.exp_main import Exp_Main

import random
import torch


def get_args() -> Dict:
    args = dotdict()
    args.is_training = 1
    args.model_id = "faster_rcnn_train1 "
    args.model = "Faster_RCNN"
    args.data = "custom"
    args.root_path = "/content/drive/MyDrive/Datasets/"
    args.data_path = "SORDI_2022_Single_Assets"
    args.ratio = 0.8
    args.checkpoints = "/checkpoints/ "
    args.batch_size = 3
    args.num_workers = 0
    args.itr = 1
    args.train_epochs = 2
    args.patience = 7
    args.learning_rate = 0.0001
    args.des = "Exp"
    args.lradj = "type1"
    args.use_gpu = True
    args.gpu = 0
    args.devices = 0

    args.use_gpu = bool(torch.cuda.is_available() and args.use_gpu)

    logger.info("Args in experiment:")
    logger.info(args)
    return args


def test_exp() -> None:
    args = get_args()
    exp = Exp_Main(args)

    setting = f"{args.model_id}_{args.model}_{args.data}_{args.des}_0"
    exp.train(setting)


def test_dataset_output() -> None:
    full_dataset = SordiAiDataset(root_path="./data/", transforms=get_transform())
    index = random.randint(len(full_dataset))
    image, target = full_dataset[index]
    assert image.shape == torch.size([3, 600, 1024])
    # assert label set of dicts, boxes scaled,
    # show image


def test_dataloader_output() -> None:
    pass


def test_experiment_eval() -> None:
    pass


model, weights = create_model(len(CLASSES))

preprocess = weights.transforms()

full_dataset = SordiAiDataset(
    root_path="./data/", transforms=get_transform()
)  # , transforms=preprocess)
train_dataset, test_dataset = train_test_split(full_dataset, 0.8)

eval_dataset = SordiAiDatasetEval(root_path="./data/", transforms=get_transform())
train_dataloder = DataLoader(
    train_dataset,
    batch_size=3,
    shuffle=config["shuffle"],
    num_workers=config["num_workers"],
    drop_last=config["drop_last"],
)
