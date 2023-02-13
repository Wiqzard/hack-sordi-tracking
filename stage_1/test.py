import torch

from torch.utils.data import DataLoader
from data_provider.data_factoy import SordiAiDataset, SordiAiDatasetEval
from utils.config import config
from network.pretrained import create_model, get_transform  # , model
from typing import Dict, List

from utils.tools import (
    transform_label,
    train_test_split,
    # show_tranformed_image,
    collate_fn,
    dotdict,
)
from utils.constants import CLASSES
from exp.exp_main import Exp_Main


model, weights = create_model(len(CLASSES))

preprocess = weights.transforms()

full_dataset = SordiAiDataset(
    root_path="./data/",
    transforms=get_transform(),
    ignore_redundant=False,
    partion_single_assets=1,
)  # , transforms=preprocess)
train_dataset, test_dataset = train_test_split(full_dataset, 0.8)

eval_dataset = SordiAiDatasetEval(root_path="./data/", transforms=get_transform())
train_dataloder = DataLoader(
    train_dataset,
    batch_size=3,
    shuffle=config["shuffle"],
    num_workers=config["num_workers"],
    drop_last=config["drop_last"],
    collate_fn=collate_fn,
)
print(len(full_dataset))
# print(next(iter(train_dataloder)))

# print(len(full_dataset))
# print(trafo(full_dataset[-1][1]))


# print(full_dataset[-1][0].shape)

# print(train_dataset[0][0].shape)
# print(test_dataset[0])
# print(eval_dataset[0][-1].shape)
# batch = next(iter(train_dataloder))
# print(batch)
## print(transform_label(CLASSES, batch[1][0]))
# show_tranformed_image(train_dataset, CLASSES)

# show_tranformed_image(train_dataloder, CLASSES)


def train_model() -> None:
    model.train()
    """
    input:
    images, targets   -> 
    [Batch, C, H, W],   
    [Batch * Dict["boxes": [x1,y1,x2,y2], "labels": [label1, label2,...]]
    """
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"{total_trainable_params:,} training parameters.\n")
    for image, label in train_dataloder:
        target = transform_label(label)  # torch.tensor([1])
        prediction = model(image, target)
        # loss_classifier = prediction["loss_classifier"]
        # loss_box_re = prediction["loss_box_re"]
        # loss_objectnes = prediction["loss_objectnes"]
        # loss_rpn_box_reg = prediction["loss_rpn_box_reg"]

        #    image, label = train_dataset[0]
        #    image = image  # .unsqueeze(0)
        #    boxes = torch.tensor(
        #        [label["Left"], label["Top"], label["Right"], label["Bottom"]]
        #    )  # .unsqueeze(0)
        #    labels = torch.tensor([classes[label["ObjectClassId"]]])  # torch.tensor([1])
        #    targets = [{"boxes": boxes, "labels": labels}]
        #    prediction = model(image, targets)
        print(target)

        print(prediction)


# train_model()
