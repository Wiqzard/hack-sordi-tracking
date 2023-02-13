from typing import List, Dict, Tuple, Optional
import numpy as np
import logging
import time
import torch
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import pandas as pd
from tqdm import tqdm

# import cv2
from utils.constants import CLASSES, CLASSES_ID, CLASSES_RE, CLASSES_ID_RE
import random
import matplotlib.pyplot as plt

# from data_provider.data_factoy import SordiAiDataset

logger = logging.getLogger("__name__")
level = logging.INFO
logger.setLevel(level)
ch = logging.StreamHandler()
ch.setLevel(level)
logger.addHandler(ch)


def dict_list_to_list_dict(labels):
    return [dict(zip(labels, t)) for t in zip(*labels.values())]


def train_test_split(dataset, ratio: float = 0.8):
    train_size = int(ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    return train_dataset, test_dataset


def collate_fn(batch):
    """
    To handle the data loading as different images may have different number
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))


# if flag == "train":
#     return train_dataset
# elif flag == "test":
#     return test_dataset
# else:
#     raise NameError
class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def transform_label(
    classes: Dict,
    labels: Dict,
) -> List[Dict]:  # sourcery skip: inline-immediately-returned-variable
    # targets = [
    #    {
    #        "boxes": torch.tensor([x1, y1, x2, y2]).unsqueeze(0),
    #        "labels": torch.tensor([classes[str(label)]]),
    #    }
    #    for (x1, y1, x2, y2), label in zip(
    #        zip(labels["Left"], labels["Top"], labels["Right"], labels["Bottom"]),
    #        labels["ObjectClassName"],
    #    )
    # ]

    return targets


def transform_target(targets) -> List[Dict]:
    """Dict of list to list of dicts"""
    return [dict(zip(targets, t)) for t in zip(*targets.values())]


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == "type1":
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == "type2":
        lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print(f"Updating learning rate to {lr}")


def store_losses(storage: Dict, source: Dict) -> None:
    storage["loss_classifier"].append(
        np.average(source["loss_classifier"].detach().cpu())
    )
    storage["loss_box_reg"].append(np.average(source["loss_box_reg"].detach().cpu()))
    storage["loss_objectness"].append(
        np.average(source["loss_objectness"].detach().cpu())
    )
    storage["loss_rpn_box_reg"].append(
        np.average(source["loss_rpn_box_reg"].detach().cpu())
    )
    return storage


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.train_loss = []
        self.train_losses = {
            "loss_classifier": [],
            "loss_box_reg": [],
            "loss_objectness": [],
            "loss_rpn_box_reg": [],
        }
        self.test_losses = {
            "loss_classifier": [],
            "loss_box_reg": [],
            "loss_objectness": [],
            "loss_rpn_box_reg": [],
        }
        self.test_loss = []

    def __call__(
        self, loss_dict_train, loss_dict_test, train_loss, val_loss, model, path
    ):
        self.store_loss(loss_dict_train, loss_dict_test, train_loss, val_loss)
        self.log_loss(train=False)

        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(
                (f"EarlyStopping counter: {self.counter} out of {self.patience}")
            )
            if self.counter >= self.patience:
                self.early_stop = True
                # self.plot_result()
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def log_loss(self, train=True) -> None:
        loss = self.train_losses if train else self.test_losses
        logger.info(
            """
            Classifier Loss: {0:.6f} --- Box-Reg Loss: {1:.6f} 
            Objectness Loss: {2:.6f} --- RPN-Box-Reg Loss: {2:.6f}
            """.format(
                loss["loss_classifier"][-1],
                loss["loss_box_reg"][-1],
                loss["loss_objectness"][-1],
                loss["loss_rpn_box_reg"][-1],
            )
        )

    def store_loss(self, loss_dict_train, loss_dict_test, train_loss, val_loss) -> None:
        self.train_losses["loss_classifier"].append(
            np.average(loss_dict_train["loss_classifier"])
        )
        self.train_losses["loss_box_reg"].append(
            np.average(loss_dict_train["loss_box_reg"])
        )
        self.train_losses["loss_objectness"].append(
            np.average(loss_dict_train["loss_objectness"])
        )
        self.train_losses["loss_rpn_box_reg"].append(
            np.average(loss_dict_train["loss_rpn_box_reg"])
        )
        self.test_losses["loss_classifier"].append(
            np.average(loss_dict_test["loss_classifier"])
        )
        self.test_losses["loss_box_reg"].append(
            np.average(loss_dict_test["loss_box_reg"])
        )
        self.test_losses["loss_objectness"].append(
            np.average(loss_dict_test["loss_objectness"])
        )
        self.test_losses["loss_rpn_box_reg"].append(
            np.average(loss_dict_test["loss_rpn_box_reg"])
        )
        # self.train_losses = store_losses(self.train_losses, loss_dict_train)
        # self.test_losses = store_losses(self.test_losses, loss_dict_test)
        self.train_loss.append(train_loss)
        self.test_loss.append(val_loss)

    def plot_result(self) -> None:

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
        axes[0].plot(
            self.train_losses.values(),
            label=[
                "loss_classifier",
                "loss_box_reg",
                "loss_objectness",
                "loss_rpn_box_reg",
            ],
        )
        axes[0].legend(loc="upper right")
        axes[1].plot(self.train_loss, label="train loss")
        axes[1].legend(loc="upper right")
        axes[2].plot(self.test, label="test loss")
        axes[2].legend(loc="upper right")

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            logger.info(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), f"{path}/checkpoint.pth")
        self.val_loss_min = val_loss


def log_train_progress(args, time_now, loss, epoch, train_steps, i, iter_count) -> None:
    logger.info(
        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item())
    )
    speed = (time.time() - time_now) / iter_count
    left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
    # logger.info("\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(speed, left_time))


def log_train_epoch(epoch, train_steps, train_loss, test_loss, scheduler) -> None:
    logger.info(
        "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Test Loss: {3:.7f} Learning Rate {4:.7f}".format(
            epoch + 1,
            train_steps,
            train_loss,
            test_loss,
            scheduler.optimizer.param_groups[0]["lr"],
        )
    )


#
# def log_loss(loss) -> None:
#    logger.info(
#        f"""Classifier Loss: {loss["loss_classifier"]} --- Box-Reg Loss: {loss["loss_box_reg"]}  \n
#            Objectness Loss: {loss["loss_objectness"]} --- RPN-Box-Reg Loss: {loss["loss_rpn_box_reg"]} """
#    )


def falsy_path(directory: str) -> bool:
    return bool(
        (
            directory.startswith(".")
            or directory.endswith("json")
            or directory.endswith("zip")
        )
    )


import csv


def write_to_csv(idx, image_name, image_width, image_height, label) -> None:
    """writes prection to csv,
    label: {'boxes': tensor([], size=(0, 4)),
             'labels': tensor([], dtype=torch.int64),
             'scores': tensor([]}
    """
    num_predictions = label["labels"].shape[0]
    with open("src/output/" + "submission.csv", "a") as submission:
        csv_writer = csv.writer(submission, delimiter=",")
        if num_predictions == 0:
            csv.writer.writerow()
        for i in range(num_predictions):
            label_num = label["labels"][i].item()
            if label_num != 0:
                idx += 1
                object_class_id = CLASSES_ID.index(label_num)
                object_class_name = CLASSES.index(label_num)
                boxes = label["boxes"][i, :]
                score = label["scores"][i].item()

                row = {
                    "detection_id": idx,
                    "image_name": image_name,
                    "image_width": image_width,
                    "image_height": image_height,
                    "object_class_id": object_class_id,
                    "object_class_name": object_class_name,
                    "bbox_left": boxes[0].item(),
                    "bbox_top": boxes[1].item(),
                    "bbox_right": boxes[2].item(),
                    "bbox_bottom": boxes[3].item(),
                    "confidence": score,
                }
                csv_writer.writerow(row)
    return idx
    # print(type(img))


import cv2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer


def show_predictions(
    predictor, dataset_name, path, num_predictions: int, save_path: str
) -> None:
    metadata = MetadataCatalog.get(dataset_name)
    images, image_names = get_eval_images(path)
    for idx in tqdm(
        random.sample(list(range(len(images))), num_predictions),
        total=num_predictions,
        leave=True,
    ):
        image_id = images[idx]
        im = cv2.imread(images[idx])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=0.8)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # file_path = os.path.join(save_path, f"prediction_{image_id}")
        file_path = os.path.join(
            "/Users/sebastian/Documents/Projects/sordi_ai/output/prediction_images",
            f"prediction_{image_id}.jpg",
        )
        directory = os.path.dirname("output/prediction_images")
        # if not os.path.exists(directory):
        #   os.makedirs(directory)

        #  if not cv2.imwrite(file_path, out.get_image()[:, :, ::-1]):
        #     raise Exception("Could not save image")

        cv2.imshow(str(image_id), out.get_image()[:, :, ::-1])

        cv2.waitKey()


def get_eval_images(path="data/eval/images"):
    images, image_names = [], []
    for image in sorted(os.listdir(path)):
        if falsy_path(image):
            continue
        if image.startswith("."):
            continue
        image_path = os.path.join(path, image)
        images.append(image_path)
        image_names.append(image)
    return images, image_names


def predict_images(predictor, path) -> None:
    idx = 0
    images, image_names = get_eval_images(path=path)
    with open("output/" + "submission.csv", "a+", newline="") as submission:
        csv_writer = csv.writer(submission, delimiter=",")
        entries = [
            "detection_id",
            "image_name",
            "image_width",
            "image_height",
            "object_class_id",
            "object_class_name",
            "bbox_left",
            "bbox_top",
            "bbox_right",
            "bbox_bottom",
            "confidence",
        ]
        csv_writer.writerow(entries)
    for image, image_name in tqdm(
        zip(images, image_names), total=len(images), leave=True
    ):
        im = cv2.imread(image)
        output = predictor(im)
        (
            num_instances,
            image_height,
            image_width,
            pred_boxes,
            scores,
            pred_classes,
        ) = transform_output(output)

        idx = detectron_write_to_csv(
            idx=idx,
            image_name=image_name,
            num_instances=num_instances,
            image_width=image_width,
            image_height=image_height,
            pred_boxes=pred_boxes,
            scores=scores,
            pred_classes=pred_classes,
        )


def transform_output(output):
    instance = output["instances"]

    num_instances = len(instance)
    image_height, image_width = instance.image_size
    fields = instance.get_fields()
    pred_boxes = fields["pred_boxes"].tensor.clone()  # Nx4 Boxes
    scores = fields["scores"]  # Tensor
    pred_classes = fields["pred_classes"]  # Tensor

    assert (
        num_instances == len(scores) == len(pred_classes) == len(pred_boxes)
    ), "danger for missmatch"
    return (
        num_instances,
        image_height,
        image_width,
        pred_boxes,
        scores,
        pred_classes,
    )


def detectron_write_to_csv(
    idx,
    image_name,
    num_instances,
    image_width,
    image_height,
    pred_boxes,
    scores,
    pred_classes,
) -> int:
    idx = idx
    with open("output/" + "submission.csv", "a+", newline="") as submission:
        csv_writer = csv.writer(submission, delimiter=",")
        if num_instances == 0:
            idx += 1
            object_class_id = CLASSES_ID_RE[3]
            object_class_name = CLASSES_RE[3 + 1]
            row = {
                "detection_id": idx,
                "image_name": image_name,
                "image_width": image_width,
                "image_height": image_height,
                "object_class_id": object_class_id,
                "object_class_name": object_class_name,
            }
            csv_writer.writerow(row.values())

        else:
            for i in range(num_instances):
                idx += 1
                box = pred_boxes[i]
                if check_area2(box=box, area_threshold=600):
                    score = scores[i] * 100
                    pred_class = pred_classes[i].item()
                    object_class_id = CLASSES_ID_RE[pred_class]
                    object_class_name = CLASSES_RE[
                        pred_class + 1
                    ]  # CLASSES.index(pred_class - 1)

                    row = {
                        "detection_id": idx,
                        "image_name": image_name,
                        "image_width": image_width,
                        "image_height": image_height,
                        "object_class_id": object_class_id,
                        "object_class_name": object_class_name,
                        "bbox_left": box[0].item(),
                        "bbox_top": box[1].item(),
                        "bbox_right": box[2].item(),
                        "bbox_bottom": box[3].item(),
                        "confidence": score.item(),
                    }
                    csv_writer.writerow(row.values())
    return idx


import os
import json


def check_area2(box, area_threshold):
    x1, y1, x2, y2 = (
        box[0],
        box[1],
        box[2],
        box[3],
    )
    return (x2 - x1) * (y2 - y1) > area_threshold


def check_area(label_path: str, area_threshold: float) -> bool:
    with open(label_path, "rb") as json_file:
        meta_data = json.load(json_file)
        removed_labels = 0
        # if all targets are
        for target in meta_data:
            x1, y1, x2, y2 = (
                target["Left"],
                target["Top"],
                target["Right"],
                target["Bottom"],
            )
            if (x2 - x1) * (y2 - y1) < area_threshold:
                removed_labels += 1
        return len(meta_data) > removed_labels


#    This function shows the transformed images from the `train_loader`.
#    Helps to check whether the tranformed images along with the corresponding
#    labels are correct or not.
#
#    """
#    colors = np.random.uniform(0, 1, size=(200, 3))
#    for _ in range(2):
#        index = random.randint(0, len(train_dataset) - 1)
#        images, targets = train_dataset[index]  # next(iter(train_loader))
#        # targets = dict_list_to_list_dict(targets)
#        #        targets = transform_target(targets)
#        print(targets)
#        boxes = targets["boxes"].cpu().numpy().astype(np.int32)
#        labels = targets["labels"].cpu().numpy().astype(np.int32)
#        # Get all the predicited class names.
#        classes_rev = {v: k for k, v in classes.items()}
#        pred_classes = [
#            classes_rev[label] for label in labels
#        ]  # classes_rev[labels]  # [classes_rev[label] for label in labels]
#        sample = images.permute(1, 2, 0).cpu().numpy()
#        sample = cv2.cvtColor(sample, cv2.COLOR_RGB2BGR)
#        for box_num, box in enumerate(boxes):
#            class_name = pred_classes[box_num]
#            color = colors[box_num]
#            cv2.rectangle(
#                sample, (box[0], box[1]), (box[2], box[3]), color, 2, cv2.LINE_AA
#            )
#            cv2.putText(
#                sample,
#                class_name,
#                (box[0], box[1] - 10),
#                cv2.FONT_HERSHEY_SIMPLEX,
#                1.0,
#                color,
#                2,
#                cv2.LINE_AA,
#            )
#        cv2.imshow("Transformed image", sample)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()


# row["detection_id"]
# def show_prediction(image, index: int, dataset: Optional) -> None:
#    model.eval()
#    image, _ = train_dataset[index]
#    img = train_dataset.get_raw_image(index)
#    prediction = model(image.unsqueeze(0))[0]
#    print(prediction)
#    labels = [weights.meta["categories"][i] for i in prediction["labels"]]
#    box = draw_bounding_boxes(
#        img,
#        boxes=prediction["boxes"],
#        labels=labels,
#        colors="red",
#        width=4,
#        font_size=30,
#    )
#    im = to_pil_image(box.detach())
#    im.show()
#
