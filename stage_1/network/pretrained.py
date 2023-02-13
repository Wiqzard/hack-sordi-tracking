import torch
import torchvision

from torchvision.io.image import read_image
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import torchvision.transforms as T

# import torchvision.detection

# import torchvision.prototype.transforms as T

weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)


def create_model(num_classes: int):
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.95)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # in_features = model.roi_heads.box_predictor.cls_score.in_features
    # model.roi_heads.box_predictor.cls_score = torch.nn.Linear(
    #     in_features=in_features, out_features=NUM_CLASSES, bias=True
    # )
    # model.roi_heads.box_predictor.bbox_pred = torch.nn.Linear(
    #     in_features=in_features, out_features=NUM_CLASSES*4, bias=True
    # )
    return model, weights


# def get_transform():
#    transforms = []
#    transforms.append(T.Resize((600, 1024)))
#    # transforms.append(T.PILToTensor())
#    transforms.append(T.ToTensor())
#    transforms.append(T.ConvertImageDtype(torch.float))
#    return T.Compose(transforms)
#


def get_transform():
    transforms = []
    transforms.append(T.PILToTensor())
    # transforms.append(T.Resize((600, 1024)))
    transforms.append(T.ConvertImageDtype(torch.float))
    return T.Compose(transforms)


# pretrained on the COCO set
# fastercnn_resnet50_fpn


# model = fasterrcnn_resnet50_fpn_v2(
#    pretrained=True
# )  # , prgress=True, num_classes=91)# , pretrained_backbone=True,)
"""
https://pytorch.org/vision/0.11/models.html#torchvision.models.detection.fasterrcnn_resnet50_fpn

input: [CHANNEL, HEIGHT, WIDTH] for each image
input during training: input tensors, targets with boxes and labels
output during training: Dict[Tensor] containing the calssification and regression losses 

INFERENCE: only input tensors, N for the number of detections
"""
"""images, boxes = torch.rand(4, 3, 600, 1200), torch.rand(4, 11, 4)
labels = torch.randint(1, 91, (4, 11))
images = list(image for image in images)
targets = []
for i in range(len(images)):
    print(i)
    d = {"boxes": boxes[i], "labels": labels[i]}
targets.append(d)
output = model(images, targets)
"""
