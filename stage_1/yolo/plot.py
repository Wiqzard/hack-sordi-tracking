import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image, ImageDraw

CLASSES_RE = {
    1: "stillage_close",
    2: "stillage_open",
    3: "l_klt_6147",
    4: "l_klt_8210",
    5: "l_klt_4147",
    6: "pallet",
    7: "jack",
    8: "forklift",
    9: "str",
    10: "bicycle",
    11: "dolly",
    12: "exit_sign",
    13: "fire_extinguisher",
    14: "spring_post",
    15: "locker",
    16: "cabinet",
    17: "cardboard_box",
}


def plot_bounding_box(image, annotation_list):
    annotations = np.array(annotation_list)
    w, h = image.size

    plotted_image = ImageDraw.Draw(image)

    transformed_annotations = np.copy(annotations)
    transformed_annotations[:, [1, 3]] = annotations[:, [1, 3]] * w
    transformed_annotations[:, [2, 4]] = annotations[:, [2, 4]] * h

    transformed_annotations[:, 1] = transformed_annotations[:, 1] - (
        transformed_annotations[:, 3] / 2
    )
    transformed_annotations[:, 2] = transformed_annotations[:, 2] - (
        transformed_annotations[:, 4] / 2
    )
    transformed_annotations[:, 3] = (
        transformed_annotations[:, 1] + transformed_annotations[:, 3]
    )
    transformed_annotations[:, 4] = (
        transformed_annotations[:, 2] + transformed_annotations[:, 4]
    )

    for ann in transformed_annotations:
        obj_cls, x0, y0, x1, y1 = ann
        plotted_image.rectangle(((x0, y0), (x1, y1)), width=2)

        plotted_image.text((x0, y0 - 10), CLASSES_RE[(int(obj_cls))])

    plt.imshow(np.array(image))
    plt.show()


# Get any random annotation file
num = random.randint(0, 100)
mode = random.randint(1, 3)
annotation_file = f"data/test_destination/dataset_{mode}/train/labels/{num}"  # random.choice(annotations)
with open(annotation_file, "r") as file:
    annotation_list = file.read().split("\n")[:-1]
    annotation_list = [x.split(" ") for x in annotation_list]
    annotation_list = [[float(y) for y in x] for x in annotation_list]

# Get the corresponding image file
image_file = f"data/test_destination/dataset_{mode}/train/images/{num}.jpg"  # annotation_file.replace("labels", "images").replace("txt", "png")
assert os.path.exists(image_file)

# Load the image
image = Image.open(image_file)

# Plot the Bounding Box
plot_bounding_box(image, annotation_list)
