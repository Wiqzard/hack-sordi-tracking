import itertools
import json
import os
from typing import List, Dict
import shutil
from tqdm import tqdm
import argparse

# if ran from on directory up


# CLASS_NAMES = ["klt_box_empty", "klt_box_full", "rack_1", "rack_2", "rack_3"]
# CLASS_IDS = [1008, 1009, 1200, 1205, 1210]
# CLASSES = dict(zip(CLASS_NAMES, CLASS_IDS))
# CLASSES_ID = {ID : i for i, ID in enumerate(CLASS_IDS)}
# CLASSES_ID_RE = {value: key for key, value in CLASSES_ID.items()}
# CLASSES_RE = {value: key for key, value in CLASSES.items()}
#
#
# reversed_model_classes = {
#    model_num: {value: key for key, value in classes.items()}
#    for model_num, classes in model_classes.items()
# }
# class_name_to_id = {v: k for (k, v) in zip(CLASSES_ID.keys(), CLASSES_RE.values())}
CLASSES = {
    "stillage_close": 0,
    "stillage_open": 2 - 1,
    "l_klt_6147": 3 - 1,
    "l_klt_8210": 4 - 1,
    "l_klt_4147": 5 - 1,
    "pallet": 6 - 1,
    "jack": 7 - 1,
    "forklift": 8 - 1,
    "str": 9 - 1,
    "bicycle": 10 - 1,
    "dolly": 11 - 1,
    "exit_sign": 12 - 1,
    "fire_extinguisher": 13 - 1,
    "spring_post": 14 - 1,
    "locker": 15 - 1,
    "cabinet": 16 - 1,
    "cardboard_box": 17 - 1,
}
CLASSES = ["stillage_close","stillage_open","l_klt_6147","l_klt_8210","l_klt_4147","pallet","jack","forklift","str","bicycle","dolly","exit_sign","fire_extinguisher","spring_post","locker","cabinet","cardboard_box"]
CLASSES_ID = {
    1002: 0,
    1003: 2 - 1,
    1012: 3 - 1,
    1013: 4 - 1,
    1011: 5 - 1,
    1100: 6 - 1,
    1120: 7 - 1,
    2010: 8 - 1,
    2050: 9 - 1,
    2000: 10 - 1,
    1110: 11 - 1,
    4000: 12 - 1,
    5010: 13 - 1,
    1135: 14 - 1,
    1030: 15 - 1,
    1040: 16 - 1,
    1070: 17 - 1,
}

model_classes = {
    1: {
        0: "stillage_close",
        1: "stillage_open",
        2: "l_klt_6147",
        3: "l_klt_8210",
        4: "l_klt_4147",
        5: "pallet",
        6: "jack",
        7: "forklift",
        8: "str",
        9: "bicycle",
        10: "dolly",
        11: "exit_sign",
        12: "fire_extinguisher",
        13: "spring_post",
        14: "locker",
        15: "cabinet",
        16: "cardboard_box",
    }
}
CLASS_NAMES = CLASSES.keys()
CLASS_IDS = CLASSES_ID.keys()
CLASSES = dict(zip(CLASS_NAMES, CLASS_IDS))
CLASSES_ID = {ID: i for i, ID in enumerate(CLASS_IDS)}
CLASSES_ID_RE = {value: key for key, value in CLASSES_ID.items()}
CLASSES_RE = {value: key for key, value in CLASSES.items()}


reversed_model_classes = {
    model_num: {value: key for key, value in classes.items()}
    for model_num, classes in model_classes.items()
}
class_name_to_id = {v: k for (k, v) in zip(CLASSES_ID.keys(), CLASSES_RE.values())}


def falsy_path(directory: str) -> bool:
    """
    If the directory starts with a dot, ends with json, or ends with zip, then it's a falsy path

    :param directory: str
    :type directory: str
    :return: A boolean value.
    """
    return bool(
        (
            directory.startswith(".")
            or directory.endswith("json")
            or directory.endswith("zip")
        )
    )


# def check_area2(box, area_threshold):
#    x1, y1, x2, y2 = (
#        box[0],
#        box[1],
#        box[2],
#        box[3],
#    )
#    return (x2 - x1) * (y2 - y1) > area_threshold


def make_dataset(
    root_path,
    target_path,
    partition_assets,
    area_min,
    area_max,
    mode: str = "train",
) -> List[Dict]:
    """Create Folder dataset_{model}
    Create Folder train, eval
    In both create folder images, labels
    Choose image, based on labels
    Choose labels based on which classifications
    Transform boxes to xyhw
    Put image in train, label as txt in label"""

    # root_path/data/train or eval

    ######################## STAGE 1
    #    source_path = os.path.join(root_path, mode)
    source_path = root_path
    ########################

    # target_path/dataset_1,2,3/train, eval
    assert mode in {"train", "eval"}, "specified mode does not exist"

    destination_path = target_path
    os.makedirs(destination_path, exist_ok=True)
    # target_path/dataset_1/train/images, labels

    train_images_destination_path = os.path.join(destination_path, "train", "images")
    train_labels_destination_path = os.path.join(destination_path, "train", "labels")

    val_images_destination_path = os.path.join(destination_path, "val", "images")
    val_labels_destination_path = os.path.join(destination_path, "val", "labels")

    os.makedirs(train_images_destination_path, exist_ok=True)
    os.makedirs(train_labels_destination_path, exist_ok=True)

    os.makedirs(val_images_destination_path, exist_ok=True)
    os.makedirs(val_labels_destination_path, exist_ok=True)

    make_train_dataset(
        source_path,
        partition_assets,
        area_min,
        area_max,
        train_images_destination_path,
        train_labels_destination_path,
    )


def get_annotations(
    label_path: str, model: int, area_min: int, area_max: int
) -> List[Dict]:
    image_width, image_height = 1280, 720
    with open(label_path, "rb") as json_file:
        labels = json.load(json_file)
        annotations = []
        for label in labels:
            x1, y1, x2, y2 = (
                label["Left"] + 1,
                label["Top"] + 1,
                label["Right"] + 1,
                label["Bottom"] + 1,
            )
            if area_min > (x2 - x1) * (y2 - y1) or (x2 - x1) * (y2 - y1) > area_max:
                continue
            # prediction_class = CLASSES_ID[int(label["ObjectClassId"])]
            #            classes = model_classes[model]
            classes = reversed_model_classes[model]
            if label["ObjectClassName"] not in classes.keys():
                continue
            prediction_class = classes[label["ObjectClassName"]]
            # if prediction_class not in models[model]:
            #    continue
            w = (x2 - x1) / image_width
            h = (y2 - y1) / image_height
            x_center = (x1 + (x2 - x1) / 2) / image_width
            y_center = (y1 + (y2 - y1) / 2) / image_height
            # class x_center y_center width height
            annotation = {
                "prediction_class": prediction_class,
                "x_center": x_center,
                "y_center": y_center,
                "width": w,
                "height": h,
            }
            annotations.append(annotation)
    return annotations


def make_train_dataset(
    source_path,
    partition_assets,
    area_min,
    area_max,
    images_destination_path,
    labels_destination_path,
):
    model = 1
    idx = 1
    step = 1
    # for dir in os.listdir(source_path):
    #    # dir Single_Assets Plant etc..
    #    if falsy_path(directory=dir):
    #        continue
    #    path_to_data_part = os.path.join(source_path, dir)
    #    print(dir)
    path_to_data_part = source_path
    for directory in sorted(os.listdir(path_to_data_part)):
        # Bicycle etc.
        if falsy_path(directory=directory):
            continue

        print(directory)
        directory = os.fsdecode(directory)
        # images, labels
        images_path = os.path.join(path_to_data_part, directory, "images")
        labels_path = os.path.join(path_to_data_part, directory, "labels/json")
        print(images_path)
        images_paths = sorted(os.listdir(images_path))
        labels_paths = sorted(os.listdir(labels_path))

        for image, label in zip(images_paths, labels_paths):

            if image.startswith(".") or label.startswith("."):
                continue
            if (
                dir in ["SORDI_2022_Single_Assets", "SORDI_2022_Regensburg_plant"]
                and step % partition_assets != 0
            ):
                step += 1
                continue
            image_name, label_name = image.split(".")[0], label.split(".")[0]

            if image_name == label_name:
                #                    label = os.path.join((idx))
                image_path = os.path.join(images_path, image)
                label_path = os.path.join(labels_path, label)

                annotations = get_annotations(label_path, model, area_min, area_max)
                if len(annotations) == 0:
                    continue
                image_destination_path = os.path.join(
                    images_destination_path, f"{str(idx)}.jpg"
                )
                label_destination_path = os.path.join(
                    labels_destination_path, f"{str(idx)}.txt"
                )
                if not os.path.exists(image_destination_path):
                    shutil.copy(image_path, image_destination_path)
                write_labels(annotations, label_destination_path)
                idx += 1
                step += 1


def make_eval_dataset(
    source_path, model, images_destination_path, labels_destination_path
):
    idx = 1
    images_path = os.path.join(source_path, "dataset", "images")
    labels_path = os.path.join(source_path, "dataset", "labels", "json")

    images_paths = sorted(os.listdir(images_path))
    labels_paths = sorted(os.listdir(labels_path))

    for image, label in zip(images_paths, labels_paths):
        if image.startswith(".") or label.startswith("."):
            continue
        image_name, label_name = image.split(".")[0], label.split(".")[0]

        assert image_name == label_name, "image, label missmatch"
        image_path = os.path.join(images_path, image)
        label_path = os.path.join(labels_path, label)

        image_destination_path = os.path.join(
            images_destination_path, f"{str(idx)}.jpg"
        )
        label_destination_path = os.path.join(
            labels_destination_path, f"{str(idx)}.txt"
        )
        #        if not delete_rows(label_path, label_destination_path, model):
        #            continue

        if not os.path.exists(os.path.join(image_destination_path)):
            shutil.copy(image_path, image_destination_path)
        if not os.path.exists(os.path.join(label_destination_path)):
            shutil.copy(label_path, label_destination_path)
        idx += 1


def delete_rows(file_name, destination_name, model):

    if not os.path.exists(file_name):
        print("Error: file does not exist.")
        return False

    with open(file_name, "r") as input_file:
        lines = input_file.readlines()

    with open(destination_name, "w+") as output_file:
        num_lines = 0
        for line in lines:
            values = line.split(" ")
            if int(values[0]) in models[model]:
                mapped_class = str(
                    reversed_model_classes[model][CLASSES_RE[int(values[0])]]
                )
                new_line = f"{mapped_class} " + " ".join(values[1:])
                output_file.write(new_line)
                num_lines += 1
        if num_lines != 0:
            return True
        if os.path.exists(destination_name):
            os.remove(destination_name)
        return False


def write_labels(annotations, label_destination_path):
    for annotation in annotations:
        with open(label_destination_path, "a+") as text_writer:
            dict_entry = {
                "prediction_class": annotation["prediction_class"],
                "x_center": annotation["x_center"],
                "y_center": annotation["y_center"],
                "w": annotation["width"],
                "h": annotation["height"],
            }
            text = " ".join(map(str, list(dict_entry.values()))) + "\n"

            text_writer.write(text)


def make_all_datasets(args):
    modes = ["train", "eval"]
    # with tqdm(total=1, leave=True) as pbar:
    #    for model, mode in itertools.product(models.keys(), modes):
    #        make_dataset(
    #            args.source,
    #            args.destination,
    #            partition_assets=args.partition_assets,
    #            area_min=args.area_threshold_min,
    #            area_max=args.area_threshold_max,
    #            mode=mode,
    #            model=model,
    #        )
    #        pbar.update(1)
    make_dataset(
        args.source,
        args.destination,
        partition_assets=args.partition_assets,
        area_min=args.area_threshold_min,
        area_max=args.area_threshold_max,
        mode="train",
    )


# make_dataset("data/test_source", "data/test_destination", mode="train", model=3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        default="data/test_source",
        help="source data path ",
    )
    parser.add_argument(
        "--destination",
        type=str,
        required=True,
        default="data/test_destinatin",
        help="destination data path",
    )
    parser.add_argument(
        "--area_threshold_min",
        type=int,
        required=False,
        default=2000,
        help="The minimum area threshold",
    )
    parser.add_argument(
        "--area_threshold_max",
        type=int,
        required=False,
        default=800000,
        help="The maximum area threshold",
    )
    parser.add_argument(
        "--partition_assets",
        type=int,
        required=False,
        default=1,
        help="The number of partitioned assets",
    )

    args = parser.parse_args()
    make_all_datasets(args)


if __name__ == "__main__":
    main()
