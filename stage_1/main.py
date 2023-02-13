import random
import torch
import argparse
import numpy as np
import os
import logging
import warnings

from detectron2.config import get_cfg
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    hooks,
    launch,
)
from detectron2.engine import DefaultPredictor

from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog


from detectron.src.utils import setup
from detectron.src.trainer import MyTrainer
from detectron.src.data_set import DataSet, register_dataset
from detectron.src.constants import CLASSES

warnings.filterwarnings("ignore")
logger = logging.getLogger("__name__")
level = logging.INFO
logger.setLevel(level)
ch = logging.StreamHandler()
ch.setLevel(level)
logger.addHandler(ch)


def main():  # sourcery skip: extract-method
    fix_seed = 1401
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description="Warehouse [Object Classification]")

    parser.add_argument(
        "--is_training", type=bool, required=True, default=True, help="status"
    )
    parser.add_argument(
        "--register_data",
        type=bool,
        required=True,
        default=True,
        help="register data for training and testing",
    )
    parser.add_argument(
        "--test", type=bool, required=True, default=False, help="test on val data"
    )
    parser.add_argument(
        "--inference",
        type=bool,
        required=False,
        default=False,
        help="make submission and plot imagesj",
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        required=False,
        default="model_final.pth",
        help="saved model to test",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="resume training from checkpoint",
    )
    # parser.add_argument(
    #    "--writer_period", type=int, required=False, default=100, help="period to log"
    # )
    #   <------------- data loader ------------->
    parser.add_argument(
        "--root_path",
        type=str,
        default="./data/",
        help="root path of the data directory",
    )
    parser.add_argument(
        "--ignore_redundant",
        action="store_true",
        default=False,
        help="ignore warehouse, on stack, on rack data",
    )
    parser.add_argument(
        "--partion_single_assets",
        type=int,
        default=1,
        help="only use every n'th image from single assets",
    )
    parser.add_argument(
        "--ratio", type=float, default=0.95, help="train-test split ratio"
    )
    parser.add_argument(
        "--area_threshold_min",
        type=int,
        default=3000,
        help="sort out boxes with smaller area",
    )
    parser.add_argument(
        "--area_threshold_max",
        type=int,
        default=700000,
        help="sort out boxes with bigger area",
    )

    #   <------------- trainer ------------->
    # parser.add_argument(
    #     "--datasets_train", type=str, default="data_train", help="train dataset name"
    # )
    # parser.add_argument(
    #     "--datasets_test", type=str, default="data_val", help="validation dataset name"
    # )
    parser.add_argument(
        "--model",
        type=str,
        default="COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml",
        help="modelzoo zaml file",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="dataloader number of workers"
    )
    parser.add_argument("--ims_per_batch", type=int, default=2, help="batch size")
    parser.add_argument("--base_lr", type=float, default=0.002, help="learning rate")
    parser.add_argument(
        "--max_iter", type=int, default=200000, help="iterations per epoch"
    )
    parser.add_argument(
        "--batch_per_img", type=int, default=512, help="roi heads per image"
    )
    parser.add_argument("--patience", type=int, default=1000, help="early stopping")
    parser.add_argument(
        "--eval_period", type=int, default=5000, help="after periods evaluate model"
    )
    parser.add_argument(
        "--checkpoint_period", type=int, default=5000, help="checkpoint after n periods"
    )
    parser.add_argument("--use_gpu", action="store_true", default=True, help="use gpu")
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="use automatic mixed precision",
    )
    parser.add_argument("--warmup_steps", type=int, default=1000, help="warmup steps")
    parser.add_argument(
        "--gamma", type=float, default=0.1, help="learning rate reducer"
    )
    args = parser.parse_args()
    args.use_gpu = bool(torch.cuda.is_available() and args.use_gpu)

    # <--------------------------------------------------------->

    logger.info("Args in experiment:")
    logger.info(args)

    cfg = setup(args)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # if args.register_data:
    # dataset = DataSet(args)
    # for d in ["train", "val"]:
    # logger.info(f">>>>>>> registering data_{d} >>>>>>> ")
    # DatasetCatalog.register(
    # f"data_{d}", lambda d=d: dataset.dataset_function(mode=d)
    # )
    # MetadataCatalog.get(f"data_{d}").set(thing_classes=CLASSES)

    #    trainer = DefaultTrainer(cfg)
    #
    #    trainer.resume_or_load(resume=args.resume)

    #    if args.is_training:
    #        logger.info(f">>>>>>> start training : {args.model} >>>>>>>>>>>>>>>>>>>>>>>>>>")
    #        # trainer = MyTrainer(cfg)
    #        trainer.train()
    #
    #    if args.test:
    #        logger.info(f">>>>>>> start testing: {args.model} >>>>>>>>>>>>>>>>>>>>>>>>>>")
    #        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, args.model_checkpoint)
    #        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
    #
    #        predictor = DefaultPredictor(cfg)
    #        evaluator = COCOEvaluator(
    #            "data_val", cfg, False, output_dir="./output/inference/"
    #        )
    #        val_loader = build_detection_test_loader(cfg, "data_val")
    #        inference_on_dataset(trainer.model, val_loader, evaluator)

    if args.inference:
        logger.info(">>>>>>> registering data_val >>>>>>> ")
        dataset = DataSet(args)
        d = "val"
        DatasetCatalog.register(
            "data_val", lambda d=d: dataset.dataset_function(mode=d)
        )
        MetadataCatalog.get("data_val").set(thing_classes=CLASSES)

        cfg.MODEL.WEIGHTS = "trained_models/trained_model_model_final.pth"  # os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        cfg.DATASETS.TEST = ("data_val",)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (
            0.45  # set the testing threshold for this model
        )
        cfg.TEST.DETECTIONS_PER_IMAGE = 1000
        predictor = DefaultPredictor(cfg)
        test_metadata = MetadataCatalog.get("data_val")
        dataset_dicts = DatasetCatalog.get("data_val")

        from utils.tools import predict_images, show_predictions

        # predict_images(predictor=predictor, path="data/eval/images")
        save_path = os.path.join(cfg.OUTPUT_DIR, "prediction_images")
        print("MAKE IMAGES")
        show_predictions(
            predictor=predictor,
            dataset_name="data_val",
            path="data/eval/images",
            num_predictions=10,
            save_path=save_path,
        )


if __name__ == "__main__":
    main()
