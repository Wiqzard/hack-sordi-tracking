import logging
import argparse
import os
import torch
import numpy as np
import random
import sys
from exp.exp_main import Exp_Main
from utils.tools import logger
from pathlib import Path


def main():
    fix_seed = 1401
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description="Warehouse [Object Classification]")

    # basic config
    parser.add_argument(
        "--is_training", type=int, required=True, default=1, help="status"
    )
    parser.add_argument(
        "--model_id", type=str, required=True, default="test", help="model id"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        default="Faster_RCNN",
        help="model name, options: [Faster_RCNN]",
    )

    # data loader
    parser.add_argument(
        "--data", type=str, required=True, default="sordi.ai", help="dataset type"
    )
    parser.add_argument(
        "--root_path", type=str, default="./data/", help="root path of the data file"
    )
    # parser.add_argument(
    #    "--data_path", type=str, default="SORDI_2022_Single_Assets", help="data file"
    # )
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
        "--ratio", type=float, default=0.8, help="train-test split ratio"
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="./checkpoints/",
        help="location of model checkpoints",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size of train input data"
    )
    parser.add_argument(
        "--num_workers", type=int, default=10, help="data loader num workers"
    )
    # model define

    # optimization
    parser.add_argument("--itr", type=int, default=2, help="experiments times")
    parser.add_argument("--train_epochs", type=int, default=10, help="train epochs")
    parser.add_argument(
        "--patience", type=int, default=3, help="early stopping patience"
    )
    parser.add_argument(
        "--optimizer", type=str, default="adam", help="optimizer: [adam, sgd] "
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum for sgd ")
    parser.add_argument(
        "--weight_decay", type=float, default=0.0005, help="weight decay for sgd "
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001, help="optimizer learning rate"
    )
    parser.add_argument("--des", type=str, default="test", help="exp description")
    parser.add_argument("--loss", type=str, default="mse", help="loss function")
    parser.add_argument(
        "--lradj", type=str, default="type1", help="adjust learning rate"
    )
    # parser.add_argument(
    #     "--use_amp", type=bool, default=False, help="use automatic mixed precision"
    # )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="use automatic mixed precision training",
        default=False,
    )

    # GPU
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    # parser.add_argument(
    #    "--use_multi_gpu", action="store_true", help="use multiple gpus", default=False
    # )
    # parser.add_argument(
    #    "--devices", type=str, default="0,1,2,3", help="device ids of multile gpus"
    # )

    args = parser.parse_args()

    args.use_gpu = bool(torch.cuda.is_available() and args.use_gpu)

    logger.info("Args in experiment:")
    logger.info(args)
    # Path(args.checkpoints).mkdir(parents=True, exist_ok=True)
    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            setting = f"{args.model_id}_{args.model}_{args.data}_{args.des}_{ii}"

            exp = Exp(args)  # set experiments
            logger.info(
                f">>>>>>> start training : {setting} >>>>>>>>>>>>>>>>>>>>>>>>>>"
            )
            exp.train(setting)

            # logger.info(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            # exp.test(setting)

            if args.do_predict:
                logger.info(
                    f">>>>>>> predicting : {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
                )
                exp.predict(setting, True)

        ii = 0
        # setting = f"{args.model_id}_{args.model}_{args.data}_{args.des}_{ii}"

        # exp = Exp(args)  # set experiments
        # logger.info(f" >>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< ")
        # exp.test(setting, test=1)
        torch.cuda.empty_cache()

    else:
        exp = Exp(args)
        setting = f"{args.model_id}_{args.model}_{args.data}_{args.des}"
        logger.info(f">>>>>>> evaluating: {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        exp.evaluation(setting)


if __name__ == "__main__":
    main()
