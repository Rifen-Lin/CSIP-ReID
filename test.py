import os
import argparse
import torch
import numpy as np
import random
import warnings

warnings.filterwarnings("ignore")

from config import cfg
from datasets.make_dataloader import make_eval_dense_dataloader, make_dataloader
from model.make_model import make_model
from processor.processor_stage2 import do_inference_dense, do_inference_rrs
from utils.logger import setup_logger


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument("--config_file", default="configs/xxx/xxx.yml", help="path to config file", type=str)
    parser.add_argument(
        "opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER
    )
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("CSIP-ReID", output_dir, if_train=False)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, "r") as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID

    (
        train_loader_stage2,
        train_loader_stage1_bs1,
        train_loader_stage1_bs,
        val_loader,
        num_query,
        num_classes,
        camera_num,
        view_num,
    ) = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    model.load_param("./logs/mars/stage2/ckpts/stage2_best.pth")

    do_inference_rrs(cfg, model, val_loader, num_query)

    # val_loader, num_query, num_classes, camera_num, view_num = make_eval_dense_dataloader(cfg)
    # do_inference_dense(cfg, model, val_loader, num_query)
