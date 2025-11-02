"""
Training script for NOCS-based 6DoF pose estimation.
"""

import logging
import os
from pathlib import Path

import torch
from configs.nocs_config import add_nocs_config
from data.dataset_mapper import NOCSDatasetMapper
from data.register_dataset import register_nocs_datasets
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import DatasetEvaluators
from detectron2.utils import comm
from evaluation.nocs_evaluator import NOCSEvaluator
from models.nocs_head import NOCSHead

# Import our custom modules
from models.nocs_rcnn import NOCSROIHeads

logger = logging.getLogger("detectron2")


class NOCSTrainer(DefaultTrainer):
    """
    Custom trainer for NOCS model.

    Extends Detectron2's DefaultTrainer to use our custom:
    - Dataset mapper
    - Evaluator
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """Build custom NOCS evaluator."""
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

        return NOCSEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        """Build training dataloader with custom mapper."""
        from detectron2.data import build_detection_train_loader

        mapper = NOCSDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """Build test dataloader with custom mapper."""
        from detectron2.data import build_detection_test_loader

        mapper = NOCSDatasetMapper(cfg, is_train=False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    from detectron2 import model_zoo

    cfg = get_cfg()
    add_nocs_config(cfg)  # Add our custom config first

    # Load base Mask R-CNN config from model zoo
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )

    # Now merge our custom config
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    # Setup logger
    logger = logging.getLogger("detectron2")
    logger.setLevel(logging.INFO)

    return cfg


def main(args):
    cfg = setup(args)

    # Register datasets
    register_nocs_datasets(data_dir=args.data_dir, splits=["train", "val"])

    if args.eval_only:
        # Evaluation only
        model = NOCSTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = NOCSTrainer.test(cfg, model)
        return res

    # Training
    trainer = NOCSTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to directory containing HDF5 dataset files",
    )
    args = parser.parse_args()

    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
