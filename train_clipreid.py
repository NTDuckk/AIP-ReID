from utils.logger import setup_logger
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
from solver.make_optimizer_prompt import make_optimizer_1stage, make_optimizer_2stage
from solver.scheduler_factory import create_scheduler
from solver.lr_scheduler import WarmupMultiStepLR
from loss.make_loss import make_loss
from processor.processor_clipreid_stage1 import do_train_stage1
from processor.processor_clipreid_stage2 import do_train_stage2
import random
import torch
import numpy as np
import os
import argparse
from config import cfg
import logging
import torch.nn as nn
from torch.cuda import amp


def strip_module_prefix(state_dict):
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned[k[len("module."):]] = v
        else:
            cleaned[k] = v
    return cleaned

def optimizer_to(optim, device):
    for state in optim.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def build_text_features_from_stage1_checkpoint(
    cfg,
    model,
    train_loader_stage1,
    stage1_ckpt_path,
    local_rank
):
    device = "cuda"
    logger = logging.getLogger("transreid.train")
    logger.info(f"Load stage1 checkpoint from: {stage1_ckpt_path}")

    # load stage1 weights into bare model first
    model.load_param(stage1_ckpt_path)

    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print("Using {} GPUs for prototype extraction".format(torch.cuda.device_count()))
            model = nn.DataParallel(model)

    core_model = model.module if isinstance(model, nn.DataParallel) else model
    model.eval()

    # giống Step 1 của stage1: cache image token features
    image_token_features = []
    labels = []

    with torch.no_grad():
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader_stage1):
            img = img.to(device)
            target = vid.to(device)

            with amp.autocast(enabled=True):
                cls_feats, token_feats = model(img, target, get_image=True)

            for pid, tok_f in zip(target, token_feats):
                labels.append(pid)
                image_token_features.append(tok_f.cpu())

    labels_list = torch.stack(labels, dim=0).cuda()
    image_token_features_list = torch.stack(image_token_features, dim=0).cuda()

    batch = cfg.SOLVER.STAGE1.IMS_PER_BATCH
    num_image = labels_list.shape[0]
    num_classes = labels_list.max().item() + 1

    logger.info("Recomputing averaged text features per ID from loaded stage1 checkpoint...")

    all_text_features = []
    with torch.no_grad():
        for i in range(0, num_image, batch):
            end = min(i + batch, num_image)
            image_token_feats = image_token_features_list[i:end]

            if core_model.s1_id_flag:
                with amp.autocast(enabled=True):
                    text_features_, text_feat_, text_score_ = model(
                        image_features_for_inversion=image_token_feats,
                        get_text_inversion=True
                    )
                all_text_features.append(text_feat_.float().cpu())
            else:
                with amp.autocast(enabled=True):
                    text_features_ = model(
                        image_features_for_inversion=image_token_feats,
                        get_text_inversion=True
                    )
                all_text_features.append(text_features_.float().cpu())

    all_text_features = torch.cat(all_text_features, dim=0)  # [N, D]

    avg_text_features = torch.zeros(num_classes, all_text_features.shape[-1])
    labels_cpu = labels_list.cpu()
    for c in range(num_classes):
        mask = (labels_cpu == c)
        if mask.sum() > 0:
            avg_text_features[c] = all_text_features[mask].mean(dim=0)

    avg_text_features = avg_text_features.cuda()
    logger.info("Loaded-stage1 prototypes ready, shape: {}".format(tuple(avg_text_features.shape)))
    return avg_text_features

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configs/person/vit_clipreid.yml", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--stage1_resume", default="", type=str,
                    help="path to stage1 checkpoint, e.g. ViT-B-16_stage1_120.pth")
    parser.add_argument("--stage2_resume", default="", type=str,
                    help="path to stage2 checkpoint with full training states")
    parser.add_argument("--local_rank", default=0, type=int)
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

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    train_loader_stage2, train_loader_stage1, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    stage2_checkpoint = None
    start_epoch_stage2 = 1
    resume_states = None

    if args.stage2_resume:
        logger.info("Resuming stage2 from checkpoint: {}".format(args.stage2_resume))
        stage2_checkpoint = torch.load(args.stage2_resume, map_location="cpu")
        model_state_dict = stage2_checkpoint.get("model_state_dict", stage2_checkpoint)
        model.load_state_dict(strip_module_prefix(model_state_dict))

        if "text_features" in stage2_checkpoint and stage2_checkpoint["text_features"] is not None:
            text_features = stage2_checkpoint["text_features"].float().cuda()
        else:
            if args.stage1_resume:
                logger.warning(
                    "stage2 checkpoint does not contain text_features. "
                    "Falling back to rebuilding text features from stage1 checkpoint: {}".format(args.stage1_resume)
                )
                text_features = build_text_features_from_stage1_checkpoint(
                    cfg,
                    model,
                    train_loader_stage1,
                    args.stage1_resume,
                    args.local_rank
                )
            else:
                raise ValueError(
                    "stage2 checkpoint does not contain text_features. "
                    "Please pass --stage1_resume to rebuild text_features, "
                    "or use a newer stage2 checkpoint that already stores text_features."
                )
    elif args.stage1_resume:
        text_features = build_text_features_from_stage1_checkpoint(
            cfg,
            model,
            train_loader_stage1,
            args.stage1_resume,
            args.local_rank
        )
    else:
        optimizer_1stage = make_optimizer_1stage(cfg, model)
        scheduler_1stage = create_scheduler(
            optimizer_1stage,
            num_epochs=cfg.SOLVER.STAGE1.MAX_EPOCHS,
            lr_min=cfg.SOLVER.STAGE1.LR_MIN,
            warmup_lr_init=cfg.SOLVER.STAGE1.WARMUP_LR_INIT,
            warmup_t=cfg.SOLVER.STAGE1.WARMUP_EPOCHS,
            noise_range=None
        )

        text_features = do_train_stage1(
            cfg,
            model,
            train_loader_stage1,
            optimizer_1stage,
            scheduler_1stage,
            args.local_rank
        )

    optimizer_2stage, optimizer_center_2stage = make_optimizer_2stage(cfg, model, center_criterion)
    scheduler_2stage = WarmupMultiStepLR(
        optimizer_2stage,
        cfg.SOLVER.STAGE2.STEPS,
        cfg.SOLVER.STAGE2.GAMMA,
        cfg.SOLVER.STAGE2.WARMUP_FACTOR,
        cfg.SOLVER.STAGE2.WARMUP_ITERS,
        cfg.SOLVER.STAGE2.WARMUP_METHOD
    )

    if stage2_checkpoint is not None:
        if "optimizer_state_dict" in stage2_checkpoint:
            optimizer_2stage.load_state_dict(stage2_checkpoint["optimizer_state_dict"])
        if "optimizer_center_state_dict" in stage2_checkpoint:
            optimizer_center_2stage.load_state_dict(stage2_checkpoint["optimizer_center_state_dict"])
        if "scheduler_state_dict" in stage2_checkpoint:
            scheduler_2stage.load_state_dict(stage2_checkpoint["scheduler_state_dict"])

        device = torch.device("cuda", args.local_rank)
        optimizer_to(optimizer_2stage, device)
        optimizer_to(optimizer_center_2stage, device)

        start_epoch_stage2 = int(stage2_checkpoint.get("epoch", 0)) + 1
        resume_states = {
            "scaler_state_dict": stage2_checkpoint.get("scaler_state_dict", None)
        }
        logger.info("Stage2 resume start epoch: {}".format(start_epoch_stage2))

        start_epoch_stage2 = int(stage2_checkpoint.get("epoch", 0)) + 1
        resume_states = {
            "scaler_state_dict": stage2_checkpoint.get("scaler_state_dict", None)
        }
        logger.info("Stage2 resume start epoch: {}".format(start_epoch_stage2))

    do_train_stage2(
        cfg,
        model,
        center_criterion,
        train_loader_stage2,
        val_loader,
        optimizer_2stage,
        optimizer_center_2stage,
        scheduler_2stage,
        loss_func,
        num_query,
        args.local_rank,
        precomputed_text_features=text_features,
        start_epoch=start_epoch_stage2,
        resume_states=resume_states
    )
