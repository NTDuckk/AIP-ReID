import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from torch.nn import functional as F
from loss.supcontrast import SupConLoss

def do_train_stage2(cfg,
             model,
             center_criterion,
             train_loader_stage2,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank,
             precomputed_text_features=None,
             start_epoch=1,
             resume_states=None):
    log_period = cfg.SOLVER.STAGE2.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.STAGE2.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.STAGE2.EVAL_PERIOD
    instance = cfg.DATALOADER.NUM_INSTANCE

    device = "cuda"
    epochs = cfg.SOLVER.STAGE2.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
            num_classes = model.module.num_classes
        else:
            num_classes = model.num_classes

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    if resume_states is not None and resume_states.get("scaler_state_dict") is not None:
        scaler.load_state_dict(resume_states["scaler_state_dict"])
        logger.info("Loaded AMP scaler state from stage2 checkpoint")
    xent = SupConLoss(device)

    from datetime import timedelta
    all_start_time = time.monotonic()

    if precomputed_text_features is not None:
        # text_features = precomputed_text_features.to(device).float()
        text_features = precomputed_text_features.cuda()
        logger.info(
            "Using precomputed stage1 prototypes for stage2, shape: {}".format(tuple(text_features.shape))
        )
    else:
        raise ValueError("precomputed_text_features is required for stage2 training/resume.")
    # else:
    #     logger.info("No precomputed stage1 prototypes provided. Recomputing text prototypes at stage2 start.")
    #     model.eval()

    #     text_feature_sum = None
    #     text_feature_count = torch.zeros(num_classes, device=device, dtype=torch.float32)

    #     with torch.no_grad():
    #         for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader_stage2):
    #             img = img.to(device)
    #             target = vid.to(device)

    #             if cfg.MODEL.SIE_CAMERA:
    #                 target_cam = target_cam.to(device)
    #             else:
    #                 target_cam = None

    #             if cfg.MODEL.SIE_VIEW:
    #                 target_view = target_view.to(device)
    #             else:
    #                 target_view = None

    #             with amp.autocast(enabled=True):
    #                 pid_list, _, mean_text_feature = model(
    #                     x=img,
    #                     label=target,
    #                     get_text_inversion_stage2=True,
    #                     cam_label=target_cam,
    #                     view_label=target_view
    #                 )

    #             pid_list = pid_list.long()
    #             mean_text_feature = mean_text_feature.float()

    #             if text_feature_sum is None:
    #                 feat_dim = mean_text_feature.size(1)
    #                 text_feature_sum = torch.zeros(
    #                     num_classes,
    #                     feat_dim,
    #                     device=device,
    #                     dtype=mean_text_feature.dtype
    #                 )

    #             text_feature_sum.index_add_(0, pid_list, mean_text_feature)
    #             text_feature_count.index_add_(
    #                 0,
    #                 pid_list,
    #                 torch.ones(pid_list.size(0), device=device, dtype=text_feature_sum.dtype)
    #             )

    #     text_features = torch.zeros_like(text_feature_sum)
    #     valid_mask = text_feature_count > 0
    #     text_features[valid_mask] = (
    #         text_feature_sum[valid_mask] /
    #         text_feature_count[valid_mask].unsqueeze(1)
    #     )

    for epoch in range(start_epoch, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()

        scheduler.step()

        model.train()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader_stage2):
            optimizer.zero_grad()
            optimizer_center.zero_grad()

            img = img.to(device)
            target = vid.to(device)

            if cfg.MODEL.SIE_CAMERA:
                target_cam = target_cam.to(device)
            else:
                target_cam = None

            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else:
                target_view = None

            with amp.autocast(enabled=True):
                score, feat, image_features = model(
                    x=img,
                    label=target,
                    cam_label=target_cam,
                    view_label=target_view
                )
                logits = image_features @ text_features.t()
                loss = loss_fn(score, feat, target, target_cam, logits)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()

            acc = (logits.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info(
                    "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}".format(
                        epoch, (n_iter + 1), len(train_loader_stage2),
                        loss_meter.avg, acc_meter.avg, scheduler.get_lr()[0]
                    )
                )

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(
                    epoch, time_per_batch, train_loader_stage2.batch_size / time_per_batch
                )
            )

        if epoch % checkpoint_period == 0:
            model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'optimizer_center_state_dict': optimizer_center.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'text_features': text_features.detach().cpu(),
                'cfg_dump': cfg.dump(),
            }
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(
                        checkpoint,
                        os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch))
                    )
            else:
                torch.save(
                    checkpoint,
                    os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch))
                )

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            if cfg.MODEL.SIE_CAMERA:
                                camids = camids.to(device)
                            else:
                                camids = None
                            if cfg.MODEL.SIE_VIEW:
                                target_view = target_view.to(device)
                            else:
                                target_view = None
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        if cfg.MODEL.SIE_CAMERA:
                            camids = camids.to(device)
                        else:
                            camids = None
                        if cfg.MODEL.SIE_VIEW:
                            target_view = target_view.to(device)
                        else:
                            target_view = None
                        feat = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Total running time: {}".format(total_time))
    print(cfg.OUTPUT_DIR)


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            if cfg.MODEL.SIE_CAMERA:
                camids = camids.to(device)
            else:
                camids = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else:
                target_view = None
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]