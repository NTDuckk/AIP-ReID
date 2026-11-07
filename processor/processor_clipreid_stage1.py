import logging
import os
import time
from datetime import timedelta

import torch
import torch.nn as nn
from torch.cuda import amp
import torch.distributed as dist
from torch.nn import functional as F

from utils.meter import AverageMeter
from loss.supcontrast import SupConLoss
from loss.softmax_loss import CrossEntropyLabelSmooth

def do_train_stage1(cfg,
             model,
             train_loader_stage1,
             optimizer,
             scheduler,
             local_rank):
    checkpoint_period = cfg.SOLVER.STAGE1.CHECKPOINT_PERIOD
    epochs = cfg.SOLVER.STAGE1.MAX_EPOCHS
    log_period = cfg.SOLVER.STAGE1.LOG_PERIOD
    device = "cuda"

    logger = logging.getLogger("transreid.train")
    logger.info("start training stage1 (inversion prompt learning)")

    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print("Using {} GPUs for training".format(torch.cuda.device_count()))
            model = nn.DataParallel(model)

    core_model = model.module if isinstance(model, nn.DataParallel) else model

    loss_meter = AverageMeter()
    loss_i2t_meter = AverageMeter()
    loss_t2i_meter = AverageMeter()

    scaler = amp.GradScaler()
    xent = SupConLoss(device)
    xent_id = CrossEntropyLabelSmooth(num_classes=model.module.num_classes if hasattr(model, 'module') else model.num_classes)
    
    all_start_time = time.monotonic()
    logger.info("model: {}".format(model))

    # Step 1: Pre-extract image features (frozen image encoder)
    image_cls_features = []
    image_token_features = []
    labels = []
    with torch.no_grad():
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader_stage1):
            img = img.to(device)
            target = vid.to(device)
            with amp.autocast(enabled=True):
                cls_feats, token_feats  = model(img, target, get_image = True)
                for pid, cls_f, tok_f in zip(target, cls_feats, token_feats):
                    labels.append(pid)
                    image_cls_features.append(cls_f.cpu())
                    image_token_features.append(tok_f.cpu())
        labels_list = torch.stack(labels, dim=0).cuda()
        image_cls_features_list = torch.stack(image_cls_features, dim=0).cuda()        # [N, D]
        image_token_features_list = torch.stack(image_token_features, dim=0).cuda()
        batch = cfg.SOLVER.STAGE1.IMS_PER_BATCH
        num_image = labels_list.shape[0]
        i_ter = num_image // batch
    del labels, image_cls_features, image_token_features

    # Step 2: Train inversion networks with contrastive loss
    for epoch in range(1, epochs + 1):
        loss_meter.reset()
        scheduler.step(epoch)
        model.train()

        iter_list = torch.randperm(num_image).to(device)
        for i in range(i_ter+1):
            optimizer.zero_grad()
            if i != i_ter:
                b_list = iter_list[i*batch:(i+1)* batch]
            else:
                b_list = iter_list[i*batch:num_image]
            
            target = labels_list[b_list]
            image_cls_feats = image_cls_features_list[b_list]        # [B, D]
            image_token_feats = image_token_features_list[b_list]    # [B, 1+Np, D]

            if core_model.s1_id_flag:
                with amp.autocast(enabled=True):
                    text_features, text_feat , text_score = model(
                        image_features_for_inversion=image_token_feats,
                        get_text_inversion=True
                    )
                loss_i2t = xent(image_cls_feats, text_feat, target, target)
                loss_t2i = xent(text_feat, image_cls_feats, target, target)
                loss_id = xent_id(text_score, target)
                loss = loss_i2t + loss_t2i + loss_id 
            else:
                with amp.autocast(enabled=True):
                    text_features = model(
                        image_features_for_inversion=image_token_feats,
                        get_text_inversion=True
                    )
                loss_i2t = xent(image_cls_feats, text_features, target, target)
                loss_t2i = xent(text_features, image_cls_feats, target, target)
                loss = loss_i2t + loss_t2i
            

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), b_list.shape[0])

            torch.cuda.synchronize()
            if (i + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (i + 1), i_ter + 1,
                                    loss_meter.avg, scheduler._get_lr(epoch)[0]))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_stage1_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_stage1_{}.pth'.format(epoch)))

    # Step 3: Compute averaged text features per ID
    logger.info("Computing averaged text features per ID...")
    num_classes = labels_list.max().item() + 1
    model.eval()
    all_text_features = []
    with torch.no_grad():
        for i in range(0, num_image, batch):
            end = min(i + batch, num_image)
            image_token_feats = image_token_features_list[i:end]
            if core_model.s1_id_flag:
                with amp.autocast(enabled=True):
                    text_features_, text_feat_ , text_score_  = model(
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
    all_text_features = torch.cat(all_text_features, dim=0)  # [N, proj_dim]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
    avg_text_features = torch.zeros(num_classes, all_text_features.shape[-1])
    for c in range(num_classes):
        mask = (labels_list.cpu() == c)
        if mask.sum() > 0:
            avg_text_features[c] = all_text_features[mask].mean(dim=0)
    avg_text_features = avg_text_features.cuda()
    logger.info("Averaged text features computed for {} classes".format(num_classes))

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Stage1 running time: {}".format(total_time))

    return avg_text_features