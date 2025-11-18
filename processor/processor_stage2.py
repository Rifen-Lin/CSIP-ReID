import logging
import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
import swanlab
from torch.nn import functional as F

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def do_train_stage2_skeleton_fixed(
    cfg,
    model,
    center_criterion,
    train_loader_stage1_dense,
    train_loader_stage1_rrs,
    train_loader_stage2,
    val_loader,
    optimizer,
    optimizer_center,
    scheduler,
    loss_fn,
    num_query,
    local_rank,
):
    log_period = cfg.SOLVER.STAGE2.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.STAGE2.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.STAGE2.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.STAGE2.MAX_EPOCHS

    logger = logging.getLogger("CSIP-ReID.train")
    logger.info("start training")

    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print("Using {} GPUs for training".format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
            num_classes = model.module.num_classes
        else:
            num_classes = model.num_classes

    loss_meter_total = AverageMeter()
    loss_meter_id_vis = AverageMeter()
    loss_meter_triplet_vis = AverageMeter()
    loss_meter_clip = AverageMeter()
    loss_meter_frame = AverageMeter()
    loss_meter_ske = AverageMeter()

    acc_meter_clip = AverageMeter()
    acc_meter_id_vis = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()

    @torch.no_grad()
    def generate_cluster_features(labels, features):
        import collections

        centers = collections.defaultdict(list)
        for i, label in enumerate(labels):
            if label == -1:
                continue
            centers[labels[i]].append(features[i])

        centers = [torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())]

        centers = torch.stack(centers, dim=0)
        return centers

    import time
    from datetime import timedelta

    all_start_time = time.monotonic()

    print("=> Automatically generating Memory (might take a while, have a coffe)")
    skeleton_features = []
    image_features = []
    labels = []
    with torch.no_grad():
        for n_iter, (datas, vid, target_cam, target_view, data_paths) in enumerate(
            tqdm(train_loader_stage1_rrs, desc=f"Processing train_loader_stage1")
        ):
            img = datas["RGB"].to(device)
            ske = datas["SKELETON"].to(device)
            target = vid.to(device)
            if len(img.size()) == 6:
                b, n, s, j, c = ske.size()
                assert b == 1
                ske = ske.view(b * n, s, j, c)
                with amp.autocast(enabled=True):
                    skeleton_feature = model(ske=ske, label=target, get_skeleton=True)
                    skeleton_feature = skeleton_feature.view(-1, skeleton_feature.size(1))
                    skeleton_feature = torch.mean(skeleton_feature, 0, keepdim=True)
                    for i, ske_feat in zip(target, skeleton_feature):
                        labels.append(i)
                        skeleton_features.append(ske_feat.detach().cpu())

                b, n, s, c, h, w = img.size()
                assert b == 1
                img = img.view(b * n, s, c, h, w)
                with amp.autocast(enabled=True):
                    image_feature = model(img=img, label=target, get_image=True)
                    image_feature = torch.mean(image_feature, dim=0, keepdim=True)
                    for i, img_feat in zip(target, image_feature):
                        image_features.append(img_feat.detach().cpu())

            else:
                with amp.autocast(enabled=True):
                    skeleton_feature = model(ske=ske, label=target, get_skeleton=True)
                    for i, ske_feat in zip(target, skeleton_feature):
                        labels.append(i)
                        skeleton_features.append(ske_feat.detach().cpu())

                    b, t, c, h, w = img.size()
                    image_feature = model(img=img, label=target, get_image=True)
                    image_feature = image_feature.view(b, t, -1)
                    image_feature = torch.mean(image_feature, dim=1)
                    for i, img_feat in zip(target, image_feature):
                        image_features.append(img_feat.detach().cpu())

        labels_list = torch.stack(labels, dim=0).cuda()
        skeleton_features_list = torch.stack(skeleton_features, dim=0).cuda()
        image_features_list = torch.stack(image_features, dim=0).cuda()

    ske_prototype = generate_cluster_features(labels_list.cpu().numpy(), skeleton_features_list).detach()

    vis_prototype = generate_cluster_features(labels_list.cpu().numpy(), image_features_list).detach()

    best_performance = 0.0
    best_epoch = 1

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        loss_meter_total.reset()
        loss_meter_id_vis.reset()
        loss_meter_triplet_vis.reset()
        loss_meter_clip.reset()
        loss_meter_frame.reset()

        acc_meter_clip.reset()
        acc_meter_id_vis.reset()

        evaluator.reset()

        scheduler.step()

        model.train()
        for n_iter, (datas, vid, target_cam, target_view, data_paths) in enumerate(
            tqdm(train_loader_stage2, desc=f"Processing train_loader_stage2")
        ):
            optimizer.zero_grad()
            optimizer_center.zero_grad()

            img = datas["RGB"].to(device)
            ske = datas["SKELETON"].to(device)
            target = vid.to(device)

            if cfg.MODEL.IMG.SIE_CAMERA:
                target_cam = target_cam.to(device)
            else:
                target_cam = None
            if cfg.MODEL.IMG.SIE_VIEW:
                target_view = target_view.to(device)
            else:
                target_view = None
            with amp.autocast(enabled=True):
                B, T, C, H, W = img.shape
                score, feat, logits, loss_frame = model(
                    img=img,
                    ske=ske,
                    label=target,
                    cam_label=target_cam,
                    view_label=target_view,
                    ske_prototype=ske_prototype,
                    vis_prototype=vis_prototype,
                )

                loss, split_loss_dict = loss_fn(score, feat, target, target_cam, logits)

            frame_loss_weight = 1.3  # 1.3
            loss = loss + frame_loss_weight * loss_frame
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if "center" in cfg.MODEL.IMG.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= 1.0 / cfg.SOLVER.CENTER_LOSS_WEIGHT
                scaler.step(optimizer_center)
                scaler.update()

            acc_clip = (logits.max(1)[1] == target).float().mean()
            acc_vis = (score[0].max(1)[1] == target).float().mean()

            loss_meter_total.update(loss.item(), img.shape[0])
            loss_meter_id_vis.update(split_loss_dict["ID_LOSS"], img.shape[0])
            loss_meter_triplet_vis.update(split_loss_dict["TRI_LOSS"], img.shape[0])
            loss_meter_clip.update(split_loss_dict["CLIP_LOSS"], img.shape[0])
            loss_meter_frame.update(loss_frame.item(), img.shape[0])

            acc_meter_clip.update(acc_clip, 1)
            acc_meter_id_vis.update(acc_vis, 1)

            swanlab.log(
                {
                    "stage2_total_loss": loss_meter_total.avg,
                    "stage2_id_loss_vis": loss_meter_id_vis.avg,
                    "stage2_triplet_loss_vis": loss_meter_triplet_vis.avg,
                    "stage2_clip_loss": loss_meter_clip.avg,
                    "stage2_frame_loss": loss_meter_frame.avg,
                    "stage2_train_acc_clip": acc_meter_clip.avg,
                    "stage2_train_acc_id_vis": acc_meter_id_vis.avg,
                    "stage2_learning_rate": optimizer.param_groups[0]["lr"],
                    "stage2_epoch": epoch,
                }
            )

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info(
                    "Epoch[{}] Iteration[{}/{}] "
                    "Total Loss: {:.3f} | Acc I2S: {:.3f} | Acc ID: {:.3f} | LR: {:.2e}".format(
                        epoch,
                        (n_iter + 1),
                        len(train_loader_stage2),
                        loss_meter_total.avg,
                        acc_meter_clip.avg,
                        acc_meter_id_vis.avg,
                        optimizer.param_groups[0]["lr"],
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
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model_save_path = f"{cfg.OUTPUT_DIR}/ckpts"
                    os.makedirs(model_save_path, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(model_save_path, "epoch_{}_stage2.pth".format(epoch)))
            else:
                model_save_path = f"{cfg.OUTPUT_DIR}/ckpts"
                os.makedirs(model_save_path, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(model_save_path, "epoch_{}_stage2.pth".format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (datas, vid, camid, camids, target_view, data_paths) in enumerate(
                        tqdm(val_loader, desc=f"Processing val_loader")
                    ):
                        with torch.no_grad():
                            img = datas["RGB"].to(device)
                            ske = datas["SKELETON"].to(device)
                            target = vid.to(device)

                            if cfg.MODEL.IMG.SIE_CAMERA:
                                camids = camids.to(device)
                            else:
                                camids = None
                            if cfg.MODEL.IMG.SIE_VIEW:
                                target_view = target_view.to(device)
                            else:
                                target_view = None

                            feat = model(img=img, ske=ske, label=target, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10, 20]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    swanlab.log(
                        {
                            "mAP": mAP,
                            "Rank-1": cmc[0],
                            "Rank-5": cmc[4],
                            "Rank-10": cmc[9],
                            "Rank-20": cmc[19],
                            "epoch": epoch,
                        }
                    )
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (datas, vid, camid, camids, target_view, data_paths) in enumerate(
                    tqdm(val_loader, desc=f"Processing val_loader")
                ):
                    with torch.no_grad():
                        img = datas["RGB"].to(device)
                        ske = datas["SKELETON"].to(device)
                        target = vid.to(device)

                        if cfg.MODEL.IMG.SIE_CAMERA:
                            camids = camids.to(device)
                        else:
                            camids = None
                        if cfg.MODEL.IMG.SIE_VIEW:
                            target_view = target_view.to(device)
                        else:
                            target_view = None

                        feat = model(img=img, ske=ske, label=target, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))

                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10, 20]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                swanlab.log({"mAP": mAP, "Rank-1": cmc[0], "Rank-5": cmc[4], "Rank-10": cmc[9], "Rank-20": cmc[19]})
                torch.cuda.empty_cache()

            prec1 = cmc[0] + mAP
            is_best = prec1 > best_performance
            best_performance = max(prec1, best_performance)
            if is_best:
                best_epoch = epoch
            model_save_path = f"{cfg.OUTPUT_DIR}/ckpts"
            os.makedirs(model_save_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(model_save_path, "stage2_best.pth"))

    logger.info("==> Best Perform {:.1%}, achieved at epoch {}".format(best_performance, best_epoch))
    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Total running time: {}".format(total_time))
    print(cfg.OUTPUT_DIR)


def do_inference_rrs(cfg, model, val_loader, num_query):
    device = "cuda"
    logger = logging.getLogger("CSIP-ReID.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print("Using {} GPUs for inference".format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    for n_iter, (datas, vid, camid, camids, target_view, data_paths) in enumerate(
        tqdm(val_loader, desc=f"Processing val_loader")
    ):
        with torch.no_grad():
            img = datas["RGB"].to(device)
            ske = datas["SKELETON"].to(device)
            target = vid.to(device)

            if cfg.MODEL.IMG.SIE_CAMERA:
                camids = camids.to(device)
            else:
                camids = None
            if cfg.MODEL.IMG.SIE_VIEW:
                target_view = target_view.to(device)
            else:
                target_view = None

            feat = model(img=img, ske=ske, label=target, cam_label=camids, view_label=target_view)

            evaluator.update((feat, vid, camids.cpu()))

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10, 20]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    return cmc[0], cmc[4]


def do_inference_dense(cfg, model, val_loader, num_query):
    device = "cuda"
    logger = logging.getLogger("CSIP-ReID.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print("Using {} GPUs for inference".format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    for n_iter, (datas, vid, camid, camids, target_view, data_paths) in enumerate(
        tqdm(val_loader, desc=f"Processing val_loader")
    ):
        with torch.no_grad():
            img = datas["RGB"].to(device)
            ske = datas["SKELETON"].to(device)
            target = vid.to(device)
            b, n, s, c, h, w = img.size()
            assert b == 1
            img = img.view(b * n, s, c, h, w)
            b, n, s, j, c = ske.size()
            assert b == 1
            ske = ske.view(b * n, s, j, c)

            if cfg.MODEL.IMG.SIE_CAMERA:
                camids = camids.to(device)
                camids = camids.repeat(n)
            else:
                camids = None
            if cfg.MODEL.IMG.SIE_VIEW:
                target_view = target_view.to(device)
            else:
                target_view = None

            feat = model(img=img, ske=ske, label=target, cam_label=camids, view_label=target_view)
            feat = torch.mean(feat, 0, keepdim=True)

            evaluator.update((feat, vid, camid))

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10, 20]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]
