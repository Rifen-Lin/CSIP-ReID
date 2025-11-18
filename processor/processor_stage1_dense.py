import logging
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from utils.meter import AverageMeter
from torch.cuda import amp
import torch.distributed as dist
import swanlab
from torch.nn import functional as F
from loss.supcontrast import SupConLoss


def do_train_stage1(
    cfg, model, train_loader_stage1_dense, train_loader_stage1_rrs, optimizer, scheduler, local_rank, start_epoch
):
    checkpoint_period = cfg.SOLVER.STAGE1.CHECKPOINT_PERIOD
    device = "cuda"
    epochs = cfg.SOLVER.STAGE1.MAX_EPOCHS
    log_period = cfg.SOLVER.STAGE1.LOG_PERIOD

    logger = logging.getLogger("CSIP-ReID.train")
    logger.info("start training")

    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print("Using {} GPUs for training".format(torch.cuda.device_count()))
            model = nn.DataParallel(model)

    loss_meter = AverageMeter()
    scaler = amp.GradScaler()
    xent = SupConLoss(device)

    import time
    from datetime import timedelta

    all_start_time = time.monotonic()
    logger.info("model: {}".format(model))

    for epoch in range(start_epoch, epochs + 1):
        loss_meter.reset()
        scheduler.step(epoch)
        model.train()

        buffer_image_features = []
        buffer_skeleton_features = []
        buffer_labels = []
        batch_size = 64

        for n_iter, (datas, vid, _, _, _) in enumerate(tqdm(train_loader_stage1_dense, desc=f"Epoch {epoch}")):

            img = datas["RGB"].to(device)
            ske = datas["SKELETON"].to(device)
            target = vid.to(device)

            b, n, t, c, h, w = img.size()
            assert b == 1
            img = img.view(b * n, t, c, h, w)

            b, n, t, j, c = ske.size()
            assert b == 1
            ske = ske.view(b * n, t, j, c)

            with amp.autocast(enabled=True):

                with torch.no_grad():
                    image_feature = model(img=img, label=target, get_image=True)
                    image_feature = image_feature.view(b * n, t, -1)
                    image_feature = torch.mean(image_feature, dim=1)
                    image_feature = torch.mean(image_feature, dim=0, keepdim=True)

                skeleton_feature = model(ske=ske, label=target, get_skeleton=True)
                skeleton_feature = torch.mean(skeleton_feature, dim=0, keepdim=True)

                buffer_image_features.append(image_feature)
                buffer_skeleton_features.append(skeleton_feature)
                buffer_labels.append(target)

                if len(buffer_labels) == batch_size:
                    optimizer.zero_grad()

                    batch_image = torch.cat(buffer_image_features, dim=0)
                    batch_skeleton = torch.cat(buffer_skeleton_features, dim=0)
                    batch_labels = torch.cat(buffer_labels, dim=0)

                    loss_i2s = xent(batch_image, batch_skeleton, batch_labels, batch_labels)
                    loss_s2i = xent(batch_skeleton, batch_image, batch_labels, batch_labels)
                    loss = loss_i2s + loss_s2i

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    loss_meter.update(loss.item(), batch_labels.shape[0])

                    buffer_image_features.clear()
                    buffer_skeleton_features.clear()
                    buffer_labels.clear()

                    swanlab.log(
                        {
                            "stage1_loss": loss_meter.avg,
                            "stage1_learning_rate": scheduler._get_lr(epoch)[0],
                            "stage1_epoch": epoch,
                        }
                    )

                    torch.cuda.synchronize()
                    if (loss_meter.count // batch_size) % log_period == 0:
                        logger.info(
                            "Epoch [{}] Step [{}] Loss: {:.3f}, Base LR: {:.2e}".format(
                                epoch,
                                loss_meter.count,
                                loss_meter.avg,
                                scheduler._get_lr(epoch)[0],
                            )
                        )

        if len(buffer_labels) > 1:
            optimizer.zero_grad()

            batch_image = torch.cat(buffer_image_features, dim=0)
            batch_skeleton = torch.cat(buffer_skeleton_features, dim=0)
            batch_labels = torch.cat(buffer_labels, dim=0)

            loss_i2s = xent(batch_image, batch_skeleton, batch_labels, batch_labels)
            loss_s2i = xent(batch_skeleton, batch_image, batch_labels, batch_labels)
            loss = loss_i2s + loss_s2i

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), batch_labels.shape[0])

            buffer_image_features.clear()
            buffer_skeleton_features.clear()
            buffer_labels.clear()

            swanlab.log(
                {
                    "stage1_loss": loss_meter.avg,
                    "stage1_learning_rate": scheduler._get_lr(epoch)[0],
                    "stage1_epoch": epoch,
                }
            )

            torch.cuda.synchronize()
            logger.info(
                "Epoch[{}] Final Small Batch Loss: {:.3f}, Size: {}".format(
                    epoch,
                    loss_meter.avg,
                    batch_labels.shape[0],
                )
            )

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(
                        model.state_dict(),
                        os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + "_stage1_{}.pth".format(epoch)),
                    )
            else:
                model_save_path = f"{cfg.OUTPUT_DIR}/ckpts"
                os.makedirs(model_save_path, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(model_save_path, "epoch_{}_stage1.pth".format(epoch)))

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Stage1 running time: {}".format(total_time))
