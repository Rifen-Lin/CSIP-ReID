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


def do_train_stage1(cfg, model, train_loader_stage1_dense, train_loader_stage1_rrs, optimizer, scheduler, local_rank):
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

    for epoch in range(1, epochs + 1):
        loss_meter.reset()
        scheduler.step(epoch)
        model.train()

        for n_iter, (datas, vid, _, _, _) in enumerate(tqdm(train_loader_stage1_rrs, desc=f"Epoch {epoch}")):
            optimizer.zero_grad()

            img = datas["RGB"].to(device)
            ske = datas["SKELETON"].to(device)
            target = vid.to(device)

            b, t, c, h, w = img.size()

            with torch.no_grad():
                image_feature = model(img=img, label=target, get_image=True)
                image_feature = image_feature.view(b, t, -1)
                image_feature = torch.mean(image_feature, dim=1)

            with amp.autocast(enabled=True):
                # image_feature = model(img=img, label=target, get_image=True)
                # image_feature = image_feature.view(b, t, -1)
                # image_feature = torch.mean(image_feature, dim=1)

                skeleton_feature = model(ske=ske, label=target, get_skeleton=True)

                loss_i2s = xent(image_feature, skeleton_feature, target, target)
                loss_s2i = xent(skeleton_feature, image_feature, target, target)
                loss = loss_i2s + loss_s2i

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                loss_meter.update(loss.item(), target.shape[0])

                swanlab.log(
                    {
                        "stage1_loss": loss_meter.avg,
                        "stage1_learning_rate": scheduler._get_lr(epoch)[0],
                        "stage1_epoch": epoch,
                    }
                )

                torch.cuda.synchronize()
                if (n_iter + 1) % log_period == 0:
                    logger.info(
                        "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}".format(
                            epoch,
                            (n_iter + 1),
                            len(train_loader_stage1_rrs),
                            loss_meter.avg,
                            scheduler._get_lr(epoch)[0],
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
