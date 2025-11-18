import torch


def make_optimizer_1stage(cfg, model):
    params = []
    keys = []
    for key, value in model.named_parameters():
        if "image_encoder" in key:
            value.requires_grad_(False)
            continue
        lr = cfg.SOLVER.STAGE1.BASE_LR
        weight_decay = cfg.SOLVER.STAGE1.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.STAGE1.BASE_LR * cfg.SOLVER.STAGE1.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.STAGE1.WEIGHT_DECAY_BIAS
        if cfg.SOLVER.STAGE1.LARGE_FC_LR:
            if "classifier" in key or "arcface" in key:
                lr = cfg.SOLVER.STAGE1.BASE_LR * 2
                print("Using two times learning rate for fc ")

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        keys += [key]

    if cfg.SOLVER.STAGE1.OPTIMIZER_NAME == "SGD":
        optimizer = getattr(torch.optim, cfg.SOLVER.STAGE1.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.STAGE1.MOMENTUM)
    elif cfg.SOLVER.STAGE1.OPTIMIZER_NAME == "AdamW":
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.STAGE1.BASE_LR, weight_decay=cfg.SOLVER.STAGE1.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.STAGE1.OPTIMIZER_NAME)(params)

    return optimizer


def make_optimizer_2stage(cfg, model, center_criterion):
    params = []
    keys = []
    for key, value in model.named_parameters():
        # if "skeleton_encoder" in key:
        #     value.requires_grad_(False)
        #     continue
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.STAGE2.BASE_LR
        weight_decay = cfg.SOLVER.STAGE2.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.STAGE2.BASE_LR * cfg.SOLVER.STAGE2.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.STAGE2.WEIGHT_DECAY_BIAS
        if cfg.SOLVER.STAGE2.LARGE_FC_LR:
            if "classifier_proj_temp" in key or "arcface" in key:
                lr = lr * 10
                print("Using 10 times learning rate for fc ")
            if "VTU" in key or "TMM" in key or "PF" in key:
                lr = lr * 10
                print("Using 10 times learning rate for modules ")

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        keys += [key]

    if cfg.SOLVER.STAGE2.OPTIMIZER_NAME == "SGD":
        optimizer = getattr(torch.optim, cfg.SOLVER.STAGE2.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.STAGE2.MOMENTUM)
    elif cfg.SOLVER.STAGE2.OPTIMIZER_NAME == "AdamW":
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.STAGE2.BASE_LR, weight_decay=cfg.SOLVER.STAGE2.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.STAGE2.OPTIMIZER_NAME)(params)

    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.STAGE2.CENTER_LR)

    return optimizer, optimizer_center
