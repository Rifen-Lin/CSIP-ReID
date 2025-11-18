import torch
from torch.utils.data import DataLoader
import utils.spatial_transforms as ST
import utils.temporal_transforms as TT
import utils.transforms as T
import utils.seqtransforms as SeqT
from datasets.video_loader import VideoDataset
from datasets.samplers import RandomIdentitySampler, RandomIdentitySamplerForSeq, RandomIdentitySamplerWYQ
from datasets.seqpreprocessor import SeqTrainPreprocessor, SeqTestPreprocessor

from datasets.set.mars import Mars
from datasets.set.ilidsvidsequence import iLIDSVIDSEQUENCE
from datasets.set.lsvid import LSVID

__factory = {
    "mars": Mars,
    "ilidsvidsequence": iLIDSVIDSEQUENCE,
    "lsvid": LSVID,
}


def train_collate_fn(batch):
    skeletpn_rgb_datas, pids, camids, viewids, img_ske_paths = zip(*batch)

    pids = torch.tensor(pids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)

    RGB_list = []
    SKELETON_list = []
    for data in skeletpn_rgb_datas:
        RGB_list.append(data[0])
        SKELETON_list.append(data[1])

    RGB = torch.stack(RGB_list, dim=0)
    SKELETON = torch.stack(SKELETON_list, dim=0)

    datas = {"RGB": RGB, "SKELETON": SKELETON}
    return datas, pids, camids, viewids, img_ske_paths


def val_collate_fn(batch):
    skeletpn_rgb_datas, pids, camids, viewids, img_ske_paths = zip(*batch)

    pids = torch.tensor(pids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)

    RGB_list = []
    SKELETON_list = []
    for data in skeletpn_rgb_datas:
        RGB_list.append(data[0])
        SKELETON_list.append(data[1])

    RGB = torch.stack(RGB_list, dim=0)
    SKELETON = torch.stack(SKELETON_list, dim=0)

    datas = {"RGB": RGB, "SKELETON": SKELETON}
    return datas, pids, camids, camids_batch, viewids, img_ske_paths


def make_dataloader(cfg):
    split_id = cfg.DATASETS.SPLIT
    seq_srd = cfg.INPUT.SEQ_SRD
    seq_len = cfg.INPUT.SEQ_LEN
    num_workers = cfg.DATALOADER.NUM_WORKERS

    if cfg.DATASETS.NAMES == "ilidsvidsequence":

        dataset = __factory[cfg.DATASETS.NAMES](
            root=cfg.DATASETS.ROOT_DIR, split_id=split_id, seq_len=seq_len, seq_srd=seq_srd, num_val=1
        )

        num_classes = dataset.num_trainval_ids
        cam_num = dataset.num_train_cams
        view_num = dataset.num_train_vids

        print(f"len(dataset.trainval): {len(dataset.trainval)}")
        print(f"len(dataset.query): {len(dataset.query)}")
        print(f"len(dataset.gallery): {len(dataset.gallery)}")

        train_set = SeqTrainPreprocessor(
            dataset.trainval,
            dataset,
            seq_len,
            transform=SeqT.Compose(
                [
                    SeqT.RectScale(256, 128),
                    SeqT.RandomHorizontalFlip(),
                    SeqT.RandomSizedEarser(),
                    SeqT.ToTensor(),
                    SeqT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            ),
        )

        train_set_normal = SeqTrainPreprocessor(
            dataset.trainval,
            dataset,
            seq_len,
            transform=SeqT.Compose(
                [
                    SeqT.RectScale(256, 128),
                    SeqT.ToTensor(),
                    SeqT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            ),
        )

        val_set = SeqTestPreprocessor(
            dataset.query + dataset.gallery,
            dataset,
            seq_len,
            transform=SeqT.Compose(
                [
                    SeqT.RectScale(256, 128),
                    SeqT.ToTensor(),
                    SeqT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            ),
        )

        train_loader_stage2 = DataLoader(
            train_set,
            sampler=RandomIdentitySamplerForSeq(
                dataset.trainval, batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH, num_instances=cfg.DATALOADER.NUM_INSTANCE
            ),
            batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH,
            num_workers=num_workers,
            drop_last=True,
            collate_fn=train_collate_fn,
        )

        train_loader_stage1 = DataLoader(
            train_set_normal,
            batch_size=cfg.SOLVER.STAGE1.IMS_PER_BATCH,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            collate_fn=train_collate_fn,
        )

        val_loader = DataLoader(
            val_set,
            batch_size=cfg.TEST.IMS_PER_BATCH,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            collate_fn=val_collate_fn,
        )

        return (
            train_loader_stage2,
            train_loader_stage1,
            train_loader_stage1,
            val_loader,
            len(dataset.query),
            num_classes,
            cam_num,
            view_num,
        )

    else:
        dataset = __factory[cfg.DATASETS.NAMES](args=cfg)

        transform_train = SeqT.Compose(
            [
                SeqT.RectScale(cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1]),
                SeqT.RandomHorizontalFlip(),
                SeqT.RandomSizedEarser(),
                SeqT.ToTensor(),
                SeqT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        transform_test = SeqT.Compose(
            [
                SeqT.RectScale(cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1]),
                SeqT.ToTensor(),
                SeqT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        num_classes = dataset.num_train_pids
        cam_num = dataset.num_train_cams
        view_num = dataset.num_train_vids

        train_set = VideoDataset(
            dataset.train, seq_len=seq_len, sample="rrs_train", transform=transform_train, config=cfg
        )
        train_loader_stage2 = DataLoader(
            train_set,
            sampler=RandomIdentitySampler(
                dataset.train, batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH, num_instances=cfg.DATALOADER.NUM_INSTANCE
            ),
            batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
            collate_fn=train_collate_fn,
        )

        train_set_normal_dense = VideoDataset(
            dataset.train, seq_len=seq_len, sample="dense", transform=transform_test, config=cfg
        )
        train_loader_stage1_bs1 = DataLoader(
            train_set_normal_dense,
            batch_size=1,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
            collate_fn=train_collate_fn,
        )
        train_set_normal = VideoDataset(
            dataset.train, seq_len=seq_len, sample="rrs_train", transform=transform_test, config=cfg
        )
        train_loader_stage1_bs = DataLoader(
            train_set_normal,
            batch_size=cfg.SOLVER.STAGE1.IMS_PER_BATCH,
            shuffle=True,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True,
            collate_fn=train_collate_fn,
        )

        val_set = VideoDataset(
            dataset.query + dataset.gallery, seq_len=seq_len, sample="rrs_test", transform=transform_test, config=cfg
        )
        val_loader = DataLoader(
            val_set,
            batch_size=30,
            # batch_size=16,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            drop_last=False,
            collate_fn=val_collate_fn,
        )

        return (
            train_loader_stage2,
            train_loader_stage1_bs1,
            train_loader_stage1_bs,
            val_loader,
            len(dataset.query),
            num_classes,
            cam_num,
            view_num,
        )


def make_eval_dense_dataloader(cfg):
    split_id = cfg.DATASETS.SPLIT
    seq_srd = cfg.INPUT.SEQ_SRD
    seq_len = cfg.INPUT.SEQ_LEN
    num_workers = cfg.DATALOADER.NUM_WORKERS

    if cfg.DATASETS.NAMES == "ilidsvidsequence":

        dataset = __factory[cfg.DATASETS.NAMES](
            root=cfg.DATASETS.ROOT_DIR, split_id=split_id, seq_len=seq_len, seq_srd=seq_srd, num_val=1
        )

        num_classes = dataset.num_trainval_ids
        cam_num = dataset.num_train_cams
        view_num = dataset.num_train_vids

        val_set = SeqTestPreprocessor(
            dataset.query + dataset.gallery,
            dataset,
            seq_len,
            transform=SeqT.Compose(
                [
                    SeqT.RectScale(256, 128),
                    SeqT.ToTensor(),
                    SeqT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            ),
        )
        val_loader = DataLoader(
            val_set,
            batch_size=cfg.TEST.IMS_PER_BATCH,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            collate_fn=val_collate_fn,
        )
    else:
        dataset = __factory[cfg.DATASETS.NAMES](args=cfg)

        transform_test = SeqT.Compose(
            [
                SeqT.RectScale(cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1]),
                SeqT.ToTensor(),
                SeqT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        num_classes = dataset.num_train_pids
        cam_num = dataset.num_train_cams
        view_num = dataset.num_train_vids

        val_set = VideoDataset(
            dataset.query + dataset.gallery, seq_len=seq_len, sample="dense", transform=transform_test
        )
        val_loader = DataLoader(
            val_set,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            drop_last=False,
            collate_fn=val_collate_fn,
        )

    return val_loader, len(dataset.query), num_classes, cam_num, view_num
