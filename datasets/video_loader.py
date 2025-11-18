from __future__ import print_function, absolute_import
import os
from PIL import Image
import numpy as np
import math
import torch
from torch.utils.data import Dataset
import random
import json
import logging


def read_mars_3d_skeleton(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
        joints_3d = np.array(data["keypoints"])

    return joints_3d


def root_centered_normalization_h36m(skeleton_data):
    root_joint = skeleton_data[:, 0, :]
    return skeleton_data - root_joint[:, np.newaxis, :]


def build_mask_from_json(json_paths, args):
    mask = []

    for json_path in json_paths:
        try:
            if args.DATASETS.NAMES in ["mars"]:
                skeleton = read_mars_3d_skeleton(json_path)
            else:
                raise ValueError(f"Unsupported dataset: {args.DATASET.NAME}")

            skeleton = np.array(skeleton)

            if np.all(skeleton == 0):
                mask.append(True)
            else:
                mask.append(False)

        except Exception as e:
            print(f"[WARN] Error processing {json_path}: {e}")
            mask.append(True)

    return mask


class VideoDataset(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """

    sample_methods = ["evenly", "random", "dense"]

    def __init__(self, dataset, seq_len=15, sample="evenly", transform=None, config=None):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
        self.args = config

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self.__get_single_item__(index) for index in indices]
        return self.__get_single_item__(indices)

    def __get_single_item__(self, index):
        S = self.seq_len  # 8
        data_paths, pid, camid, trackid = self.dataset[index]
        img_paths = data_paths[0]
        skeleton_paths = data_paths[1]

        ## ----------------------------------------------------------------------------------------------
        # mask = build_mask_from_json(skeleton_paths, self.args)
        # img_paths = [p for p, m in zip(img_paths, mask) if not m]
        # skeleton_paths = [p for p, m in zip(skeleton_paths, mask) if not m]
        # logger = logging.getLogger("CSIP-ReID.train")
        # num_deleted = np.sum(mask)
        # if num_deleted > 0:
        #     total_frames = len(mask)
        #     num_remaining = total_frames - num_deleted
        #     tmp_sample_name = img_paths[0].split("/")[-2]
        #     tmp_filename = img_paths[0].split("/")[-1]
        #     tmp_camid = tmp_filename[4:6]
        #     tmp_trackletid = tmp_filename[6:11]
        #     logger.info(f"ID: {tmp_sample_name} - camera: {tmp_camid} - tracklet: {tmp_trackletid} - delete_nums: {num_deleted} - original_nums: {total_frames} - remain_nums: {num_remaining}")
        ## ----------------------------------------------------------------------------------------------

        num = len(img_paths)

        sample_clip = []
        frame_indices = list(range(num))
        if num < S:
            strip = list(range(num)) + [frame_indices[-1]] * (S - num)
            for s in range(S):
                pool = strip[s * 1 : (s + 1) * 1]
                sample_clip.append(list(pool))
        else:
            inter_val = math.ceil(num / S)
            strip = list(range(num)) + [frame_indices[-1]] * (inter_val * S - num)
            for s in range(S):
                pool = strip[inter_val * s : inter_val * (s + 1)]
                sample_clip.append(list(pool))

        sample_clip = np.array(sample_clip)

        if self.sample == "random":
            """
            Randomly sample seq_len consecutive frames from num frames,
            if num is smaller than seq_len, then replicate items.
            This sampling strategy is used in training phase.
            """
            frame_indices = list(range(num))
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))

            indices = frame_indices[begin_index:end_index]

            for index in indices:
                if len(indices) >= self.seq_len:
                    break
                indices.append(index)
            indices = np.array(indices)
            imgseq = []
            for index in indices:
                index = int(index)
                img_path = img_paths[index]
                img = Image.open(img_path).convert("RGB")
                imgseq.append(img)

            seq = [imgseq]
            if self.transform is not None:
                seq = self.transform(seq)

            img_tensor = torch.stack(seq[0], dim=0)
            flow_tensor = None

            return img_tensor, pid, camid

        elif self.sample == "dense":
            """
            Sample all frames in a video into a list of clips, each clip contains seq_len frames, batch_size needs to be set to 1.
            This sampling strategy is used in test phase.
            """
            cur_index = 0
            frame_indices = list(range(num))
            indices_list = []
            while num - cur_index > self.seq_len:
                indices_list.append(frame_indices[cur_index : cur_index + self.seq_len])
                cur_index += self.seq_len

            last_seq = frame_indices[cur_index:]

            for index in last_seq:
                if len(last_seq) >= self.seq_len:
                    break
                last_seq.append(index)

            indices_list.append(last_seq)
            imgs_list = []
            skes_list = []
            for indices in indices_list:
                imgs = []
                skes = []
                for index in indices:
                    index = int(index)
                    img_path = img_paths[index]
                    ske_path = skeleton_paths[index]
                    img = Image.open(img_path).convert("RGB")
                    ske = read_mars_3d_skeleton(ske_path)
                    ske = torch.tensor(ske, dtype=torch.float32)
                    imgs.append(img)
                    skes.append(ske)

                imgs = [imgs]
                skes = [skes]
                if self.transform is not None:
                    imgs = self.transform(imgs)
                imgs = torch.stack(imgs[0], 0)
                skes = torch.stack(skes[0], 0)
                skes = root_centered_normalization_h36m(skes)

                imgs_list.append(imgs)
                skes_list.append(skes)

            imgs_tensor = torch.stack(imgs_list)
            skes_tensor = torch.stack(skes_list)

            return (imgs_tensor, skes_tensor), pid, camid, trackid, (img_paths, skeleton_paths)

        elif self.sample == "rrs_train":
            idx = np.random.choice(sample_clip.shape[1], sample_clip.shape[0])
            number = sample_clip[np.arange(len(sample_clip)), idx]

            img_paths = np.array(list(img_paths))
            ske_paths = np.array(list(skeleton_paths))

            imgseq = [Image.open(img_path).convert("RGB") for img_path in img_paths[number]]
            skeseq = [
                torch.tensor(read_mars_3d_skeleton(ske_path), dtype=torch.float32) for ske_path in ske_paths[number]
            ]

            img_seq = [imgseq]
            ske_seq = [skeseq]

            if self.transform is not None:
                img_seq = self.transform(img_seq)

            img_tensor = torch.stack(img_seq[0], dim=0)
            ske_tensor = torch.stack(ske_seq[0], dim=0)

            ske_tensor = root_centered_normalization_h36m(ske_tensor)

            return_img_paths = [img_paths[idx_] for idx_ in number]
            return_skeleton_paths = [skeleton_paths[idx_] for idx_ in number]

            return (img_tensor, ske_tensor), pid, camid, trackid, (return_img_paths, return_skeleton_paths)

        elif self.sample == "rrs_test":
            number = sample_clip[:, 0]

            img_paths = np.array(list(img_paths))
            ske_paths = np.array(list(skeleton_paths))

            imgseq = [Image.open(img_path).convert("RGB") for img_path in img_paths[number]]
            skeseq = [
                torch.tensor(read_mars_3d_skeleton(ske_path), dtype=torch.float32) for ske_path in ske_paths[number]
            ]

            img_seq = [imgseq]
            ske_seq = [skeseq]

            if self.transform is not None:
                img_seq = self.transform(img_seq)

            img_tensor = torch.stack(img_seq[0], dim=0)
            ske_tensor = torch.stack(ske_seq[0], dim=0)

            ske_tensor = root_centered_normalization_h36m(ske_tensor)

            return_img_paths = [img_paths[idx_] for idx_ in number]
            return_skeleton_paths = [skeleton_paths[idx_] for idx_ in number]

            return (img_tensor, ske_tensor), pid, camid, trackid, (return_img_paths, return_skeleton_paths)
        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))
