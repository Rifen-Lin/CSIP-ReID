from __future__ import absolute_import
import os.path as osp
import torch
from PIL import Image
import json
import numpy as np


def read_ilids_3d_skeleton(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
        joints_3d = np.array(data["keypoints"])

    return joints_3d


def root_centered_normalization_h36m(skeleton_data):
    root_joint = skeleton_data[:, 0, :]
    return skeleton_data - root_joint[:, np.newaxis, :]


class SeqTrainPreprocessor(object):
    def __init__(self, seqset, dataset, seq_len, transform=None):
        super(SeqTrainPreprocessor, self).__init__()
        self.seqset = seqset
        self.identities = dataset.identities
        self.identities_ske = dataset.identities_ske
        self.transform = transform
        self.seq_len = seq_len
        self.root = [dataset.images_dir]
        self.root.append(dataset.skes_dir)

    def __len__(self):
        return len(self.seqset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):

        start_ind, end_ind, pid, label, camid = self.seqset[index]

        imgseq = []
        skeseq = []
        ret_img_paths = []
        ret_ske_paths = []

        for ind in range(start_ind, end_ind):
            fname = self.identities[pid][camid][ind]
            fpath_img = osp.join(self.root[0], fname)
            ret_img_paths.append(fpath_img)
            imgrgb = Image.open(fpath_img).convert("RGB")
            fname_ske = self.identities_ske[pid][camid][ind]
            fpath_ske = osp.join(self.root[1], fname_ske)
            ret_ske_paths.append(fpath_ske)
            ske_data = torch.tensor(read_ilids_3d_skeleton(fpath_ske), dtype=torch.float32)
            imgseq.append(imgrgb)
            skeseq.append(ske_data)

        while len(imgseq) < self.seq_len:
            imgseq.append(imgrgb)
            skeseq.append(ske_data)

        imgseq = [imgseq]
        skeseq = [skeseq]

        if self.transform is not None:
            imgseq = self.transform(imgseq)

        img_tensor = torch.stack(imgseq[0], 0)
        ske_tensor = torch.stack(skeseq[0], 0)

        ske_tensor = root_centered_normalization_h36m(ske_tensor)

        return (img_tensor, ske_tensor), label, camid, 1, (ret_img_paths, ret_ske_paths)


class SeqTestPreprocessor(object):

    def __init__(self, seqset, dataset, seq_len, transform=None):
        super(SeqTestPreprocessor, self).__init__()
        self.seqset = seqset
        self.identities = dataset.identities
        self.identities_ske = dataset.identities_ske
        self.transform = transform
        self.seq_len = seq_len
        self.root = [dataset.images_dir]
        self.root.append(dataset.skes_dir)

    def __len__(self):
        return len(self.seqset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):

        start_ind, end_ind, pid, label, camid = self.seqset[index]

        imgseq = []
        skeseq = []
        ret_img_paths = []
        ret_ske_paths = []

        for ind in range(start_ind, end_ind):
            fname = self.identities[pid][camid][ind]
            fpath_img = osp.join(self.root[0], fname)
            ret_img_paths.append(fpath_img)
            imgrgb = Image.open(fpath_img).convert("RGB")
            fname_ske = self.identities_ske[pid][camid][ind]
            fpath_ske = osp.join(self.root[1], fname_ske)
            ret_ske_paths.append(fpath_ske)
            ske_data = torch.tensor(read_ilids_3d_skeleton(fpath_ske), dtype=torch.float32)
            imgseq.append(imgrgb)
            skeseq.append(ske_data)

        while len(imgseq) < self.seq_len:
            imgseq.append(imgrgb)
            skeseq.append(ske_data)

        imgseq = [imgseq]
        skeseq = [skeseq]

        if self.transform is not None:
            imgseq = self.transform(imgseq)

        img_tensor = torch.stack(imgseq[0], 0)
        ske_tensor = torch.stack(skeseq[0], 0)

        ske_tensor = root_centered_normalization_h36m(ske_tensor)

        return (img_tensor, ske_tensor), label, camid, 1, (ret_img_paths, ret_ske_paths)
