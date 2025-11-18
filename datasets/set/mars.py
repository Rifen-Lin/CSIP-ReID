from __future__ import print_function, absolute_import

import os.path as osp
from scipy.io import loadmat
import numpy as np

from utils.serialization import write_json, read_json

import json
import os
import torch
import shutil
import errno


class infostruct(object):
    pass


class Mars(object):

    def __init__(self, args, min_seq_len=0):
        self.args = args
        self.root = self.args.DATASETS.ROOT_DIR
        self.train_name_path = osp.join(self.root, "info/train_name.txt")
        self.test_name_path = osp.join(self.root, "info/test_name.txt")
        self.track_train_info_path = osp.join(self.root, "info/tracks_train_info.mat")
        self.track_test_info_path = osp.join(self.root, "info/tracks_test_info.mat")
        self.query_IDX_path = osp.join(self.root, "info/query_IDX.mat")

        self.split_train_json_path = osp.join(self.root, "cashed_dataset/split_train.json")
        self.split_query_json_path = osp.join(self.root, "cashed_dataset/split_query.json")
        self.split_gallery_json_path = osp.join(self.root, "cashed_dataset/split_gallery.json")

        self.filtered_split_train_json_path = osp.join(self.root, "filtered_cashed_dataset/split_train_cleaned.json")
        self.filtered_split_query_json_path = osp.join(self.root, "filtered_cashed_dataset/split_query_cleaned.json")
        self.filtered_split_gallery_json_path = osp.join(
            self.root, "filtered_cashed_dataset/split_gallery_cleaned.json"
        )

        self._check_before_run()

        train_names = self._get_names(self.train_name_path)
        test_names = self._get_names(self.test_name_path)

        track_train = loadmat(self.track_train_info_path)["track_train_info"]
        track_test = loadmat(self.track_test_info_path)["track_test_info"]

        query_IDX = loadmat(self.query_IDX_path)["query_IDX"].squeeze()
        query_IDX -= 1  # index from 0
        track_query = track_test[query_IDX, :]

        gallery_IDX = [i for i in range(track_test.shape[0]) if i not in query_IDX]
        track_gallery = track_test[gallery_IDX, :]

        train, num_train_tracklets, num_train_pids, num_train_imgs, num_train_cams, num_train_vids = self._process_data(
            train_names,
            track_train,
            home_dir="bbox_train",
            relabel=True,
            min_seq_len=min_seq_len,
            out_path=self.split_train_json_path,
            filtered_out_path=self.filtered_split_train_json_path,
        )

        query, num_query_tracklets, num_query_pids, num_query_imgs, query_pid, query_camid = self._process_gallery_data(
            test_names,
            track_query,
            home_dir="bbox_test",
            relabel=False,
            min_seq_len=min_seq_len,
            out_path=self.split_query_json_path,
            filtered_out_path=self.filtered_split_query_json_path,
        )

        gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs, gallery_pid, gallery_camid = (
            self._process_gallery_data(
                test_names,
                track_gallery,
                home_dir="bbox_test",
                relabel=False,
                min_seq_len=min_seq_len,
                out_path=self.split_gallery_json_path,
                filtered_out_path=self.filtered_split_gallery_json_path,
            )
        )

        num_imgs_per_tracklet = num_train_imgs + num_query_imgs + num_gallery_imgs
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        print("=> MARS loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

        self.queryinfo = infostruct()
        self.queryinfo.pid = query_pid
        self.queryinfo.camid = query_camid
        self.queryinfo.tranum = num_query_imgs

        self.galleryinfo = infostruct()
        self.galleryinfo.pid = gallery_pid
        self.galleryinfo.camid = gallery_camid
        self.galleryinfo.tranum = num_gallery_imgs

        self.num_train_cams = num_train_cams
        self.num_train_vids = num_train_vids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.train_name_path):
            raise RuntimeError("'{}' is not available".format(self.train_name_path))
        if not osp.exists(self.test_name_path):
            raise RuntimeError("'{}' is not available".format(self.test_name_path))
        if not osp.exists(self.track_train_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_train_info_path))
        if not osp.exists(self.track_test_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_test_info_path))
        if not osp.exists(self.query_IDX_path):
            raise RuntimeError("'{}' is not available".format(self.query_IDX_path))

    def _get_names(self, fpath):
        names = []
        with open(fpath, "r") as f:
            for line in f:
                new_line = line.rstrip()
                names.append(new_line)
        return names

    def _process_data(
        self, names, meta_data, home_dir=None, relabel=False, min_seq_len=0, out_path="", filtered_out_path=""
    ):

        # if osp.exists(filtered_out_path):
        #     print("=> {} generated before, awesome! Using filtered cashed dataset!".format(filtered_out_path))
        #     split = read_json(filtered_out_path)
        #     return split['tracklets'], split['num_tracklets'], split['num_pids'], split['num_imgs_per_tracklet'], split['num_cams'], split['num_tracks']

        if osp.exists(out_path):
            print("=> {} generated before, awesome!".format(out_path))
            split = read_json(out_path)
            return (
                split["tracklets"],
                split["num_tracklets"],
                split["num_pids"],
                split["num_imgs_per_tracklet"],
                split["num_cams"],
                split["num_tracks"],
            )
        print("=> Automatically generating split (might take a while for the first time, have a coffe)")

        assert home_dir in ["bbox_train", "bbox_test"]
        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:, 2].tolist()))
        num_pids = len(pid_list)

        if relabel:
            pid2label = {pid: label for label, pid in enumerate(pid_list)}

        tracklets = []
        num_imgs_per_tracklet = []
        cams = []

        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx, ...]
            start_index, end_index, pid, camid = data

            cams += [int(camid)]

            if pid == -1:
                continue  # junk images are just ignored
            assert 1 <= camid <= 6
            if relabel:
                pid = pid2label[pid]
            camid -= 1
            # index starts from 0
            img_names = names[start_index - 1 : end_index]

            # make sure image names correspond to the same person
            pnames = [img_name[:4] for img_name in img_names]
            assert len(set(pnames)) == 1, "Error: a single tracklet contains different person images"

            # make sure all images are captured under the same camera
            camnames = [img_name[5] for img_name in img_names]
            assert len(set(camnames)) == 1, "Error: images are captured under different cameras!"

            # append image names with directory information
            img_paths = [osp.join(self.root, home_dir, img_name[:4], img_name) for img_name in img_names]  # list<16>
            skeleton_paths = [
                osp.join(self.root, "skeleton_3d_hmr2.0", home_dir, img_name[:4], img_name.split(".")[0] + ".json")
                for img_name in img_names
            ]

            if len(img_paths) >= min_seq_len:
                img_paths = tuple(img_paths)
                skeleton_paths = tuple(skeleton_paths)
                tracklets.append(((img_paths, skeleton_paths), int(pid), int(camid), 1))
                num_imgs_per_tracklet.append(len(img_paths))

        num_tracklets = len(tracklets)

        cams = set(cams)
        num_cams = len(cams)

        print("Saving split to {}".format(out_path))
        split_dict = {
            "tracklets": tracklets,
            "num_tracklets": num_tracklets,
            "num_pids": num_pids,
            "num_imgs_per_tracklet": num_imgs_per_tracklet,
            "num_cams": num_cams,
            "num_tracks": 1,
        }
        write_json(split_dict, out_path)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet, num_cams, 1

    def _process_gallery_data(
        self, names, meta_data, home_dir=None, relabel=False, min_seq_len=0, out_path="", filtered_out_path=""
    ):

        # if osp.exists(filtered_out_path):
        #     print("=> {} generated before, awesome! Using filtered cashed dataset!".format(filtered_out_path))
        #     split = read_json(filtered_out_path)
        #     return (
        #         split["tracklets"],
        #         split["num_tracklets"],
        #         split["num_pids"],
        #         split["num_imgs_per_tracklet"],
        #         split["pids"],
        #         split["camid"],
        #     )

        if osp.exists(out_path):
            print("=> {} generated before, awesome!".format(out_path))
            split = read_json(out_path)
            return (
                split["tracklets"],
                split["num_tracklets"],
                split["num_pids"],
                split["num_imgs_per_tracklet"],
                split["pids"],
                split["camid"],
            )

        assert home_dir in ["bbox_train", "bbox_test"]
        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:, 2].tolist()))
        num_pids = len(pid_list)

        if relabel:
            pid2label = {pid: label for label, pid in enumerate(pid_list)}

        tracklets = []
        num_imgs_per_tracklet = []
        gallery_pid = []
        gallery_camid = []

        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx, ...]
            start_index, end_index, pid, camid = data

            if pid == -1:
                continue  # junk images are just ignored
            assert 1 <= camid <= 6
            if relabel:
                pid = pid2label[pid]
            camid -= 1
            # index starts from 0
            img_names = names[start_index - 1 : end_index]

            # make sure image names correspond to the same person
            pnames = [img_name[:4] for img_name in img_names]
            assert len(set(pnames)) == 1, "Error: a single tracklet contains different person images"

            # make sure all images are captured under the same camera
            camnames = [img_name[5] for img_name in img_names]
            assert len(set(camnames)) == 1, "Error: images are captured under different cameras!"

            # append image names with directory information
            img_paths = [osp.join(self.root, home_dir, img_name[:4], img_name) for img_name in img_names]  # list<16>
            skeleton_paths = [
                osp.join(self.root, "skeleton_3d_hmr2.0", home_dir, img_name[:4], img_name.split(".")[0] + ".json")
                for img_name in img_names
            ]

            if len(img_paths) >= min_seq_len:
                img_paths = tuple(img_paths)
                skeleton_paths = tuple(skeleton_paths)
                tracklets.append(((img_paths, skeleton_paths), int(pid), int(camid), 1))
                num_imgs_per_tracklet.append(len(img_paths))

            gallery_pid.append(int(pid))
            gallery_camid.append(int(camid))

        num_tracklets = len(tracklets)

        print("Saving split to {}".format(out_path))
        split_dict = {
            "tracklets": tracklets,
            "num_tracklets": num_tracklets,
            "num_pids": num_pids,
            "num_imgs_per_tracklet": num_imgs_per_tracklet,
            "pids": gallery_pid,
            "camid": gallery_camid,
        }
        write_json(split_dict, out_path)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet, gallery_pid, gallery_camid


if __name__ == "__main__":
    # test
    dataset = Mars()
