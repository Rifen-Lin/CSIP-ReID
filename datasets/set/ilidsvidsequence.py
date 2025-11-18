from __future__ import absolute_import
import os
import os.path as osp
from datasets.set.datasequence import Datasequence
from utils.osutils import mkdir_if_missing
from utils.serialization import write_json
import tarfile
from glob import glob
import shutil
import scipy.io as sio

datasetname = "iLIDS-VID"


class infostruct(object):
    pass


class iLIDSVIDSEQUENCE(Datasequence):

    def __init__(self, root, split_id=0, seq_len=12, seq_srd=6, num_val=1, download=False):
        super(iLIDSVIDSEQUENCE, self).__init__(root, split_id=split_id)

        if not self._check_integrity():
            self.imgextract()
        self.load(seq_len, seq_srd, num_val)
        self.num_train_cams = 2
        self.num_train_vids = 1

        self.query, query_pid, query_camid, query_num = self._pluckseq_cam(
            self.identities, self.split["query"], seq_len, seq_srd, 0
        )
        self.queryinfo = infostruct()
        self.queryinfo.pid = query_pid
        self.queryinfo.camid = query_camid
        self.queryinfo.tranum = query_num

        self.gallery, gallery_pid, gallery_camid, gallery_num = self._pluckseq_cam(
            self.identities, self.split["gallery"], seq_len, seq_srd, 1
        )
        self.galleryinfo = infostruct()
        self.galleryinfo.pid = gallery_pid
        self.galleryinfo.camid = gallery_camid
        self.galleryinfo.tranum = gallery_num

    def imgextract(self):

        raw_dir = osp.join(self.root, "raw")
        exdir1 = osp.join(raw_dir, datasetname)
        fpath1 = osp.join(raw_dir, datasetname + ".tar")

        if not osp.isdir(exdir1):
            print("Extracting tar file")
            cwd = os.getcwd()
            tar = tarfile.open(fpath1)
            mkdir_if_missing(exdir1)
            os.chdir(exdir1)
            tar.extractall()
            tar.close()
            os.chdir(cwd)

        temp_images_dir = osp.join(self.root, "temp_images")
        mkdir_if_missing(temp_images_dir)

        temp_skes_dir = osp.join(self.root, "temp_skes")
        mkdir_if_missing(temp_skes_dir)

        images_dir = osp.join(self.root, "images")
        mkdir_if_missing(images_dir)

        skes_dir = osp.join(self.root, "skes")
        mkdir_if_missing(skes_dir)

        fpaths1 = sorted(glob(osp.join(exdir1, "i-LIDS-VID/sequences", "*/*/*.png")))

        fpaths2 = sorted(glob(osp.join(self.root, "i-LIDS-VID/sequences_skeleton_3d_hmr2.0", "*/*/*.json")))

        identities_imgraw = [[[] for _ in range(2)] for _ in range(319)]
        identities_skeraw = [[[] for _ in range(2)] for _ in range(319)]

        # image information
        for fpath in fpaths1:
            fname = osp.basename(fpath)
            fname_list = fname.split("_")
            cam_name = fname_list[0]
            pid_name = fname_list[1]
            cam = int(cam_name[-1])
            pid = int(pid_name[-3:])
            temp_fname = "{:08d}_{:02d}_{:04d}.png".format(pid, cam, len(identities_imgraw[pid - 1][cam - 1]))
            identities_imgraw[pid - 1][cam - 1].append(temp_fname)
            shutil.copy(fpath, osp.join(temp_images_dir, temp_fname))

        identities_temp = [x for x in identities_imgraw if x != [[], []]]
        identities_images = identities_temp

        for pid in range(len(identities_temp)):
            for cam in range(2):
                for img in range(len(identities_images[pid][cam])):
                    temp_fname = identities_temp[pid][cam][img]
                    fname = "{:08d}_{:02d}_{:04d}.png".format(pid, cam, img)
                    identities_images[pid][cam][img] = fname
                    shutil.copy(osp.join(temp_images_dir, temp_fname), osp.join(images_dir, fname))

        shutil.rmtree(temp_images_dir)

        # ske information
        for fpath in fpaths2:
            fname = osp.basename(fpath)
            fname_list = fname.split("_")
            cam_name = fname_list[0]
            pid_name = fname_list[1]
            cam = int(cam_name[-1])
            pid = int(pid_name[-3:])
            temp_fname = "{:08d}_{:02d}_{:04d}.json".format(pid, cam, len(identities_skeraw[pid - 1][cam - 1]))
            identities_skeraw[pid - 1][cam - 1].append(temp_fname)
            shutil.copy(fpath, osp.join(temp_skes_dir, temp_fname))

        identities_temp = [x for x in identities_skeraw if x != [[], []]]
        identities_skes = identities_temp

        for pid in range(len(identities_temp)):
            for cam in range(2):
                for img in range(len(identities_skes[pid][cam])):
                    temp_fname = identities_temp[pid][cam][img]
                    fname = "{:08d}_{:02d}_{:04d}.json".format(pid, cam, img)
                    identities_skes[pid][cam][img] = fname
                    shutil.copy(osp.join(temp_skes_dir, temp_fname), osp.join(skes_dir, fname))

        shutil.rmtree(temp_skes_dir)

        meta = {
            "name": "iLIDS-sequence",
            "shot": "sequence",
            "num_cameras": 2,
            "identities": identities_images,
            "identities_skes": identities_skes,
        }

        write_json(meta, osp.join(self.root, "meta.json"))

        # Consider fixed training and testing split
        splitmat_name = osp.join(exdir1, "train-test people splits", "train_test_splits_ilidsvid.mat")
        data = sio.loadmat(splitmat_name)
        person_list = data["ls_set"]
        num = len(identities_images)
        splits = []

        for i in range(10):
            pids = (person_list[i] - 1).tolist()
            trainval_pids = sorted(pids[: num // 2])
            test_pids = sorted(pids[num // 2 :])
            split = {"trainval": trainval_pids, "query": test_pids, "gallery": test_pids}

            splits.append(split)
        write_json(splits, osp.join(self.root, "splits.json"))

    def _pluckseq_cam(self, identities, indices, seq_len, seq_str, camid):

        ret = []
        per_id = []
        cam_id = []
        tra_num = []

        for index, pid in enumerate(indices):
            pid_images = identities[pid]
            cam_images = pid_images[camid]
            seqall = len(cam_images)
            seq_inds = [(start_ind, start_ind + seq_len) for start_ind in range(0, seqall - seq_len, seq_str)]
            if not seq_inds:
                seq_inds = [(0, seqall)]
            for seq_ind in seq_inds:
                ret.append((seq_ind[0], seq_ind[1], pid, index, camid))
            per_id.append(pid)
            cam_id.append(camid)
            tra_num.append(len(seq_inds))
        return ret, per_id, cam_id, tra_num
