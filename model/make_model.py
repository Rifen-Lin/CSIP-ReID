import os
import cv2
import torch
import torch.nn as nn
from torch.nn import LayerNorm
import numpy as np
import scipy.sparse
from collections import OrderedDict
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from loss.softmax_loss import CrossEntropyLabelSmooth

_tokenizer = _Tokenizer()
from .clip.model import QuickGELU, LayerNorm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .SkeGraphTrans import build_skeleton_encoder


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        nn.init.constant_(m.bias, 0.0)

    elif classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def generate_adj_matrices_h36m():
    j_pair_1 = np.array(
        [
            10,
            9,
            9,
            8,
            8,
            11,
            11,
            12,
            12,
            13,
            8,
            14,
            14,
            15,
            15,
            16,
            8,
            7,
            7,
            0,
            0,
            1,
            1,
            2,
            2,
            3,
            0,
            4,
            4,
            5,
            5,
            6,
        ]
    )
    j_pair_2 = np.array(
        [
            9,
            10,
            8,
            9,
            11,
            8,
            12,
            11,
            13,
            12,
            14,
            8,
            15,
            14,
            16,
            15,
            7,
            8,
            0,
            7,
            1,
            0,
            2,
            1,
            3,
            2,
            4,
            0,
            5,
            4,
            6,
            5,
        ]
    )
    con_matrix = np.ones([len(j_pair_1)])
    adj_J = scipy.sparse.coo_matrix((con_matrix, (j_pair_1, j_pair_2)), shape=(17, 17)).toarray()
    return adj_J


def generate_pos_enc(adj_joint, k=10):
    def normalized_laplacian(adj_matrix):
        R = np.sum(adj_matrix, axis=1)
        R_sqrt = 1 / np.sqrt(R)
        D_sqrt = np.diag(R_sqrt)
        I = np.eye(adj_matrix.shape[0])
        return I - np.matmul(np.matmul(D_sqrt, adj_matrix), D_sqrt)

    L = normalized_laplacian(adj_joint)
    EigVal, EigVec = np.linalg.eig(L)
    idx = EigVal.argsort()
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    # exit()
    pos_enc_ori = EigVec[:, 1 : k + 1]

    sign_flip = np.random.rand(pos_enc_ori.shape[1])
    sign_flip[sign_flip >= 0.5] = 1.0
    sign_flip[sign_flip < 0.5] = -1.0
    pos_enc_ori = pos_enc_ori * sign_flip

    return pos_enc_ori


def generate_random_seq_mask(time_step=6, mask_num=2):
    mask = torch.zeros(time_step, dtype=torch.bool)
    keep_indices = torch.randperm(time_step)[: time_step - mask_num]
    mask[keep_indices] = True
    return mask


def generate_random_joint_mask(joint_num=17, drop_num=7):
    mask = torch.ones(joint_num, dtype=torch.bool)
    idx = torch.randperm(joint_num)[:drop_num]
    mask[idx] = False
    return mask


class SkeletonEncoder(nn.Module):
    def __init__(self, cfg, num_classes, skeleton_type="h36m", skeleton_dim=3, encoder_type="ST-GCN"):
        super(SkeletonEncoder, self).__init__()
        self.cfg = cfg
        self.num_classes = num_classes

        assert skeleton_type in ["h36m", "ntu-rgb+d"], "Error: not a supported skeleton_type"
        assert skeleton_dim in [2, 3], "Error: not a supported skeleton_dim"
        assert encoder_type in ["ST-GCN", "GAT", "SGT"], "Error: not a supported encoder_type"

        self.skeleton_type = skeleton_type
        self.skeleton_dim = skeleton_dim
        self.encoder_type = encoder_type

        # self.proj_ske = nn.Linear(self.cfg.MODEL.SKE.HIDDEN_DIM, 512)
        self.proj_ske = nn.Parameter(torch.empty(self.cfg.MODEL.SKE.HIDDEN_DIM, 512))
        nn.init.normal_(self.proj_ske, std=self.cfg.MODEL.SKE.HIDDEN_DIM**-0.5)

        if self.encoder_type == "ST-GCN":
            self.st_gcn = ST_GCN(
                in_channels=self.skeleton_dim,
                num_class=self.num_classes,
                graph_args={"layout": self.skeleton_type, "strategy": "spatial"},
                edge_importance_weighting=True,
            )
            # self.load_pretrained_stgcn("pretrained_models/st_gcn.kinetics-6fa43f73.pth")
        elif self.encoder_type == "SGT":
            self.skeleton_graph_transformer = build_skeleton_encoder(cfg=self.cfg)
        else:
            raise KeyError(f"Unknown encoder type {encoder_type}")

    def forward(self, x, finetune_ske=False, labels=None, ske_prototype=None):
        batch_size, n_frames, n_joints, n_dim = x.shape
        if self.encoder_type == "ST-GCN":
            feature = self.st_gcn.extract_feature(x.permute(0, 3, 1, 2).unsqueeze(-1))
            feature = feature.mean(dim=2).squeeze(-1).mean(dim=-1)
            feature = self.fc_out_stgcn(feature)
            return feature
        elif self.encoder_type == "SGT":
            if finetune_ske == False:
                adj_J = generate_adj_matrices_h36m()
                device = self.cfg.MODEL.DEVICE
                adj_J = torch.tensor(adj_J, dtype=torch.float32).to(device)
                pos_enc = generate_pos_enc(adj_J.cpu().numpy(), k=self.cfg.MODEL.SKE.K_POS_ENC)
                pos_enc = torch.tensor(pos_enc, dtype=torch.float32).to(device)
                _, out_dict = self.skeleton_graph_transformer(skeleton=x, pos_enc=pos_enc)
                ori_feature = out_dict["seq_ftr"]
                ori_feature = ori_feature @ self.proj_ske
                frame_feature = ori_feature
                feature = ori_feature.mean(dim=2).mean(dim=1)
                return feature, frame_feature
            else:
                adj_J = generate_adj_matrices_h36m()
                device = self.cfg.MODEL.DEVICE
                adj_J = torch.tensor(adj_J, dtype=torch.float32).to(device)
                pos_enc = generate_pos_enc(adj_J.cpu().numpy(), k=self.cfg.MODEL.SKE.K_POS_ENC)
                pos_enc = torch.tensor(pos_enc, dtype=torch.float32).to(device)
                seq_mask = generate_random_seq_mask(time_step=n_frames, mask_num=2)
                node_mask = generate_random_joint_mask(joint_num=n_joints, drop_num=7)
                loss_dict, out_dict = self.skeleton_graph_transformer(
                    skeleton=x,
                    pos_enc=pos_enc,
                    node_mask=node_mask,
                    seq_mask=seq_mask,
                    gt_lab=labels,
                    gt_class_ftr=ske_prototype,
                )
                ori_feature = out_dict["seq_ftr"]
                ori_feature = ori_feature @ self.proj_ske
                frame_feature = ori_feature
                feature = ori_feature.mean(dim=2).mean(dim=1)
                return loss_dict["total_loss"], feature, frame_feature


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(build_transformer, self).__init__()
        self.cfg = cfg
        self.model_name = cfg.MODEL.IMG.NAME
        self.cos_layer = cfg.MODEL.IMG.COS_LAYER
        self.neck = cfg.MODEL.IMG.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        if self.model_name == "ViT-B-16":
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == "RN50":
            self.in_planes = 2048
            self.in_planes_proj = 1024
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = cfg.MODEL.IMG.SIE_COE

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0] - 16) // cfg.MODEL.IMG.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1] - 16) // cfg.MODEL.IMG.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.IMG.STRIDE_SIZE[0]
        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")

        self.image_encoder = clip_model.visual

        if cfg.MODEL.IMG.SIE_CAMERA and cfg.MODEL.IMG.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=0.02)
            print("camera number is : {}".format(camera_num))
        elif cfg.MODEL.IMG.SIE_CAMERA:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=0.02)
            print("camera number is : {}".format(camera_num))
        elif cfg.MODEL.IMG.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=0.02)
            print("camera number is : {}".format(view_num))

        self.skeleton_encoder = SkeletonEncoder(
            cfg=cfg, num_classes=self.num_classes, skeleton_type="h36m", skeleton_dim=3, encoder_type="SGT"
        )

        self.VTU = VisualTokenUpdater()

        self.TMM = SkeletonGuidedTemporalModelingModule()
        self.xent_frame = CrossEntropyLabelSmooth(num_classes=self.num_classes)
        self.classifier_proj_temp = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_proj_temp.apply(weights_init_classifier)
        self.classifier_proj_temp2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_proj_temp2.apply(weights_init_classifier)
        self.bottleneck_proj_temp = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_proj_temp.bias.requires_grad_(False)
        self.bottleneck_proj_temp.apply(weights_init_kaiming)
        self.bottleneck_proj_temp2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_proj_temp2.bias.requires_grad_(False)
        self.bottleneck_proj_temp2.apply(weights_init_kaiming)

        self.PF = PrototypeFusion(self.in_planes_proj)

        self.classifier_ske = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_ske.apply(weights_init_classifier)
        self.bottleneck_ske = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_ske.bias.requires_grad_(False)
        self.bottleneck_ske.apply(weights_init_kaiming)

    def forward(
        self,
        img=None,
        ske=None,
        label=None,
        get_image=False,
        get_skeleton=False,
        cam_label=None,
        view_label=None,
        ske_prototype=None,
        vis_prototype=None,
        data_paths=None,
    ):
        if get_skeleton == True:
            B, T, J, C = ske.shape
            if self.skeleton_encoder.encoder_type == "ST-GCN":
                skeleton_feature = self.skeleton_encoder(ske)
            elif self.skeleton_encoder.encoder_type == "SGT":
                skeleton_feature, skeleton_frame_feature = self.skeleton_encoder(
                    ske
                )
            return skeleton_feature

        if get_image == True:
            B, T, C, H, W = img.shape
            img = img.view(-1, C, H, W)
            image_features_last, image_features, image_features_proj, RGB_attn = self.image_encoder(
                img
            )
            if self.model_name == "RN50":
                return image_features_proj[0]
            elif self.model_name == "ViT-B-16":
                return image_features_proj[:, 0]

        if self.model_name == "RN50":
            image_features_last, image_features, image_features_proj = self.image_encoder(img)
            img_feature_last = nn.functional.avg_pool2d(image_features_last, image_features_last.shape[2:4]).view(
                img.shape[0], -1
            )
            img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(img.shape[0], -1)
            img_feature_proj = image_features_proj[0]

        elif self.model_name == "ViT-B-16":

            B, T, C, H, W = img.shape

            if cam_label != None and view_label != None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
            elif cam_label != None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label]
            elif view_label != None:
                cv_embed = self.sie_coe * self.cv_embed[view_label]
            else:
                cv_embed = None

            cv_embed = cv_embed.repeat((1, T)).view(B * T, -1)

            img = img.view(-1, C, H, W)

            _, image_features, image_features_proj_raw, RGB_attn = self.image_encoder(
                img, cv_embed
            )

            img_feature = image_features[:, 0]
            img_feature_proj = image_features_proj_raw[:, 0]

            img_feature = img_feature.view(B, T, -1)
            img_feature_proj = img_feature_proj.view(B, T, -1)

            img_feature = img_feature.mean(dim=1)
            img_feature_proj = img_feature_proj.mean(dim=1)

        feat = self.bottleneck(img_feature)
        feat_proj = self.bottleneck_proj(img_feature_proj)

        if self.training:

            _, ske_frame_feat = self.skeleton_encoder(ske, finetune_ske=False)

            fusion_prototype = self.PF(ske_prototype, vis_prototype)

            fusion_prototype = fusion_prototype.unsqueeze(0).expand(B, -1, -1)
            img_k_v = image_features_proj_raw.view(B, T, -1, image_features_proj_raw.shape[-1]).mean(
                dim=1
            )
            ske_k_v = ske_frame_feat.mean(dim=1)
            fusion_k_v = torch.cat([img_k_v, ske_k_v], dim=1)
            fusion_prototype = fusion_prototype + self.VTU(fusion_prototype, fusion_k_v)
            logits = torch.einsum("bd,bkd->bk", img_feature_proj, fusion_prototype)

            cls_score = self.classifier(feat)
            cls_score_proj = self.classifier_proj(feat_proj)

            ft_for_another_branch = image_features.detach()
            image_features_SAT = ft_for_another_branch.permute(1, 0, 2)

            cls_f_sp = self.TMM(image_features_SAT, ske_frame_feat, train=True)
            cls_f_sp_tap = cls_f_sp.view(B, T, -1)
            cls_f_tp = cls_f_sp_tap.mean(1)
            feat_proj_frame = self.bottleneck_proj_temp(cls_f_sp)
            feat_proj_temp = self.bottleneck_proj_temp2(cls_f_tp)
            cls_score_proj_frame = self.classifier_proj_temp(feat_proj_frame)
            cls_score_proj_temp = self.classifier_proj_temp(feat_proj_temp)
            targetX = label.unsqueeze(1)
            targetX = targetX.expand(B, T)
            targetX = targetX.contiguous()
            targetX = targetX.view(B * T, -1)
            targetX = targetX.squeeze(1)
            loss_frame = self.xent_frame(cls_score_proj_frame, targetX)
            loss_frame = loss_frame / T

            return (
                [cls_score, cls_score_proj, cls_score_proj_temp],
                [img_feature, img_feature_proj, cls_f_tp],
                logits,
                loss_frame,
            )

        else:
            if self.neck_feat == "after":
                # print("Test with feature after BN")
                return torch.cat([feat, feat_proj], dim=1)
            else:
                ft_for_another_branch = image_features.detach()
                image_features_SAT = ft_for_another_branch.permute(1, 0, 2)
                cls_f_sp = self.TMM(image_features_SAT, None, train=False)
                cls_f_sp_tap = cls_f_sp.view(B, T, -1)
                cls_f_tp = cls_f_sp_tap.mean(1)

                return torch.cat([img_feature, img_feature_proj, cls_f_tp], dim=1)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location="cpu")
        model_dict = self.state_dict()

        loaded_params = []
        skipped_params = []

        for name in param_dict:
            clean_name = name.replace("module.", "")
            if clean_name in model_dict:
                try:
                    model_dict[clean_name].copy_(param_dict[name])
                    loaded_params.append(clean_name)
                except Exception as e:
                    print(f"[!] Failed to load parameter: {clean_name} — shape mismatch or other error: {e}")
                    skipped_params.append(clean_name)
            else:
                print(f"[!] Skip loading '{clean_name}': not found in current model.")
                skipped_params.append(clean_name)

        print(f"\n Successfully loaded {len(loaded_params)} parameters.")
        print(f"Skipped {len(skipped_params)} parameters.")
        print("Loaded pretrained model from {}".format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print("Loading pretrained model for finetuning from {}".format(model_path))

    def overlay_attention_on_sequence_overlap_avg_heads(
        RGB_attn, batch_idx, img_json_path, patch_embed, save_dir="./attn_maps_avg", save_video=True
    ):
        def expand_patch_attention(attn_map, num_x, num_y, img_w, img_h):
            patch_h = img_h // num_y
            patch_w = img_w // num_x
            full_map = np.zeros((img_h, img_w), dtype=np.float32)

            for i in range(num_y):
                for j in range(num_x):
                    y_start = i * patch_h
                    y_end = (i + 1) * patch_h
                    x_start = j * patch_w
                    x_end = (j + 1) * patch_w
                    full_map[y_start:y_end, x_start:x_end] = attn_map[i, j]

            return full_map

        os.makedirs(save_dir, exist_ok=True)
        num_x, num_y = patch_embed.num_x, patch_embed.num_y
        frames = len(img_json_path[batch_idx][0])
        print(f"Processing batch {batch_idx}, {frames} frames, Patch Grid: {num_y} x {num_x}")

        frame_images_for_video = []

        for frame_idx in range(frames):
            image_path = img_json_path[batch_idx][0][frame_idx]
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img_h, img_w = 256, 128
            img_resized = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_CUBIC)

            attn_index = batch_idx * frames + frame_idx
            att_mat = []
            for layer in range(len(RGB_attn)):
                layer_attn = RGB_attn[layer][attn_index]
                attn_mean = layer_attn.mean(dim=0)
                att_mat.append(attn_mean)
            att_mat = torch.stack(att_mat, dim=0)

            residual_att = torch.eye(att_mat.shape[1], device=att_mat.device)
            aug_att_mat = att_mat + residual_att
            aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1, keepdim=True)

            joint_attn = torch.zeros_like(aug_att_mat)
            joint_attn[0] = aug_att_mat[0]
            for n in range(1, aug_att_mat.shape[0]):
                joint_attn[n] = aug_att_mat[n] @ joint_attn[n - 1]
            final_attn = joint_attn[-1]

            attn_cls = final_attn[0, 1:].detach().cpu().numpy()
            attn_map_resized = attn_cls.reshape(num_y, num_x).astype(np.float32)
            # attn_map_resized = cv2.resize(attn_map_resized, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
            attn_map_resized = expand_patch_attention(attn_map_resized, num_x, num_y, img_w, img_h)
            attn_map_resized = (attn_map_resized - attn_map_resized.min()) / (
                attn_map_resized.max() - attn_map_resized.min() + 1e-8
            )
            attn_map_resized = np.uint8(255 * attn_map_resized)
            heatmap = cv2.applyColorMap(attn_map_resized, cv2.COLORMAP_JET)
            overlayed_img = cv2.addWeighted(img_resized, 0.7, heatmap, 0.3, 0)

            cv2.putText(
                overlayed_img,
                f"Frame {frame_idx}",
                (10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(255, 255, 255),
                thickness=2,
                lineType=cv2.LINE_AA,
            )
            
            filename = os.path.join(save_dir, f"batch{batch_idx}_frame{frame_idx}.png")
            cv2.imwrite(filename, cv2.cvtColor(overlayed_img, cv2.COLOR_RGB2BGR))
            print(f"Saved: {filename}")

            frame_images_for_video.append(overlayed_img)

        if save_video and len(frame_images_for_video) > 0:
            video_path = os.path.join(save_dir, f"batch{batch_idx}_attn_avg_residual_video.mp4")
            h, w, _ = frame_images_for_video[0].shape
            writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 2, (w, h))
            for frame_img in frame_images_for_video:
                writer.write(cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR))
            writer.release()
            print(f"Saved video: {video_path}")


def make_model(cfg, num_class, camera_num, view_num):
    model = build_transformer(num_class, camera_num, view_num, cfg)
    return model


from .clip import clip


def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model


class MulitHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim**-0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape
        B, M, C = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PromptGeneratorLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.1,
    ):
        super().__init__()
        self.self_attn = MulitHeadAttention(d_model, nhead, proj_drop=dropout)
        self.cross_attn = MulitHeadAttention(d_model, nhead, proj_drop=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4), QuickGELU(), nn.Dropout(dropout), nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x, visual):
        q = k = v = self.norm1(x)
        x = x + self.self_attn(q, k, v)
        q = self.norm2(x)
        x = x + self.cross_attn(q, visual, visual)
        x = x + self.dropout(self.mlp(self.norm3(x)))
        return x


class VisualTokenUpdater(nn.Module):

    def __init__(
        self,
        layers=2,
        embed_dim=512,
        alpha=0.1,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(embed_dim)
        self.memory_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.text_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )
        self.out_proj = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim))
        self.decoder = nn.ModuleList([PromptGeneratorLayer(embed_dim, embed_dim // 64) for _ in range(layers)])
        self.alpha = nn.Parameter(torch.ones(embed_dim) * alpha)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, text, visual):

        # B, N, C = visual.shape
        visual = self.memory_proj(visual)
        text = self.text_proj(text)
        # visual = self.norm(visual)
        for layer in self.decoder:
            text = layer(text, visual)
        text = self.out_proj(text)
        return text


class SkeletonGuidedTemporalModelingModule(nn.Module):
    def __init__(self, dim_vis=768, dim_ske=512, n_heads=8, dropout=0.1, time_step=8):
        super(SkeletonGuidedTemporalModelingModule, self).__init__()

        self.T = time_step
        self.dim_vis = dim_vis
        self.dim_ske = dim_ske
        self.common_dim = dim_vis

        self.msg_fc_vis = nn.Linear(dim_vis, self.common_dim)
        self.msg_ln_vis = LayerNorm(self.common_dim)
        self.msg_attn_vis = nn.MultiheadAttention(
            embed_dim=self.common_dim, num_heads=n_heads, dropout=dropout, batch_first=False
        )

        self.msg_fc_ske = nn.Linear(dim_ske, self.common_dim)
        self.msg_ln_ske = LayerNorm(self.common_dim)
        self.msg_attn_ske = nn.MultiheadAttention(
            embed_dim=self.common_dim, num_heads=n_heads, dropout=dropout, batch_first=False
        )

        self.cross_attn_ske_to_vis = nn.MultiheadAttention(
            embed_dim=self.common_dim, num_heads=n_heads, dropout=dropout, batch_first=True
        )

        self.norm_vis = LayerNorm(self.common_dim)
        self.norm_ske = LayerNorm(self.common_dim)

        self.attention = nn.MultiheadAttention(self.common_dim, n_heads, dropout=dropout)
        self.ln_1 = LayerNorm(self.common_dim)
        self.ln_2 = LayerNorm(self.common_dim)

        self.mlp = nn.Sequential(
            nn.Linear(self.common_dim, self.common_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.common_dim * 4, self.common_dim),
            nn.Dropout(dropout),
        )

        self.drop_path = nn.Identity()

        self.token_type_embed = nn.Embedding(4, dim_vis)

        self.token_attn_score = nn.Sequential(nn.Linear(dim_vis, 1), nn.Sigmoid())

    def forward(self, x_vis, x_ske=None, train=False):
        l, bt, d = x_vis.size()
        b = bt // self.T

        x_vis = x_vis.permute(1, 0, 2).contiguous().view(b, self.T, -1, self.common_dim)
        x_vis_mean = x_vis.mean(dim=2)
        msg_vis = self.msg_fc_vis(x_vis_mean).view(b, self.T, 1, self.common_dim)
        msg_vis = msg_vis.permute(1, 2, 0, 3).reshape(self.T, b, self.common_dim)
        msg_vis = (
            msg_vis
            + self.msg_attn_vis(
                self.msg_ln_vis(msg_vis), self.msg_ln_vis(msg_vis), self.msg_ln_vis(msg_vis), need_weights=False
            )[0]
        )
        msg_vis = msg_vis.permute(1, 0, 2)

        if train and x_ske is not None:
            b, t, j, d_ske = x_ske.size()
            x_ske = self.msg_fc_ske(x_ske)
            x_ske_mean = x_ske.mean(dim=2)
            msg_ske = x_ske_mean.view(b, self.T, 1, self.common_dim)
            msg_ske = msg_ske.permute(1, 2, 0, 3).reshape(self.T, b, self.common_dim)
            msg_ske = (
                msg_ske
                + self.msg_attn_ske(
                    self.msg_ln_ske(msg_ske), self.msg_ln_ske(msg_ske), self.msg_ln_ske(msg_ske), need_weights=False
                )[0]
            )
            msg_ske = msg_ske.permute(1, 0, 2)

            vis_from_ske, _ = self.cross_attn_ske_to_vis(
                self.norm_vis(msg_vis), self.norm_ske(msg_ske), self.norm_ske(msg_ske)
            )

            msg_vis = msg_vis + vis_from_ske

            msg_vis = msg_vis.view(1, b, self.T, self.common_dim)
            msg_ske = msg_ske.view(1, b, self.T, self.common_dim)
            x_ske = x_ske.view(j, b, self.T, self.common_dim)
            x_vis = x_vis.permute(2, 0, 1, 3)

            x_vis = x_vis + self.token_type_embed(torch.tensor(0, device=x_vis.device)).view(1, 1, 1, -1)
            msg_vis = msg_vis + self.token_type_embed(torch.tensor(1, device=x_vis.device)).view(1, 1, 1, -1)
            x_ske = x_ske + self.token_type_embed(torch.tensor(2, device=x_vis.device)).view(1, 1, 1, -1)
            msg_ske = msg_ske + self.token_type_embed(torch.tensor(3, device=x_vis.device)).view(1, 1, 1, -1)

            x_tm = torch.cat([x_vis, msg_vis, x_ske, msg_ske], dim=0)
        else:
            x_vis = x_vis.permute(2, 0, 1, 3)
            msg_vis = msg_vis.view(1, b, self.T, self.common_dim)

            x_vis = x_vis + self.token_type_embed(torch.tensor(0, device=x_vis.device)).view(1, 1, 1, -1)
            msg_vis = msg_vis + self.token_type_embed(torch.tensor(1, device=x_vis.device)).view(1, 1, 1, -1)

            x_tm = torch.cat([x_vis, msg_vis], dim=0)

        x_tm = x_tm.view(x_tm.size(0), -1, self.common_dim)
        x_tm = x_tm + self.drop_path(self.attention(self.ln_1(x_tm), self.ln_1(x_tm), self.ln_1(x_tm))[0])
        x_tm = x_tm + self.drop_path(self.mlp(self.ln_2(x_tm)))

        attn_score = self.token_attn_score(x_tm)
        attn_score = attn_score / (attn_score.sum(dim=0, keepdim=True) + 1e-6)
        x_weighted = (x_tm * attn_score).sum(dim=0)

        return x_weighted


class TAggregate(nn.Module):
    def __init__(self, clip_length=None, embed_dim=2048, n_layers=6):
        super(TAggregate, self).__init__()
        self.clip_length = clip_length
        drop_rate = 0.0
        enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8)
        self.transformer_enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers, norm=nn.LayerNorm(embed_dim))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, clip_length + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        with torch.no_grad():
            trunc_normal_(self.pos_embed, std=0.02)
            trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        nvids = x.shape[0]

        cls_tokens = self.cls_token.expand(nvids, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x.transpose_(1, 0)
        o = self.transformer_enc(x)

        return o[0]


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TemporalTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks((x))


class visual_prompt(nn.Module):
    def __init__(self, sim_head, clip_state_dict, T):
        super().__init__()
        self.sim_header = sim_head
        self.T = T
        assert sim_head in ["meanP", "Transf", "Conv_1D", "Transf_cls"]

        if self.sim_header == "Transf" or self.sim_header == "Transf_cls" or self.sim_header == "Conv_1D":
            # embed_dim = clip_state_dict["text_projection"].shape[1]
            # context_length = clip_state_dict["positional_embedding"].shape[0]
            # transformer_width = clip_state_dict["ln_final.weight"].shape[0]
            embed_dim = 768
            context_length = 77
            transformer_width = 512

            transformer_heads = transformer_width // 64
            self.frame_position_embeddings = nn.Embedding(context_length, embed_dim)

        if self.sim_header == "Transf":
            self.transformer = TemporalTransformer(width=embed_dim, layers=6, heads=transformer_heads)
            print("layer=6")

        self.apply(self.init_weights)

        if self.sim_header == "Transf_cls":
            self.transformer = TAggregate(clip_length=self.T, embed_dim=embed_dim, n_layers=6)

        if self.sim_header == "Conv_1D":
            self.shift = nn.Conv1d(embed_dim, embed_dim, 3, padding=1, groups=embed_dim, bias=False)
            weight = torch.zeros(embed_dim, 1, 3)
            weight[: embed_dim // 4, 0, 0] = 1.0
            weight[embed_dim // 4 : embed_dim // 4 + embed_dim // 2, 0, 1] = 1.0
            weight[-embed_dim // 4 :, 0, 2] = 1.0
            self.shift.weight = nn.Parameter(weight)

    def init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            if "beta" in dir(module) and "gamma" in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x):
        b, t, c = x.size()
        x = x.contiguous()
        if self.sim_header == "meanP":
            pass

        elif self.sim_header == "Conv_1D":
            x_original = x
            x = x.view(-1, c, t)
            x = self.shift(x.float())
            x = x.permute(0, 2, 1)
            x = x.type(x_original.dtype) + x_original

        elif self.sim_header == "Transf":
            x_original = x
            seq_length = t
            position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
            position_ids = position_ids.unsqueeze(0).expand(x.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            x = x + frame_position_embeddings

            x = x.permute(1, 0, 2)
            x = self.transformer(x)
            x = x.permute(1, 0, 2)
            x = x.type(x_original.dtype) + x_original

        elif self.sim_header == "Transf_cls":
            x_original = x
            return self.transformer(x).type(x_original.dtype)

        else:
            raise ValueError("Unknown optimizer: {}".format(self.sim_header))

        return x.mean(dim=1, keepdim=False)


class PrototypeFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU(), nn.Linear(dim, 1), nn.Sigmoid())

    def forward(self, p_ske, p_img):
        concat = torch.cat([p_ske, p_img], dim=1)
        alpha = self.fc(concat)
        fused = alpha * p_ske + (1 - alpha) * p_img
        return fused


class PromptLearner(nn.Module):
    def __init__(self, num_class, dataset_name, dtype, token_embedding):
        super().__init__()
        if dataset_name == "VehicleID" or dataset_name == "veri":
            ctx_init = "A photo of a X X X X vehicle."
        else:
            ctx_init = "A photo of a X X X X person."

        ctx_dim = 512
        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = 4

        tokenized_prompts = clip.tokenize(ctx_init).cuda()
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)
        self.tokenized_prompts = tokenized_prompts

        n_cls_ctx = 4
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors)

        self.register_buffer("token_prefix", embedding[:, : n_ctx + 1, :])
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx :, :])
        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx

    def forward(self, label):
        cls_ctx = self.cls_ctx[label]
        b = label.shape[0]
        prefix = self.token_prefix.expand(b, -1, -1)
        suffix = self.token_suffix.expand(b, -1, -1)

        prompts = torch.cat(
            [
                prefix,
                cls_ctx,
                suffix,
            ],
            dim=1,
        )

        return prompts


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x
