import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads=8, dropout=0.5):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)

        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim))

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, J, C = x.shape
        x_ = x.view(B * T, J, C)
        qkv = self.qkv_proj(x_)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B * T, J, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B * T, J, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B * T, J, self.n_heads, self.head_dim).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores.clamp(-5, 5), dim=-1)
        attn_out = attn_weights @ v
        attn_out = attn_out.transpose(1, 2).contiguous().view(B * T, J, C)
        attn_out = self.out_proj(attn_out)
        attn_out = x_ + self.dropout(attn_out)
        out = self.norm1(attn_out)

        ffn_out = self.ffn(out)
        out = out + self.dropout(ffn_out)
        out = self.norm2(out)
        return out.view(B, T, J, C)


class JointPromptModule(nn.Module):
    def __init__(self, feature_dim, joint_num, loss_type="l2"):
        super(JointPromptModule, self).__init__()
        self.joint_num = joint_num
        self.loss_type = loss_type

        self.fc1 = nn.Linear(feature_dim, feature_dim)
        self.fc2 = nn.Linear(feature_dim, joint_num * 3)

    def forward(self, h, node_mask, gt_pos):
        B, T, J, C = h.shape

        node_mask = torch.tensor(node_mask, dtype=torch.float32, device=h.device)
        node_mask = node_mask.view(1, 1, J, 1).expand(B, T, J, C)
        node_mask = node_mask.float()

        h_masked = h * node_mask
        valid_counts = node_mask[..., 0].sum(dim=-1, keepdim=True).clamp(min=1e-6)
        h_pooled = h_masked.sum(dim=2) / valid_counts

        out = F.relu(self.fc1(h_pooled))
        pred = self.fc2(out)

        if self.loss_type == "l1":
            loss = F.l1_loss(pred, gt_pos, reduction="mean")
        elif self.loss_type == "l2":
            loss = torch.mean((pred - gt_pos) ** 2)
        elif self.loss_type == "MSE":
            loss = F.mse_loss(pred, gt_pos, reduction="mean")
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")

        return loss, pred


class TrajectoryPromptModule(nn.Module):
    def __init__(self, feature_dim, time_step, loss_type="l2"):
        super(TrajectoryPromptModule, self).__init__()
        self.time_step = time_step
        self.loss_type = loss_type

        self.fc1 = nn.Linear(feature_dim, feature_dim // 2)
        self.fc2 = nn.Linear(feature_dim // 2, time_step * 3)

    def forward(self, h_ori, seq_mask, gt_pos):
        B, T, J, C = h_ori.shape

        seq_mask = torch.tensor(seq_mask, dtype=torch.float32, device=h_ori.device)
        seq_mask = seq_mask.view(1, T, 1, 1).expand(B, T, J, C)
        seq_mask = seq_mask.float()
        h_masked = h_ori * seq_mask

        h_masked = h_masked.permute(0, 2, 1, 3)
        valid_counts = seq_mask.permute(0, 2, 1, 3).sum(dim=2).clamp(min=1e-6)
        pooled = h_masked.sum(dim=2) / valid_counts.squeeze(-1)

        out = F.relu(self.fc1(pooled))
        pred = self.fc2(out)

        target = gt_pos.permute(0, 2, 1, 3).reshape(B, J, T * 3)

        if self.loss_type == "l1":
            loss = F.l1_loss(pred, target, reduction="mean")
        elif self.loss_type == "l2":
            loss = torch.mean((pred - target) ** 2)
        elif self.loss_type == "MSE":
            loss = F.mse_loss(pred, target, reduction="mean")
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")

        return loss, pred


def GPC_seq_loss(t: float, pseudo_lab: torch.Tensor, all_ftr: torch.Tensor, cluster_ftr: torch.Tensor) -> torch.Tensor:
    all_ftr = F.normalize(all_ftr, dim=-1)
    cluster_ftr = F.normalize(cluster_ftr, dim=-1).to(all_ftr.device)

    logits = torch.matmul(all_ftr, cluster_ftr.T) / t
    loss = F.cross_entropy(logits, pseudo_lab)
    return loss


def GPC_ske_loss(t: float, labels: torch.Tensor, all_ftr: torch.Tensor, cluster_ftr: torch.Tensor) -> torch.Tensor:
    B, T, C = all_ftr.shape
    all_ftr = all_ftr.reshape(B * T, C)
    cluster_ftr = cluster_ftr.to(all_ftr.device)

    logits = torch.matmul(all_ftr, cluster_ftr.T) / t

    labels_framewise = labels.view(-1, 1).repeat(1, T).view(-1)

    loss = F.cross_entropy(logits, labels_framewise)
    return loss


class SkeletonGraphTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        C = self.config.MODEL.SKE.HIDDEN_DIM
        J = self.config.MODEL.SKE.JOINT_NUM
        T = self.config.MODEL.SKE.TIME_STEP

        self.input_fc1 = nn.Linear(3, C)
        self.input_fc2 = nn.Linear(C, C)

        self.pos_fc = nn.Linear(self.config.MODEL.SKE.K_POS_ENC, C)

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(dim=C, n_heads=self.config.MODEL.SKE.N_HEADS, dropout=self.config.MODEL.SKE.DROPOUT)
                for _ in range(self.config.MODEL.SKE.NUM_LAYERS)
            ]
        )

        self.joint_prompt = JointPromptModule(C, J, config.MODEL.SKE.S_TYPE)
        self.traj_prompt = TrajectoryPromptModule(C, T, config.MODEL.SKE.T_TYPE)

    def forward(self, skeleton, pos_enc, node_mask=None, seq_mask=None, gt_lab=None, gt_class_ftr=None):
        B, T, J, _ = skeleton.shape
        C = self.config.MODEL.SKE.HIDDEN_DIM

        x = skeleton.view(B * T * J, 3)
        x = F.relu(self.input_fc1(x))
        x = self.input_fc2(x)
        seq_ftr = x.view(B, T, J, C)

        if self.config.MODEL.SKE.USE_POS_ENC:
            pos_enc = pos_enc.view(1, 1, J, -1).repeat(B, T, 1, 1)
            pos_enc = self.pos_fc(pos_enc)
            seq_ftr = seq_ftr + pos_enc

        h = seq_ftr
        for block in self.transformer_blocks:
            h = block(h)

        h_ori = h

        loss_dict = {}
        out_dict = {}

        if self.config.MODEL.SKE.USE_S_PROMPT and node_mask is not None:
            gt_joint = skeleton.view(B, T, J * 3)
            joint_loss, joint_pred = self.joint_prompt(h, node_mask, gt_joint)
            loss_dict["joint_prompt"] = joint_loss
            out_dict["joint_pred"] = joint_pred
        else:
            joint_loss = torch.tensor(0.0, device=h.device)

        if self.config.MODEL.SKE.USE_T_PROMPT and seq_mask is not None:
            traj_loss, traj_pred = self.traj_prompt(h_ori, seq_mask, skeleton)
            loss_dict["traj_prompt"] = traj_loss
            out_dict["traj_pred"] = traj_pred
        else:
            traj_loss = torch.tensor(0.0, device=h.device)

        if gt_lab is not None and gt_class_ftr is not None:
            h_seq = h.mean(dim=2)
            seq_ftr_global = h_seq.mean(dim=1)
            ske_contrast_loss = GPC_ske_loss(self.config.MODEL.SKE.TEMP_SKE, gt_lab, h_seq, gt_class_ftr)
            seq_contrast_loss = GPC_seq_loss(self.config.MODEL.SKE.TEMP_SEQ, gt_lab, seq_ftr_global, gt_class_ftr)
            H_loss = (
                1 - self.config.MODEL.SKE.SEQ_LAMBDA
            ) * ske_contrast_loss + self.config.MODEL.SKE.SEQ_LAMBDA * seq_contrast_loss
            loss_dict["ske_contrast"] = ske_contrast_loss
            loss_dict["seq_contrast"] = seq_contrast_loss
            loss_dict["H_loss"] = H_loss

            recon_loss = joint_loss * self.config.MODEL.SKE.PROMPT_LAMBDA + traj_loss * (
                1 - self.config.MODEL.SKE.PROMPT_LAMBDA
            )
            total_loss = self.config.MODEL.SKE.GPC_LAMBDA * H_loss + (1 - self.config.MODEL.SKE.GPC_LAMBDA) * recon_loss
            loss_dict["recon_loss"] = recon_loss
            loss_dict["total_loss"] = total_loss

            out_dict["seq_ftr"] = h
        else:
            out_dict["seq_ftr"] = h

        return loss_dict, out_dict


def build_skeleton_encoder(cfg):

    model = SkeletonGraphTransformer(cfg)
    print("===========Building SkeletonGraphTransformer as Skeleton-Encoder===========")
    return model
