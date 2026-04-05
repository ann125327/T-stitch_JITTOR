import numpy as np
import jittor as jt
from jittor import nn


def _conv_act(in_ch, out_ch, k=3, s=1, p=1, act=True):
    layers = [nn.Conv(in_ch, out_ch, k, s, p)]
    if act:
        layers.append(nn.LeakyReLU(scale=0.1))
    return nn.Sequential(*layers)


def _make_base_grid(b, h, w):
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    grid = np.stack([x, y], axis=-1).astype(np.float32)  # H, W, 2
    grid = jt.array(grid).unsqueeze(0).broadcast([b, h, w, 2])
    return grid


def flow_warp(x, flow):
    """
    x:    [B, C, H, W]
    flow: [B, 2, H, W]  (pixel unit, dx/dy)
    """
    b, _, h, w = x.shape
    base_grid = _make_base_grid(b, h, w)
    flow = flow.permute(0, 2, 3, 1)
    grid = base_grid + flow
    grid_x = 2.0 * grid[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_y = 2.0 * grid[:, :, :, 1] / max(h - 1, 1) - 1.0
    norm_grid = jt.stack([grid_x, grid_y], dim=-1)
    return nn.grid_sample(x, norm_grid, mode="bilinear", padding_mode="border", align_corners=True)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = _conv_act(channels, channels, 3, 1, 1, act=True)
        self.conv2 = _conv_act(channels, channels, 3, 1, 1, act=False)
        self.act = nn.LeakyReLU(scale=0.1)

    def execute(self, x):
        out = self.conv2(self.conv1(x))
        return self.act(x + out)


class PyramidEncoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=48):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.level1 = nn.Sequential(
            _conv_act(in_channels, c1, 3, 1, 1),
            ResidualBlock(c1),
            ResidualBlock(c1),
        )
        self.down1 = _conv_act(c1, c2, 3, 2, 1)
        self.level2 = nn.Sequential(
            ResidualBlock(c2),
            ResidualBlock(c2),
        )
        self.down2 = _conv_act(c2, c3, 3, 2, 1)
        self.level3 = nn.Sequential(
            ResidualBlock(c3),
            ResidualBlock(c3),
        )

    def execute(self, x):
        f1 = self.level1(x)
        f2 = self.level2(self.down1(f1))
        f3 = self.level3(self.down2(f2))
        return [f1, f2, f3]


class TemporalAlignment(nn.Module):
    """
    时序特征对齐模块：
    使用轻量 flow 预测 + grid_sample 对齐相邻帧特征到中心帧特征坐标系。
    """
    def __init__(self, channels):
        super().__init__()
        hidden = max(32, channels // 2)
        self.offset_net = nn.Sequential(
            _conv_act(channels * 2, hidden, 3, 1, 1),
            _conv_act(hidden, hidden, 3, 1, 1),
            nn.Conv(hidden, 2, 3, 1, 1),
        )
        self.mask_net = nn.Sequential(
            _conv_act(channels * 2, hidden, 3, 1, 1),
            nn.Conv(hidden, 1, 3, 1, 1),
        )

    def execute(self, ref_feat, nbr_feat):
        inp = jt.concat([ref_feat, nbr_feat], dim=1)
        flow = self.offset_net(inp)
        mask = nn.Sigmoid()(self.mask_net(inp))
        aligned = flow_warp(nbr_feat, flow) * mask + ref_feat * (1.0 - mask)
        align_err = jt.abs(aligned - ref_feat).mean()
        return aligned, flow, align_err


class StitchModule(nn.Module):
    """
    Stitch 模块：
    将左邻帧对齐特征、中心帧特征、右邻帧对齐特征进行门控拼接融合。
    """
    def __init__(self, channels):
        super().__init__()
        hidden = max(32, channels // 2)
        self.gate = nn.Sequential(
            _conv_act(channels * 3, hidden, 3, 1, 1),
            nn.Conv(hidden, 3, 3, 1, 1),
        )
        self.refine = nn.Sequential(
            _conv_act(channels * 2, channels, 3, 1, 1),
            ResidualBlock(channels),
            nn.Conv(channels, channels, 3, 1, 1),
        )

    def execute(self, left_feat, center_feat, right_feat):
        cat_feat = jt.concat([left_feat, center_feat, right_feat], dim=1)
        weights = nn.softmax(self.gate(cat_feat), dim=1)
        w_left = weights[:, 0:1, :, :]
        w_center = weights[:, 1:2, :, :]
        w_right = weights[:, 2:3, :, :]
        fused = left_feat * w_left + center_feat * w_center + right_feat * w_right
        out = self.refine(jt.concat([fused, center_feat], dim=1)) + center_feat
        return out, weights


class MultiScaleFusion(nn.Module):
    """
    多尺度融合（FPN-style）：从低分辨率语义特征向高分辨率细节特征传递。
    """
    def __init__(self, base_channels=48):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        self.fuse_2 = nn.Sequential(
            _conv_act(c2 + c3, c2, 3, 1, 1),
            ResidualBlock(c2),
        )
        self.fuse_1 = nn.Sequential(
            _conv_act(c1 + c2, c1, 3, 1, 1),
            ResidualBlock(c1),
        )

    def execute(self, f1, f2, f3):
        up3 = nn.interpolate(f3, size=(f2.shape[2], f2.shape[3]), mode="bilinear", align_corners=False)
        f2_out = self.fuse_2(jt.concat([f2, up3], dim=1))
        up2 = nn.interpolate(f2_out, size=(f1.shape[2], f1.shape[3]), mode="bilinear", align_corners=False)
        f1_out = self.fuse_1(jt.concat([f1, up2], dim=1))
        return f1_out, f2_out, f3


class TStitchNet(nn.Module):
    """
    可训练版 T-Stitch 主干（时序对齐 + Stitch + 多尺度融合）
    输入:
        frames [B, T, C, H, W], 默认 T=3
    输出:
        dict:
            pred         [B, C, H, W]
            pred_pyramid [pred_l1, pred_l2_up, pred_l3_up]
            align_errors list of scalar losses
    """
    def __init__(self, in_channels=3, out_channels=3, base_channels=48, num_frames=3):
        super().__init__()
        if num_frames < 3 or num_frames % 2 == 0:
            raise ValueError("num_frames must be odd and >= 3.")
        self.num_frames = num_frames
        self.center_idx = num_frames // 2

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.encoder = PyramidEncoder(in_channels=in_channels, base_channels=base_channels)
        self.align_l1 = TemporalAlignment(c1)
        self.align_l2 = TemporalAlignment(c2)
        self.align_l3 = TemporalAlignment(c3)
        self.stitch_l1 = StitchModule(c1)
        self.stitch_l2 = StitchModule(c2)
        self.stitch_l3 = StitchModule(c3)
        self.ms_fusion = MultiScaleFusion(base_channels=base_channels)

        self.recon_l1 = nn.Sequential(
            ResidualBlock(c1),
            nn.Conv(c1, out_channels, 3, 1, 1),
        )
        self.recon_l2 = nn.Conv(c2, out_channels, 3, 1, 1)
        self.recon_l3 = nn.Conv(c3, out_channels, 3, 1, 1)

    def _check_shape(self, frames):
        if len(frames.shape) != 5:
            raise ValueError(f"Expected [B,T,C,H,W], got shape={frames.shape}")
        b, t, c, h, w = frames.shape
        if t != self.num_frames:
            raise ValueError(f"Expected T={self.num_frames}, got T={t}")
        if c != 3:
            raise ValueError(f"Expected C=3 RGB input, got C={c}")
        if h % 4 != 0 or w % 4 != 0:
            raise ValueError(f"Input H/W must be divisible by 4, got H={h}, W={w}")
        return b, t, c, h, w

    def _align_and_aggregate(self, align_module, ref_feat, neighbor_feats):
        if len(neighbor_feats) == 0:
            return ref_feat, [], jt.array([0.0]).mean()
        aligned_list = []
        losses = []
        for feat in neighbor_feats:
            aligned, _, loss = align_module(ref_feat, feat)
            aligned_list.append(aligned)
            losses.append(loss)
        aggregated = aligned_list[0]
        for k in range(1, len(aligned_list)):
            aggregated = aggregated + aligned_list[k]
        aggregated = aggregated / float(len(aligned_list))
        align_loss = jt.stack(losses).mean() if len(losses) > 1 else losses[0]
        return aggregated, aligned_list, align_loss

    def execute(self, frames):
        _, _, _, h, w = self._check_shape(frames)

        frame_list = [frames[:, i, :, :, :] for i in range(self.num_frames)]
        center = frame_list[self.center_idx]
        left_frames = frame_list[:self.center_idx]
        right_frames = frame_list[self.center_idx + 1:]

        feat_list = [self.encoder(f) for f in frame_list]
        center_feats = feat_list[self.center_idx]
        left_feats = feat_list[:self.center_idx]
        right_feats = feat_list[self.center_idx + 1:]

        align_errors = []

        # level 1
        left_l1 = [f[0] for f in left_feats]
        right_l1 = [f[0] for f in right_feats]
        left_aligned_l1, _, left_loss_l1 = self._align_and_aggregate(self.align_l1, center_feats[0], left_l1)
        right_aligned_l1, _, right_loss_l1 = self._align_and_aggregate(self.align_l1, center_feats[0], right_l1)
        stitch_l1, _ = self.stitch_l1(left_aligned_l1, center_feats[0], right_aligned_l1)
        align_errors.extend([left_loss_l1, right_loss_l1])

        # level 2
        left_l2 = [f[1] for f in left_feats]
        right_l2 = [f[1] for f in right_feats]
        left_aligned_l2, _, left_loss_l2 = self._align_and_aggregate(self.align_l2, center_feats[1], left_l2)
        right_aligned_l2, _, right_loss_l2 = self._align_and_aggregate(self.align_l2, center_feats[1], right_l2)
        stitch_l2, _ = self.stitch_l2(left_aligned_l2, center_feats[1], right_aligned_l2)
        align_errors.extend([left_loss_l2, right_loss_l2])

        # level 3
        left_l3 = [f[2] for f in left_feats]
        right_l3 = [f[2] for f in right_feats]
        left_aligned_l3, _, left_loss_l3 = self._align_and_aggregate(self.align_l3, center_feats[2], left_l3)
        right_aligned_l3, _, right_loss_l3 = self._align_and_aggregate(self.align_l3, center_feats[2], right_l3)
        stitch_l3, _ = self.stitch_l3(left_aligned_l3, center_feats[2], right_aligned_l3)
        align_errors.extend([left_loss_l3, right_loss_l3])

        f1, f2, f3 = self.ms_fusion(stitch_l1, stitch_l2, stitch_l3)

        pred_l1 = jt.clamp(center + self.recon_l1(f1), 0.0, 1.0)
        pred_l2 = jt.clamp(center + nn.interpolate(self.recon_l2(f2), size=(h, w), mode="bilinear", align_corners=False), 0.0, 1.0)
        pred_l3 = jt.clamp(center + nn.interpolate(self.recon_l3(f3), size=(h, w), mode="bilinear", align_corners=False), 0.0, 1.0)

        return {
            "pred": pred_l1,
            "pred_pyramid": [pred_l1, pred_l2, pred_l3],
            "align_errors": align_errors,
            "center": center,
            "left_count": len(left_frames),
            "right_count": len(right_frames),
        }
