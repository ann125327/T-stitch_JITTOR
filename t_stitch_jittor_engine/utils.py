import json
import logging
import os
import random
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

import jittor as jt


def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def set_random_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    if hasattr(jt, "set_global_seed"):
        jt.set_global_seed(seed)
    elif hasattr(jt, "set_seed"):
        jt.set_seed(seed)


def safe_float(v) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(v)
    except Exception:
        try:
            return float(v.numpy())
        except Exception:
            return float(v.data)


def setup_logger(log_dir: str, name: str = "train") -> logging.Logger:
    ensure_dir(log_dir)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    log_path = os.path.join(log_dir, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val: float, n: int = 1):
        self.sum += float(val) * n
        self.count += int(n)
        self.avg = self.sum / max(self.count, 1)


def charbonnier_loss(pred, target, eps: float = 1e-6):
    diff = pred - target
    return jt.sqrt(diff * diff + eps).mean()


def edge_loss(pred, target):
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    tgt_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    tgt_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return (jt.abs(pred_dx - tgt_dx).mean() + jt.abs(pred_dy - tgt_dy).mean()) * 0.5


class TStitchLoss:
    def __init__(
        self,
        w_rec: float = 1.0,
        w_edge: float = 0.05,
        w_temporal: float = 0.1,
        w_pyramid: float = 0.2,
    ):
        self.w_rec = w_rec
        self.w_edge = w_edge
        self.w_temporal = w_temporal
        self.w_pyramid = w_pyramid

    def __call__(self, outputs: Dict, gt):
        pred = outputs["pred"]
        pred_pyramid = outputs.get("pred_pyramid", [pred])
        align_errors = outputs.get("align_errors", [])

        l_rec = charbonnier_loss(pred, gt)
        l_edge = edge_loss(pred, gt)

        pyr_terms = [charbonnier_loss(p, gt) for p in pred_pyramid]
        l_pyr = jt.stack(pyr_terms).mean() if len(pyr_terms) > 1 else pyr_terms[0]

        if len(align_errors) == 0:
            l_temp = jt.array([0.0]).mean()
        else:
            l_temp = jt.stack(align_errors).mean()

        total = self.w_rec * l_rec + self.w_edge * l_edge + self.w_temporal * l_temp + self.w_pyramid * l_pyr

        return {
            "total": total,
            "rec": l_rec,
            "edge": l_edge,
            "temporal": l_temp,
            "pyramid": l_pyr,
        }


def compute_psnr(pred, gt, max_val: float = 1.0) -> float:
    mse = safe_float(((pred - gt) ** 2).mean())
    mse = max(mse, 1e-10)
    return 10.0 * np.log10((max_val * max_val) / mse)


def compute_ssim_fast(pred, gt) -> float:
    """
    轻量近似 SSIM（全图统计版），用于训练期监控。
    """
    x = pred.numpy()
    y = gt.numpy()
    x = np.clip(x, 0.0, 1.0).astype(np.float32)
    y = np.clip(y, 0.0, 1.0).astype(np.float32)
    mu_x = x.mean()
    mu_y = y.mean()
    var_x = x.var()
    var_y = y.var()
    cov_xy = ((x - mu_x) * (y - mu_y)).mean()
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2))
    return float(ssim)


def save_checkpoint(
    path: str,
    model,
    optimizer,
    epoch: int,
    best_psnr: float,
    scheduler_state: Optional[Dict] = None,
):
    ckpt = {
        "epoch": int(epoch),
        "best_psnr": float(best_psnr),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler_state or {},
    }
    ensure_dir(os.path.dirname(path))
    jt.save(ckpt, path)


def load_checkpoint(path: str, model, optimizer=None) -> Tuple[int, float, Dict]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    ckpt = jt.load(path)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    epoch = int(ckpt.get("epoch", 0))
    best_psnr = float(ckpt.get("best_psnr", 0.0))
    scheduler_state = ckpt.get("scheduler", {})
    return epoch, best_psnr, scheduler_state


def dump_json(obj: Dict, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def tensor_to_image_uint8(tensor_chw) -> np.ndarray:
    arr = tensor_chw.numpy()
    arr = np.transpose(arr, (1, 2, 0))
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0 + 0.5).astype(np.uint8)
    return arr


def save_image(tensor_chw, path: str):
    ensure_dir(os.path.dirname(path))
    img = tensor_to_image_uint8(tensor_chw)
    Image.fromarray(img).save(path)
