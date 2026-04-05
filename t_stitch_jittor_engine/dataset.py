import os
import random
from typing import Dict, List

import numpy as np
from PIL import Image, ImageFilter

from jittor.dataset import Dataset


IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _is_image(path: str) -> bool:
    return os.path.splitext(path.lower())[1] in IMAGE_EXT


def _sorted_images(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    files = [os.path.join(folder, x) for x in os.listdir(folder) if _is_image(x)]
    return sorted(files)


def read_rgb(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr


def to_chw(img_hwc: np.ndarray) -> np.ndarray:
    return np.transpose(img_hwc, (2, 0, 1)).astype(np.float32)


class VideoSequenceDataset(Dataset):
    """
    目录结构（推荐）:
    data_root/
      train/
        seq_001/
          input/000000.png ...
          target/000000.png ...
      val/
        seq_101/
          input/...
          target/...

    兼容形式:
    - 只有 target 或只有 frames，会自动做退化生成 input 进行监督训练。
    """
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        num_frames: int = 3,
        crop_size: int = 256,
        synthetic_degrade: bool = True,
        noise_std: float = 8.0,
        batch_size: int = 4,
        shuffle: bool = True,
        num_workers: int = 4,
    ):
        super().__init__()
        if num_frames < 3 or num_frames % 2 == 0:
            raise ValueError("num_frames must be odd and >= 3.")
        self.data_root = data_root
        self.split = split
        self.num_frames = num_frames
        self.radius = num_frames // 2
        self.crop_size = crop_size
        self.synthetic_degrade = synthetic_degrade
        self.noise_std = noise_std
        self.is_train = split.lower() == "train"

        self.samples = self._scan_samples()
        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid samples found in '{data_root}' split='{split}'. "
                "Please check dataset structure and image files."
            )

        self.set_attrs(
            total_len=len(self.samples),
            batch_size=batch_size,
            shuffle=shuffle if self.is_train else False,
            drop_last=self.is_train,
            num_workers=num_workers,
        )

    def _resolve_split_dir(self) -> str:
        split_dir = os.path.join(self.data_root, self.split)
        if os.path.isdir(split_dir):
            return split_dir
        return self.data_root

    def _find_dirs(self, seq_dir: str):
        input_candidates = [
            os.path.join(seq_dir, "input"),
            os.path.join(seq_dir, "lq"),
            os.path.join(seq_dir, "blur"),
            os.path.join(seq_dir, "frames"),
            seq_dir,
        ]
        target_candidates = [
            os.path.join(seq_dir, "target"),
            os.path.join(seq_dir, "gt"),
            os.path.join(seq_dir, "hr"),
            os.path.join(seq_dir, "frames"),
            seq_dir,
        ]
        input_dir = next((d for d in input_candidates if len(_sorted_images(d)) > 0), None)
        target_dir = next((d for d in target_candidates if len(_sorted_images(d)) > 0), None)
        return input_dir, target_dir

    def _scan_samples(self) -> List[Dict]:
        split_dir = self._resolve_split_dir()
        seq_dirs = [os.path.join(split_dir, x) for x in os.listdir(split_dir)]
        seq_dirs = [d for d in seq_dirs if os.path.isdir(d)]
        seq_dirs = sorted(seq_dirs)
        if len(seq_dirs) == 0 and len(_sorted_images(split_dir)) > 0:
            seq_dirs = [split_dir]

        all_samples: List[Dict] = []
        for seq_dir in seq_dirs:
            input_dir, target_dir = self._find_dirs(seq_dir)
            if target_dir is None:
                continue

            target_frames = _sorted_images(target_dir)
            if len(target_frames) < self.num_frames:
                continue

            input_frames = _sorted_images(input_dir) if input_dir is not None else []
            input_map = {os.path.basename(p): p for p in input_frames}

            for i in range(self.radius, len(target_frames) - self.radius):
                center_name = os.path.basename(target_frames[i])
                frame_paths = []
                from_target = False
                for j in range(i - self.radius, i + self.radius + 1):
                    name_j = os.path.basename(target_frames[j])
                    if name_j in input_map:
                        frame_paths.append(input_map[name_j])
                    else:
                        frame_paths.append(target_frames[j])
                        from_target = True

                sample = {
                    "seq": os.path.basename(seq_dir),
                    "frame_paths": frame_paths,
                    "target_path": target_frames[i],
                    "from_target": from_target,
                }
                all_samples.append(sample)
        return all_samples

    def _degrade(self, img: np.ndarray) -> np.ndarray:
        if not self.synthetic_degrade:
            return img
        h, w = img.shape[:2]
        pil = Image.fromarray((img * 255.0).astype(np.uint8))

        if random.random() < 0.5:
            pil = pil.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.2)))

        if random.random() < 0.8:
            scale = random.choice([1, 2, 3])
            if min(h, w) // scale >= 8 and scale > 1:
                lr = pil.resize((w // scale, h // scale), Image.BICUBIC)
                pil = lr.resize((w, h), Image.BICUBIC)

        out = np.asarray(pil).astype(np.float32) / 255.0
        noise = np.random.randn(*out.shape).astype(np.float32) * (random.uniform(0.0, self.noise_std) / 255.0)
        out = np.clip(out + noise, 0.0, 1.0)
        return out

    def _paired_crop(self, frames: List[np.ndarray], target: np.ndarray):
        h, w = target.shape[:2]
        crop = self.crop_size
        if (not self.is_train) or crop <= 0:
            return frames, target
        if h < crop or w < crop:
            return frames, target

        top = random.randint(0, h - crop)
        left = random.randint(0, w - crop)
        frames = [x[top:top + crop, left:left + crop, :] for x in frames]
        target = target[top:top + crop, left:left + crop, :]
        return frames, target

    def _augment(self, frames: List[np.ndarray], target: np.ndarray):
        if not self.is_train:
            return frames, target
        if random.random() < 0.5:
            frames = [np.flip(x, axis=1).copy() for x in frames]
            target = np.flip(target, axis=1).copy()
        if random.random() < 0.5:
            frames = [np.flip(x, axis=0).copy() for x in frames]
            target = np.flip(target, axis=0).copy()
        if random.random() < 0.3:
            frames = list(reversed(frames))
        return frames, target

    @staticmethod
    def _ensure_divisible(img: np.ndarray, div: int = 4) -> np.ndarray:
        h, w = img.shape[:2]
        h2 = h - (h % div)
        w2 = w - (w % div)
        if h2 == h and w2 == w:
            return img
        top = (h - h2) // 2
        left = (w - w2) // 2
        return img[top:top + h2, left:left + w2, :]

    def __getitem__(self, idx):
        item = self.samples[idx]
        target = read_rgb(item["target_path"])

        frames = []
        for p in item["frame_paths"]:
            frm = read_rgb(p)
            if item["from_target"]:
                frm = self._degrade(frm)
            frames.append(frm)

        # 尺寸对齐到 target
        th, tw = target.shape[:2]
        frames = [x[:th, :tw, :] for x in frames]

        frames, target = self._paired_crop(frames, target)
        frames, target = self._augment(frames, target)

        # 保证网络下采样兼容
        target = self._ensure_divisible(target, 4)
        h, w = target.shape[:2]
        frames = [self._ensure_divisible(x[:h, :w, :], 4) for x in frames]

        lq = np.stack([to_chw(x) for x in frames], axis=0).astype(np.float32)  # [T,C,H,W]
        gt = to_chw(target).astype(np.float32)  # [C,H,W]
        return lq, gt
