# -*- coding: utf-8 -*-
"""
2_build_dataset.py
─────────────────────────────────────────────────────────
PyTorch Dataset 定义 + 数据增强 + DataLoader 工厂函数。

本模块不独立运行，由 3_train.py 和 4_evaluate.py 导入使用。
也可以单独执行做快速健康检查：
    python 2_build_dataset.py

⚠ 注意：本模块使用 grid_category/train_dataset/ 目录
（由 1_generate_dataset.py 合成生成）。
"""

import os
import sys
import random
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as CFG


# ──────────────────────────────────────────────
#  标签映射工具
# ──────────────────────────────────────────────

def load_label_map(label_map_path: str = CFG.LABEL_MAP_PATH) -> Tuple[Dict[str, int], List[str]]:
    """
    读取 label_map.txt，返回：
      id_to_idx: {"A01": 0, "A02": 1, ...}
      idx_to_id: ["A01", "A02", ...]   （下标即类别索引）
    """
    if not os.path.exists(label_map_path):
        raise FileNotFoundError(
            f"找不到标签映射文件：{label_map_path}\n"
            "请先运行 python 1_generate_dataset.py 生成合成数据集。"
        )
    id_to_idx: Dict[str, int] = {}
    idx_to_id: List[str] = []
    with open(label_map_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                idx = int(parts[0])
                bead_id = parts[1]
                id_to_idx[bead_id] = idx
                # 扩展列表到正确长度
                while len(idx_to_id) <= idx:
                    idx_to_id.append("")
                idx_to_id[idx] = bead_id
    return id_to_idx, idx_to_id


def num_classes(label_map_path: str = CFG.LABEL_MAP_PATH) -> int:
    _, idx_to_id = load_label_map(label_map_path)
    return len(idx_to_id)


# ──────────────────────────────────────────────
#  数据增强：高斯噪声（torchvision 没有内置）
# ──────────────────────────────────────────────

class AddGaussianNoise:
    """给 Tensor 添加高斯噪声"""
    def __init__(self, std: float = 0.01):
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.std > 0:
            noise = torch.randn_like(tensor) * self.std
            return torch.clamp(tensor + noise, 0.0, 1.0)
        return tensor

    def __repr__(self):
        return f"AddGaussianNoise(std={self.std})"


# ──────────────────────────────────────────────
#  Transform 工厂
# ──────────────────────────────────────────────

def build_transform(split: str = "train") -> T.Compose:
    """
    根据 split 返回对应的 torchvision Transform。

    split: "train" | "val" | "test"

    训练集使用 config.AUGMENT_CONFIG 中定义的增强策略；
    验证/测试集只做 Resize + ToTensor + Normalize。
    """
    aug = CFG.AUGMENT_CONFIG
    size = CFG.INPUT_SIZE

    # ImageNet 均值/标准差（迁移学习预训练模型需要）
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if split == "train":
        ops = [T.Resize((size, size))]

        if aug.get("random_horizontal_flip"):
            ops.append(T.RandomHorizontalFlip())
        if aug.get("random_vertical_flip"):
            ops.append(T.RandomVerticalFlip())
        if aug.get("random_rotation"):
            deg = aug.get("rotation_degrees", 5)
            ops.append(T.RandomRotation(degrees=deg))
        if aug.get("color_jitter"):
            jitter_p = aug.get("color_jitter_params", {})
            ops.append(T.ColorJitter(**jitter_p))

        ops.append(T.ToTensor())
        ops.append(T.Normalize(mean=mean, std=std))

        if aug.get("gaussian_noise_std", 0) > 0:
            ops.append(AddGaussianNoise(std=aug["gaussian_noise_std"]))
        if aug.get("random_erasing"):
            p = aug.get("random_erasing_p", 0.2)
            ops.append(T.RandomErasing(p=p, scale=(0.02, 0.15), ratio=(0.3, 3.3)))

    else:
        # val / test：只做标准化
        ops = [
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]

    return T.Compose(ops)


# ──────────────────────────────────────────────
#  Dataset
# ──────────────────────────────────────────────

class BeadCellDataset(Dataset):
    """
    拼豆格子图像分类数据集。

    读取 dataset/split/{split}.txt 中的文件列表，
    图片实际路径 = images_dir / 相对路径。

    每条样本：(image_tensor, label_idx)
    """

    def __init__(
        self,
        split: str = "train",
        images_dir: str = CFG.DATASET_IMAGES_DIR,
        split_dir:  str = CFG.DATASET_SPLIT_DIR,
        transform: Optional[T.Compose] = None,
    ):
        self.split = split
        self.images_dir = images_dir
        self.transform = transform or build_transform(split)

        split_file = os.path.join(split_dir, f"{split}.txt")
        if not os.path.exists(split_file):
            raise FileNotFoundError(
                f"找不到划分文件：{split_file}\n"
                "请先运行 python 1_generate_dataset.py"
            )

        self.samples: List[Tuple[str, int]] = []  # (abs_path, label_idx)
        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 2:
                    continue
                rel_path, idx_str = parts
                abs_path = os.path.join(images_dir, rel_path)
                self.samples.append((abs_path, int(idx_str)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        img_path, label = self.samples[index]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    def __repr__(self):
        return (
            f"BeadCellDataset(split='{self.split}', "
            f"n_samples={len(self.samples)})"
        )


# ──────────────────────────────────────────────
#  DataLoader 工厂
# ──────────────────────────────────────────────

def build_dataloader(
    split: str = "train",
    batch_size: int = CFG.BATCH_SIZE,
    num_workers: int = CFG.NUM_WORKERS,
    **dataset_kwargs,
) -> DataLoader:
    """
    快速创建 DataLoader。

    Args:
        split: "train" | "val" | "test"
        batch_size: 批大小
        num_workers: 数据加载并行 worker 数
        **dataset_kwargs: 传递给 BeadCellDataset 的额外参数

    Returns:
        torch.utils.data.DataLoader
    """
    dataset = BeadCellDataset(split=split, **dataset_kwargs)
    shuffle = (split == "train")
    loader = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = (split == "train"),
    )
    return loader


# ──────────────────────────────────────────────
#  计算数据集的类别权重（应对样本不均衡）
# ──────────────────────────────────────────────

def compute_class_weights(
    split_dir: str = CFG.DATASET_SPLIT_DIR,
    n_classes: Optional[int] = None,
) -> torch.Tensor:
    """
    根据 train.txt 中各类别的样本数，计算反频率权重，
    用于 CrossEntropyLoss 的 weight 参数。

    返回：shape = (n_classes,) 的 FloatTensor
    """
    train_file = os.path.join(split_dir, "train.txt")
    counts = {}
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                idx = int(parts[1])
                counts[idx] = counts.get(idx, 0) + 1

    if n_classes is None:
        n_classes = max(counts.keys()) + 1

    freq = np.zeros(n_classes, dtype=np.float32)
    for idx, cnt in counts.items():
        freq[idx] = cnt

    # 避免除零
    freq = np.where(freq == 0, 1, freq)
    weights = 1.0 / freq
    weights = weights / weights.sum() * n_classes   # 归一化
    return torch.from_numpy(weights)


# ──────────────────────────────────────────────
#  健康检查（直接运行本文件时执行）
# ──────────────────────────────────────────────

def _health_check():
    print("=" * 50)
    print("2_build_dataset.py - 健康检查")
    print("=" * 50)

    try:
        id_to_idx, idx_to_id = load_label_map()
        print(f"标签映射加载成功：{len(idx_to_id)} 个类别")
        print(f"  前5个：{idx_to_id[:5]}")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    for split in ["train", "val", "test"]:
        try:
            ds = BeadCellDataset(split=split)
            print(f"\n{split} 集：{len(ds)} 条样本")
            img, label = ds[0]
            print(f"  样本0 → tensor形状: {img.shape}, 标签: {label} ({idx_to_id[label]})")
        except FileNotFoundError as e:
            print(f"  [{split}] 跳过：{e}")

    try:
        loader = build_dataloader("train", batch_size=8, num_workers=0)
        batch_imgs, batch_labels = next(iter(loader))
        print(f"\nDataLoader batch → imgs: {batch_imgs.shape}, labels: {batch_labels.shape}")
    except Exception as e:
        print(f"\n[DataLoader ERROR] {e}")

    print("\n✓ 健康检查完成")


if __name__ == "__main__":
    _health_check()
