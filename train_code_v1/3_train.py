# -*- coding: utf-8 -*-
"""
3_train.py
─────────────────────────────────────────────────────────
训练拼豆格子颜色分类器。

核心改动（v3）：
  1. 全参数训练，不做冻结热身
  2. 学习率：线性 warmup 0→target_lr（warmup_epochs），然后保持不变
  3. 训练集：每 epoch 在内存中在线随机生成（无限多样性），不写磁盘
  4. 测试集：固定磁盘文件，只生成一次
  5. 网络保留 ResNet-18（内置 BatchNorm 齐全）

用法
────
  python 3_train.py                         # 全默认
  python 3_train.py --lr 1e-4               # 指定目标学习率
  python 3_train.py --resume models/last.pth
"""

import os
import sys
import time
import argparse
import json
import importlib

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.models as models
import torchvision.transforms as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as CFG

# 导入生成器（不依赖磁盘数据）
_gen_mod = importlib.import_module('1_generate_dataset')
NormalBeadGenerator = _gen_mod.NormalBeadGenerator
EmptyBeadGenerator  = _gen_mod.EmptyBeadGenerator
load_bead_list      = _gen_mod.load_bead_list

# 导入 label map 工具
_ds_mod = importlib.import_module('2_build_dataset')
load_label_map = _ds_mod.load_label_map
BeadCellDataset = _ds_mod.BeadCellDataset
build_dataloader = _ds_mod.build_dataloader


# ──────────────────────────────────────────────
#  在线生成 Dataset（训练集专用）
# ──────────────────────────────────────────────

class OnlineBeadDataset(Dataset):
    """
    每次 __getitem__ 都实时生成一张新图，无需磁盘。
    每个 epoch 传入不同的 epoch_seed，保证数据多样性。
    """

    def __init__(self, bead_list, id_to_idx, samples_per_class=500,
                 epoch_seed=0, transform=None):
        """
        bead_list: [(bead_id, rgb), ...]  共221个
        id_to_idx: {"A01": 0, ..., "空": 221}
        samples_per_class: 每类生成多少张（每 epoch）
        epoch_seed: 本 epoch 的随机种子
        """
        self.bead_list = bead_list
        self.id_to_idx = id_to_idx
        self.samples_per_class = samples_per_class
        self.epoch_seed = epoch_seed
        self.transform = transform

        # 构建索引：(class_idx_in_list, sample_idx)
        # 0..220 = 普通色号，221 = 空
        n_classes = len(bead_list) + 1  # 221 + 1 = 222
        self._indices = []
        for cls_i in range(n_classes):
            for s_i in range(samples_per_class):
                self._indices.append((cls_i, s_i))

        # 预先为每个类别初始化生成器（不含 rng，rng 在 __getitem__ 里按需创建）
        self._bead_list = bead_list  # 已有

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, index):
        cls_i, s_i = self._indices[index]

        # 用 epoch_seed + index 做种，保证同 epoch 内可复现、不同 epoch 不同
        sample_seed = self.epoch_seed * 100_000_000 + index
        rng = np.random.default_rng(sample_seed)

        if cls_i < len(self._bead_list):
            # 普通色号
            bead_id, rgb = self._bead_list[cls_i]
            gen = NormalBeadGenerator(bead_id, rgb, rng)
            label = self.id_to_idx[bead_id]
        else:
            # 空类
            gen = EmptyBeadGenerator(rng)
            label = self.id_to_idx["空"]

        img = gen.generate_one()  # PIL Image

        if self.transform:
            img = self.transform(img)

        return img, label

    def set_epoch(self, epoch):
        """在每个 epoch 开始前调用，更新种子"""
        self.epoch_seed = epoch


# ──────────────────────────────────────────────
#  Transform（与 2_build_dataset 保持一致）
# ──────────────────────────────────────────────

def build_train_transform():
    size = CFG.INPUT_SIZE
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    return T.Compose([
        T.Resize((size, size)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(degrees=5),
        T.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05, hue=0.02),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

def build_eval_transform():
    size = CFG.INPUT_SIZE
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


# ──────────────────────────────────────────────
#  模型构建
# ──────────────────────────────────────────────

def build_model(backbone: str, n_classes: int, pretrained: bool = True,
                dropout: float = 0.3) -> nn.Module:
    weights_arg = "DEFAULT" if pretrained else None

    if backbone == "resnet18":
        model = models.resnet18(weights=weights_arg)
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(model.fc.in_features, n_classes)
        )
    elif backbone == "resnet34":
        model = models.resnet34(weights=weights_arg)
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(model.fc.in_features, n_classes)
        )
    elif backbone == "resnet50":
        model = models.resnet50(weights=weights_arg)
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(model.fc.in_features, n_classes)
        )
    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights_arg)
        in_f = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_f, n_classes)
        )
    else:
        raise ValueError(f"不支持的骨干：{backbone}")
    return model


# ──────────────────────────────────────────────
#  学习率调度：线性 warmup → 保持恒定
# ──────────────────────────────────────────────

def get_lr(epoch: int, warmup_epochs: int, target_lr: float) -> float:
    """
    epoch: 当前 epoch（从 1 开始）
    warmup_epochs: 线性从 0 增长到 target_lr 的 epoch 数
    target_lr: 目标学习率
    """
    if epoch <= warmup_epochs:
        return target_lr * (epoch / warmup_epochs)
    return target_lr


def set_lr(optimizer, lr: float):
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# ──────────────────────────────────────────────
#  单 epoch 训练 / 验证
# ──────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

        if (batch_idx + 1) % 100 == 0:
            print(f"  Epoch {epoch} [{batch_idx+1}/{len(loader)}] "
                  f"loss={loss.item():.4f} acc={correct/total:.4f}")

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


# ──────────────────────────────────────────────
#  Checkpoint
# ──────────────────────────────────────────────

def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    start_epoch = ckpt.get("epoch", 0) + 1
    best_val_acc = ckpt.get("best_val_acc", 0.0)
    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    print(f"  已加载 checkpoint：{path}（epoch={start_epoch-1}, best={best_val_acc:.4f}）")
    return start_epoch, best_val_acc


# ──────────────────────────────────────────────
#  TrainLogger
# ──────────────────────────────────────────────

class TrainLogger:
    def __init__(self, log_path):
        self.log_path = log_path
        self.history = []

    def log(self, epoch, train_loss, train_acc, val_loss, val_acc, lr):
        entry = dict(epoch=epoch,
                     train_loss=round(train_loss, 6),
                     train_acc=round(train_acc, 6),
                     val_loss=round(val_loss, 6),
                     val_acc=round(val_acc, 6),
                     lr=lr)
        self.history.append(entry)
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

    def print_latest(self):
        e = self.history[-1]
        print(f"  Epoch {e['epoch']:3d} | "
              f"train loss={e['train_loss']:.4f} acc={e['train_acc']:.4f} | "
              f"val loss={e['val_loss']:.4f} acc={e['val_acc']:.4f} | "
              f"lr={e['lr']:.2e}")


# ──────────────────────────────────────────────
#  主训练流程
# ──────────────────────────────────────────────

def train(args):
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # 标签
    id_to_idx, idx_to_id = load_label_map()
    n_cls = len(idx_to_id)
    print(f"类别数：{n_cls}")

    # 色板（用于在线生成）
    bead_list = load_bead_list()
    print(f"色板加载：{len(bead_list)} 色")

    # 在线训练集 & 固定测试集
    train_transform = build_train_transform()
    eval_transform  = build_eval_transform()

    train_dataset = OnlineBeadDataset(
        bead_list=bead_list,
        id_to_idx=id_to_idx,
        samples_per_class=args.samples_per_class,
        epoch_seed=0,
        transform=train_transform,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.workers > 0),
    )

    # 测试集：用磁盘上固定的 test.txt（先确保存在）
    try:
        test_loader = build_dataloader("test", batch_size=args.batch_size,
                                       num_workers=args.workers,
                                       transform=eval_transform)
        print(f"测试集：{len(test_loader.dataset)} 条（固定，来自磁盘）")
    except FileNotFoundError:
        print("[警告] 找不到 test.txt，将跳过测试集评估")
        test_loader = None

    print(f"训练集：每 epoch 在线生成 {len(train_dataset):,} 条")

    # 模型（全参数训练，不冻结）
    model = build_model(args.backbone, n_cls,
                        pretrained=not args.no_pretrain,
                        dropout=args.dropout)
    model = model.to(device)
    print(f"骨干：{args.backbone}（预训练={'否' if args.no_pretrain else '是'}，Dropout={args.dropout}）")
    print("全参数训练（不冻结骨干）")

    # 损失
    criterion = nn.CrossEntropyLoss()

    # 优化器（全参数）
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-8,   # 初始很小，warmup 逐步升到 target_lr
        weight_decay=args.weight_decay,
    )

    # 断点续训
    start_epoch = 1
    best_val_acc = 0.0
    if args.resume and os.path.exists(args.resume):
        start_epoch, best_val_acc = load_checkpoint(args.resume, model, optimizer)
        print(f"  从 epoch {start_epoch} 继续训练")

    logger = TrainLogger(os.path.join(CFG.CHECKPOINTS_DIR, "train_log.json"))
    patience_counter = 0
    best_ckpt = CFG.BEST_MODEL_PATH

    print(f"\n开始训练（共 {args.epochs} epochs，warmup={args.warmup_epochs} epochs，"
          f"目标 lr={args.lr:.1e}）")
    print("=" * 65)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        # 1. 更新学习率（warmup → 恒定）
        cur_lr = get_lr(epoch, args.warmup_epochs, args.lr)
        set_lr(optimizer, cur_lr)

        # 2. 更新训练集 epoch seed（确保每 epoch 数据不同）
        train_dataset.set_epoch(epoch)

        # 3. 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch)

        # 4. 测试集评估
        if test_loader:
            val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        else:
            val_loss, val_acc = 0.0, 0.0

        logger.log(epoch, train_loss, train_acc, val_loss, val_acc, cur_lr)
        logger.print_latest()
        print(f"  耗时：{time.time()-t0:.1f}s")

        # 5. 保存最优
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            save_checkpoint({
                "epoch": epoch,
                "backbone": args.backbone,
                "n_classes": n_cls,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
            }, best_ckpt)
            print(f"  ★ 新最优 test_acc={best_val_acc:.4f} → {best_ckpt}")
        else:
            patience_counter += 1
            print(f"  test_acc 未提升，patience={patience_counter}/{args.patience}")

        # 6. 保存 last
        save_checkpoint({
            "epoch": epoch,
            "backbone": args.backbone,
            "n_classes": n_cls,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
        }, os.path.join(CFG.CHECKPOINTS_DIR, "last.pth"))

        # 7. 早停
        if patience_counter >= args.patience:
            print(f"\n早停（{args.patience} epochs 无提升），训练结束。")
            break

    print("=" * 65)
    print(f"训练完成！最优 test_acc={best_val_acc:.4f}")


# ──────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="训练拼豆格子颜色分类器（在线生成数据）")
    parser.add_argument("--backbone",    default=CFG.BACKBONE)
    parser.add_argument("--epochs",      type=int,   default=200)
    parser.add_argument("--batch-size",  type=int,   default=CFG.BATCH_SIZE)
    parser.add_argument("--lr",          type=float, default=1e-4,
                        help="目标学习率（warmup 结束后保持此值）")
    parser.add_argument("--warmup-epochs", type=int, default=3,
                        help="线性 warmup epoch 数")
    parser.add_argument("--samples-per-class", type=int, default=500,
                        help="每 epoch 每类生成多少张训练样本")
    parser.add_argument("--workers",     type=int,   default=4)
    parser.add_argument("--no-pretrain", action="store_true")
    parser.add_argument("--resume",      default="",  help="断点续训路径")
    parser.add_argument("--dropout",     type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience",    type=int,   default=20,
                        help="早停 patience（无提升 epoch 数）")
    args = parser.parse_args()

    print("=" * 55)
    print("Step 3 - 训练分类模型（在线生成 + 全参数训练）")
    print("=" * 55)
    print(f"  backbone:          {args.backbone}")
    print(f"  epochs:            {args.epochs}")
    print(f"  batch_size:        {args.batch_size}")
    print(f"  target_lr:         {args.lr:.1e}")
    print(f"  warmup_epochs:     {args.warmup_epochs}")
    print(f"  samples_per_class: {args.samples_per_class}")
    print(f"  dropout:           {args.dropout}")
    print(f"  weight_decay:      {args.weight_decay}")
    print(f"  patience:          {args.patience}")
    print()

    train(args)


if __name__ == "__main__":
    main()
