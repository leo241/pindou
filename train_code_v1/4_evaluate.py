# -*- coding: utf-8 -*-
"""
4_evaluate.py
─────────────────────────────────────────────────────────
在测试集上评估训练好的模型，输出：
  - 整体 Top-1 / Top-3 准确率
  - 每类的 Precision / Recall / F1
  - 混淆矩阵热图（保存到 checkpoints/confusion_matrix.png）
  - 预测最差的 N 个类别（便于 debug 和后续 SFT）

用法
────
  python 4_evaluate.py
  python 4_evaluate.py --model checkpoints/best.pth --split test
  python 4_evaluate.py --top-k 3 --save-errors   # 同时保存预测错误的样本
"""

import os
import sys
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as CFG
# 数字开头模块名需要用 importlib
import importlib
_build_dataset = importlib.import_module('2_build_dataset')
build_dataloader = _build_dataset.build_dataloader
load_label_map = _build_dataset.load_label_map
BeadCellDataset = _build_dataset.BeadCellDataset
build_transform = _build_dataset.build_transform
_train = importlib.import_module('3_train')
build_model = _train.build_model


# ──────────────────────────────────────────────
#  推理
# ──────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, loader, device, top_k: int = 1):
    """
    对 loader 中所有样本跑一遍推理。

    Returns:
        all_preds:  np.ndarray shape (N,)    Top-1 预测
        all_topk:   np.ndarray shape (N, k)  Top-k 预测
        all_labels: np.ndarray shape (N,)    真实标签
        all_probs:  np.ndarray shape (N, C)  Softmax 概率
    """
    model.eval()
    all_preds  = []
    all_topk   = []
    all_labels = []
    all_probs  = []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs  = F.softmax(logits, dim=1).cpu().numpy()
        topk_preds = np.argsort(-probs, axis=1)[:, :top_k]

        all_preds.append(topk_preds[:, 0])
        all_topk.append(topk_preds)
        all_labels.append(labels.numpy())
        all_probs.append(probs)

    return (
        np.concatenate(all_preds),
        np.concatenate(all_topk, axis=0),
        np.concatenate(all_labels),
        np.concatenate(all_probs, axis=0),
    )


# ──────────────────────────────────────────────
#  指标计算
# ──────────────────────────────────────────────

def compute_metrics(preds, labels, topk_preds, n_classes: int):
    """
    返回字典：{
        "top1_acc": float,
        "topk_acc": float,
        "per_class": {idx: {"precision", "recall", "f1", "support"}},
        "confusion_matrix": np.ndarray shape (n_classes, n_classes)
    }
    """
    top1_acc = (preds == labels).mean()
    topk_acc = sum(labels[i] in topk_preds[i] for i in range(len(labels))) / len(labels)

    # 混淆矩阵
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(labels, preds):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1

    # Per-class
    per_class = {}
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[i] = {
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
            "support":   int(support),
        }

    return {
        "top1_acc":        round(float(top1_acc), 6),
        "topk_acc":        round(float(topk_acc), 6),
        "per_class":       per_class,
        "confusion_matrix": cm,
    }


# ──────────────────────────────────────────────
#  混淆矩阵可视化（仅 matplotlib 可用时）
# ──────────────────────────────────────────────

def save_confusion_matrix(cm: np.ndarray, idx_to_id: list, save_path: str,
                           max_classes: int = 50):
    """
    如果类别太多（>max_classes），只显示样本数最多的 max_classes 个类。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
    except ImportError:
        print("  [跳过] matplotlib 未安装，无法生成混淆矩阵图")
        return

    n = cm.shape[0]
    if n > max_classes:
        # 选最多样本的类
        top_idx = np.argsort(-cm.sum(axis=1))[:max_classes]
        cm = cm[np.ix_(top_idx, top_idx)]
        labels = [idx_to_id[i] if i < len(idx_to_id) else str(i) for i in top_idx]
    else:
        labels = [idx_to_id[i] if i < len(idx_to_id) else str(i) for i in range(n)]

    fig_size = max(10, len(labels) * 0.35)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    # 使用 log scale 防止高频类覆盖低频类
    cm_plot = cm.astype(float)
    cm_plot[cm_plot == 0] = np.nan
    im = ax.imshow(cm_plot, aspect="auto",
                   norm=LogNorm(vmin=1, vmax=cm.max() + 1),
                   cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("预测标签")
    ax.set_ylabel("真实标签")
    ax.set_title("混淆矩阵（对角=正确，对数色阶）")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  混淆矩阵已保存：{save_path}")


# ──────────────────────────────────────────────
#  打印报告
# ──────────────────────────────────────────────

def print_report(metrics: dict, idx_to_id: list, top_k: int, worst_n: int = 10):
    print(f"\n{'─'*50}")
    print(f"Top-1 准确率：{metrics['top1_acc']*100:.2f}%")
    print(f"Top-{top_k} 准确率：{metrics['topk_acc']*100:.2f}%")
    print(f"{'─'*50}")

    # 按 F1 升序排，取最差 worst_n 个
    per_class = metrics["per_class"]
    sorted_classes = sorted(per_class.items(), key=lambda x: x[1]["f1"])

    print(f"\n▼ 预测最差的 {worst_n} 个色号（F1 最低）：")
    print(f"  {'色号':<8} {'Precision':>10} {'Recall':>10} {'F1':>8} {'样本数':>8}")
    for idx, stat in sorted_classes[:worst_n]:
        name = idx_to_id[idx] if idx < len(idx_to_id) else str(idx)
        if stat["support"] == 0:
            continue
        print(
            f"  {name:<8} {stat['precision']:>10.4f} {stat['recall']:>10.4f} "
            f"{stat['f1']:>8.4f} {stat['support']:>8}"
        )

    # 宏平均
    valid = [v for v in per_class.values() if v["support"] > 0]
    macro_p  = np.mean([v["precision"] for v in valid])
    macro_r  = np.mean([v["recall"]    for v in valid])
    macro_f1 = np.mean([v["f1"]        for v in valid])
    print(f"\n宏平均 Precision={macro_p:.4f}  Recall={macro_r:.4f}  F1={macro_f1:.4f}")
    print(f"{'─'*50}")


# ──────────────────────────────────────────────
#  保存预测错误样本（便于 debug）
# ──────────────────────────────────────────────

def save_error_samples(dataset: BeadCellDataset, preds: np.ndarray,
                       labels: np.ndarray, idx_to_id: list,
                       save_dir: str, max_errors: int = 200):
    """
    把预测错误的样本保存到 save_dir/errors/{true_id}/{pred_id}_xxx.png。
    """
    os.makedirs(save_dir, exist_ok=True)
    count = 0
    for i, (pred, label) in enumerate(zip(preds, labels)):
        if pred == label:
            continue
        img_path, _ = dataset.samples[i]
        true_name = idx_to_id[label] if label < len(idx_to_id) else str(label)
        pred_name = idx_to_id[pred]  if pred  < len(idx_to_id) else str(pred)
        dest_dir = os.path.join(save_dir, true_name)
        os.makedirs(dest_dir, exist_ok=True)
        dest_name = f"pred_{pred_name}_{i:05d}.png"
        try:
            img = Image.open(img_path)
            img.save(os.path.join(dest_dir, dest_name))
        except Exception:
            pass
        count += 1
        if count >= max_errors:
            break
    print(f"  错误样本已保存（{count} 张）→ {save_dir}")


# ──────────────────────────────────────────────
#  主流程
# ──────────────────────────────────────────────

def evaluate(args):
    # ── 设备 ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")

    # ── 标签映射 ──
    _, idx_to_id = load_label_map()
    n_cls = len(idx_to_id)
    print(f"类别数：{n_cls}")

    # ── 加载模型 ──
    if not os.path.exists(args.model):
        print(f"[ERROR] 找不到模型文件：{args.model}")
        print("请先运行 3_train.py 训练模型。")
        return

    ckpt = torch.load(args.model, map_location="cpu")
    backbone = ckpt.get("backbone", CFG.BACKBONE)
    model = build_model(backbone, n_cls, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    print(f"模型加载成功：{args.model}（backbone={backbone}）")

    # ── 数据集 ──
    loader = build_dataloader(args.split, batch_size=args.batch_size, num_workers=args.workers)
    print(f"评估集（{args.split}）：{len(loader.dataset)} 条")

    # ── 推理 ──
    print("开始推理...")
    preds, topk_preds, labels, probs = run_inference(model, loader, device, top_k=args.top_k)

    # ── 指标 ──
    metrics = compute_metrics(preds, labels, topk_preds, n_classes=n_cls)
    print_report(metrics, idx_to_id, top_k=args.top_k)

    # ── 保存评估结果 JSON ──
    result_path = os.path.join(CFG.CHECKPOINTS_DIR, f"eval_{args.split}.json")
    save_data = {
        "split":    args.split,
        "top1_acc": metrics["top1_acc"],
        "topk_acc": metrics["topk_acc"],
        "top_k":    args.top_k,
        "per_class": {
            idx_to_id[int(k)] if int(k) < len(idx_to_id) else k: v
            for k, v in metrics["per_class"].items()
        },
    }
    os.makedirs(CFG.CHECKPOINTS_DIR, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"\n评估结果已保存：{result_path}")

    # ── 混淆矩阵 ──
    cm_path = os.path.join(CFG.CHECKPOINTS_DIR, "confusion_matrix.png")
    save_confusion_matrix(metrics["confusion_matrix"], idx_to_id, cm_path)

    # ── 保存错误样本 ──
    if args.save_errors:
        dataset = BeadCellDataset(split=args.split)
        errors_dir = os.path.join(CFG.CHECKPOINTS_DIR, "error_samples")
        save_error_samples(dataset, preds, labels, idx_to_id, errors_dir)

    print("\n评估完成！下一步运行：python 5_predict.py")


# ──────────────────────────────────────────────
#  CLI 入口
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="评估拼豆格子颜色分类模型")
    parser.add_argument("--model",       default=CFG.BEST_MODEL_PATH, help="模型 checkpoint 路径")
    parser.add_argument("--split",       default="test",  help="评估集：train/val/test")
    parser.add_argument("--batch-size",  type=int, default=128, help="批大小")
    parser.add_argument("--workers",     type=int, default=CFG.NUM_WORKERS)
    parser.add_argument("--top-k",       type=int, default=3,   help="Top-K 准确率的 K 值")
    parser.add_argument("--save-errors", action="store_true",   help="保存预测错误的样本图片")
    args = parser.parse_args()

    print("=" * 50)
    print("Step 4 - 模型评估（合成数据集 train_dataset/）")
    print("=" * 50)
    evaluate(args)


if __name__ == "__main__":
    main()
