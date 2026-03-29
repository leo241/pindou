# -*- coding: utf-8 -*-
"""
5_predict.py
─────────────────────────────────────────────────────────
推理接口：给定格子图片或图纸图片区域，输出色号预测。

提供两种使用方式：
  1. 作为模块导入：BeadClassifier 类
  2. 命令行：对单张图片预测 / 对图纸区域批量预测

与 grid_parse.py 集成
─────────────────────
在 grid_parse.py 的 parse_grid_region() 中，可以用
  from grid_category.predict import BeadClassifier
  classifier = BeadClassifier()
  bead_id = classifier.predict_cell(cell_arr)
替换原来的 find_closest_bead_color() 调用。

用法
────
  # 单张格子图片
  python 5_predict.py --image path/to/cell.png

  # 对图纸PNG + JSON 批量重新预测（测试分类器效果）
  python 5_predict.py --grid-png output/bear_processed.png \
                      --grid-json output/bear_grid.json

  # Python 模块使用
  from grid_category.predict import BeadClassifier
  clf = BeadClassifier()
  bead_id, confidence = clf.predict_cell(cell_np_array)
"""

import os
import sys
import argparse
import json
from typing import Optional, Tuple, List, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as CFG
from build_dataset import load_label_map, build_transform
from train import build_model


# ──────────────────────────────────────────────
#  BeadClassifier
# ──────────────────────────────────────────────

class BeadClassifier:
    """
    拼豆格子颜色分类器（封装好的推理接口）。

    Example:
        clf = BeadClassifier()
        bead_id, conf = clf.predict_cell(cell_img_array)
        # bead_id: "A05"
        # conf:    0.97 (置信度)

        # Top-K 预测
        results = clf.predict_cell_topk(cell_img_array, k=3)
        # [("A05", 0.97), ("A04", 0.02), ("A06", 0.01)]
    """

    def __init__(
        self,
        model_path:      str = CFG.BEST_MODEL_PATH,
        label_map_path:  str = CFG.LABEL_MAP_PATH,
        device:          Optional[str] = None,
        confidence_threshold: float = 0.0,   # 低于此阈值时返回 None
    ):
        """
        Args:
            model_path: 训练好的 checkpoint 路径
            label_map_path: 标签映射文件路径
            device: "cuda" / "cpu" / None（自动选）
            confidence_threshold: 置信度阈值，低于此值返回 None
        """
        # 设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 标签映射
        if not os.path.exists(label_map_path):
            raise FileNotFoundError(
                f"找不到标签映射文件：{label_map_path}\n"
                "请先运行 python 1_generate_dataset.py + python 3_train.py"
            )
        self.id_to_idx, self.idx_to_id = load_label_map(label_map_path)
        self.n_classes = len(self.idx_to_id)

        # 模型
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"找不到模型文件：{model_path}\n"
                "请先运行 3_train.py 训练模型"
            )
        ckpt = torch.load(model_path, map_location="cpu")
        backbone = ckpt.get("backbone", CFG.BACKBONE)
        self.model = build_model(backbone, self.n_classes, pretrained=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device)
        self.model.eval()

        # 预处理
        self.transform = build_transform("val")   # val 模式：只 resize + normalize
        self.confidence_threshold = confidence_threshold

        print(
            f"BeadClassifier 已加载：backbone={backbone}, "
            f"classes={self.n_classes}, device={self.device}"
        )

    # ──── 单格预测 ────

    def predict_cell(
        self,
        cell: Union[np.ndarray, Image.Image],
    ) -> Tuple[Optional[str], float]:
        """
        预测单个格子的色号。

        Args:
            cell: RGB numpy array (H, W, 3) 或 PIL.Image

        Returns:
            (bead_id, confidence)  如果置信度低于阈值，bead_id 为 None
        """
        results = self.predict_cell_topk(cell, k=1)
        bead_id, conf = results[0]
        if conf < self.confidence_threshold:
            return None, conf
        return bead_id, conf

    def predict_cell_topk(
        self,
        cell: Union[np.ndarray, Image.Image],
        k: int = 3,
    ) -> List[Tuple[str, float]]:
        """
        返回 Top-K 预测结果。

        Returns:
            [(bead_id, confidence), ...]  按置信度降序
        """
        if isinstance(cell, np.ndarray):
            if cell.ndim == 2:   # 灰度
                cell = np.stack([cell] * 3, axis=-1)
            elif cell.shape[2] == 4:  # RGBA
                cell = cell[:, :, :3]
            img = Image.fromarray(cell.astype(np.uint8))
        else:
            img = cell.convert("RGB")

        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()

        top_indices = np.argsort(-probs)[:k]
        return [
            (self.idx_to_id[i] if i < len(self.idx_to_id) else str(i), float(probs[i]))
            for i in top_indices
        ]

    # ──── 批量预测 ────

    def predict_batch(
        self,
        cells: List[Union[np.ndarray, Image.Image]],
        batch_size: int = 128,
    ) -> List[Tuple[Optional[str], float]]:
        """
        批量预测多个格子（比逐个调用 predict_cell 更快）。

        Returns:
            [(bead_id, confidence), ...]  与输入等长
        """
        tensors = []
        for cell in cells:
            if isinstance(cell, np.ndarray):
                if cell.ndim == 2:
                    cell = np.stack([cell] * 3, axis=-1)
                elif cell.shape[2] == 4:
                    cell = cell[:, :, :3]
                img = Image.fromarray(cell.astype(np.uint8))
            else:
                img = cell.convert("RGB")
            tensors.append(self.transform(img))

        results = []
        for i in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[i:i + batch_size]).to(self.device)
            with torch.no_grad():
                logits = self.model(batch)
                probs = F.softmax(logits, dim=1).cpu().numpy()
            for p in probs:
                top_idx = np.argmax(p)
                conf = float(p[top_idx])
                bead_id = self.idx_to_id[top_idx] if top_idx < len(self.idx_to_id) else str(top_idx)
                if conf < self.confidence_threshold:
                    bead_id = None
                results.append((bead_id, conf))

        return results

    # ──── 对图纸区域批量预测（替换 parse_grid_region 中的颜色匹配） ────

    def predict_grid_region(
        self,
        img_arr:     np.ndarray,
        region_rect: Tuple[int, int, int, int],
        cols:        int,
        rows:        int,
    ) -> dict:
        """
        给定原图 numpy array 和框选区域，按行列数均匀切格，
        对每个格子做分类预测，返回与 parse_grid_region() 兼容的 grid_data。

        Args:
            img_arr:     原图 RGB numpy array (H, W, 3)
            region_rect: (x, y, w, h) 框选区域（原图坐标）
            cols:        列数
            rows:        行数

        Returns:
            grid_data dict，结构与 grid_parse.parse_grid_region() 一致
        """
        x0, y0, rw, rh = region_rect
        h_full, w_full = img_arr.shape[:2]

        x1 = max(0, x0)
        y1 = max(0, y0)
        x2 = min(w_full, x0 + rw)
        y2 = min(h_full, y0 + rh)
        region = img_arr[y1:y2, x1:x2]
        rh_actual, rw_actual = region.shape[:2]

        cell_w = rw_actual / cols
        cell_h = rh_actual / rows

        # 先裁切所有格子
        cell_list = []
        positions = []
        for row in range(rows):
            for col in range(cols):
                cx1 = int(col * cell_w)
                cy1 = int(row * cell_h)
                cx2 = min(int((col + 1) * cell_w), rw_actual)
                cy2 = min(int((row + 1) * cell_h), rh_actual)
                cell = region[cy1:cy2, cx1:cx2]
                cell_list.append(cell)
                positions.append((row, col))

        # 批量推理
        preds = self.predict_batch(cell_list)

        # 组装 grid
        grid = [[None] * cols for _ in range(rows)]
        for (row, col), (bead_id, conf) in zip(positions, preds):
            grid[row][col] = {
                "bead_id":    bead_id,
                "confidence": round(conf, 4),
            }

        return {
            "width":     cols,
            "height":    rows,
            "cell_size": {"width": round(cell_w, 2), "height": round(cell_h, 2)},
            "grid":      grid,
            "method":    "classifier",
        }


# ──────────────────────────────────────────────
#  命令行工具
# ──────────────────────────────────────────────

def _predict_single_image(clf: BeadClassifier, image_path: str, top_k: int):
    img = Image.open(image_path).convert("RGB")
    results = clf.predict_cell_topk(img, k=top_k)
    print(f"\n图片：{image_path}")
    print(f"Top-{top_k} 预测：")
    for rank, (bead_id, conf) in enumerate(results, 1):
        print(f"  {rank}. {bead_id:6s}  置信度 {conf*100:.2f}%")


def _predict_grid(clf: BeadClassifier, png_path: str, json_path: str):
    """对已有图纸用分类器重新预测，与原始色号比对"""
    from collections import Counter

    img = Image.open(png_path).convert("RGB")
    img_arr = np.array(img)

    with open(json_path, "r", encoding="utf-8") as f:
        grid_true = json.load(f)

    rows = len(grid_true)
    cols = len(grid_true[0]) if rows > 0 else 0
    print(f"\n图纸大小：{cols} x {rows}")

    # 用 28px cell size（render_pattern 默认）
    cell_size = 28

    # 采样部分格子做验证
    correct = 0
    total = 0
    sample_results = []

    for row in range(rows):
        for col in range(cols):
            true_val = grid_true[row][col]
            if isinstance(true_val, dict):
                true_id = true_val.get("bead_id")
            else:
                true_id = str(true_val)
            if not true_id:
                continue

            x0 = (col + 1) * cell_size
            y0 = (row + 1) * cell_size
            x1, y1 = x0 + cell_size, y0 + cell_size
            if x1 > img_arr.shape[1] or y1 > img_arr.shape[0]:
                continue

            cell = img_arr[y0:y1, x0:x1]
            pred_id, conf = clf.predict_cell(cell)
            total += 1
            if pred_id == true_id:
                correct += 1
            sample_results.append((true_id, pred_id, conf))

    acc = correct / total if total > 0 else 0
    print(f"准确率（对比JSON标签）：{acc*100:.2f}%  ({correct}/{total})")

    # 统计错误
    errors = [(t, p) for t, p, _ in sample_results if t != p]
    if errors:
        most_common_errors = Counter(errors).most_common(10)
        print(f"\n最常见的错误（真实→预测）：")
        for (t, p), cnt in most_common_errors:
            print(f"  {t} → {p}  ({cnt}次)")


def main():
    parser = argparse.ArgumentParser(description="拼豆格子颜色分类推理")
    parser.add_argument("--image",      default="",  help="单张格子图片路径")
    parser.add_argument("--grid-png",   default="",  help="图纸渲染PNG路径（批量验证）")
    parser.add_argument("--grid-json",  default="",  help="图纸JSON路径（批量验证）")
    parser.add_argument("--model",      default=CFG.BEST_MODEL_PATH, help="模型 checkpoint 路径")
    parser.add_argument("--top-k",      type=int, default=3, help="Top-K 预测数量")
    parser.add_argument("--threshold",  type=float, default=0.0, help="置信度阈值")
    args = parser.parse_args()

    print("=" * 50)
    print("Step 5 - 模型推理（合成数据集 train_dataset/）")
    print("=" * 50)

    try:
        clf = BeadClassifier(
            model_path=args.model,
            confidence_threshold=args.threshold,
        )
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    if args.image:
        if not os.path.exists(args.image):
            print(f"[ERROR] 图片文件不存在：{args.image}")
            return
        _predict_single_image(clf, args.image, args.top_k)

    elif args.grid_png and args.grid_json:
        if not os.path.exists(args.grid_png):
            print(f"[ERROR] PNG 不存在：{args.grid_png}")
            return
        if not os.path.exists(args.grid_json):
            print(f"[ERROR] JSON 不存在：{args.grid_json}")
            return
        _predict_grid(clf, args.grid_png, args.grid_json)

    else:
        print("请指定 --image 或 --grid-png + --grid-json 参数")
        print("示例：python 5_predict.py --image path/to/cell.png")


if __name__ == "__main__":
    main()
