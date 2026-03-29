# -*- coding: utf-8 -*-
"""
grid_parse.py - 交互式图纸解析工具（PyQt5版 + 深度学习分类器）

流程：
  Step 1: 选择图片文件
  Step 2: 两阶段框选（先点左上角 → 再拖右下角 → 选框固定）
  Step 3: 输入列数（宽）和行数（高）
  Step 4: 模型批量推理 → 格子视图展示（含完整图例）→ 保存
          → 弹出识别结果审查对话框（聚类+置信度）→ 保存 tmp.jpg
"""

import sys
import os
import json
import copy
import time
import traceback
import numpy as np
from collections import Counter, defaultdict
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from concurrent.futures import ThreadPoolExecutor

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QFileDialog, QVBoxLayout, QHBoxLayout, QSpinBox, QDialog,
    QDialogButtonBox, QFormLayout, QMessageBox, QScrollArea,
    QSizePolicy, QFrame, QProgressBar, QCheckBox, QSlider,
    QTextEdit, QGroupBox, QListWidget, QListWidgetItem
)
from PyQt5.QtCore import Qt, QRect, QPoint, QSize, pyqtSignal, QThread, QObject
from PyQt5.QtGui import (
    QPixmap, QPainter, QPen, QColor, QImage, QBrush, QFont, QCursor,
    QPainterPath, QLinearGradient
)


# ─────────────────────────────────────────────────────────────────────────────
#  0. 路径 & 全局配置
# ─────────────────────────────────────────────────────────────────────────────

def _get_base_dir():
    return os.path.dirname(os.path.abspath(__file__))

CLASSIFIER_MODEL_PATH = os.path.join(_get_base_dir(), "load", "bead_classifier.pth")
PALETTE_JSON_PATH    = os.path.join(_get_base_dir(), "load", "beads_palette_221_correct.json")
LABEL_MAP_PATH       = os.path.join(_get_base_dir(), "grid_category", "train_dataset", "label_map.txt")

# 222个色号列表（含空）
FULL_BEAD_IDS = []
for prefix, count in [("A", 26), ("B", 32), ("C", 29), ("D", 26),
                      ("E", 24), ("F", 25), ("G", 21), ("H", 23), ("M", 15)]:
    for i in range(1, count + 1):
        FULL_BEAD_IDS.append(f"{prefix}{i:02d}")
FULL_BEAD_IDS.append("空")  # 222个
FULL_BEAD_IDS_SET = set(FULL_BEAD_IDS)


# ─────────────────────────────────────────────────────────────────────────────
#  1. 工具函数（原保留不动）
# ─────────────────────────────────────────────────────────────────────────────

def extract_dominant_color(cell_arr):
    """统计格子内出现最多的RGB颜色"""
    arr = cell_arr.astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.ndim != 3 or arr.shape[2] != 3:
        return (128, 128, 128)
    pixels = arr.reshape(-1, 3)
    rgb_tuples = [tuple(p) for p in pixels]
    counter = Counter(rgb_tuples)
    most_common = counter.most_common(1)[0][0]
    return int(most_common[0]), int(most_common[1]), int(most_common[2])


def load_bead_palette(palette_path=None):
    """加载珠子色板 JSON，返回 {色号: (R,G,B)}"""
    path = palette_path or PALETTE_JSON_PATH
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        palette = json.load(f)
    # 标准化为 {色号: (R,G,B)}
    result = {}
    if isinstance(palette, dict):
        for kid, val in palette.items():
            if isinstance(val, (list, tuple)) and len(val) >= 3:
                result[kid] = tuple(int(v) for v in val[:3])
            elif isinstance(val, dict):
                r = int(val.get("r", 128))
                g = int(val.get("g", 128))
                b = int(val.get("b", 128))
                result[kid] = (r, g, b)
    elif isinstance(palette, list):
        for i, item in enumerate(palette):
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                result[f"Color_{i}"] = tuple(int(v) for v in item[:3])
            elif isinstance(item, dict):
                name = item.get("id") or item.get("name") or f"Color_{i}"
                r = int(item.get("r", 128))
                g = int(item.get("g", 128))
                b = int(item.get("b", 128))
                result[str(name)] = (r, g, b)
    return result


def find_closest_bead_color(rgb, palette_data):
    """欧氏距离匹配最近色珠（原逻辑，保留备用）"""
    if palette_data is None:
        return None, list(rgb)
    rgb_array = np.array(rgb, dtype=np.float32)
    bead_ids = list(palette_data.keys())
    colors = np.array(list(palette_data.values()), dtype=np.float32)
    distances = np.sqrt(np.sum((colors - rgb_array) ** 2, axis=1))
    min_idx   = np.argmin(distances)
    bead_id   = bead_ids[min_idx]
    bead_rgb  = list(palette_data[bead_id])
    return bead_id, bead_rgb


# ─────────────────────────────────────────────────────────────────────────────
#  2. 深度学习分类器
# ─────────────────────────────────────────────────────────────────────────────

def _load_label_map(path=None):
    """加载 label_map.txt，返回 idx_to_id 列表"""
    path = path or LABEL_MAP_PATH
    if not os.path.exists(path):
        with open(PALETTE_JSON_PATH, encoding="utf-8") as f:
            beads = json.load(f)
        ids = sorted(beads.keys()) if isinstance(beads, dict) else [f"Color_{i}" for i in range(len(beads))]
        return ids
    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    idx_to_id = [None] * len(lines)
    for line in lines:
        parts = line.split("\t")
        if len(parts) == 2:
            idx, bid = parts
            idx_to_id[int(idx)] = bid
    return idx_to_id


class BeadClassifier:
    """
    拼豆格子颜色分类器（内置版本，不依赖 5_predict.py）。
    支持 GPU/CPU 自动切换、批量推理、进度回调。
    """

    def __init__(self, model_path=None, batch_size=256, device=None,
                 progress_callback=None):
        self.model_path      = model_path or CLASSIFIER_MODEL_PATH
        self.batch_size      = batch_size
        self.progress_callback = progress_callback
        self.device = torch.device(device) if device else \
            torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.idx_to_id = _load_label_map()

        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(self.model.fc.in_features, len(self.idx_to_id))
        )

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"找不到分类器模型：{self.model_path}\n"
                "请先运行 python 1_generate_dataset.py + python 3_train.py 生成模型，"
                "然后复制 best.pth 到 load/bead_classifier.pth"
            )

        ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device)
        self.model.eval()

        self.transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        gpu_name = torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU"
        print(f"[BeadClassifier] 已加载: device={self.device}, GPU={gpu_name}, classes={len(self.idx_to_id)}")

    def predict_cells(self, cells):
        """
        批量预测多个格子图片。
        Args:
            cells: list of numpy array (H,W,3) 或 PIL.Image
        Returns:
            list of (bead_id, confidence)
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
        total   = len(tensors)
        n_batches = (total + self.batch_size - 1) // self.batch_size

        for b in range(n_batches):
            batch = tensors[b * self.batch_size : (b + 1) * self.batch_size]
            batch_t = torch.stack(batch).to(self.device)
            with torch.no_grad():
                logits  = self.model(batch_t)
                probs   = torch.softmax(logits, dim=1).cpu().numpy()

            for p in probs:
                top_idx = int(np.argmax(p))
                conf    = float(p[top_idx])
                bead_id = self.idx_to_id[top_idx] if top_idx < len(self.idx_to_id) else str(top_idx)
                results.append((bead_id, conf))

            if self.progress_callback:
                done = min((b + 1) * self.batch_size, total)
                self.progress_callback(done, total)

        return results


# ─────────────────────────────────────────────────────────────────────────────
#  3. 解析函数
# ─────────────────────────────────────────────────────────────────────────────

def parse_grid_region_by_color(img_arr, region_rect, cols, rows, palette_data):
    """原版主色匹配逻辑（保留备用）"""
    x0, y0, rw, rh = region_rect
    h_full, w_full = img_arr.shape[:2]
    x1, y1 = max(0, x0), max(0, y0)
    x2, y2 = min(w_full, x0 + rw), min(h_full, y0 + rh)
    region = img_arr[y1:y2, x1:x2]
    rh_actual, rw_actual = region.shape[:2]
    cell_w = rw_actual / cols
    cell_h = rh_actual / rows

    grid = []
    for row in range(rows):
        row_data = []
        for col in range(cols):
            cx1, cy1 = int(col * cell_w), int(row * cell_h)
            cx2, cy2 = int((col + 1) * cell_w), int((row + 1) * cell_h)
            cx2 = min(cx2, rw_actual); cy2 = min(cy2, rh_actual)
            cell = region[cy1:cy2, cx1:cx2]
            if cell.size == 0:
                row_data.append({"rgb": [128, 128, 128], "bead_id": None, "bead_rgb": [128, 128, 128]})
                continue
            dominant_rgb = extract_dominant_color(cell)
            bead_id, bead_rgb = find_closest_bead_color(dominant_rgb, palette_data)
            row_data.append({
                "rgb": list(dominant_rgb),
                "bead_id": bead_id,
                "bead_rgb": bead_rgb
            })
        grid.append(row_data)
    return {"width": cols, "height": rows,
            "cell_size": {"width": round(cell_w, 2), "height": round(cell_h, 2)}, "grid": grid}


def parse_grid_region_by_model(img_arr, region_rect, cols, rows, classifier, palette_data=None):
    """
    模型推理版：切格子 → 批量模型分类 → 组装 grid_data。
    格子内容（rgb）同时记录，方便后续审查。
    """
    x0, y0, rw, rh = region_rect
    h_full, w_full  = img_arr.shape[:2]
    x1, y1 = max(0, x0), max(0, y0)
    x2, y2 = min(w_full, x0 + rw), min(h_full, y0 + rh)
    region = img_arr[y1:y2, x1:x2]
    rh_actual, rw_actual = region.shape[:2]
    cell_w = rw_actual / cols
    cell_h = rh_actual / rows

    # 收集所有格子
    cell_list  = []
    positions   = []
    for row in range(rows):
        for col in range(cols):
            cx1, cy1 = int(col * cell_w), int(row * cell_h)
            cx2, cy2 = int((col + 1) * cell_w), int((row + 1) * cell_h)
            cx2 = min(cx2, rw_actual); cy2 = min(cy2, rh_actual)
            cell = region[cy1:cy2, cx1:cx2]
            cell_list.append(cell)
            positions.append((row, col))

    # 批量模型推理
    total_cells = len(cell_list)
    results = classifier.predict_cells(cell_list)

    # 组装 grid（同时记录格子rgb）
    grid = [[None] * cols for _ in range(rows)]
    for (row, col), (bead_id, conf) in zip(positions, results):
        # 记录格子平均色（用于聚类展示）
        avg_rgb = extract_dominant_color(cell_list[row * cols + col]) \
            if (row * cols + col) < len(cell_list) else (128, 128, 128)
        grid[row][col] = {
            "bead_id":    bead_id,
            "bead_rgb":   bead_id,
            "confidence": round(conf, 4),
            "rgb":        list(avg_rgb),
            "cell_arr":   cell_list[row * cols + col] if (row * cols + col) < len(cell_list) else None,
        }

    return {
        "width":    cols,
        "height":   rows,
        "cell_size": {"width": round(cell_w, 2), "height": round(cell_h, 2)},
        "grid":     grid,
    }


def save_grid_json(grid_data, output_path):
    """保存 grid_data 为 JSON（排除 cell_arr 等不可序列化字段）"""
    data = copy.deepcopy(grid_data)
    for row in data.get("grid", []):
        for cell in row:
            if cell:
                cell.pop("cell_arr", None)
                cell.pop("_orig_arr", None)
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, tuple):
            return list(obj)
        return obj
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=convert)


# ─────────────────────────────────────────────────────────────────────────────
#  4. 图纸渲染（参照 pindou_UI.py 的 render_pattern + add_color_legend）
# ─────────────────────────────────────────────────────────────────────────────

def render_grid_to_pil(grid_2d, beads, cell=28):
    """
    将 grid_2d（2D list of bead_id）渲染为 PIL Image，含完整图例。
    完全参照 pindou_UI.py / utils/tools.py 的逻辑。
    """
    height = len(grid_2d)
    width  = len(grid_2d[0]) if height > 0 else 0
    new_height, new_width = height + 2, width + 2

    img = Image.new("RGB", (new_width * cell, new_height * cell), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("Arial.ttf", int(cell * 0.35))
    except Exception:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", int(cell * 0.35))
        except Exception:
            font = ImageFont.load_default()

    # ── 第一步：填充每个格子的颜色 ──
    for y in range(new_height):
        for x in range(new_width):
            if 1 <= x < new_width - 1 and 1 <= y < new_height - 1:
                code = grid_2d[y - 1][x - 1]
                rgb  = beads.get(code, (255, 255, 255))
            else:
                rgb = (200, 200, 200)

            x0, y0 = x * cell, y * cell
            x1, y1 = x0 + cell, y0 + cell
            draw.rectangle([x0, y0, x1, y1], fill=rgb)

            if 1 <= x < new_width - 1 and 1 <= y < new_height - 1:
                continue
            elif y == 0:
                text = str(x)
            elif y == new_height - 1:
                text = str(x)
            elif x == 0:
                text = str(y)
            elif x == new_width - 1:
                text = str(y)
            else:
                continue

            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            tx = x0 + (cell - tw) / 2
            ty = y0 + (cell - th) / 2
            draw.text((tx, ty), text, fill=(0, 0, 0), font=font)

    # ── 第二步：画网格线 ──
    for y in range(new_height + 1):
        line_y = y * cell
        is_5 = (y % 5 == 0)
        color = (0, 180, 255) if is_5 else (180, 180, 180)
        lw = 2 if is_5 else 1
        draw.line([0, line_y, new_width * cell, line_y], fill=color, width=lw)

    for x in range(new_width + 1):
        line_x = x * cell
        is_5 = (x % 5 == 0)
        color = (0, 180, 255) if is_5 else (180, 180, 180)
        lw = 2 if is_5 else 1
        draw.line([line_x, 0, line_x, new_height * cell], fill=color, width=lw)

    # ── 第三步：写入色号（仅限主图区域）──
    for y in range(height):
        for x in range(width):
            code = grid_2d[y][x]
            if code == '空':
                continue
            rgb = beads.get(code, (255, 255, 255))
            x0, y0 = (x + 1) * cell, (y + 1) * cell
            text = code
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            tx = x0 + (cell - tw) / 2
            ty = y0 + (cell - th) / 2
            brightness = sum(rgb) / 3
            tc = (255, 255, 255) if brightness < 140 else (0, 0, 0)
            draw.text((tx, ty), text, fill=tc, font=font)

    # ── 第四步：添加图例（使用内联的 add_color_legend）──
    img = _add_color_legend(img, beads, grid_2d, cell)
    return img


def _add_color_legend(img, beads, grid_2d, cell=28):
    """
    在图像底部追加色号图例（与 tools.py add_color_legend 逻辑一致）。
    """
    code_counts = Counter(code for row in grid_2d for code in row if code != '空')
    sorted_items = sorted(code_counts.items(), key=lambda x: x[1], reverse=True)
    if not sorted_items:
        return img

    block_size         = max(40, int(cell * 1.2))
    margin_between     = max(16, int(block_size * 0.4))
    text_gap           = max(6, int(block_size * 0.15))
    bottom_padding     = max(30, block_size)
    font_size_code     = max(14, int(block_size * 0.35))
    font_size_count    = max(12, int(block_size * 0.3))

    try:
        font_code  = ImageFont.truetype("Arial.ttf", font_size_code)
        font_count = ImageFont.truetype("Arial.ttf", font_size_count)
    except Exception:
        try:
            font_code  = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size_code)
            font_count = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size_count)
        except Exception:
            font_code  = ImageFont.load_default()
            font_count = ImageFont.load_default()

    item_total_width = block_size + margin_between
    max_per_line = max(1, (img.width + margin_between) // item_total_width)

    lines = []
    current_line = []
    for item in sorted_items:
        current_line.append(item)
        if len(current_line) >= max_per_line:
            lines.append(current_line)
            current_line = []
    if current_line:
        lines.append(current_line)

    legend_content_h = len(lines) * (block_size + text_gap + font_size_count) + \
                        (len(lines) - 1) * margin_between
    legend_total_h    = legend_content_h + bottom_padding
    total_h = img.height + legend_total_h
    combined = Image.new("RGB", (img.width, total_h), (255, 255, 255))
    combined.paste(img, (0, 0))
    draw = ImageDraw.Draw(combined)

    y_start = img.height
    for line_idx, line in enumerate(lines):
        y_line = y_start + line_idx * (block_size + text_gap + font_size_count + margin_between)
        line_width = len(line) * item_total_width - margin_between
        start_x = (img.width - line_width) // 2

        for col_idx, (code, count) in enumerate(line):
            x = start_x + col_idx * item_total_width
            y = y_line
            rgb = beads.get(code, (200, 200, 200))

            draw.rectangle([x, y, x + block_size, y + block_size],
                           fill=rgb, outline=(100, 100, 100), width=1)

            bbox_code = draw.textbbox((0, 0), code, font=font_code)
            tw_code = bbox_code[2] - bbox_code[0]
            th_code = bbox_code[3] - bbox_code[1]
            tx_code = x + (block_size - tw_code) / 2
            ty_code = y + (block_size - th_code) / 2
            brightness = sum(rgb) / 3
            tc = (255, 255, 255) if brightness < 160 else (0, 0, 0)
            draw.text((tx_code, ty_code), code, fill=tc, font=font_code)

            count_text = f"{count}"
            bbox_count = draw.textbbox((0, 0), count_text, font=font_count)
            tw_count = bbox_count[2] - bbox_count[0]
            th_count = bbox_count[3] - bbox_count[1]
            tx_count = x + (block_size - tw_count) / 2
            ty_count = y + block_size + text_gap
            draw.text((tx_count, ty_count), count_text, fill=(0, 0, 0), font=font_count)

    return combined


# ─────────────────────────────────────────────────────────────────────────────
#  5. Step 3：输入行列数对话框
# ─────────────────────────────────────────────────────────────────────────────

class GridSizeDialog(QDialog):
    def __init__(self, parent=None, suggested_cols=50, suggested_rows=50):
        super().__init__(parent)
        self.setWindowTitle("Step 3 - 输入格子数量")
        self.setFixedSize(340, 200)
        self.setStyleSheet("""
            QDialog { background: #1e1e2e; color: #cdd6f4; }
            QLabel { color: #cdd6f4; font-size: 13px; }
            QSpinBox {
                background: #313244; color: #cdd6f4;
                border: 1px solid #7c3aed; border-radius: 5px;
                padding: 4px 8px; font-size: 13px;
            }
            QSpinBox::up-button, QSpinBox::down-button { width: 18px; }
            QPushButton {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #7c3aed, stop:1 #9c27b0);
                color: white; border: none; border-radius: 6px;
                padding: 6px 18px; font-size: 13px;
            }
            QPushButton:hover { background: #9c27b0; }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(14)

        tip = QLabel("请输入框选区域内格子的数量：")
        tip.setStyleSheet("color: #a6adc8; font-size: 12px;")
        layout.addWidget(tip)

        form = QFormLayout(); form.setSpacing(10)
        self.spin_cols = QSpinBox(); self.spin_cols.setRange(1, 5000)
        self.spin_cols.setValue(suggested_cols)
        self.spin_rows = QSpinBox(); self.spin_rows.setRange(1, 5000)
        self.spin_rows.setValue(suggested_rows)
        form.addRow("列数（宽）：", self.spin_cols)
        form.addRow("行数（高）：", self.spin_rows)
        layout.addLayout(form)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        btns.setStyleSheet("QPushButton { min-width: 70px; }")
        layout.addWidget(btns, alignment=Qt.AlignRight)

    def get_values(self):
        return self.spin_cols.value(), self.spin_rows.value()


# ─────────────────────────────────────────────────────────────────────────────
#  6. 格子视图预览对话框（完整图纸，含图例）
# ─────────────────────────────────────────────────────────────────────────────

class GridPreviewDialog(QDialog):
    """
    图纸预览对话框：显示完整格子视图（参照 pindou_UI.py，含完整图例）。
    使用 render_grid_to_pil + add_color_legend 渲染。
    """
    def __init__(self, grid_data, beads, source_img_arr=None, region=None, parent=None):
        super().__init__(parent)
        self.grid_data   = grid_data
        self.beads       = beads
        self._src_arr    = source_img_arr
        self._region     = region  # (rx, ry, rw, rh)
        self._zoom       = 1.0

        rows = grid_data["height"]
        cols = grid_data["width"]

        self.setWindowTitle(f"图纸预览  {cols} × {rows}  [完整图例版]")
        self.setMinimumSize(800, 700)
        self.setStyleSheet("""
            QDialog { background: #1e1e2e; }
            QLabel { color: #cdd6f4; }
            QPushButton {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #7c3aed,stop:1 #9c27b0);
                color: white; border: none; border-radius: 6px;
                padding: 6px 14px; font-size: 12px;
            }
            QPushButton:hover { background: #9333ea; }
            QPushButton#btn_sec {
                background: #313244; color: #cdd6f4;
                border: 1px solid #45475a; padding: 6px 14px;
            }
            QPushButton#btn_sec:hover { background: #45475a; }
        """)

        # 构建 2D bead_id list
        grid_2d = []
        for row in grid_data["grid"]:
            row_ids = []
            for cell in row:
                row_ids.append(cell.get("bead_id", "空") if cell else "空")
            grid_2d.append(row_ids)

        # 用 render_grid_to_pil 渲染（含完整图例），使用更大格子确保清晰
        cell_size = max(36, min(56, 800 // max(cols, rows)))
        self._pil_img = render_grid_to_pil(grid_2d, beads, cell=cell_size)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # 工具栏
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        self.zoom_label = QLabel(f"缩放: {self._zoom:.1f}x")
        toolbar.addWidget(self.zoom_label)

        btn_zoom_in = QPushButton("🔍+ 放大")
        btn_zoom_in.clicked.connect(self._zoom_in)
        toolbar.addWidget(btn_zoom_in)

        btn_zoom_out = QPushButton("🔍- 缩小")
        btn_zoom_out.clicked.connect(self._zoom_out)
        toolbar.addWidget(btn_zoom_out)

        btn_reset = QPushButton("↺ 原始")
        btn_reset.clicked.connect(self._zoom_reset)
        toolbar.addWidget(btn_reset)

        toolbar.addStretch()

        btn_review = QPushButton("📊 审查识别结果")
        btn_review.setObjectName("btn_sec")
        btn_review.clicked.connect(self._open_review)
        toolbar.addWidget(btn_review)

        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.close)
        toolbar.addWidget(btn_close)

        layout.addLayout(toolbar)

        # PIL Image → QPixmap 展示
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(False)
        self.scroll.setAlignment(Qt.AlignCenter)
        self.scroll.setStyleSheet("background: #181825; border: none;")

        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignCenter)
        self._update_pixmap()
        self.scroll.setWidget(self.img_label)
        layout.addWidget(self.scroll)

        self._review_dialog = None

    def _pil_to_pixmap(self, pil_img, scale=1.0):
        w = int(pil_img.width  * scale)
        h = int(pil_img.height * scale)
        resized = pil_img.resize((w, h), Image.Resampling.LANCZOS)
        qimg = QImage(resized.tobytes(), w, h, w * 3, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def _update_pixmap(self):
        pm = self._pil_to_pixmap(self._pil_img, scale=self._zoom)
        self.img_label.setPixmap(pm)
        self.img_label.resize(pm.size())

    def _zoom_in(self):
        self._zoom = min(self._zoom + 0.25, 5.0)
        self.zoom_label.setText(f"缩放: {self._zoom:.1f}x")
        self._update_pixmap()

    def _zoom_out(self):
        self._zoom = max(self._zoom - 0.25, 0.1)
        self.zoom_label.setText(f"缩放: {self._zoom:.1f}x")
        self._update_pixmap()

    def _zoom_reset(self):
        self._zoom = 1.0
        self.zoom_label.setText("1.0x")
        self._update_pixmap()

    def _open_review(self):
        if self._review_dialog is None:
            self._review_dialog = RecognitionReviewDialog(
                self.grid_data, self.beads,
                source_img_arr=self._src_arr, region=self._region,
                parent=self.parent()
            )
        self._review_dialog.show()
        self._review_dialog.raise_()
        self._review_dialog.activateWindow()


# ─────────────────────────────────────────────────────────────────────────────
#  7. 识别结果聚类审查对话框（新增）
# ─────────────────────────────────────────────────────────────────────────────

class RecognitionReviewDialog(QDialog):
    """
    识别结果聚类审查：
    - 左侧：按预测色号分组的颜色列表（背景=颜色，显示数量/置信度）
    - 右侧（上方）：选中色号的原始裁切格子缩略图（网格排列）
    - 右侧（下方）：选中格子的详细信息
    - 底部：批量修正按钮 + 保存 tmp.jpg（含所有分组的原始格子图）
    """

    def __init__(self, grid_data, beads, source_img_arr=None, region=None, parent=None):
        super().__init__(parent)
        self.grid_data = grid_data
        self.beads     = beads
        self._src_arr  = source_img_arr
        self._region   = region  # (rx, ry, rw, rh)
        rows = grid_data["height"]
        cols = grid_data["width"]

        self.setWindowTitle(f"识别结果审查  {cols} × {rows}")
        self.setMinimumSize(1100, 750)
        self.setStyleSheet("""
            QDialog { background: #1e1e2e; }
            QLabel { color: #cdd6f4; }
            QPushButton {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #7c3aed,stop:1 #9c27b0);
                color: white; border: none; border-radius: 6px;
                padding: 6px 14px; font-size: 12px;
            }
            QPushButton:hover { background: #9333ea; }
            QPushButton#btn_sec {
                background: #313244; color: #cdd6f4;
                border: 1px solid #45475a; padding: 6px 14px; font-size: 12px;
            }
            QPushButton#btn_sec:hover { background: #45475a; }
            QPushButton#btn_warn {
                background: #6b2128; color: #f38ba8;
                border: 1px solid #f38ba8; padding: 6px 14px; font-size: 12px;
            }
            QPushButton#btn_warn:hover { background: #7c2d37; }
            QListWidget {
                background: #181825; color: #cdd6f4;
                border: 1px solid #313244; border-radius: 6px;
                font-size: 13px;
            }
            QListWidget::item { padding: 6px; border-bottom: 1px solid #313244; }
            QListWidget::item:selected { background: #3b2f5e; color: #cba6f7; }
            QTextEdit {
                background: #181825; color: #cdd6f4;
                border: 1px solid #313244; border-radius: 6px; font-size: 12px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # ── 顶部统计信息 ──
        info_layout = QHBoxLayout()
        info_layout.setSpacing(12)

        groups = self._build_groups()
        self.groups = groups
        total_cells = rows * cols
        avg_conf = np.mean([g["avg_conf"] for g in groups]) if groups else 0
        low_conf_count = sum(
            1 for row in grid_data["grid"] for cell in row
            if cell and cell.get("confidence", 1) < 0.8
        )

        for label, val, color in [
            ("总格子数", str(total_cells), "#89b4fa"),
            ("颜色数", str(len(groups)), "#a6e3a1"),
            ("平均置信", f"{avg_conf:.1%}", "#f9e2af"),
            ("低置信格(<0.8)", str(low_conf_count), "#f38ba8" if low_conf_count > 0 else "#a6e3a1"),
        ]:
            lbl = QLabel(f"<b>{label}:</b> {val}")
            lbl.setStyleSheet(
                f"color: {color}; font-size: 13px; padding: 4px 8px; "
                f"background: #2a2a3e; border-radius: 6px;"
            )
            info_layout.addWidget(lbl)

        info_layout.addStretch()
        layout.addLayout(info_layout)

        # ── 主区域：左侧列表 + 右侧详情 ──
        center = QHBoxLayout()
        center.setSpacing(10)

        # 左侧：分组列表
        list_widget = QListWidget()
        list_widget.setMinimumWidth(260)
        self.list_widget = list_widget

        for g in groups:
            rgb = self.beads.get(g["bead_id"], (128, 128, 128))
            brightness = sum(rgb) / 3
            tc_c = "white" if brightness < 128 else "black"
            flag = "⚠️" if g["avg_conf"] < 0.8 else "✅"
            item_text = (
                f"  {g['bead_id']}  │  {g['count']}格  │  "
                f"{flag} {g['avg_conf']:.0%}"
            )
            item = QListWidgetItem(item_text)
            item.setBackground(QColor(*rgb))
            item.setForeground(QColor(60 if brightness >= 128 else 220, 60, 60))
            item.setData(Qt.UserRole, g["bead_id"])
            list_widget.addItem(item)

        list_widget.itemClicked.connect(self._on_item_clicked)
        center.addWidget(list_widget)

        # 右侧：格子预览 + 详情
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        # 上部：原始格子缩略图画布（QLabel 支持滚动）
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(False)
        scroll_area.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.cell_preview_label = QLabel()
        self.cell_preview_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        scroll_area.setWidget(self.cell_preview_label)
        right_layout.addWidget(scroll_area, stretch=3)

        # 下部：详情文字
        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setMaximumHeight(120)
        right_layout.addWidget(self.detail_text, stretch=1)

        center.addWidget(right_panel, stretch=1)
        layout.addLayout(center)

        # ── 底部操作区 ──
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)

        self.btn_correct = QPushButton("🔧 批量修正：将此色号修正为...")
        self.btn_correct.setObjectName("btn_warn")
        self.btn_correct.clicked.connect(self._on_batch_correct)
        btn_layout.addWidget(self.btn_correct)

        btn_layout.addStretch()

        btn_save = QPushButton("💾 保存 tmp.jpg（含所有格子图）")
        btn_save.clicked.connect(self._save_review_image)
        btn_layout.addWidget(btn_save)

        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.close)
        btn_layout.addWidget(btn_close)

        layout.addLayout(btn_layout)

        # 默认选中第一个
        if list_widget.count() > 0:
            list_widget.setCurrentRow(0)
            self._on_item_clicked(list_widget.item(0))

    # ── 构建分组（含裁切数据）──
    def _build_groups(self):
        groups_map = defaultdict(lambda: {"count": 0, "confs": [], "positions": []})
        rows = self.grid_data["height"]
        cols = self.grid_data["width"]

        for r in range(rows):
            for c in range(cols):
                cell = self.grid_data["grid"][r][c]
                if not cell:
                    continue
                bead_id = cell.get("bead_id", "空")
                conf    = cell.get("confidence", 0)
                groups_map[bead_id]["count"]     += 1
                groups_map[bead_id]["confs"].append(conf)
                groups_map[bead_id]["positions"].append((r, c))
                groups_map[bead_id]["avg_rgb"]    = cell.get("rgb", [128, 128, 128])

        groups = []
        for bead_id, g in groups_map.items():
            groups.append({
                "bead_id":   bead_id,
                "count":     g["count"],
                "avg_conf":  np.mean(g["confs"]) if g["confs"] else 0,
                "min_conf":  min(g["confs"]) if g["confs"] else 0,
                "positions": g["positions"],
                "avg_rgb":   g.get("avg_rgb", [128, 128, 128]),
            })
        groups.sort(key=lambda x: -x["count"])
        return groups

    def _crop_cell(self, row, col):
        """从原图数组中裁切指定格子的图像区域"""
        if self._src_arr is None or self._region is None:
            return None
        rx, ry, rw, rh = self._region
        rows_n = self.grid_data["height"]
        cols_n = self.grid_data["width"]
        cw = rw / cols_n
        ch = rh / rows_n
        x0 = int(rx + col * cw)
        y0 = int(ry + row * ch)
        x1 = int(rx + (col + 1) * cw)
        y1 = int(ry + (row + 1) * ch)
        x0_c = max(0, min(x0, self._src_arr.shape[1] - 1))
        y0_c = max(0, min(y0, self._src_arr.shape[0] - 1))
        x1_c = max(x0_c + 1, min(x1, self._src_arr.shape[1]))
        y1_c = max(y0_c + 1, min(y1, self._src_arr.shape[0]))
        cell_arr = self._src_arr[y0_c:y1_c, x0_c:x1_c]
        return Image.fromarray(cell_arr.astype(np.uint8)) if cell_arr.size > 0 else None

    # ── 点击分组 → 右侧显示原始格子缩略图 ──
    def _on_item_clicked(self, item):
        bead_id = item.data(Qt.UserRole)
        g = next((x for x in self.groups if x["bead_id"] == bead_id), None)
        if not g:
            return

        rgb = self.beads.get(bead_id, tuple(int(v) for v in g["avg_rgb"][:3]))
        rgb_hex = "#{:02X}{:02X}{:02X}".format(*rgb)
        brightness = sum(rgb) / 3
        tc = "white" if brightness < 128 else "black"

        # 构建格子预览图：所有该色号的原始裁切格子排列成网格
        cell_preview_img = self._render_cell_grid(g["positions"], bead_id)
        if cell_preview_img:
            self.cell_preview_label.setPixmap(
                QPixmap.fromImage(
                    QImage(cell_preview_img.tobytes("raw", "RGB"),
                           cell_preview_img.width, cell_preview_img.height,
                           cell_preview_img.width * 3, QImage.Format_RGB888)
                )
            )
        else:
            self.cell_preview_label.setText("（无原始图片，无法显示裁切格子）")
            self.cell_preview_label.setStyleSheet("color: #6c7086; font-size: 13px;")

        low_conf_pos = [
            (r, c) for r, c in g["positions"]
            if self.grid_data["grid"][r][c].get("confidence", 1) < 0.8
        ]

        text = (
            f"<b>预测色号：{bead_id}</b><br>"
            f"<font style='background:{rgb_hex}; color:{tc}; padding:2px 6px; border-radius:4px;'>"
            f"&nbsp;■&nbsp;</font>  RGB{tuple(rgb)}<br><br>"
            f"<b>格子数量：</b>{g['count']}格<br>"
            f"<b>平均置信：</b>{g['avg_conf']:.2%} &nbsp; "
            f"<b>最低置信：</b>{g['min_conf']:.2%}<br>"
            f"<b>低置信格子：</b>{len(low_conf_pos)}个"
            f"{('<br><b>低置信位置：</b>' + str(low_conf_pos[:5]) + ('…' if len(low_conf_pos) > 5 else '')) if low_conf_pos else ''}"
        )
        self.detail_text.setHtml(text)

    def _render_cell_grid(self, positions, bead_id):
        """将一组格子的原始裁切图排列成网格 PIL Image"""
        if not positions:
            return None
        CELL = 72          # 每个格子缩略图的尺寸
        CELLS_PER_ROW = 12  # 每行最多显示格子数
        PADDING = 4
        cols_per_row = min(len(positions), CELLS_PER_ROW)
        rows_n = (len(positions) + cols_per_row - 1) // cols_per_row
        total_w = cols_per_row * (CELL + PADDING) + PADDING
        total_h = rows_n * (CELL + PADDING) + PADDING

        img = Image.new("RGB", (total_w, total_h), (24, 24, 46))
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 10)
        except Exception:
            font = ImageFont.load_default()

        for idx, (r, c) in enumerate(sorted(positions)):
            row_i = idx // cols_per_row
            col_i = idx % cols_per_row
            x = PADDING + col_i * (CELL + PADDING)
            y = PADDING + row_i * (CELL + PADDING)

            cell_img = self._crop_cell(r, c)
            if cell_img:
                cell_img = cell_img.resize((CELL, CELL), Image.LANCZOS)
                img.paste(cell_img, (x, y))
            else:
                draw.rectangle([x, y, x + CELL, y + CELL], fill=(60, 60, 80))

            # 边框
            draw.rectangle([x, y, x + CELL, y + CELL],
                           outline=(80, 80, 100), width=1)

        return img

    # ── 批量修正 ──
    def _on_batch_correct(self):
        """将当前选中色号的所有格子批量修正为另一个色号"""
        current = self.list_widget.currentItem()
        if not current:
            return
        from_bead_id = current.data(Qt.UserRole)
        g = next((x for x in self.groups if x["bead_id"] == from_bead_id), None)
        if not g:
            return

        # 弹出修正对话框
        dlg = QDialog(self)
        dlg.setWindowTitle(f"批量修正 {from_bead_id} → ?")
        dlg.setMinimumSize(400, 300)
        dlg.setStyleSheet(self.styleSheet())

        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel(
            f"<b>将 {g['count']} 个格子从 <font color='#f38ba8'>{from_bead_id}</font> "
            f"修正为：</b>（共 {len(self.beads)} 种色号）"
        ))

        list_widget = QListWidget()
        list_widget.setSelectionMode(QListWidget.SingleSelection)
        for bid in sorted(self.beads.keys()):
            item = QListWidgetItem(bid)
            rgb = self.beads[bid]
            item.setBackground(QColor(*rgb))
            brightness = sum(rgb) / 3
            item.setForeground(QColor(60 if brightness >= 128 else 220, 60, 60))
            list_widget.addItem(item)

        layout.addWidget(list_widget)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)

        if dlg.exec_() != QDialog.Accepted:
            return
        selected = list_widget.currentItem()
        if not selected:
            return
        to_bead_id = selected.data(Qt.UserRole)
        if to_bead_id == from_bead_id:
            return

        count = 0
        for (r, c) in g["positions"]:
            cell = self.grid_data["grid"][r][c]
            if cell:
                old_rgb = cell.get("rgb", [128, 128, 128])
                old_bead_id = cell.get("bead_id", from_bead_id)
                # 修正为新色号
                new_rgb = self.beads.get(to_bead_id, old_rgb)
                cell["bead_id"] = to_bead_id
                cell["rgb"] = list(new_rgb)
                count += 1

        # 重新构建 grid_2d 并刷新界面
        rows_n = self.grid_data["height"]
        cols_n = self.grid_data["width"]
        grid_2d = [
            [cell.get("bead_id", "空") if cell else "空" for cell in row]
            for row in self.grid_data["grid"]
        ]

        # 刷新审查对话框，重建分组列表
        self.groups = self._build_groups()
        self.list_widget.clear()
        for g2 in self.groups:
            rgb2 = self.beads.get(g2["bead_id"], (128, 128, 128))
            brightness2 = sum(rgb2) / 3
            flag2 = "⚠️" if g2["avg_conf"] < 0.8 else "✅"
            item_text = (
                f"  {g2['bead_id']}  │  {g2['count']}格  │  "
                f"{flag2} {g2['avg_conf']:.0%}"
            )
            item = QListWidgetItem(item_text)
            item.setBackground(QColor(*rgb2))
            item.setForeground(QColor(60 if brightness2 >= 128 else 220, 60, 60))
            item.setData(Qt.UserRole, g2["bead_id"])
            self.list_widget.addItem(item)

        if self.list_widget.count() > 0:
            self.list_widget.setCurrentRow(0)
            self._on_item_clicked(self.list_widget.item(0))

        QMessageBox.information(
            self, "修正完成",
            f"已将 {count} 个格子从 {from_bead_id} → {to_bead_id}。\n"
            f"请关闭审查对话框后，重新点击「生成图纸」保存修正后的结果。"
        )

    # ── 保存聚类图（含所有原始格子）──
    def _save_review_image(self):
        """渲染含所有分组原始格子的聚类图，保存为 tmp.jpg"""
        groups = self.groups
        if not groups:
            QMessageBox.warning(self, "提示", "没有可保存的识别结果")
            return

        CELL  = 64
        MARGIN = 10
        HEADER = 50   # 每组标题行高度
        cols_per_row = 10
        max_cells_shown = 20  # 每组最多显示格子数（避免太大）

        try:
            font_title = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)
            font_id    = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 12)
            font_info  = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 11)
        except Exception:
            font_title = ImageFont.load_default()
            font_id    = ImageFont.load_default()
            font_info  = ImageFont.load_default()

        # 计算总尺寸：按列分页，每列一组
        total_w = cols_per_row * (CELL + MARGIN) + MARGIN

        # 先收集所有组渲染数据的高度
        group_heights = []
        for g in groups:
            shown = min(len(g["positions"]), max_cells_shown)
            rows_n = (shown + cols_per_row - 1) // cols_per_row
            h = HEADER + rows_n * (CELL + MARGIN) + MARGIN
            group_heights.append(h)

        total_h = sum(group_heights) + MARGIN + 30

        img = Image.new("RGB", (total_w, total_h), (20, 20, 35))
        draw = ImageDraw.Draw(img)

        y = MARGIN
        for gi, g in enumerate(groups):
            bead_id = g["bead_id"]
            rgb = self.beads.get(bead_id, tuple(int(v) for v in g["avg_rgb"][:3]))
            rgb_tuple = tuple(int(v) for v in rgb)
            brightness = sum(rgb) / 3
            tc = (255, 255, 255) if brightness < 128 else (0, 0, 0)
            low = g["avg_conf"] < 0.8

            # 组标题栏
            bar_h = HEADER
            draw.rectangle([0, y, total_w, y + bar_h],
                           fill=(40, 40, 60) if not low else (80, 20, 30))
            # 颜色条
            draw.rectangle([0, y, 12, y + bar_h], fill=rgb_tuple)
            # 色号文字
            draw.text((16, y + 4), f"{bead_id}", fill=tc, font=font_title)
            # 信息
            conf_color = (244, 114, 182) if low else (166, 227, 161)
            info_txt = f"{g['count']}格  avg={g['avg_conf']:.0%}  min={g['min_conf']:.0%}"
            draw.text((16, y + bar_h - 18), info_txt, fill=conf_color, font=font_info)
            y += bar_h

            # 格子缩略图
            positions = sorted(g["positions"])[:max_cells_shown]
            for idx, (r, c) in enumerate(positions):
                col_i = idx % cols_per_row
                row_i = idx // cols_per_row
                x = MARGIN + col_i * (CELL + MARGIN)
                cy = y + row_i * (CELL + MARGIN)

                cell_img = self._crop_cell(r, c)
                if cell_img:
                    cell_img = cell_img.resize((CELL, CELL), Image.LANCZOS)
                    img.paste(cell_img, (x, cy))
                else:
                    draw.rectangle([x, cy, x + CELL, cy + CELL], fill=(50, 50, 70))

                # 格子边框
                draw.rectangle([x, cy, x + CELL, cy + CELL],
                               outline=(80, 80, 110), width=1)

            y += group_heights[gi] - bar_h

        out_dir = _get_base_dir()
        out_path = os.path.join(out_dir, "tmp.jpg")
        img.save(out_path, quality=95)
        QMessageBox.information(
            self, "已保存",
            f"聚类图（含原始格子截图）已保存至：\n{out_path}\n\n"
            "每组最多显示前 20 个格子，粉色标题栏 = 低置信度颜色。"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  8. Step 2：图片预览 + 两阶段框选 + 缩放/平移
# ─────────────────────────────────────────────────────────────────────────────

class ImageSelectWidget(QLabel):
    """
    两阶段框选 + 缩放/平移。
    嵌入 QScrollArea 使用其滚动条平移。
    Phase 1: 鼠标点击 → 固定左上角（显示①号handle）
    Phase 2: 拖拽 → 拉出右下角，选框完成（显示①+②号handle）

    缩放/平移规则：
    - Ctrl + 滚轮  → 缩放
    - 纯滚轮        → 上下滚动
    - Shift + 滚轮  → 左右滚动
    - 右侧/底部滚动条 → 任意方向平移
    """
    selection_done = pyqtSignal(object)   # (x, y, w, h)
    zoom_changed   = pyqtSignal(float)

    HANDLE_SIZE = 10
    MIN_SCALE   = 0.1
    MAX_SCALE   = 10.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.setCursor(QCursor(Qt.CrossCursor))
        self.setMouseTracking(True)

        self._orig_pixmap   = None
        self._scale         = 1.0
        self._fit_scale     = 1.0

        # 两阶段选区
        self._phase          = 0
        self._tl_img         = None
        self._br_img         = None
        self._mouse_start    = None

        # 平移（空格键/Ctrl）
        self._panning      = False
        self._pan_start    = QPoint()   # 鼠标按下时的widget坐标
        self._pan_sb_start_h = 0        # 按下时的水平滚动条值
        self._pan_sb_start_v = 0        # 按下时的垂直滚动条值
        self._space_held   = False

        # 实时预览（Phase1拖拽时）
        self._preview_br_w  = None

        # ScrollArea 引用（由 GridParseWindow 传入）
        self._scroll_area   = None

        self.setStyleSheet("background: #181825; border: 1px solid #313244;")

    def set_scroll_area(self, sa):
        """由 GridParseWindow 注入 ScrollArea 引用，用于控制滚动条"""
        self._scroll_area = sa

    def load_image(self, pixmap: QPixmap):
        self._orig_pixmap = pixmap
        self._phase       = 0
        self._tl_img      = None
        self._br_img      = None
        self._preview_br_w = None
        self._scale       = self._fit_scale if hasattr(self, "_fit_scale") else 1.0
        self._fit_to_size()
        self.update()

    def _fit_to_size(self):
        if self._orig_pixmap is None:
            return
        ow = self._orig_pixmap.width()
        oh = self._orig_pixmap.height()
        if self._scroll_area:
            sa = self._scroll_area
            ww = max(sa.viewport().width(), 1)
            wh = max(sa.viewport().height(), 1)
        else:
            ww = max(self.width(), 1)
            wh = max(self.height(), 1)
        self._fit_scale = min(ww / ow, wh / oh, 1.0)
        if self._scale < self._fit_scale * 0.99:
            self._scale = self._fit_scale
        self._resize_to_content()

    def _resize_to_content(self):
        """更新 widget 尺寸 = 缩放后的图片尺寸，触发滚动条"""
        if self._orig_pixmap is None:
            return
        w = int(self._orig_pixmap.width() * self._scale)
        h = int(self._orig_pixmap.height() * self._scale)
        self.setFixedSize(w, h)

    # ── 坐标换算（widget 左上角 = (0,0)，无 offset） ──
    def _widget_to_img(self, pt: QPoint) -> QPoint:
        return QPoint(
            int(pt.x() / self._scale),
            int(pt.y() / self._scale)
        )

    def _img_to_widget(self, pt: QPoint) -> QPoint:
        return QPoint(
            int(pt.x() * self._scale),
            int(pt.y() * self._scale)
        )

    def _sel_rect_in_img(self):
        """返回图片坐标系下的选区 QRect（Phase2有效）"""
        if self._phase < 2 or self._tl_img is None or self._br_img is None:
            return QRect()
        r = QRect(self._tl_img, self._br_img).normalized()
        if self._orig_pixmap:
            r = r.intersected(QRect(0, 0, self._orig_pixmap.width(), self._orig_pixmap.height()))
        return r

    # ── 鼠标事件 ──
    def mousePressEvent(self, event):
        if self._orig_pixmap is None:
            return

        # Ctrl+左键 或 空格+左键 = 平移（通过滚动条实现）
        if (event.modifiers() & Qt.ControlModifier) or self._space_held:
            self._panning = True
            self._pan_start = event.pos()
            sa = self._scroll_area
            if sa:
                self._pan_sb_start_h = sa.horizontalScrollBar().value()
                self._pan_sb_start_v = sa.verticalScrollBar().value()
            else:
                self._pan_sb_start_h = 0
                self._pan_sb_start_v = 0
            self.setCursor(QCursor(Qt.SizeAllCursor))
            return

        if event.button() != Qt.LeftButton:
            return

        img_pt = self._widget_to_img(event.pos())
        img_pt.setX(max(0, min(img_pt.x(), self._orig_pixmap.width() - 1)))
        img_pt.setY(max(0, min(img_pt.y(), self._orig_pixmap.height() - 1)))

        if self._phase == 0:
            # Phase 1：固定左上角
            self._phase   = 1
            self._tl_img  = img_pt
            self._br_img  = img_pt
            self._mouse_start = event.pos()
            self.setCursor(QCursor(Qt.CrossCursor))

        elif self._phase == 1:
            # Phase 1 再次点击：重新固定左上角
            self._tl_img  = img_pt
            self._br_img  = img_pt
            self._mouse_start = event.pos()

        elif self._phase == 2:
            # Phase 2：检查是否点到了 handle 做微调
            hit = self._hit_handle(event.pos())
            if hit:
                self._drag_handle     = hit
                self._drag_start_w    = event.pos()
                self._drag_tl_start   = QPoint(self._tl_img)
                self._drag_br_start   = QPoint(self._br_img)
            else:
                # 点在框外/框内，重新开始
                self._phase   = 1
                self._tl_img  = img_pt
                self._br_img  = img_pt
                self._mouse_start = event.pos()
            self.setCursor(QCursor(Qt.CrossCursor))

        self.update()

    def mouseMoveEvent(self, event):
        if self._orig_pixmap is None:
            return

        # 平移（空格/Ctrl 拖动）：鼠标移动距离 = 滚动条减少的距离
        if self._panning:
            sa = self._scroll_area
            if sa:
                sb_h = sa.horizontalScrollBar()
                sb_v = sa.verticalScrollBar()
                sb_h.setValue(self._pan_sb_start_h - (event.pos().x() - self._pan_start.x()))
                sb_v.setValue(self._pan_sb_start_v - (event.pos().y() - self._pan_start.y()))
            return

        # Phase 1：根据鼠标位置实时更新预览选框
        if self._phase == 1 and self._tl_img is not None:
            img_pt = self._widget_to_img(event.pos())
            img_pt.setX(max(0, min(img_pt.x(), self._orig_pixmap.width())))
            img_pt.setY(max(0, min(img_pt.y(), self._orig_pixmap.height())))
            self._preview_br_w = self._img_to_widget(img_pt)
            self.update()

        # 拖拽 handle
        if hasattr(self, "_drag_handle") and self._drag_handle is not None:
            delta_img = self._widget_to_img(event.pos()) - self._widget_to_img(self._drag_start_w)
            iw = self._orig_pixmap.width()
            ih = self._orig_pixmap.height()

            if self._drag_handle == 'tl':
                self._tl_img = QPoint(
                    max(0, min(self._drag_tl_start.x() + delta_img.x(), iw - 1)),
                    max(0, min(self._drag_tl_start.y() + delta_img.y(), ih - 1))
                )
            elif self._drag_handle == 'br':
                self._br_img = QPoint(
                    max(0, min(self._drag_br_start.x() + delta_img.x(), iw)),
                    max(0, min(self._drag_br_start.y() + delta_img.y(), ih))
                )
            elif self._drag_handle == 'tm':
                self._tl_img.setY(max(0, min(self._drag_tl_start.y() + delta_img.y(), self._br_img.y() - 1)))
            elif self._drag_handle == 'bm':
                self._br_img.setY(max(0, min(self._drag_br_start.y() + delta_img.y(), ih)))
            elif self._drag_handle == 'lm':
                self._tl_img.setX(max(0, min(self._drag_tl_start.x() + delta_img.x(), self._br_img.x() - 1)))
            elif self._drag_handle == 'rm':
                self._br_img.setX(max(0, min(self._drag_br_start.x() + delta_img.x(), iw)))
            elif self._drag_handle == 'move':
                dx = delta_img.x(); dy = delta_img.y()
                new_tl_x = max(0, min(self._drag_tl_start.x() + dx, iw - 1))
                new_tl_y = max(0, min(self._drag_tl_start.y() + dy, ih - 1))
                new_br_x = max(0, min(self._drag_br_start.x() + dx, iw))
                new_br_y = max(0, min(self._drag_br_start.y() + dy, ih))
                self._tl_img = QPoint(new_tl_x, new_tl_y)
                self._br_img = QPoint(new_br_x, new_br_y)
            self.update()
            return

        # 更新鼠标样式
        hit = self._hit_handle(event.pos()) if self._phase >= 2 else None
        if self._space_held or (event.modifiers() & Qt.ControlModifier):
            self.setCursor(QCursor(Qt.SizeAllCursor))
        elif hit:
            cur_map = {
                'tl': Qt.SizeFDiagCursor, 'br': Qt.SizeFDiagCursor,
                'tr': Qt.SizeBDiagCursor, 'bl': Qt.SizeBDiagCursor,
                'tm': Qt.SizeVerCursor,   'bm': Qt.SizeVerCursor,
                'lm': Qt.SizeHorCursor,   'rm': Qt.SizeHorCursor,
                'move': Qt.SizeAllCursor,
            }
            self.setCursor(QCursor(cur_map.get(hit, Qt.CrossCursor)))
        else:
            self.setCursor(QCursor(Qt.CrossCursor) if not self._space_held else QCursor(Qt.SizeAllCursor))

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._panning:
            self._panning = False
            self.setCursor(QCursor(Qt.CrossCursor) if not self._space_held else QCursor(Qt.SizeAllCursor))
            return

        if event.button() != Qt.LeftButton:
            return

        # 完成 handle 拖拽
        if hasattr(self, "_drag_handle") and self._drag_handle is not None:
            self._drag_handle = None
            self.update()
            return

        if self._phase == 1 and self._tl_img is not None and self._mouse_start is not None:
            # Phase 1 松开：从 tl 拖到当前位置，形成选框
            img_pt = self._widget_to_img(event.pos())
            img_pt.setX(max(0, min(img_pt.x(), self._orig_pixmap.width())))
            img_pt.setY(max(0, min(img_pt.y(), self._orig_pixmap.height())))
            self._br_img = img_pt
            r = QRect(self._tl_img, self._br_img).normalized()
            if r.width() > 5 and r.height() > 5:
                self._phase = 2
            else:
                self._br_img = self._tl_img
            self.update()

    def wheelEvent(self, event):
        """滚轮：Ctrl=缩放，纯滚轮=上下滚动，Shift+滚轮=左右滚动"""
        if self._orig_pixmap is None:
            return

        sa = self._scroll_area
        ctrl = bool(event.modifiers() & Qt.ControlModifier)

        if ctrl:
            # Ctrl+滚轮 → 缩放（保持鼠标下像素不动）
            delta = event.angleDelta().y()
            factor = 1.15 if delta > 0 else 1.0 / 1.15
            before = self._widget_to_img(event.pos())
            self._scale = max(self.MIN_SCALE, min(self.MAX_SCALE, self._scale * factor))
            self._resize_to_content()
            after = self._img_to_widget(before)
            if sa:
                sb_h = sa.horizontalScrollBar()
                sb_v = sa.verticalScrollBar()
                sb_h.setValue(sb_h.value() + event.pos().x() - after.x())
                sb_v.setValue(sb_v.value() + event.pos().y() - after.y())
            self.update()
            self.zoom_changed.emit(self._scale)
        else:
            # 纯/Shift滚轮 → 直接操作滚动条
            if sa:
                delta = event.angleDelta()
                step_v = sa.verticalScrollBar().singleStep()
                step_h = sa.horizontalScrollBar().singleStep()
                # 纯滚轮 = 上下，Shift+滚轮 = 左右
                if bool(event.modifiers() & Qt.ShiftModifier):
                    sa.horizontalScrollBar().setValue(
                        sa.horizontalScrollBar().value() - delta.x() if delta.x() else
                        (-1 if delta.y() > 0 else 1) * step_h)
                else:
                    sa.verticalScrollBar().setValue(
                        sa.verticalScrollBar().value() - delta.y())
            event.accept()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self._space_held = True
            if not self._panning:
                self.setCursor(QCursor(Qt.SizeAllCursor))
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Space:
            self._space_held = False
            if not self._panning:
                self.setCursor(QCursor(Qt.CrossCursor))
        super().keyReleaseEvent(event)

    def resizeEvent(self, event):
        self._fit_to_size()
        self.update()

    def _hit_handle(self, pos: QPoint):
        """检查是否点到了 handle"""
        if self._phase < 2:
            return None
        r = self._current_rect_w()
        if r.isNull():
            return None
        hs = self.HANDLE_SIZE
        cx = r.center().x(); cy = r.center().y()
        handles = {
            'tl': QRect(r.left() - hs//2, r.top() - hs//2,    hs, hs),
            'tr': QRect(r.right() - hs//2, r.top() - hs//2,   hs, hs),
            'bl': QRect(r.left() - hs//2, r.bottom() - hs//2, hs, hs),
            'br': QRect(r.right() - hs//2, r.bottom() - hs//2,hs, hs),
            'tm': QRect(cx - hs//2, r.top() - hs//2,          hs, hs),
            'bm': QRect(cx - hs//2, r.bottom() - hs//2,       hs, hs),
            'lm': QRect(r.left() - hs//2, cy - hs//2,         hs, hs),
            'rm': QRect(r.right() - hs//2, cy - hs//2,        hs, hs),
        }
        for name, hr in handles.items():
            if hr.contains(pos):
                return name
        if r.contains(pos):
            return 'move'
        return None

    def _current_rect_w(self):
        """当前 widget 坐标系下的选框（Phase2用）"""
        if self._phase < 2 or self._tl_img is None or self._br_img is None:
            # Phase 1 实时预览
            if self._phase == 1 and self._tl_img is not None and self._preview_br_w is not None:
                return QRect(self._img_to_widget(self._tl_img), self._preview_br_w).normalized()
            return QRect()
        return QRect(self._img_to_widget(self._tl_img),
                     self._img_to_widget(self._br_img)).normalized()

    # ── 绘制 ──
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#181825"))

        if self._orig_pixmap is None:
            painter.setPen(QColor("#585b70"))
            painter.setFont(QFont("Arial", 14))
            painter.drawText(self.rect(), Qt.AlignCenter,
                             "← 点击左侧「选择图片」按钮加载图片")
            return

        # 绘制图片（widget 左上角即 (0,0)）
        dw = int(self._orig_pixmap.width() * self._scale)
        dh = int(self._orig_pixmap.height() * self._scale)
        painter.drawPixmap(0, 0, dw, dh, self._orig_pixmap)

        # 选框（Phase 1 & 2）
        sel_rect_w = self._current_rect_w()
        if not sel_rect_w.isNull():
            r = sel_rect_w
            # 遮罩
            mask = QColor(0, 0, 0, 80)
            img_rect = QRect(0, 0, dw, dh)
            dark_regions = [
                QRect(0, 0, img_rect.width(), max(0, r.top())),
                QRect(0, r.bottom(), img_rect.width(), max(0, img_rect.bottom() - r.bottom())),
                QRect(0, r.top(), max(0, r.left()), r.height()),
                QRect(r.right(), r.top(), max(0, img_rect.right() - r.right()), r.height()),
            ]
            for dr in dark_regions:
                if dr.isValid():
                    painter.fillRect(dr, mask)

            # 选框线
            color = QColor("#a855f7") if self._phase == 2 else QColor("#f0abfc")
            painter.setPen(QPen(color, 2, Qt.DashLine))
            painter.drawRect(r)

            # Phase 1：只画左上角①号 handle
            if self._phase == 1:
                painter.setPen(QPen(QColor("#ffffff"), 1))
                painter.setBrush(QBrush(QColor("#a855f7")))
                tl_w = self._img_to_widget(self._tl_img)
                h_rect = QRect(tl_w.x() - self.HANDLE_SIZE//2,
                                tl_w.y() - self.HANDLE_SIZE//2,
                                self.HANDLE_SIZE, self.HANDLE_SIZE)
                painter.drawRect(h_rect)
                painter.setPen(QColor("#ffffff"))
                painter.setFont(QFont("Arial", 9))
                painter.drawText(h_rect.adjusted(0, -16, 0, -16),
                                  Qt.AlignCenter, "① 左上角")
                painter.setPen(QColor("#e0e0e0"))
                painter.setFont(QFont("Arial", 10))
                painter.drawText(r.left() + 4, r.top() - 6,
                                 "拖动设定右下角...")

            # Phase 2：画 ①（左上角）+ ②（右下角）+ 中间4个 handle
            elif self._phase == 2:
                handles = {
                    'tl': self._img_to_widget(self._tl_img),
                    'br': self._img_to_widget(self._br_img),
                }
                handle_colors = {'tl': QColor("#a855f7"), 'br': QColor("#f97316")}
                handle_labels = {'tl': '①', 'br': '②'}
                painter.setFont(QFont("Arial", 8, QFont.Bold))

                for name, pt in handles.items():
                    h_rect = QRect(pt.x() - self.HANDLE_SIZE//2,
                                    pt.y() - self.HANDLE_SIZE//2,
                                    self.HANDLE_SIZE, self.HANDLE_SIZE)
                    painter.setPen(QPen(handle_colors[name], 2))
                    painter.setBrush(QBrush(handle_colors[name]))
                    painter.drawRect(h_rect)
                    lbl_rect = h_rect.adjusted(0, -14, 0, -14)
                    painter.setPen(handle_colors[name])
                    painter.drawText(lbl_rect, Qt.AlignCenter, handle_labels[name])

                # 中间4个handle
                r_norm = sel_rect_w.normalized()
                cx = r_norm.center().x()
                cy = r_norm.center().y()
                mid_handles = {
                    'tm': QPoint(cx, r_norm.top()),
                    'bm': QPoint(cx, r_norm.bottom()),
                    'lm': QPoint(r_norm.left(), cy),
                    'rm': QPoint(r_norm.right(), cy),
                }
                for pt in mid_handles.values():
                    h_rect = QRect(pt.x() - 4, pt.y() - 4, 8, 8)
                    painter.setPen(QPen(QColor("#ffffff"), 1))
                    painter.setBrush(QBrush(QColor("#a855f7")))
                    painter.drawRect(h_rect)

                # 尺寸标注
                img_r = self._sel_rect_in_img()
                if not img_r.isNull():
                    painter.setPen(QColor("#e0e0e0"))
                    painter.setFont(QFont("Arial", 10))
                    painter.drawText(r.left() + 4, r.top() - 6,
                                     f"{img_r.width()} × {img_r.height()} px")

        # 底部提示
        if self._orig_pixmap:
            painter.setPen(QColor("#585b70"))
            painter.setFont(QFont("Arial", 9))
            phase_hint = {
                0: "点击设定左上角",
                1: "拖动设定右下角",
                2: "框选完成，可拖 handle 微调",
            }
            tip = phase_hint.get(self._phase, "")
            tip += " | Ctrl+滚轮=缩放 | 滚轮/滚动条=平移"
            painter.drawText(self.rect().adjusted(6, -18, -6, 0),
                             Qt.AlignLeft | Qt.AlignBottom, tip)

        painter.end()

    def get_selection_in_img(self):
        """返回 (x, y, w, h) 或 None"""
        if self._phase < 2:
            return None
        r = self._sel_rect_in_img()
        if r.isNull() or r.width() < 2 or r.height() < 2:
            return None
        return (r.x(), r.y(), r.width(), r.height())

    def zoom_to_fit(self):
        self._fit_to_size()
        # 滚动到中心
        if self._scroll_area:
            sa = self._scroll_area
            sa.horizontalScrollBar().setValue(
                (self.width() - sa.viewport().width()) // 2)
            sa.verticalScrollBar().setValue(
                (self.height() - sa.viewport().height()) // 2)
        self.update()

    def zoom_in(self):
        self._scale = min(self.MAX_SCALE, self._scale * 1.25)
        self._resize_to_content()
        self.zoom_changed.emit(self._scale)
        self.update()

    def zoom_out(self):
        self._scale = max(self.MIN_SCALE, self._scale / 1.25)
        self._resize_to_content()
        self.zoom_changed.emit(self._scale)
        self.update()


# ─────────────────────────────────────────────────────────────────────────────
#  9. 主窗口
# ─────────────────────────────────────────────────────────────────────────────

STYLE = """
QMainWindow, QWidget#central { background: #1e1e2e; }
QLabel { color: #cdd6f4; font-size: 13px; }
QPushButton {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #7c3aed, stop:1 #9c27b0);
    color: white; border: none; border-radius: 8px;
    padding: 8px 20px; font-size: 13px; min-width: 110px;
}
QPushButton:hover { background: #9333ea; }
QPushButton:disabled { background: #45475a; color: #6c7086; }
QPushButton#btn_secondary {
    background: #313244; color: #cdd6f4; border: 1px solid #45475a;
}
QPushButton#btn_secondary:hover { background: #45475a; }
QPushButton#btn_tool {
    background: #313244; color: #cdd6f4; border: 1px solid #45475a;
    border-radius: 6px; padding: 4px 10px; min-width: 0; font-size: 12px;
}
QPushButton#btn_tool:hover { background: #45475a; }
QProgressBar {
    background: #313244; border: 1px solid #45475a;
    border-radius: 4px; height: 10px; text-align: center; color: transparent;
    font-size: 11px;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #7c3aed,stop:1 #a855f7);
    border-radius: 3px;
}
"""


class StepLabel(QLabel):
    def __init__(self, step: int, text: str, parent=None):
        super().__init__(parent)
        self._step = step; self._text = text
        self._active = False; self._done = False
        self._update_style()

    def set_active(self, active: bool):
        self._active = active; self._done = False
        self._update_style()

    def set_done(self, done: bool):
        self._done = done; self._active = False
        self._update_style()

    def _update_style(self):
        if self._done:
            icon, bg, color, bcolor = "✓", "#1a472a", "#a6e3a1", "#a6e3a1"
        elif self._active:
            icon, bg, color, bcolor = f"{self._step}", "#3b2f5e", "#cba6f7", "#a855f7"
        else:
            icon, bg, color, bcolor = f"{self._step}", "#181825", "#585b70", "#313244"
        self.setText(f"  {icon}  {self._text}")
        self.setStyleSheet(f"""
            QLabel {{
                background: {bg}; color: {color};
                border: 1px solid {bcolor}; border-radius: 6px;
                padding: 6px 10px; font-size: 13px;
                font-weight: {'bold' if self._active else 'normal'};
            }}
        """)


class GridParseWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图纸解析工具  [深度学习分类器]")
        self.resize(1200, 800)
        self.setStyleSheet(STYLE)

        self._img_path      = None
        self._img_arr       = None
        self._beads         = None   # {bead_id: (R,G,B)}
        self._grid_data     = None
        self._classifier    = None   # 延迟加载
        self._cols          = None
        self._rows          = None
        self._dataset_mode  = False  # 是否开启数据集生成模式
        self._dataset_dir   = None   # 数据集目标文件夹

        self._build_ui()
        self._beads = load_bead_palette()
        self._go_to_step(1)

    def _build_ui(self):
        central = QWidget(objectName="central")
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── 左侧面板 ──
        left = QFrame()
        left.setFixedWidth(235)
        left.setStyleSheet("background: #181825; border-right: 1px solid #313244;")
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(14, 20, 14, 20)
        left_layout.setSpacing(10)

        title = QLabel("图纸解析")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #cba6f7; margin-bottom: 8px;")
        left_layout.addWidget(title)

        self.step1_label = StepLabel(1, "选择图片")
        self.step2_label = StepLabel(2, "框选格子区域")
        self.step3_label = StepLabel(3, "输入格子数量")
        self.step4_label = StepLabel(4, "生成并保存")
        for lbl in [self.step1_label, self.step2_label,
                     self.step3_label, self.step4_label]:
            left_layout.addWidget(lbl)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #313244;")
        left_layout.addWidget(sep)

        # 数据集模式选项
        self.chk_dataset = QCheckBox("📁 生成数据集模式")
        self.chk_dataset.setStyleSheet("""
            QCheckBox {
                color: #cdd6f4;
                font-size: 12px;
                padding: 4px 0px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 4px;
                border: 1px solid #7c3aed;
                background: #313244;
            }
            QCheckBox::indicator:checked {
                background: #7c3aed;
            }
        """)
        self.chk_dataset.stateChanged.connect(self._on_dataset_mode_changed)
        left_layout.addWidget(self.chk_dataset)

        self.btn_select_dataset_dir = QPushButton("📂 选择目标文件夹")
        self.btn_select_dataset_dir.setObjectName("btn_secondary")
        self.btn_select_dataset_dir.setVisible(False)
        self.btn_select_dataset_dir.clicked.connect(self._on_select_dataset_dir)
        left_layout.addWidget(self.btn_select_dataset_dir)

        self.dataset_dir_label = QLabel("未选择目标文件夹")
        self.dataset_dir_label.setStyleSheet("color: #6c7086; font-size: 10px;")
        self.dataset_dir_label.setWordWrap(True)
        self.dataset_dir_label.setVisible(False)
        left_layout.addWidget(self.dataset_dir_label)

        sep2 = QFrame(); sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet("color: #313244;")
        left_layout.addWidget(sep2)

        self.info_label = QLabel("请选择图片文件")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("color: #a6adc8; font-size: 12px; line-height: 1.5;")
        left_layout.addWidget(self.info_label)

        self.mode_label = QLabel("🔍 分类器：未加载")
        self.mode_label.setStyleSheet("color: #89b4fa; font-size: 11px;")
        left_layout.addWidget(self.mode_label)

        left_layout.addStretch()

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setFormat("%p%")
        left_layout.addWidget(self.progress_bar)

        self.btn_open         = QPushButton("📂  选择图片")
        self.btn_confirm_sel  = QPushButton("✅  确认选区")
        self.btn_input_size   = QPushButton("🔢  输入格子数")
        self.btn_generate     = QPushButton("💾  生成图纸")
        self.btn_reset        = QPushButton("🔄  重新开始")
        self.btn_reset.setObjectName("btn_secondary")

        self.btn_confirm_sel.setEnabled(False)
        self.btn_input_size.setEnabled(False)
        self.btn_generate.setEnabled(False)

        for btn in [self.btn_open, self.btn_confirm_sel,
                     self.btn_input_size, self.btn_generate, self.btn_reset]:
            left_layout.addWidget(btn)

        self.btn_open.clicked.connect(self._on_open_image)
        self.btn_confirm_sel.clicked.connect(self._on_confirm_selection)
        self.btn_input_size.clicked.connect(self._on_input_grid_size)
        self.btn_generate.clicked.connect(self._on_generate)
        self.btn_reset.clicked.connect(self._on_reset)

        root.addWidget(left)

        # ── 右侧主区域 ──
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(12, 12, 12, 12)
        right_layout.setSpacing(8)

        self.tip_label = QLabel()
        self.tip_label.setStyleSheet("""
            background: #2a2a3e; color: #89b4fa;
            border: 1px solid #45475a; border-radius: 6px;
            padding: 6px 12px; font-size: 12px;
        """)
        right_layout.addWidget(self.tip_label)

        # 缩放工具栏
        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)
        self.zoom_fit_btn = QPushButton("⊡ 适应窗口")
        self.zoom_fit_btn.setObjectName("btn_tool")
        self.zoom_fit_btn.clicked.connect(self._on_zoom_fit)
        toolbar.addWidget(self.zoom_fit_btn)

        self.zoom_in_btn = QPushButton("🔍+")
        self.zoom_in_btn.setObjectName("btn_tool")
        self.zoom_in_btn.clicked.connect(self._on_zoom_in)
        toolbar.addWidget(self.zoom_in_btn)

        self.zoom_out_btn = QPushButton("🔍-")
        self.zoom_out_btn.setObjectName("btn_tool")
        self.zoom_out_btn.clicked.connect(self._on_zoom_out)
        toolbar.addWidget(self.zoom_out_btn)

        self.zoom_val_label = QLabel("100%")
        self.zoom_val_label.setStyleSheet("color: #89b4fa; font-size: 11px; min-width: 40px;")
        toolbar.addWidget(self.zoom_val_label)
        toolbar.addStretch()
        right_layout.addLayout(toolbar)

        self.view_scroll = QScrollArea()
        self.view_scroll.setWidgetResizable(False)
        self.view_scroll.setAlignment(Qt.AlignCenter)
        self.view_scroll.setStyleSheet("background: #181825; border: none;")

        self.img_select_widget = ImageSelectWidget()
        self.img_select_widget.set_scroll_area(self.view_scroll)
        self.img_select_widget.zoom_changed.connect(self._on_zoom_changed)
        self.view_scroll.setWidget(self.img_select_widget)
        right_layout.addWidget(self.view_scroll)

        root.addWidget(right)

    # ── 步骤管理 ──
    def _go_to_step(self, step: int):
        self._current_step = step
        labels = [self.step1_label, self.step2_label,
                  self.step3_label, self.step4_label]
        tips = [
            "Step 1：点击「选择图片」，加载要解析的图纸图片。",
            "Step 2：① 点击设定左上角 → ② 拖动设定右下角 → 确认选区。Ctrl+滚轮缩放，滚轮/滚动条平移。",
            "Step 3：点击「输入格子数」，填写该区域横向/纵向格子数量。",
            "Step 4：点击「生成图纸」，使用深度学习模型批量识别。完成后弹出格子视图+审查对话框。",
        ]
        for i, lbl in enumerate(labels):
            if i + 1 < step:    lbl.set_done(True)
            elif i + 1 == step: lbl.set_active(True)
            else:               lbl.set_active(False)
        self.tip_label.setText(f"👉 {tips[step - 1]}")
        self.btn_confirm_sel.setEnabled(step == 2)
        self.btn_input_size.setEnabled(step == 3)
        self.btn_generate.setEnabled(step == 4)

    # ── Step 1 ──
    def _on_open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图纸图片", "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.webp);;所有文件 (*)"
        )
        if not path:
            return
        self._img_path = path
        img = Image.open(path).convert("RGB")
        self._img_arr  = np.array(img)

        h, w = self._img_arr.shape[:2]
        q_img = QImage(self._img_arr.data, w, h, w * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.img_select_widget.load_image(pixmap)

        name = os.path.basename(path)
        self.info_label.setText(f"已加载：\n{name}\n尺寸：{w} × {h}")
        self._go_to_step(2)
        self.view_scroll.setVisible(True)
        self._ensure_classifier()

    def _ensure_classifier(self):
        if self._classifier is not None:
            return
        try:
            batch_size = 256 if torch.cuda.is_available() else 128

            def _progress(done, total):
                pct = int(done / total * 100) if total > 0 else 0
                self.progress_bar.setValue(max(1, pct))

            self._classifier = BeadClassifier(
                batch_size=batch_size,
                progress_callback=_progress
            )
            self.mode_label.setText(
                "🔍 分类器：✅ GPU" if self._classifier.device.type == "cuda"
                else "🔍 分类器：✅ CPU"
            )
        except Exception as e:
            self._classifier = None
            self.mode_label.setText(f"⚠️ 分类器：加载失败\n{str(e)[:40]}")
            traceback.print_exc()

    # ── Step 2 ──
    def _on_confirm_selection(self):
        sel = self.img_select_widget.get_selection_in_img()
        if sel is None:
            QMessageBox.warning(self, "提示",
                "请先在图片上完成两阶段框选：\n"
                "① 点击设定左上角 → ② 拖动设定右下角")
            return
        x, y, w, h = sel
        self._selection = sel
        self.info_label.setText(
            f"选区：\n({x}, {y})\n宽 {w} px\n高 {h} px"
        )
        self._go_to_step(3)

    # ── Step 3 ──
    def _on_input_grid_size(self):
        sel = getattr(self, "_selection", None)
        if sel:
            _, _, sw, sh = sel
            suggested = max(sw, sh) // 20
            suggested = max(5, min(suggested, 200))
        else:
            suggested = 50
        dlg = GridSizeDialog(self, suggested_cols=suggested, suggested_rows=suggested)
        if dlg.exec_() != QDialog.Accepted:
            return
        self._cols, self._rows = dlg.get_values()
        self.info_label.setText(
            f"选区：\n宽 {self._selection[2]} px\n高 {self._selection[3]} px\n"
            f"\n格子：{self._cols} 列 × {self._rows} 行"
        )
        self._go_to_step(4)

    # ── Step 4 ──
    def _on_generate(self):
        if self._img_arr is None or self._selection is None:
            return
        if self._classifier is None:
            QMessageBox.warning(self, "错误", "分类器未加载，请检查模型文件是否存在。")
            return

        total_cells = self._cols * self._rows
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(1)
        self.progress_bar.setFormat(f"正在识别... 0% ({total_cells} 格)")
        QApplication.processEvents()

        try:
            def _progress(done, total):
                pct = int(done / total * 95) + 1
                self.progress_bar.setValue(min(pct, 95))
                self.progress_bar.setFormat(f"正在识别... {pct}% ({done}/{total} 格)")
                QApplication.processEvents()

            self._classifier.progress_callback = _progress

            grid_data = parse_grid_region_by_model(
                self._img_arr, self._selection,
                self._cols, self._rows,
                self._classifier, self._beads,
            )

            self._grid_data = grid_data
            self.progress_bar.setValue(96)
            QApplication.processEvents()

            # 保存 JSON
            base_name    = os.path.splitext(os.path.basename(self._img_path))[0]
            out_dir      = os.path.dirname(self._img_path) or "."
            json_path    = os.path.join(out_dir, f"{base_name}_grid.json")
            preview_path = os.path.join(out_dir, f"{base_name}_grid_preview.png")

            save_grid_json(grid_data, json_path)

            # 保存预览图（含完整图例）
            grid_2d = []
            for row in grid_data["grid"]:
                row_ids = [cell.get("bead_id", "空") if cell else "空" for cell in row]
                grid_2d.append(row_ids)
            cell_size = max(36, min(56, 800 // max(self._cols, self._rows)))
            preview_pil = render_grid_to_pil(grid_2d, self._beads, cell=cell_size)
            preview_pil.save(preview_path)

            self.progress_bar.setValue(99)
            QApplication.processEvents()

            # 数据集模式：保存独一无二的色块
            dataset_saved = 0
            if self._dataset_mode and self._dataset_dir:
                try:
                    self.progress_bar.setFormat("正在保存数据集...")
                    QApplication.processEvents()
                    dataset_saved = self._save_cells_to_dataset(
                        grid_data, self._img_arr, self._selection
                    )
                except Exception as e:
                    print(f"保存数据集时出错：{e}")
                    traceback.print_exc()

            self.progress_bar.setValue(100)
            self.progress_bar.setFormat("完成！")
            QApplication.processEvents()

            color_count = len({
                str(cell.get("bead_id")) for row in grid_data["grid"]
                for cell in row if cell and cell.get("bead_id")
            })

            info_text = (
                f"✅ 完成！\n\n"
                f"图纸：{self._cols} × {self._rows}\n"
                f"颜色数：{color_count}\n\n"
                f"已保存：\n{os.path.basename(json_path)}"
            )
            if dataset_saved > 0:
                info_text += f"\n\n📁 数据集：+{dataset_saved} 张色块"
            self.info_label.setText(info_text)

            for lbl in [self.step1_label, self.step2_label,
                         self.step3_label, self.step4_label]:
                lbl.set_done(True)
            self.tip_label.setText(
                "✅ 图纸生成完毕！可点击「重新开始」处理下一张。"
            )
            self.progress_bar.setVisible(False)

            # 弹出格子视图（含完整图例）
            dlg_preview = GridPreviewDialog(
                grid_data, self._beads,
                source_img_arr=self._img_arr, region=self._selection,
                parent=self
            )
            dlg_preview.exec_()

        except Exception as e:
            self.progress_bar.setVisible(False)
            traceback.print_exc()
            QMessageBox.critical(self, "错误",
                                 f"生成图纸时出错：\n{traceback.format_exc()}")

    # ── 缩放控制 ──
    def _on_zoom_fit(self):
        self.img_select_widget.zoom_to_fit()
        self._on_zoom_changed(self.img_select_widget._scale)

    def _on_zoom_in(self):
        self.img_select_widget.zoom_in()

    def _on_zoom_out(self):
        self.img_select_widget.zoom_out()

    def _on_zoom_changed(self, scale):
        self.zoom_val_label.setText(f"{int(scale * 100)}%")

    # ── 重新开始 ──
    def _on_reset(self):
        self._img_path   = None
        self._img_arr    = None
        self._grid_data  = None
        self._selection  = None
        self._cols       = None
        self._rows       = None
        self.img_select_widget._orig_pixmap = None
        self.img_select_widget._phase       = 0
        self.img_select_widget._tl_img      = None
        self.img_select_widget._br_img      = None
        self.img_select_widget._preview_br_w = None
        self.img_select_widget._scale      = 1.0
        self.img_select_widget.update()
        self.info_label.setText("请选择图片文件")
        self.progress_bar.setVisible(False)
        self.view_scroll.setVisible(True)
        self._go_to_step(1)

    # ── 数据集模式 ──
    def _on_dataset_mode_changed(self, state):
        self._dataset_mode = (state == Qt.Checked)
        self.btn_select_dataset_dir.setVisible(self._dataset_mode)
        self.dataset_dir_label.setVisible(self._dataset_mode)
        if not self._dataset_mode:
            self._dataset_dir = None
            self.dataset_dir_label.setText("未选择目标文件夹")

    def _on_select_dataset_dir(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, "选择数据集目标文件夹",
            self._dataset_dir or os.path.expanduser("~")
        )
        if not dir_path:
            return
        self._dataset_dir = dir_path
        # 显示文件夹名（简短）
        short_name = os.path.basename(dir_path) or dir_path
        self.dataset_dir_label.setText(f"📁 {short_name}")
        self.dataset_dir_label.setToolTip(dir_path)
        # 检查/创建222个色号文件夹
        self._ensure_dataset_folders(dir_path)

    def _ensure_dataset_folders(self, base_dir):
        """检查并创建222个色号文件夹"""
        missing = []
        for bead_id in FULL_BEAD_IDS:
            folder = os.path.join(base_dir, bead_id)
            if not os.path.exists(folder):
                os.makedirs(folder, exist_ok=True)
                missing.append(bead_id)
        if missing:
            QMessageBox.information(
                self, "文件夹已创建",
                f"已创建 {len(missing)} 个缺失的色号文件夹。\n"
                f"示例：{missing[0]} ...\n\n"
                f"目标文件夹：{base_dir}"
            )
        else:
            self.dataset_dir_label.setText(f"✅ {os.path.basename(base_dir)} (222个文件夹已就绪)")

    def _save_cells_to_dataset(self, grid_data, img_arr, selection):
        """
        将识别出的格子聚类后保存到数据集文件夹。
        每个独一无二的色块只保存一张代表性图片到对应色号文件夹。
        """
        if not self._dataset_dir:
            return 0

        rx, ry, rw, rh = selection
        rows_n = grid_data["height"]
        cols_n = grid_data["width"]
        cw = rw / cols_n
        ch = rh / rows_n

        saved_count = 0
        # 按预测色号分组，收集独一无二的色块
        # 格式: {bead_id: {unique_rgb_tuple: (row, col)}}
        unique_cells = defaultdict(lambda: defaultdict(list))

        for r in range(rows_n):
            for c in range(cols_n):
                cell = grid_data["grid"][r][c]
                if not cell:
                    continue
                bead_id = cell.get("bead_id", "空")
                rgb = tuple(cell.get("rgb", [128, 128, 128])[:3])

                # 裁切格子
                x0 = int(rx + c * cw)
                y0 = int(ry + r * ch)
                x1 = int(rx + (c + 1) * cw)
                y1 = int(ry + (r + 1) * ch)
                x0_c = max(0, min(x0, img_arr.shape[1] - 1))
                y0_c = max(0, min(y0, img_arr.shape[0] - 1))
                x1_c = max(x0_c + 1, min(x1, img_arr.shape[1]))
                y1_c = max(y0_c + 1, min(y1, img_arr.shape[0]))
                if y1_c <= y0_c or x1_c <= x0_c:
                    continue
                cell_arr = img_arr[y0_c:y1_c, x0_c:x1_c]
                if cell_arr.size == 0:
                    continue

                # 用rgb作为唯一标识，避免重复保存相似的颜色
                key = (round(rgb[0] / 10) * 10, round(rgb[1] / 10) * 10, round(rgb[2] / 10) * 10)
                unique_cells[bead_id][key].append((r, c, cell_arr))

        # 保存每个独一无二色块的一张代表性图片
        for bead_id, color_groups in unique_cells.items():
            bead_folder = os.path.join(self._dataset_dir, bead_id)
            os.makedirs(bead_folder, exist_ok=True)

            for key, cells in color_groups.items():
                # 只保存第一张
                r, c, cell_arr = cells[0]
                try:
                    cell_img = Image.fromarray(cell_arr.astype(np.uint8))
                    # 生成文件名：bead_id_idx.png
                    existing = len([f for f in os.listdir(bead_folder)
                                   if f.startswith(f"{bead_id}_") and f.endswith(".png")])
                    filename = f"{bead_id}_{existing + 1:04d}.png"
                    filepath = os.path.join(bead_folder, filename)
                    cell_img.save(filepath)
                    saved_count += 1
                except Exception as e:
                    print(f"保存色块失败 {bead_id}: {e}")
                    continue

        return saved_count


# ─────────────────────────────────────────────────────────────────────────────
#  入口
# ─────────────────────────────────────────────────────────────────────────────

# 禁用 Qt stylesheet 警告（QScrollArea 子控件选择器兼容性）
def _suppress_qt_warnings(msg_type, context, message):
    if "Could not parse stylesheet" in str(message):
        return
    print(message, file=sys.stderr)

try:
    from PyQt5.QtCore import qInstallMessageHandler
    qInstallMessageHandler(_suppress_qt_warnings)
except Exception:
    pass

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    # 统一设置 QScrollBar 样式（避免在单个 QScrollArea 上设置触发警告）
    app.setStyleSheet("""
        QScrollBar:vertical {
            width: 10px;
            background: #181825;
            margin: 0px;
        }
        QScrollBar::handle:vertical {
            background: #45475a;
            min-height: 30px;
            border-radius: 4px;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
        QScrollBar:horizontal {
            height: 10px;
            background: #181825;
            margin: 0px;
        }
        QScrollBar::handle:horizontal {
            background: #45475a;
            min-width: 30px;
            border-radius: 4px;
        }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            width: 0px;
        }
    """)
    win = GridParseWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
