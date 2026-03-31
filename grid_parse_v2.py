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
    QFileDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QSpinBox,
    QDialog, QDialogButtonBox, QFormLayout, QMessageBox, QScrollArea,
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
    模型推理版：逐格切图 → 逐格预测 → 组装 grid_data。
    每个格子独立过一次神经网络，结果互不影响。
    """
    x0, y0, rw, rh = region_rect
    h_full, w_full  = img_arr.shape[:2]
    x1, y1 = max(0, x0), max(0, y0)
    x2, y2 = min(w_full, x0 + rw), min(h_full, y0 + rh)
    region = img_arr[y1:y2, x1:x2]
    rh_actual, rw_actual = region.shape[:2]
    cell_w = rw_actual / cols
    cell_h = rh_actual / rows

    # ── 第一步：切所有格子 ───────────────────────────────────────────────────
    all_cells  = []
    cell_info  = []  # (row, col, cell_arr, avg_rgb)

    for row in range(rows):
        for col in range(cols):
            cx1, cy1 = int(col * cell_w), int(row * cell_h)
            cx2, cy2 = int((col + 1) * cell_w), int((row + 1) * cell_h)
            cx2 = min(cx2, rw_actual); cy2 = min(cy2, rh_actual)
            cell = region[cy1:cy2, cx1:cx2]
            if cell.size == 0:
                cell = np.zeros((1, 1, 3), dtype=np.uint8)
            avg_rgb = extract_dominant_color(cell)
            all_cells.append(cell)
            cell_info.append((row, col, cell, avg_rgb))

    total_cells = len(all_cells)

    # ── 第二步：逐格预测（predict_cells 内部自动分批）───────────────────────
    results = classifier.predict_cells(all_cells)

    # ── 第三步：组装 grid_data ───────────────────────────────────────────────
    grid = [[None] * cols for _ in range(rows)]

    for info, (bead_id, conf) in zip(cell_info, results):
        row, col, cell, avg_rgb = info
        grid[row][col] = {
            "bead_id":    bead_id,
            "bead_rgb":   bead_id,
            "confidence": round(conf, 4),
            "rgb":        list(avg_rgb),
        }

    print(f"[parse_grid_region_by_model] 逐格预测: {total_cells} 格全部独立预测")

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
    def __init__(self, grid_data, beads, source_img_arr=None, region=None,
                 img_path=None, parent=None):
        super().__init__(parent)
        self.grid_data   = grid_data
        self.beads       = beads
        self._src_arr    = source_img_arr
        self._region     = region  # (rx, ry, rw, rh)
        self._img_path   = img_path
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

        btn_review = QPushButton("📊 色块聚类修改")
        btn_review.setObjectName("btn_sec")
        btn_review.clicked.connect(self._open_cluster_edit)
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

    def _open_cluster_edit(self):
        """打开色块聚类批量修改弹窗，关闭自身"""
        img_path = self._img_path or ""
        out_dir  = os.path.dirname(img_path) or "."
        base     = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(out_dir, f"{base}_grid.json")

        # 先关闭自身预览窗口
        self.close()

        dlg = ColorClusterDialog(
            self.grid_data, self.beads,
            source_img_arr=self._src_arr, region=self._region,
            output_path=out_path,
            parent=self.parent()
        )
        dlg.grid_changed.connect(self._on_grid_changed_from_cluster)
        dlg.exec_()

    def _on_grid_changed_from_cluster(self):
        """聚类弹窗修改了 grid_data，刷新预览"""
        self._refresh_preview()

    def _refresh_preview(self):
        """重新渲染预览图"""
        rows = self.grid_data["height"]
        cols = self.grid_data["width"]
        grid_2d = [
            [cell.get("bead_id", "空") if cell else "空" for cell in row]
            for row in self.grid_data["grid"]
        ]
        cell_size = max(36, min(56, 800 // max(cols, rows)))
        new_pil = render_grid_to_pil(grid_2d, self.beads, cell=cell_size)
        self._pil_img = new_pil
        self._update_pixmap()


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
#  7b. 色块聚类批量修改弹窗（全新）
# ─────────────────────────────────────────────────────────────────────────────


class ClickableLabel(QLabel):
    """支持点击事件的 QLabel，发射 clicked(QPoint) 信号"""
    clicked = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)

    def mousePressEvent(self, event):
        self.clicked.emit(event)
        super().mousePressEvent(event)


class ColorClusterDialog(QDialog):
    """
    色块聚类批量修改弹窗：
    - 聚类方式：按原始 RGB 颜色聚类（相近颜色的格子归为一组）
    - 左侧：分页色块列表，每行显示真实格子截图拼图 + 色号 + 置信度 + 数量
    - 选中效果：点击某色块 → 该组高亮（正常颜色+紫色边框），其他全部低亮（变暗）
    - 右侧：图纸预览图，同步高亮/低亮所有格子位置
    - 底部：替换（色号板选色）、撤销、随时保存
    """

    # 通知父窗口数据已变更
    grid_changed = pyqtSignal()

    def __init__(self, grid_data, beads, source_img_arr=None, region=None,
                 output_path=None, parent=None):
        super().__init__(parent)
        self.grid_data   = grid_data      # 引用传递，直接修改
        self.beads       = beads
        self._src_arr    = source_img_arr  # numpy array (H,W,3)
        self._region     = region          # (rx, ry, rw, rh)
        self._out_path   = output_path
        self._parent_win = parent

        self._undo_stack = []              # 存 deep copy 的 grid_data 快照
        self._highlight_idx = None         # 当前高亮的聚类索引（int 或 None）
        self._cell_to_cluster = {}          # 缓存：(r,c)→cluster_idx 映射
        self._cluster_cache_ver = None      # 缓存版本号（clusters 对象id）
        self._selected = set()              # Ctrl+多选选中索引集合
        self._selection_anchor = None      # Shift+框选的锚点索引
        self._preview_dlg = None            # 原图预览弹窗引用（非模态）
        self._preview_img_lbl = None        # 原图预览可点击区域
        self._preview_zoom = 1.0            # 预览缩放倍数

        # 橡皮筋框选状态
        self._rubber_band = None            # 橡皮筋选框 widget
        self._rubber_start = None           # 框选起点 (QPoint)
        self._rubber_rect = None            # 当前框选区域 (QRect)
        self._rubber_drawing = False        # 是否正在框选

        rows = grid_data["height"]
        cols = grid_data["width"]

        self.setWindowTitle(f"色块聚类修改  {cols} × {rows}")
        self.setMinimumSize(1300, 780)
        self.setStyleSheet("""
            QDialog { background: #1e1e2e; }
            QLabel { color: #cdd6f4; }
            QPushButton {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #7c3aed,stop:1 #9c27b0);
                color: white; border: none; border-radius: 6px;
                padding: 7px 16px; font-size: 12px; font-weight: bold;
            }
            QPushButton:hover { background: #9333ea; }
            QPushButton:disabled { background: #45475a; color: #6c7086; }
            QPushButton#btn_sec {
                background: #313244; color: #cdd6f4;
                border: 1px solid #45475a; padding: 7px 16px; font-size: 12px;
            }
            QPushButton#btn_sec:hover { background: #45475a; }
            QPushButton#btn_replace {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #e64a19,stop:1 #bf360c);
                color: white; border: none; border-radius: 6px;
                padding: 8px 18px; font-size: 13px; font-weight: bold;
            }
            QPushButton#btn_replace:hover { background: #d84315; }
            QPushButton#btn_undo {
                background: #1e3a5f; color: #89b4fa;
                border: 1px solid #89b4fa; padding: 7px 16px; font-size: 12px;
            }
            QPushButton#btn_undo:hover { background: #1e4976; }
            QPushButton#btn_save {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #2e7d32,stop:1 #1b5e20);
                color: white; border: none; border-radius: 6px;
                padding: 7px 16px; font-size: 12px;
            }
            QPushButton#btn_save:hover { background: #388e3c; }
            QScrollArea { background: #181825; border: 1px solid #313244; }
            QTextEdit { background: #181825; color: #cdd6f4; border: 1px solid #313244; font-size: 12px; }
        """)

        self._clusters = self._build_clusters()
        self._per_page = 30
        self._cur_page = 0
        self._total_pages = max(1, (len(self._clusters) + self._per_page - 1) // self._per_page)

        self._init_ui()
        # 默认选中第一页第一项
        if self._clusters:
            self._select_cluster(0)

    # ── 构建聚类 ──────────────────────────────────────────────────────────────
    def _build_clusters(self):
        """
        按神经网络预测前的预处理结果聚类。
        预处理与 BeadClassifier 保持一致：Resize(64x64) → ToTensor → Normalize。
        只有预处理后完全一致的格子才会被归为一类。
        """
        rows = self.grid_data["height"]
        cols = self.grid_data["width"]
        src = self._src_arr
        region = self._region

        # 复用 BeadClassifier 的 transform
        from torchvision import transforms as T
        transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # ── 1. 从原图裁剪每个格子，应用神经网络相同的预处理 ──────────────────
        cells = []
        if src is not None and region is not None:
            rx, ry, rw, rh = region
            cell_w = rw / cols
            cell_h = rh / rows
            H, W = src.shape[:2]

            for r in range(rows):
                for c in range(cols):
                    grid_cell = self.grid_data["grid"][r][c]
                    if not grid_cell:
                        continue

                    # 从原图裁剪格子
                    px1 = int(rx + c * cell_w)
                    py1 = int(ry + r * cell_h)
                    px2 = int(rx + (c + 1) * cell_w)
                    py2 = int(ry + (r + 1) * cell_h)
                    px1_c = max(0, min(px1, W - 1))
                    py1_c = max(0, min(py1, H - 1))
                    px2_c = max(px1_c + 1, min(px2, W))
                    py2_c = max(py1_c + 1, min(py2, H))

                    tile = src[py1_c:py2_c, px1_c:px2_c]
                    if tile.size == 0:
                        continue

                    # 应用与神经网络完全相同的预处理
                    from PIL import Image
                    tile_img = Image.fromarray(tile.astype(np.uint8))
                    tensor = transform(tile_img)  # (3, 64, 64)
                    
                    # 用张量的 tobytes() 作为哈希键
                    tensor_hash = tensor.numpy().tobytes()
                    
                    # 保存原始 64x64 图像用于缩略图（反归一化）
                    tile_np = tensor.numpy()
                    tile_np = tile_np * np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1) + np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
                    tile_np = (tile_np.transpose(1, 2, 0) * 255).astype(np.uint8)  # (64, 64, 3)
                    tile_np = np.clip(tile_np, 0, 255)

                    cells.append({
                        "r": r, "c": c,
                        "tensor_hash": tensor_hash,
                        "tile_arr": tile_np,
                        "bead_id": grid_cell.get("bead_id", "空"),
                        "confidence": grid_cell.get("confidence", 0),
                    })
        else:
            # 降级：没有原图时只能用 rgb 字段
            for r in range(rows):
                for c in range(cols):
                    grid_cell = self.grid_data["grid"][r][c]
                    if not grid_cell:
                        continue
                    rgb_raw = grid_cell.get("rgb", [128, 128, 128])
                    rgb = tuple(int(v) for v in rgb_raw[:3])
                    cells.append({
                        "r": r, "c": c,
                        "tensor_hash": bytes(rgb),
                        "bead_id": grid_cell.get("bead_id", "空"),
                        "confidence": grid_cell.get("confidence", 0),
                    })

        if not cells:
            return []

        # ── 2. 按预处理后的张量哈希聚类 ───────────────────────────────────────
        from collections import defaultdict
        hash_groups = defaultdict(list)
        for cell in cells:
            hash_groups[cell["tensor_hash"]].append(cell)

        # 转换为聚类列表
        all_clusters = []
        for tensor_hash, group in hash_groups.items():
            if len(group) == 0:
                continue
            bead_id = group[0]["bead_id"]
            confs = [c["confidence"] for c in group]
            
            # 取代表性格子的像素用于缩略图
            if "tile_arr" in group[0]:
                sample_tile = group[0]["tile_arr"]
                # 平均色用于背景
                avg_rgb = tuple(int(v) for v in sample_tile.mean(axis=(0, 1)).astype(int))
            else:
                avg_rgb = (128, 128, 128)
                sample_tile = None
            
            cluster = {
                "bead_id": bead_id,
                "count": len(group),
                "avg_rgb": avg_rgb,
                "tensor_hash": tensor_hash,
                "positions": [(c["r"], c["c"]) for c in group],
                "avg_conf": round(sum(confs) / len(confs), 4) if confs else 0,
                "min_conf": round(min(confs), 4) if confs else 0,
                "max_conf": round(max(confs), 4) if confs else 0,
            }
            if sample_tile is not None:
                cluster["sample_tile"] = sample_tile
            all_clusters.append(cluster)

        # ── 3. 色号排序：按图例顺序 A01→M15 ──────────────────────────────────
        BEAD_ORDER = []
        for prefix in ["A", "B", "C", "D", "E", "F", "G", "H", "M"]:
            max_num = {"A": 26, "B": 32, "C": 29, "D": 26, "E": 24,
                       "F": 25, "G": 21, "H": 23, "M": 15}[prefix]
            for num in range(1, max_num + 1):
                BEAD_ORDER.append(f"{prefix}{num:02d}")

        bead_rank = {bid: i for i, bid in enumerate(BEAD_ORDER)}
        unknown_rank = len(BEAD_ORDER)

        def get_bead_rank(bid):
            return bead_rank.get(bid, unknown_rank)

        # ── 4. 排序规则：同色号按概率降序，不同色号按色号顺序 ──────────────────
        # 先按预测色号分组，每组内按 avg_conf 降序
        from collections import defaultdict
        bead_groups = defaultdict(list)
        for c in all_clusters:
            bead_groups[c["bead_id"]].append(c)
        
        # 每组内按 avg_conf 降序排列
        for bid in bead_groups:
            bead_groups[bid].sort(key=lambda x: -x["avg_conf"])
        
        # 定义色号顺序
        BEAD_ORDER = []
        for prefix in ["A", "B", "C", "D", "E", "F", "G", "H", "M"]:
            max_num = {"A": 26, "B": 32, "C": 29, "D": 26, "E": 24,
                       "F": 25, "G": 21, "H": 23, "M": 15}[prefix]
            for num in range(1, max_num + 1):
                BEAD_ORDER.append(f"{prefix}{num:02d}")
        bead_rank = {bid: i for i, bid in enumerate(BEAD_ORDER)}
        unknown_rank = len(BEAD_ORDER)
        
        # 按色号顺序拼接所有聚类（组内已按概率降序）
        def get_bead_rank(bid):
            return bead_rank.get(bid, unknown_rank)
        
        # "空"排在最前面，然后按色号顺序
        sorted_clusters = []
        if "空" in bead_groups:
            sorted_clusters.extend(bead_groups.pop("空"))
        for bid in sorted(bead_groups.keys(), key=get_bead_rank):
            sorted_clusters.extend(bead_groups[bid])
        
        return sorted_clusters

    # ── UI 初始化 ─────────────────────────────────────────────────────────────
    def _init_ui(self):
        self._is_fullscreen = False

        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # ── 左侧：聚类分页列表 ─────────────────────────────────────────────
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(6)

        # 标题行
        title_row = QHBoxLayout()
        lbl = QLabel("色块聚类列表")
        lbl.setStyleSheet("font-size:14px; font-weight:bold; color:#cba6f7;")
        title_row.addWidget(lbl)
        title_row.addStretch()

        # 总聚类数和页码
        self.total_lbl = QLabel(f"共 {len(self._clusters)} 个聚类")
        self.total_lbl.setStyleSheet("color:#89b4fa; font-size:11px;")
        title_row.addWidget(self.total_lbl)

        self.page_lbl = QLabel(f"第 {self._cur_page+1}/{self._total_pages} 页")
        self.page_lbl.setStyleSheet("color:#cdd6f4; font-size:12px; font-weight:bold; background:#313244; padding:2px 8px; border-radius:3px;")
        title_row.addWidget(self.page_lbl)
        left_layout.addLayout(title_row)

        # 聚类列表（QScrollArea + 自定义 widget）
        self.list_scroll = QScrollArea()
        self.list_scroll.setWidgetResizable(True)     # 让 inner widget 自动扩展高度
        self.list_scroll.setAlignment(Qt.AlignTop)
        self.list_scroll.setMinimumHeight(500)         # 确保有足够高度
        self.list_widget = QWidget()
        self.list_vbox   = QVBoxLayout(self.list_widget)
        self.list_vbox.setSpacing(4)
        self.list_vbox.setContentsMargins(2, 2, 2, 2)
        self.list_vbox.addStretch()                   # 防止空时塌陷
        self.list_scroll.setWidget(self.list_widget)
        left_layout.addWidget(self.list_scroll, stretch=1)

        # 分页按钮行
        page_btn_row = QHBoxLayout()
        page_btn_row.setSpacing(6)
        self.btn_prev = QPushButton("◀ 上一页")
        self.btn_prev.setObjectName("btn_sec")
        self.btn_prev.clicked.connect(self._on_prev_page)
        page_btn_row.addWidget(self.btn_prev)

        self.btn_next = QPushButton("下一页 ▶")
        self.btn_next.setObjectName("btn_sec")
        self.btn_next.clicked.connect(self._on_next_page)
        page_btn_row.addWidget(self.btn_next)
        left_layout.addLayout(page_btn_row)

        # 底部操作行
        op_row = QHBoxLayout()
        op_row.setSpacing(6)

        self.btn_replace = QPushButton("🔧 替换选中")
        self.btn_replace.setObjectName("btn_replace")
        self.btn_replace.setEnabled(False)
        self.btn_replace.clicked.connect(self._on_replace)
        op_row.addWidget(self.btn_replace)

        self.btn_undo = QPushButton("↩ 撤销")
        self.btn_undo.setObjectName("btn_undo")
        self.btn_undo.setEnabled(False)
        self.btn_undo.clicked.connect(self._on_undo)
        op_row.addWidget(self.btn_undo)

        self.btn_save = QPushButton("💾 保存")
        self.btn_save.setObjectName("btn_save")
        self.btn_save.clicked.connect(self._on_save)
        op_row.addWidget(self.btn_save)

        left_layout.addLayout(op_row)
        left_widget.setMinimumWidth(300)
        left_layout.setStretch(1, 1)
        main_layout.addWidget(left_widget)

        # ── 右侧：预览区域 ────────────────────────────────────────────────────
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(6)
        right_layout.setContentsMargins(4, 20, 4, 4)

        # 预留空间给聚类列表（不显示）
        spacer = QWidget()
        spacer.setMinimumHeight(200)
        right_layout.addWidget(spacer)

        # 查看原图 按钮
        self.btn_show_preview = QPushButton("📋 查看原图")
        self.btn_show_preview.setObjectName("btn_sec")
        self.btn_show_preview.setStyleSheet("""
            QPushButton {
                background: #313244; color: #cdd6f4;
                border: 2px solid #9c27b0; border-radius: 8px;
                padding: 10px; font-size: 13px; font-weight: bold;
            }
            QPushButton:hover { background: #45475a; border-color: #cba6f7; }
        """)
        self.btn_show_preview.clicked.connect(self._show_preview_dialog)
        right_layout.addWidget(self.btn_show_preview, alignment=Qt.AlignCenter)

        # 查看当前图纸 按钮
        self.btn_show_grid = QPushButton("📐 查看当前图纸")
        self.btn_show_grid.setObjectName("btn_sec")
        self.btn_show_grid.setStyleSheet("""
            QPushButton {
                background: #313244; color: #cdd6f4;
                border: 2px solid #89b4fa; border-radius: 8px;
                padding: 10px; font-size: 13px; font-weight: bold;
            }
            QPushButton:hover { background: #45475a; border-color: #cba6f7; }
        """)
        self.btn_show_grid.clicked.connect(self._show_grid_preview_dialog)
        right_layout.addWidget(self.btn_show_grid, alignment=Qt.AlignCenter)

        # 当前选中提示
        self.highlight_lbl = QLabel("未选择聚类")
        self.highlight_lbl.setStyleSheet(
            "color:#f38ba8; font-size:11px; padding:4px 8px; "
            "background:#2a1a1a; border-radius:4px;")
        right_layout.addWidget(self.highlight_lbl, alignment=Qt.AlignCenter)

        # 选中色号总数量统计（显示该色号在所有聚类中的总格子数）
        self.total_count_lbl = QLabel("")
        self.total_count_lbl.setStyleSheet(
            "color:#89b4fa; font-size:11px; padding:2px 8px; "
            "background:#1a1a2a; border-radius:4px;")
        right_layout.addWidget(self.total_count_lbl, alignment=Qt.AlignCenter)

        right_layout.addStretch()
        main_layout.addWidget(right_widget)

        # 安装事件过滤器（处理卡片 Ctrl+多选 + 原图预览点击高亮）
        self.list_scroll.viewport().installEventFilter(self)

    # ── 事件过滤器：卡片 Ctrl/Shift多选 + 橡皮筋框选 + 原图预览点击 ───────────────
    def eventFilter(self, obj, event):
        # ── 聚类列表视口：橡皮筋框选 ─────────────────────────────────────────
        if obj is self.list_scroll.viewport():
            etype = event.type()

            if etype == event.MouseButtonPress and event.button() == Qt.LeftButton:
                viewport_pos = self.list_scroll.viewport().mapFromGlobal(event.globalPos())
                widget_at = self.list_scroll.viewport().childAt(viewport_pos)

                # 检查是否点到了卡片
                w = widget_at
                while w and not (w.property("is_cluster_card") and w.property("cluster_idx") >= 0):
                    w = w.parent()

                if w:
                    # 点到卡片：普通点击处理
                    idx = w.property("cluster_idx")
                    mods = QApplication.keyboardModifiers()

                    if mods & Qt.ControlModifier:
                        # Ctrl+点击：toggle 选中
                        if idx in self._selected:
                            self._selected.discard(idx)
                        else:
                            self._selected.add(idx)
                        self._render_cluster_list()
                        self._update_replace_btn()
                        return True
                    elif mods & Qt.ShiftModifier:
                        # Shift+点击：框选范围（从锚点到当前点）
                        if self._selection_anchor is not None:
                            anchor = self._selection_anchor
                            start = min(anchor, idx)
                            end = max(anchor, idx)
                            self._selected.update(range(start, end + 1))
                        else:
                            self._selected.add(idx)
                            self._selection_anchor = idx
                        self._render_cluster_list()
                        self._update_replace_btn()
                        return True
                    else:
                        # 普通点击：单选 + 取消其他 + 设置锚点
                        self._selected.clear()
                        self._selection_anchor = idx
                        self._select_cluster(idx)
                        self._render_cluster_list()
                        self._update_replace_btn()
                        return True
                else:
                    # 没有点到卡片：开始橡皮筋框选
                    self._rubber_start = viewport_pos
                    self._rubber_drawing = True
                    self._rubber_rect = QRect(viewport_pos, viewport_pos)
                    self._ensure_rubber_band()
                    self._rubber_band.setGeometry(QRect(viewport_pos, viewport_pos).normalized())
                    self._rubber_band.show()
                    return True

            # ── 右键点击：选中该色号的所有格子 ──
            elif etype == event.MouseButtonPress and event.button() == Qt.RightButton:
                viewport_pos = self.list_scroll.viewport().mapFromGlobal(event.globalPos())
                widget_at = self.list_scroll.viewport().childAt(viewport_pos)

                w = widget_at
                while w and not (w.property("is_cluster_card") and w.property("cluster_idx") >= 0):
                    w = w.parent()

                if w:
                    idx = w.property("cluster_idx")
                    self._select_all_same_bead_id(idx)
                    return True

            elif etype == event.MouseMove and self._rubber_drawing:
                if self._rubber_start:
                    cur_pos = self.list_scroll.viewport().mapFromGlobal(event.globalPos())
                    self._rubber_rect = QRect(self._rubber_start, cur_pos).normalized()
                    self._rubber_band.setGeometry(self._rubber_rect)
                return True

            elif etype == event.MouseButtonRelease and event.button() == Qt.LeftButton:
                if self._rubber_drawing and self._rubber_rect:
                    self._rubber_band.hide()
                    self._rubber_drawing = False

                    # 计算框选范围内的所有卡片索引
                    sel_rect = self._rubber_rect
                    if sel_rect.width() > 5 or sel_rect.height() > 5:  # 过滤掉过小的误触
                        newly_selected = set()
                        # 遍历当前页所有卡片
                        start = self._cur_page * self._per_page
                        end = min(start + self._per_page, len(self._clusters))
                        for idx in range(start, end):
                            cluster = self._clusters[idx]
                            card_pos = self._find_card_geometry(idx)
                            if card_pos and sel_rect.intersects(card_pos):
                                newly_selected.add(idx)

                        if newly_selected:
                            self._selected.update(newly_selected)
                            # 如果没有高亮项，设置锚点为第一个新选中的
                            if self._highlight_idx is None:
                                self._selection_anchor = min(newly_selected)
                            self._render_cluster_list()
                            self._update_replace_btn()

                    self._rubber_rect = None
                    self._rubber_start = None
                return True

        # ── 原图预览点击：点击高亮对应聚类 ──
        if self._preview_dlg and self._preview_img_lbl and obj is self._preview_img_lbl:
            if event.type() == event.MouseButtonPress and event.button() == Qt.LeftButton:
                lbl = self._preview_img_lbl
                rx, ry, rw, rh = self._region
                H, W = self._src_arr.shape[:2]
                pixmap = lbl.pixmap()
                if pixmap and not pixmap.isNull():
                    pm_size = pixmap.size()
                    lbl_size = lbl.size()
                    src_x = int(event.pos().x() * pm_size.width() / lbl_size.width())
                    src_y = int(event.pos().y() * pm_size.height() / lbl_size.height())
                    img_x = int(rx + src_x / self._preview_zoom)
                    img_y = int(ry + src_y / self._preview_zoom)
                    cell_w = rw / self.grid_data["width"]
                    cell_h = rh / self.grid_data["height"]
                    c = int((img_x - rx) / cell_w)
                    r = int((img_y - ry) / cell_h)
                    if 0 <= r < self.grid_data["height"] and 0 <= c < self.grid_data["width"]:
                        cluster_idx = self._cell_to_cluster.get((r, c))
                        if cluster_idx is not None:
                            self._jump_to_cluster(cluster_idx)
                return True

        return super().eventFilter(obj, event)

    def _ensure_rubber_band(self):
        """确保橡皮筋选框 widget 存在（每次都重新创建确保可见）"""
        # 先删除旧的
        if self._rubber_band is not None:
            self._rubber_band.deleteLater()
            self._rubber_band = None
        # 创建新的（父对象为 viewport）
        self._rubber_band = QLabel(self.list_scroll.viewport())
        self._rubber_band.setStyleSheet("""
            background: rgba(156, 39, 176, 0.25);
            border: 2px solid #9c27b0;
        """)
        self._rubber_band.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._rubber_band.hide()

    def _find_card_geometry(self, global_idx):
        """查找指定索引卡片在视口中的几何位置"""
        # 遍历视口中的卡片 widget
        for w in self.list_scroll.viewport().findChildren(QWidget):
            if w.property("is_cluster_card") and w.property("cluster_idx") == global_idx:
                # 转换为视口坐标系
                rect = QRect(QPoint(0, 0), w.size())
                pos = w.mapToParent(QPoint(0, 0))
                # 加上滚动偏移
                scroll_y = self.list_scroll.verticalScrollBar().value()
                rect.moveTopLeft(QPoint(pos.x(), pos.y() + scroll_y))
                return rect
        return None

    def _select_all_same_bead_id(self, clicked_idx):
        """选中所有与指定聚类相同色号的所有聚类（包括跨页的）"""
        if clicked_idx < 0 or clicked_idx >= len(self._clusters):
            return

        clicked_cluster = self._clusters[clicked_idx]
        target_bead_id = clicked_cluster["bead_id"]

        # 找出所有相同色号的聚类索引
        same_bead_clusters = []
        for idx, cluster in enumerate(self._clusters):
            if cluster["bead_id"] == target_bead_id:
                same_bead_clusters.append(idx)

        if not same_bead_clusters:
            return

        # 更新选中状态
        self._selected.clear()
        self._selected.update(same_bead_clusters)

        # 跳转到第一页并高亮
        first_idx = same_bead_clusters[0]
        self._selection_anchor = first_idx

        # 计算应该跳转到哪一页
        page = first_idx // self._per_page
        if page != self._cur_page:
            self._cur_page = page
            self._render_cluster_list()

        # 更新高亮显示
        self._highlight_idx = first_idx
        self._render_cluster_list()
        self._update_replace_btn()

        # 统计总格子数
        total_count = sum(self._clusters[i]["count"] for i in same_bead_clusters)

        # 更新右侧信息（使用黑色文字确保可见性）
        r, g, b = clicked_cluster["avg_rgb"]
        same_count = len(same_bead_clusters)
        self.highlight_lbl.setText(
            f"<b>{target_bead_id}</b>  共{same_count}组 {total_count}格  "
            f"RGB({r},{g},{b})  置信 {clicked_cluster['avg_conf']:.0%}")
        self.highlight_lbl.setStyleSheet(
            "color:#000000; font-size:12px; padding:4px 8px; "
            "background:rgb(255,255,255); border:2px solid #9c27b0; border-radius:4px;")

        # 更新总数量标签
        self.total_count_lbl.setText(f"「{target_bead_id}」在图纸中共 {total_count} 格（跨 {same_count} 组）")
        self.total_count_lbl.setVisible(True)

        self.btn_replace.setEnabled(True)

        # 刷新预览图
        if self._preview_dlg and self._preview_img_lbl:
            show_grid = getattr(self, '_preview_show_grid', False)
            pixmap = self._render_preview_img(show_grid=show_grid)
            self._preview_img_lbl.setPixmap(pixmap)

    def keyPressEvent(self, event):
        """F11 全屏切换"""
        if event.key() == Qt.Key_F11:
            self._toggle_fullscreen()
        super().keyPressEvent(event)

    def resizeEvent(self, event):
        """窗口大小变化时重新渲染聚类列表（动态适应新尺寸）"""
        super().resizeEvent(event)
        # 延迟到下一帧执行，确保 viewport 尺寸已更新
        QApplication.processEvents()
        if hasattr(self, '_clusters') and self._clusters:
            self._render_cluster_list()

    def _toggle_fullscreen(self):
        """全屏/退出全屏（使用Qt原生方法避免警告）"""
        if self._is_fullscreen:
            # 退出全屏
            self.showNormal()
            self._is_fullscreen = False
        else:
            # 进入全屏
            self.showFullScreen()
            self._is_fullscreen = True

    def _jump_to_cluster(self, global_idx):
        """跳转到指定聚类：选中 + 翻到对应页 + 高亮 + 刷新预览"""
        page = global_idx // self._per_page
        if page != self._cur_page:
            self._cur_page = page
            self._render_cluster_list()
        self._highlight_idx = global_idx
        self._selected.clear()
        self._render_cluster_list()

        # 同步刷新预览弹窗
        if self._preview_dlg and self._preview_img_lbl:
            show_grid = getattr(self, '_preview_show_grid', False)
            pixmap = self._render_preview_img(show_grid=show_grid)
            self._preview_img_lbl.setPixmap(pixmap)

        cluster = self._clusters[global_idx]
        r, g, b = cluster["avg_rgb"]
        self.highlight_lbl.setText(
            f"<b>{cluster['bead_id']}</b>  {cluster['count']}格  "
            f"RGB({r},{g},{b})  置信 {cluster['avg_conf']:.0%}")
        # 使用黑色文字确保可见性
        self.highlight_lbl.setStyleSheet(
            "color:#000000; font-size:12px; padding:4px 8px; "
            "background:rgb(255,255,255); border:2px solid #9c27b0; border-radius:4px;")

        # 更新总数量标签
        target_bead_id = cluster["bead_id"]
        total_count = sum(c["count"] for c in self._clusters if c["bead_id"] == target_bead_id)
        same_color_groups = sum(1 for c in self._clusters if c["bead_id"] == target_bead_id)
        if same_color_groups > 1:
            self.total_count_lbl.setText(f"「{target_bead_id}」在图纸中共 {total_count} 格（跨 {same_color_groups} 组）")
        else:
            self.total_count_lbl.setText(f"「{target_bead_id}」在图纸中共 {total_count} 格")
        self.total_count_lbl.setVisible(True)

        self.btn_replace.setEnabled(True)

    # ── 预览弹窗 ─────────────────────────────────────────────────────────────
    def _show_preview_dialog(self):
        """显示原图框选区域的预览弹窗"""
        from PyQt5.QtWidgets import QLabel, QVBoxLayout, QHBoxLayout, QPushButton

        dlg = QDialog(self)
        dlg.setWindowTitle("原图预览")
        dlg.setMinimumSize(900, 700)
        dlg.setStyleSheet("""
            QDialog { background: #1e1e2e; }
            ZoomScrollArea { background: #181825; border: none; }
        """)

        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(8, 8, 8, 8)

        # 工具栏
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        self._preview_zoom = 1.0
        zoom_label = QLabel("100%")
        zoom_label.setStyleSheet("color:#89b4fa; font-size:12px;")
        toolbar.addWidget(zoom_label)

        btn_zoom_in = QPushButton("🔍+ 放大")
        btn_zoom_in.setObjectName("btn_sec")
        btn_zoom_in.clicked.connect(lambda: self._preview_zoom_helper(dlg, zoom_label, 1.2))
        toolbar.addWidget(btn_zoom_in)

        btn_zoom_out = QPushButton("🔍- 缩小")
        btn_zoom_out.setObjectName("btn_sec")
        btn_zoom_out.clicked.connect(lambda: self._preview_zoom_helper(dlg, zoom_label, 0.8))
        toolbar.addWidget(btn_zoom_out)

        btn_reset = QPushButton("↺ 原始")
        btn_reset.setObjectName("btn_sec")
        btn_reset.clicked.connect(lambda: self._preview_zoom_helper(dlg, zoom_label, None, reset=True))
        toolbar.addWidget(btn_reset)

        # 添加滚轮提示
        wheel_hint = QLabel("滚轮:滚动  Ctrl+滚轮:缩放  Shift+滚轮:横滚")
        wheel_hint.setStyleSheet("color:#6c7086; font-size:11px;")
        toolbar.addWidget(wheel_hint)

        toolbar.addStretch()
        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(dlg.close)
        toolbar.addWidget(btn_close)
        layout.addLayout(toolbar)

        # 预览区域（使用 ZoomScrollArea）
        scroll = ZoomScrollArea()
        scroll.setWidgetResizable(False)
        scroll.setAlignment(Qt.AlignCenter)
        # 设置滚轮缩放回调
        scroll.set_zoom_callback(lambda f: self._preview_zoom_helper(dlg, zoom_label, f))

        # 用 _render_preview_img() 生成预览图
        pixmap = self._render_preview_img()
        preview_lbl = QLabel()
        preview_lbl.setPixmap(pixmap)
        preview_lbl.setAlignment(Qt.AlignCenter)
        preview_lbl.setCursor(QCursor(Qt.PointingHandCursor))
        preview_lbl.setMouseTracking(True)
        # 给 label 安装事件过滤器（点击高亮）
        preview_lbl.installEventFilter(self)
        scroll.setWidget(preview_lbl)
        layout.addWidget(scroll)

        dlg._scroll = scroll
        dlg._lbl = preview_lbl
        dlg._zoom_label = zoom_label

        # 非模态弹窗：show() + 存储引用防止被GC
        self._preview_dlg = dlg
        self._preview_img_lbl = preview_lbl
        self._preview_show_grid = False  # 标记当前是原图模式
        dlg.setAttribute(Qt.WA_DeleteOnClose, False)
        dlg.show()

    def _show_grid_preview_dialog(self):
        """显示当前图纸预览弹窗（grid状态）"""
        from PyQt5.QtWidgets import QLabel, QVBoxLayout, QHBoxLayout, QPushButton

        dlg = QDialog(self)
        dlg.setWindowTitle("当前图纸预览")
        dlg.setMinimumSize(900, 700)
        dlg.setStyleSheet("""
            QDialog { background: #1e1e2e; }
            ZoomScrollArea { background: #181825; border: none; }
        """)

        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(8, 8, 8, 8)

        # 工具栏
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        self._preview_zoom = 1.0
        zoom_label = QLabel("100%")
        zoom_label.setStyleSheet("color:#89b4fa; font-size:12px;")
        toolbar.addWidget(zoom_label)

        btn_zoom_in = QPushButton("🔍+ 放大")
        btn_zoom_in.setObjectName("btn_sec")
        btn_zoom_in.clicked.connect(lambda: self._preview_zoom_helper(dlg, zoom_label, 1.2))
        toolbar.addWidget(btn_zoom_in)

        btn_zoom_out = QPushButton("🔍- 缩小")
        btn_zoom_out.setObjectName("btn_sec")
        btn_zoom_out.clicked.connect(lambda: self._preview_zoom_helper(dlg, zoom_label, 0.8))
        toolbar.addWidget(btn_zoom_out)

        btn_reset = QPushButton("↺ 原始")
        btn_reset.setObjectName("btn_sec")
        btn_reset.clicked.connect(lambda: self._preview_zoom_helper(dlg, zoom_label, None, reset=True))
        toolbar.addWidget(btn_reset)

        # 添加滚轮提示
        wheel_hint = QLabel("滚轮:滚动  Ctrl+滚轮:缩放  Shift+滚轮:横滚")
        wheel_hint.setStyleSheet("color:#6c7086; font-size:11px;")
        toolbar.addWidget(wheel_hint)

        toolbar.addStretch()
        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(dlg.close)
        toolbar.addWidget(btn_close)
        layout.addLayout(toolbar)

        # 预览区域（使用 ZoomScrollArea）
        scroll = ZoomScrollArea()
        scroll.setWidgetResizable(False)
        scroll.setAlignment(Qt.AlignCenter)
        # 设置滚轮缩放回调
        scroll.set_zoom_callback(lambda f: self._preview_zoom_helper(dlg, zoom_label, f))

        pixmap = self._render_preview_img(show_grid=True)
        preview_lbl = QLabel()
        preview_lbl.setPixmap(pixmap)
        preview_lbl.setAlignment(Qt.AlignCenter)
        scroll.setWidget(preview_lbl)
        layout.addWidget(scroll)

        dlg._scroll = scroll
        dlg._lbl = preview_lbl
        dlg._zoom_label = zoom_label

        self._preview_dlg = dlg
        self._preview_img_lbl = preview_lbl
        self._preview_show_grid = True  # 标记当前是图纸模式
        dlg.setAttribute(Qt.WA_DeleteOnClose, False)
        dlg.show()

    def _preview_zoom_helper(self, dlg, zoom_label, factor=None, reset=False):
        """预览弹窗缩放"""
        if reset:
            self._preview_zoom = 1.0
        elif factor:
            self._preview_zoom = max(0.1, min(10, self._preview_zoom * factor))

        zoom_label.setText(f"{self._preview_zoom * 100:.0f}%")
        show_grid = getattr(self, '_preview_show_grid', False)
        pixmap = self._render_preview_img(show_grid=show_grid)
        dlg._lbl.setPixmap(pixmap)

    def _render_preview_img(self, show_grid=False):
        """
        渲染预览图。
        show_grid=False: 原图预览，支持聚类高亮/暗化
        show_grid=True:  当前图纸预览（grid状态）
        """
        from PIL import Image as _PIL_Image
        src = self._src_arr
        region = self._region
        if src is None or region is None:
            return QPixmap()

        rows = self.grid_data["height"]
        cols = self.grid_data["width"]

        # ── show_grid=True: 渲染当前图纸 ───────────────────────────────────
        if show_grid:
            grid_2d = [
                [cell.get("bead_id", "空") if cell else "空" for cell in row]
                for row in self.grid_data["grid"]
            ]
            cell_size = int(12 * self._preview_zoom)
            cell_size = max(4, min(cell_size, 60))
            pil_grid = render_grid_to_pil(grid_2d, self.beads, cell=cell_size)
            if pil_grid is None:
                return QPixmap()

            # 高亮叠加
            overlay = _PIL_Image.new("RGBA", pil_grid.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            actual_cell_w = pil_grid.width / cols
            actual_cell_h = pil_grid.height / rows

            # 收集所有高亮位置
            hl_positions = set()
            sel_positions = set()

            hl_idx = self._highlight_idx
            if hl_idx is not None and 0 <= hl_idx < len(self._clusters):
                hl_positions = set(self._clusters[hl_idx]["positions"])

            for sel_idx in self._selected:
                if 0 <= sel_idx < len(self._clusters):
                    sel_positions.update(self._clusters[sel_idx]["positions"])

            for r in range(rows):
                for c in range(cols):
                    px1 = int(c * actual_cell_w)
                    py1 = int(r * actual_cell_h)
                    px2 = int((c + 1) * actual_cell_w)
                    py2 = int((r + 1) * actual_cell_h)
                    is_hl = (r, c) in hl_positions
                    is_sel = (r, c) in sel_positions
                    if hl_idx is not None or self._selected:
                        if is_hl:
                            draw.rectangle([px1, py1, px2 - 1, py2 - 1],
                                           outline=(255, 255, 0, 255), width=max(1, cell_size // 10))
                        elif is_sel:
                            draw.rectangle([px1, py1, px2 - 1, py2 - 1],
                                           outline=(249, 226, 175, 200), width=max(1, cell_size // 12))
                        else:
                            draw.rectangle([px1, py1, px2 - 1, py2 - 1],
                                           fill=(0, 0, 0, 80))
                    else:
                        draw.rectangle([px1, py1, px2 - 1, py2 - 1],
                                       outline=(60, 60, 80, 60), width=1)

            pil_grid = pil_grid.convert("RGBA")
            pil_grid = _PIL_Image.alpha_composite(pil_grid, overlay).convert("RGB")
            qimg = QImage(pil_grid.tobytes(), pil_grid.width, pil_grid.height,
                          pil_grid.width * 3, QImage.Format_RGB888)
            return QPixmap.fromImage(qimg)

        # ── show_grid=False: 渲染原图 + 高亮/暗化 ──────────────────────────
        rx, ry, rw, rh = region
        H, W = src.shape[:2]

        # 裁剪框选区域
        rx_c = max(0, min(int(rx), W - 1))
        ry_c = max(0, min(int(ry), H - 1))
        rx2_c = max(rx_c + 1, min(int(rx + rw), W))
        ry2_c = max(ry_c + 1, min(int(ry + rh), H))

        crop = src[ry_c:ry2_c, rx_c:rx2_c]
        if crop.size == 0:
            return QPixmap()

        pil_img = _PIL_Image.fromarray(crop.astype(np.uint8))

        # 计算缩放
        max_w = int(800 * self._preview_zoom)
        max_h = int(600 * self._preview_zoom)
        w, h = pil_img.size
        scale = min(max_w / w, max_h / h, 1.0) if self._preview_zoom < 1 else self._preview_zoom
        new_w = int(w * scale)
        new_h = int(h * scale)
        pil_img = pil_img.resize((new_w, new_h), _PIL_Image.Resampling.LANCZOS)

        # 高亮/暗化叠加层
        cell_w = new_w / cols
        cell_h = new_h / rows

        overlay = _PIL_Image.new("RGBA", (new_w, new_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # 收集所有高亮位置：_highlight_idx + _selected 中的聚类
        hl_positions = set()
        sel_positions = set()

        hl_idx = self._highlight_idx
        if hl_idx is not None and 0 <= hl_idx < len(self._clusters):
            hl_positions = set(self._clusters[hl_idx]["positions"])

        # 添加所有选中的聚类位置
        for sel_idx in self._selected:
            if 0 <= sel_idx < len(self._clusters):
                sel_positions.update(self._clusters[sel_idx]["positions"])

        all_hl_positions = hl_positions | sel_positions

        for r in range(rows):
            for c in range(cols):
                px1 = int(c * cell_w)
                py1 = int(r * cell_h)
                px2 = int((c + 1) * cell_w)
                py2 = int((r + 1) * cell_h)

                is_hl = (r, c) in hl_positions
                is_sel = (r, c) in sel_positions
                if hl_idx is not None or self._selected:
                    if is_hl:
                        # 主要高亮：紫色边框
                        draw.rectangle([px1, py1, px2 - 1, py2 - 1],
                                       outline=(156, 39, 176, 220), width=2)
                    elif is_sel:
                        # 选中高亮：金色边框
                        draw.rectangle([px1, py1, px2 - 1, py2 - 1],
                                       outline=(249, 226, 175, 200), width=2)
                    else:
                        # 未选中：暗化
                        draw.rectangle([px1, py1, px2 - 1, py2 - 1],
                                       fill=(0, 0, 0, 100))
                else:
                    draw.rectangle([px1, py1, px2 - 1, py2 - 1],
                                   outline=(100, 100, 120, 60), width=1)

        pil_img = pil_img.convert("RGBA")
        pil_img = _PIL_Image.alpha_composite(pil_img, overlay).convert("RGB")
        qimg = QImage(pil_img.tobytes(), new_w, new_h, new_w * 3, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)

    # ── 分页渲染（网格布局，动态适应窗口大小）───────────────────────────────────
    def _render_cluster_list(self):
        """重新渲染当前页的聚类列表。

        策略：
        - 根据视口宽度 动态计算每行放几个卡片（min_card_w=50px）
        - 根据视口高度 动态计算每页放几行（卡片高度=card_w+间距）
        - 末行空位 用空占位 QLabel 填满整行
        """
        total = len(self._clusters)
        start = self._cur_page * self._per_page
        end   = min(start + self._per_page, total)

        # ── 动态计算网格参数 ──────────────────────────────────────────────
        scroll_w = self.list_scroll.viewport().width()
        scroll_h = self.list_scroll.viewport().height()
        if scroll_w <= 0:
            scroll_w = 500
        if scroll_h <= 0:
            scroll_h = 400

        min_card_w = 80                        # 每个卡片最小宽度（增大避免太小）
        card_spacing = 6
        header_h = 0                           # 网格内部边距（已在ContentsMargins处理）
        cols_per_row = max(1, scroll_w // min_card_w)

        # 卡片实际宽度 = scroll_w 均分 - 间距
        card_w = scroll_w // cols_per_row - card_spacing
        card_w = max(60, card_w)               # 最小60px
        # 估算卡片高度：方形主体 + 底部单行文字高度
        # 缩略图是 card_w-4 的正方形，底部文字行约 18px
        card_h = (card_w - 4) + 18             # 缩略图+单行文字标签
        rows_per_page = max(1, scroll_h // (card_h + card_spacing))
        # 确保每页至少有2行
        rows_per_page = max(2, rows_per_page)
        self._per_page = cols_per_row * rows_per_page
        self._total_pages = max(1, (total + self._per_page - 1) // self._per_page)

        # 确保当前页在有效范围内
        if self._cur_page >= self._total_pages:
            self._cur_page = max(0, self._total_pages - 1)
        start = self._cur_page * self._per_page
        end   = min(start + self._per_page, total)

        # ── 构建网格容器 widget ───────────────────────────────────────────
        # 方案：用 QGridLayout 直接管理卡片，不混用 setColumnStretch
        grid_widget = QWidget()
        grid_layout = QGridLayout(grid_widget)
        grid_layout.setSpacing(card_spacing)
        grid_layout.setContentsMargins(2, 2, 2, 2)

        # 添加当前页所有卡片
        for i, idx in enumerate(range(start, end)):
            cluster = self._clusters[idx]
            card = self._build_cluster_card(cluster, idx, compact=True, card_w=card_w)
            row = i // cols_per_row
            col = i % cols_per_row
            grid_layout.addWidget(card, row, col)

        # 替换旧的 list_widget
        if self.list_widget.layout() is not None:
            # 断开旧布局
            old_layout = self.list_widget.layout()
            while old_layout.count():
                child = old_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
            self.list_widget.deleteLater()

        self.list_widget = grid_widget
        self.list_scroll.setWidget(grid_widget)

        self.page_lbl.setText(f"第 {self._cur_page+1}/{self._total_pages} 页")
        self.total_lbl.setText(f"共 {total} 个 · 每页 {self._per_page} 个")
        self.btn_prev.setEnabled(self._cur_page > 0)
        self.btn_next.setEnabled(self._cur_page < self._total_pages - 1)
        self._update_replace_btn()

    def _update_replace_btn(self):
        """根据当前选中状态更新替换按钮文字和启用状态"""
        sel_count = len(self._selected)
        hl_count = 1 if self._highlight_idx is not None else 0
        total = sel_count + hl_count
        if total > 0:
            self.btn_replace.setEnabled(True)
            self.btn_replace.setText(f"🔧 替换选中 ({total})")
        else:
            self.btn_replace.setEnabled(False)
            self.btn_replace.setText("🔧 替换选中")

        # 同步刷新预览图（如果有打开的话）
        if self._preview_dlg and self._preview_img_lbl:
            show_grid = getattr(self, '_preview_show_grid', False)
            pixmap = self._render_preview_img(show_grid=show_grid)
            self._preview_img_lbl.setPixmap(pixmap)

    def _build_cluster_card(self, cluster, global_idx, compact=False, card_w=None):
        """
        构建单个聚类卡片 widget。
        compact=True: 紧凑网格模式（色块缩略图+外侧色号/概率）
        compact=False: 完整列表模式（缩略图+色号+置信度+数量）
        """
        avg_rgb  = cluster["avg_rgb"]
        r, g, b  = avg_rgb
        count    = cluster["count"]
        avg_conf = cluster["avg_conf"]
        bead_id  = cluster["bead_id"]

        # 判断是否为当前选中聚类
        is_hl = (global_idx == self._highlight_idx)

        if is_hl:
            border = "#9c27b0"
            border_w = 3
        else:
            border = "#45475a"
            border_w = 1

        base_r, base_g, base_b = r, g, b

        if compact:
            # ── 紧凑网格模式：色号+概率显示在正下方外侧 ──
            cw = card_w if card_w else 70
            # 外层容器（白色背景，包含卡片+下方文字）
            container = QWidget()
            container.setObjectName("cluster_card")
            container.setStyleSheet("""
                QWidget#cluster_card {
                    background: #ffffff;
                    border: 1px solid #313244;
                    border-radius: 4px;
                }
            """)
            container.setCursor(QCursor(Qt.PointingHandCursor))

            # 外层布局：垂直排列
            outer_layout = QVBoxLayout(container)
            outer_layout.setContentsMargins(1, 1, 1, 1)
            outer_layout.setSpacing(2)
            outer_layout.setAlignment(Qt.AlignHCenter)

            # 内层色块卡片（只显示缩略图）
            card = QWidget()
            card.setObjectName("cluster_card_inner")
            card.setStyleSheet(f"""
                QWidget#cluster_card_inner {{
                    background: rgb({base_r},{base_g},{base_b});
                    border: {border_w}px solid {border};
                    border-radius: 4px;
                }}
                QWidget#cluster_card_inner:hover {{ border: 2px solid #9c27b0; }}
            """)
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(2, 2, 2, 2)
            card_layout.setSpacing(0)
            card_layout.setAlignment(Qt.AlignHCenter)

            # 缩略图
            thumb_lbl = QLabel()
            thumb_size = max(20, cw - 6)
            thumb_lbl.setFixedSize(thumb_size, thumb_size)
            thumb_lbl.setAlignment(Qt.AlignCenter)
            thumb_pm = self._render_thumbnail(cluster)
            if thumb_pm:
                thumb_lbl.setPixmap(thumb_pm.scaled(thumb_size, thumb_size, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            card_layout.addWidget(thumb_lbl)

            outer_layout.addWidget(card)

            # 色号+置信度文字（显示在正下方外侧）
            font_size = max(9, min(12, cw // 7))
            info_row = QHBoxLayout()
            info_row.setSpacing(1)
            info_row.setContentsMargins(0, 0, 0, 0)

            # 色号
            bead_lbl = QLabel(f"<b>{bead_id}</b>")
            bead_lbl.setStyleSheet(
                f"color:#1e1e2e; font-size:{font_size}px; font-weight:bold; background:transparent;")
            bead_lbl.setAlignment(Qt.AlignCenter)
            bead_lbl.setTextInteractionFlags(Qt.NoTextInteraction)
            info_row.addWidget(bead_lbl, 1)

            # 分隔符
            sep = QLabel("|")
            sep.setStyleSheet(f"color:#6c7086; font-size:{font_size}px; background:transparent;")
            sep.setAlignment(Qt.AlignCenter)
            info_row.addWidget(sep)

            # 置信度
            conf_lbl = QLabel(f"{avg_conf:.0%}")
            conf_lbl.setStyleSheet(
                f"color:#9c27b0; font-size:{font_size}px; background:transparent;")
            conf_lbl.setAlignment(Qt.AlignCenter)
            conf_lbl.setTextInteractionFlags(Qt.NoTextInteraction)
            info_row.addWidget(conf_lbl, 1)

            outer_layout.addLayout(info_row)

            # 动态设置尺寸
            total_h = cw + 18  # 卡片高度 + 文字行高度
            container.setMinimumSize(cw, total_h)
            container.setMaximumSize(cw, total_h)

            # 给容器附加全局索引
            container.setProperty("is_cluster_card", True)
            container.setProperty("cluster_idx", global_idx)

            # 选中视觉
            self._update_card_selection_style(container, global_idx, compact, inner_card=card)

            return container
        else:
            # ── 完整列表模式 ──
            card.setStyleSheet(f"""
                QWidget#cluster_card {{
                    background: rgb({base_r},{base_g},{base_b});
                    border: 2px solid {border};
                    border-radius: 6px;
                    padding: 6px;
                }}
                QWidget#cluster_card:hover {{ border: 2px solid #9c27b0; }}
            """)
            card.setCursor(QCursor(Qt.PointingHandCursor))

            layout = QHBoxLayout(card)
            layout.setContentsMargins(8, 4, 8, 4)
            layout.setSpacing(10)

            # 缩略图
            thumb_lbl = QLabel()
            thumb_lbl.setFixedSize(60, 60)
            thumb_lbl.setStyleSheet(
                f"background: rgb({base_r},{base_g},{base_b}); "
                "border: 1px solid #313244; border-radius: 4px;")
            thumb_lbl.setAlignment(Qt.AlignCenter)
            thumb_pm = self._render_thumbnail(cluster)
            if thumb_pm:
                thumb_lbl.setPixmap(thumb_pm)
            layout.addWidget(thumb_lbl)

            # 信息列
            info_col = QVBoxLayout()
            info_col.setSpacing(2)

            bead_lbl = QLabel(f"<b>{bead_id}</b>")
            bead_lbl.setStyleSheet(
                f"color:{text_color}; font-size:16px; font-weight:bold; background:transparent;")
            info_col.addWidget(bead_lbl)

            conf_lbl = QLabel(f"置信 {avg_conf:.0%}")
            conf_lbl.setStyleSheet(
                f"color:{text_color}; font-size:11px; background:transparent;")
            info_col.addWidget(conf_lbl)

            count_lbl = QLabel(f"<b>{count}</b> 格")
            count_lbl.setStyleSheet(
                f"color:{text_color}; font-size:12px; background:transparent;")
            info_col.addWidget(count_lbl)

            layout.addLayout(info_col, 1)
            card.setMinimumHeight(64)

        # 给卡片附加全局索引（供 eventFilter 通过 property 读取）
        card.setProperty("is_cluster_card", True)
        card.setProperty("cluster_idx", global_idx)

        # 选中视觉：compact 模式下叠加勾号
        self._update_card_selection_style(card, global_idx, compact)

        return card

    def _update_card_selection_style(self, card, global_idx, compact, inner_card=None):
        """刷新单个卡片的选中/高亮视觉"""
        cluster = self._clusters[global_idx]
        avg_rgb = cluster["avg_rgb"]
        br, bg, bb = avg_rgb
        is_hl = (global_idx == self._highlight_idx)
        is_sel = (global_idx in self._selected)

        if is_hl:
            border = "#9c27b0"
            border_w = 3
        elif is_sel:
            border = "#f9e2af"   # 金色选中边框
            border_w = 2
        else:
            border = "#45475a"
            border_w = 1

        if compact:
            # compact 模式：容器有白色背景，内层卡片有颜色
            container_border = border if is_hl or is_sel else "#313244"
            container_border_w = border_w if is_hl or is_sel else 1
            card.setStyleSheet(f"""
                QWidget#cluster_card {{
                    background: #ffffff;
                    border: {container_border_w}px solid {container_border};
                    border-radius: 4px;
                }}
            """)
            # 更新内层卡片边框
            if inner_card:
                inner_card.setStyleSheet(f"""
                    QWidget#cluster_card_inner {{
                        background: rgb({br},{bg},{bb});
                        border: {border_w}px solid {border};
                        border-radius: 4px;
                    }}
                    QWidget#cluster_card_inner:hover {{ border: 2px solid #9c27b0; }}
                """)

            # compact 模式下叠加勾号（若被选中）- 添加到内层卡片上
            if is_sel and inner_card:
                inner_layout = inner_card.layout()
                if inner_layout:
                    # 移除旧勾号
                    for i in range(inner_layout.count()):
                        w = inner_layout.itemAt(i).widget()
                        if w and w.objectName() == "sel_chk":
                            w.deleteLater()
                            break
                    chk = QLabel("✓")
                    chk.setObjectName("sel_chk")
                    chk.setStyleSheet(
                        "color:#f9e2af; font-size:16px; font-weight:bold; background:none;")
                    chk.setAlignment(Qt.AlignCenter)
                    chk.setFixedSize(20, 20)
                    inner_layout.addWidget(chk)
        else:
            # 非 compact 模式（完整列表）
            card.setStyleSheet(f"""
                QWidget#cluster_card {{
                    background: rgb({br},{bg},{bb});
                    border: {border_w}px solid {border};
                    border-radius: 4px;
                }}
                QWidget#cluster_card:hover {{ border: 2px solid #9c27b0; }}
            """)

            # 非 compact 模式下叠加勾号
            layout = card.layout()
            if layout:
                # 移除旧勾号（名为 "sel_chk" 的 widget）
                for i in range(layout.count()):
                    w = layout.itemAt(i).widget()
                    if w and w.objectName() == "sel_chk":
                        w.deleteLater()
                        break
                if is_sel:
                    chk = QLabel("✓")
                    chk.setObjectName("sel_chk")
                    chk.setStyleSheet(
                        "color:#f9e2af; font-size:16px; font-weight:bold; background:none;")
                    chk.setAlignment(Qt.AlignCenter)
                    chk.setFixedSize(20, 20)
                    layout.addWidget(chk)

    def _render_thumbnail(self, cluster):
        """
        使用聚类中保存的 sample_tile 渲染缩略图。
        sample_tile 已经是 32x32 的标准化像素矩阵，保证逐像素完全一致。
        """
        try:
            tile_size = 60
            avg_rgb = cluster["avg_rgb"]
            bg_r, bg_g, bg_b = avg_rgb

            # 如果聚类中有 sample_tile，直接使用
            if "sample_tile" in cluster:
                tile_arr = cluster["sample_tile"]
                tile_img = Image.fromarray(tile_arr.astype(np.uint8)).resize(
                    (tile_size, tile_size), Image.Resampling.NEAREST)
                qimg = QImage(tile_img.tobytes(), tile_size, tile_size, 
                              tile_size * 3, QImage.Format_RGB888)
                return QPixmap.fromImage(qimg)
            
            # 降级：从原图裁剪
            src = self._src_arr
            region = self._region
            if src is None or region is None:
                return self._fallback_thumbnail(cluster)

            rx, ry, rw, rh = region
            rows = self.grid_data["height"]
            cols = self.grid_data["width"]
            positions = cluster["positions"]
            cell_w = rw / cols
            cell_h = rh / rows

            H, W = src.shape[:2]
            gr, gc = positions[0]
            px1 = int(rx + gc * cell_w)
            py1 = int(ry + gr * cell_h)
            px2 = int(rx + (gc + 1) * cell_w)
            py2 = int(ry + (gr + 1) * cell_h)

            px1_c = max(0, min(px1, W - 1))
            py1_c = max(0, min(py1, H - 1))
            px2_c = max(px1_c + 1, min(px2, W))
            py2_c = max(py1_c + 1, min(py2, H))

            tile = src[py1_c:py2_c, px1_c:px2_c]
            if tile.size == 0:
                return self._fallback_thumbnail(cluster)

            tile_img = Image.fromarray(tile.astype(np.uint8)).resize(
                (tile_size, tile_size), Image.Resampling.LANCZOS)
            qimg = QImage(tile_img.tobytes(), tile_size, tile_size, 
                          tile_size * 3, QImage.Format_RGB888)
            return QPixmap.fromImage(qimg)

        except Exception:
            return self._fallback_thumbnail(cluster)

    def _fallback_thumbnail(self, cluster):
        """降级显示：PIL 渲染带色号文字的纯色缩略图"""
        try:
            avg_rgb = cluster["avg_rgb"]
            r, g, b = avg_rgb

            SIZE = 60
            img = Image.new("RGB", (SIZE, SIZE), (r, g, b))
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 9)
            except Exception:
                font = ImageFont.load_default()

            bead_id = cluster["bead_id"]
            brightness = (r * 299 + g * 587 + b * 114) / 1000
            tc = (255, 255, 255) if brightness < 128 else (0, 0, 0)
            bbox = draw.textbbox((0, 0), bead_id, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text(((SIZE - tw) / 2, (SIZE - th) / 2), bead_id, fill=tc, font=font)
            qimg = QImage(img.tobytes(), SIZE, SIZE, SIZE * 3, QImage.Format_RGB888)
            return QPixmap.fromImage(qimg)
        except Exception:
            return None

    # ── 选择聚类 ─────────────────────────────────────────────────────────────
    def _select_cluster(self, global_idx):
        """选中某个聚类：高亮它 + 低亮其他所有聚类 + 刷新预览弹窗"""
        if global_idx < 0 or global_idx >= len(self._clusters):
            return
        self._highlight_idx = global_idx
        cluster = self._clusters[global_idx]

        # 刷新列表卡片样式
        self._render_cluster_list()

        # 同步刷新预览弹窗（如果已打开）
        if self._preview_dlg and self._preview_img_lbl:
            show_grid = getattr(self, '_preview_show_grid', False)
            pixmap = self._render_preview_img(show_grid=show_grid)
            self._preview_img_lbl.setPixmap(pixmap)

        # 更新高亮标签
        r, g, b = cluster["avg_rgb"]
        self.highlight_lbl.setText(
            f"<b>{cluster['bead_id']}</b>  {cluster['count']}格  "
            f"RGB({r},{g},{b})  置信 {cluster['avg_conf']:.0%}")
        # 使用黑色文字确保可见性
        self.highlight_lbl.setStyleSheet(
            "color:#000000; font-size:12px; padding:4px 8px; "
            "background:rgb(255,255,255); border:2px solid #9c27b0; border-radius:4px;")

        # 更新总数量标签（统计所有同色号的聚类总格子数）
        target_bead_id = cluster["bead_id"]
        total_count = sum(c["count"] for c in self._clusters if c["bead_id"] == target_bead_id)
        same_color_groups = sum(1 for c in self._clusters if c["bead_id"] == target_bead_id)
        if same_color_groups > 1:
            self.total_count_lbl.setText(f"「{target_bead_id}」在图纸中共 {total_count} 格（跨 {same_color_groups} 组）")
        else:
            self.total_count_lbl.setText(f"「{target_bead_id}」在图纸中共 {total_count} 格")
        self.total_count_lbl.setVisible(True)

        self.btn_replace.setEnabled(True)

    def _on_prev_page(self):
        if self._cur_page > 0:
            self._cur_page -= 1
            self._render_cluster_list()
            self._cur_page  # 确保刷新后选中第一项
            first_global = self._cur_page * self._per_page
            if first_global < len(self._clusters):
                self._select_cluster(first_global)

    def _on_next_page(self):
        if self._cur_page < self._total_pages - 1:
            self._cur_page += 1
            self._render_cluster_list()
            first_global = self._cur_page * self._per_page
            if first_global < len(self._clusters):
                self._select_cluster(first_global)

    # ── 替换 ────────────────────────────────────────────────────────────────
    def _on_replace(self):
        """用色号板替换当前选中的聚类（支持批量）"""
        # 收集要替换的聚类列表（highlight + selected）
        targets = []
        if self._highlight_idx is not None:
            targets.append(self._highlight_idx)
        targets.extend(self._selected)

        if not targets:
            return

        # 去重
        targets = sorted(set(targets))
        clusters = [self._clusters[i] for i in targets]

        # 弹出色号板选色对话框
        dlg = QDialog(self)
        if len(clusters) == 1:
            c = clusters[0]
            dlg.setWindowTitle(f"替换 {c['bead_id']} → 选择新色号")
        else:
            dlg.setWindowTitle(f"批量替换 {len(clusters)} 个聚类 → 选择新色号")
        dlg.setMinimumSize(600, 500)
        dlg.setStyleSheet(self.styleSheet())

        layout = QVBoxLayout(dlg)

        # 标题
        if len(clusters) == 1:
            avg = clusters[0]["avg_rgb"]
            r, g, b = avg
            count_text = "1 个格子" if clusters[0]["count"] == 1 else f"{clusters[0]['count']} 个格子"
            tip = QLabel(
                f"<b>将 {count_text} 从 "
                f"<font style='background:rgb({r},{g},{b}); padding:2px 8px; "
                f"border-radius:4px;'>{clusters[0]['bead_id']}</font> "
                f"替换为：</b>"
            )
        else:
            tip = QLabel(f"<b>将 {len(clusters)} 个聚类全部替换为：</b>")
        tip.setStyleSheet("color:#cdd6f4; font-size:13px; padding:4px;")
        layout.addWidget(tip)

        # 色号网格（参考色号板）
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        grid_widget = QWidget()
        grid_layout = QGridLayout(grid_widget)
        grid_layout.setSpacing(4)

        # 排序列出
        all_codes = sorted(self.beads.keys(), key=lambda x: (
            x == '空', x[0] if x != '空' else '',  # 先按字母
            int(x[1:]) if x[1:].isdigit() else 0   # 再按数字
        ))
        # 把空排第一位
        all_codes = ['空'] + [c for c in all_codes if c != '空']

        cols_per_row = 10
        for idx, code in enumerate(all_codes):
            row_i, col_i = divmod(idx, cols_per_row)
            rgb = self.beads.get(code, (200, 200, 200))
            if not isinstance(rgb, (tuple, list)) or len(rgb) != 3:
                rgb = (200, 200, 200)

            btn = QPushButton(code)
            brightness = (rgb[0]*299 + rgb[1]*587 + rgb[2]*114) / 1000
            tc = "white" if brightness < 128 else "black"
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: rgb({rgb[0]},{rgb[1]},{rgb[2]});
                    color: {tc};
                    border: 1px solid #313244;
                    border-radius: 4px;
                    padding: 6px 4px;
                    font-size: 11px;
                    font-weight: bold;
                    min-width: 44px;
                    max-width: 56px;
                }}
                QPushButton:hover {{
                    border: 2px solid #9c27b0;
                }}
            """)
            btn.setFixedHeight(40)
            btn.clicked.connect(lambda _=None, _code=code: self._do_replace_batch(clusters, _code, dlg))
            grid_layout.addWidget(btn, row_i, col_i)

        scroll.setWidget(grid_widget)
        layout.addWidget(scroll, 1)

        # 取消按钮
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_cancel = QPushButton("取消")
        btn_cancel.setObjectName("btn_sec")
        btn_cancel.clicked.connect(dlg.reject)
        btn_row.addWidget(btn_cancel)
        layout.addLayout(btn_row)

        dlg.exec_()

    def _do_replace_batch(self, clusters, to_code, dlg):
        """批量执行替换"""
        # 检查是否所有聚类都是同一色号（才允许替换）
        unique_from = set(c["bead_id"] for c in clusters)
        if len(unique_from) == 1 and list(unique_from)[0] == to_code:
            dlg.reject()
            return

        # 保存撤销快照
        self._undo_stack.append(copy.deepcopy(self.grid_data))
        self.btn_undo.setEnabled(True)

        # 执行替换
        to_rgb = self.beads.get(to_code, (200, 200, 200))
        if not isinstance(to_rgb, (tuple, list)) or len(to_rgb) != 3:
            to_rgb = (200, 200, 200)

        for cluster in clusters:
            for (r, c) in cluster["positions"]:
                cell = self.grid_data["grid"][r][c]
                if cell:
                    cell["bead_id"]    = to_code
                    cell["rgb"]        = list(to_rgb)
                    cell["confidence"] = 1.0  # 替换后置信度置1

        dlg.accept()

        # 立即刷新主窗口预览图
        main_win = self.parent()
        if main_win and hasattr(main_win, '_refresh_preview'):
            main_win._refresh_preview()

        self.grid_changed.emit()

        # 重建聚类
        self._clusters = self._build_clusters()
        self._total_pages = max(1, (len(self._clusters) + self._per_page - 1) // self._per_page)
        if self._cur_page >= self._total_pages:
            self._cur_page = max(0, self._total_pages - 1)
        self._highlight_idx = None
        self._selected.clear()
        self._render_cluster_list()
        self.highlight_lbl.setText(f"已替换 {len(clusters)} 个聚类，等待选择")
        self.highlight_lbl.setStyleSheet(
            "color:#a6e3a1; font-size:12px; padding:4px 8px; "
            "background:#1a2a1a; border-radius:4px;")
        self.btn_replace.setEnabled(False)

        self.grid_changed.emit()

    # ── 撤销 ────────────────────────────────────────────────────────────────
    def _on_undo(self):
        if not self._undo_stack:
            return
        self.grid_data = self._undo_stack.pop()
        self.btn_undo.setEnabled(len(self._undo_stack) > 0)

        self._clusters = self._build_clusters()
        self._total_pages = max(1, (len(self._clusters) + self._per_page - 1) // self._per_page)
        if self._cur_page >= self._total_pages:
            self._cur_page = max(0, self._total_pages - 1)
        self._highlight_idx = None
        self._render_cluster_list()
        self.highlight_lbl.setText("已撤销，等待选择新聚类")
        self.highlight_lbl.setStyleSheet(
            "color:#f9e2af; font-size:12px; padding:4px 8px; "
            "background:#2a2a1a; border-radius:4px;")
        self.btn_replace.setEnabled(False)

        self.grid_changed.emit()

    # ── 保存 ────────────────────────────────────────────────────────────────
    def _on_save(self):
        """保存当前 grid_data 到 output 路径"""
        out_path = self._out_path
        if not out_path:
            QMessageBox.warning(self, "提示", "未设置输出路径，无法保存。")
            return

        try:
            save_grid_json(self.grid_data, out_path)

            # 同时保存预览图 PNG（与 JSON 同目录同名）
            preview_path = out_path.replace("_grid.json", "_grid_preview.png")
            grid_2d = [
                [cell.get("bead_id", "空") if cell else "空" for cell in row]
                for row in self.grid_data["grid"]
            ]
            rows_n = self.grid_data["height"]
            cols_n = self.grid_data["width"]
            cell_size = max(36, min(56, 800 // max(cols_n, rows_n)))
            preview_pil = render_grid_to_pil(grid_2d, self.beads, cell=cell_size)
            preview_pil.save(preview_path)

            # 立即刷新主窗口预览图（dialog 还没关闭）
            main_win = self.parent()
            if main_win and hasattr(main_win, '_refresh_preview'):
                main_win._refresh_preview()
            self.grid_changed.emit()
            QMessageBox.information(
                self, "已保存",
                f"图纸已保存至：\n{out_path}\n{preview_path}"
            )
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存出错：\n{e}")


# ─────────────────────────────────────────────────────────────────────────────
#  7.1 支持滚轮缩放的 ScrollArea（用于预览弹窗）
# ─────────────────────────────────────────────────────────────────────────────
class ZoomScrollArea(QScrollArea):
    """
    支持滚轮交互的 ScrollArea：
    - Ctrl+滚轮：缩放（调用 zoom_in/zoom_out 回调）
    - 纯滚轮：上下滚动
    - Shift+滚轮：左右滚动
    """
    wheel_zoomed = pyqtSignal(float)  # 传递缩放因子

    def __init__(self, parent=None):
        super().__init__(parent)
        self._zoom_callback = None

    def set_zoom_callback(self, callback):
        """设置缩放回调函数，接受缩放因子参数"""
        self._zoom_callback = callback

    def wheelEvent(self, event):
        modifiers = QApplication.keyboardModifiers()
        
        if modifiers & Qt.ControlModifier:
            # Ctrl+滚轮：缩放
            if self._zoom_callback:
                delta = event.angleDelta().y()
                factor = 1.1 if delta > 0 else 0.9
                self._zoom_callback(factor)
            event.accept()
        else:
            # 纯滚轮或Shift+滚轮：交给父类处理（正常滚动）
            super().wheelEvent(event)


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

        self.btn_warehouse     = QPushButton("🧺  我的仓库")
        self.btn_warehouse.setObjectName("btn_secondary")

        self.btn_confirm_sel.setEnabled(False)
        self.btn_input_size.setEnabled(False)
        self.btn_generate.setEnabled(False)

        for btn in [self.btn_open, self.btn_confirm_sel,
                     self.btn_input_size, self.btn_generate, self.btn_reset]:
            left_layout.addWidget(btn)

        # 我的仓库按钮单独一行，放在底部
        left_layout.addWidget(self.btn_warehouse)

        self.btn_open.clicked.connect(self._on_open_image)
        self.btn_confirm_sel.clicked.connect(self._on_confirm_selection)
        self.btn_input_size.clicked.connect(self._on_input_grid_size)
        self.btn_generate.clicked.connect(self._on_generate)
        self.btn_reset.clicked.connect(self._on_reset)
        self.btn_warehouse.clicked.connect(self._on_open_warehouse)

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
                img_path=self._img_path,
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

    # ── 打开我的仓库 ──
    def _on_open_warehouse(self):
        """打开我的仓库弹窗"""
        dlg = BeadWarehouseDialog(self, beads=self._beads)
        dlg.exec_()

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


# ─────────────────────────────────────────────────────────────────────────────
#  8. 我的仓库 - 拼豆库存管理
# ─────────────────────────────────────────────────────────────────────────────

class BeadWarehouseDialog(QDialog):
    """
    我的仓库：拼豆库存管理弹窗
    - 管理221种色号的库存数量（初始1000）
    - 支持选择图纸加入/挪出"已出库"列表
    - 已出库的图纸色号自动做减法
    - 数据持久化到JSON文件
    """

    # 221个色号列表
    FULL_BEAD_IDS = []
    for prefix, count in [("A", 26), ("B", 32), ("C", 29), ("D", 26),
                          ("E", 24), ("F", 25), ("G", 21), ("H", 23), ("M", 15)]:
        for i in range(1, count + 1):
            FULL_BEAD_IDS.append(f"{prefix}{i:02d}")

    def __init__(self, parent=None, beads=None):
        super().__init__(parent)
        self.beads = beads or {}
        self.setWindowTitle("🧺 我的仓库 - 拼豆库存管理")
        self.setMinimumSize(1000, 700)
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
            QPushButton#btn_danger {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #e74c3c,stop:1 #c0392b);
                color: white; border: none; border-radius: 6px;
                padding: 6px 14px; font-size: 12px;
            }
            QPushButton#btn_danger:hover { background: #c0392b; }
            QScrollArea { background: #181825; border: 1px solid #313244; }
            QListWidget { background: #181825; color: #cdd6f4; border: 1px solid #313244; }
            QListWidget::item { padding: 4px; }
            QListWidget::item:selected { background: #7c3aed; }
        """)

        # 加载数据
        self._data_file = os.path.join(_get_base_dir(), "warehouse_data.json")
        self._load_data()

        self._init_ui()

    def _load_data(self):
        """从JSON文件加载数据"""
        if os.path.exists(self._data_file):
            try:
                with open(self._data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self._stock = data.get('stock', {})
                self._shipped = data.get('shipped', [])  # 已出库图纸列表
            except:
                self._stock = {}
                self._shipped = []
        else:
            self._stock = {}
            self._shipped = []

        # 初始化所有色号为1000
        for bid in self.FULL_BEAD_IDS:
            if bid not in self._stock:
                self._stock[bid] = 1000

    def _save_data(self):
        """保存数据到JSON文件"""
        data = {
            'stock': self._stock,
            'shipped': self._shipped
        }
        try:
            with open(self._data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            QMessageBox.warning(self, "保存失败", f"无法保存数据：\n{str(e)}")

    def _recalc_stock(self):
        """根据已出库图纸重新计算库存"""
        # 重置所有库存为1000
        self._stock = {bid: 1000 for bid in self.FULL_BEAD_IDS}

        # 遍历已出库图纸，减去消耗
        for json_path in self._shipped:
            if json_path and os.path.isfile(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        grid_data = json.load(f)
                    # 统计该图纸各色号数量
                    for row in grid_data.get('grid', []):
                        for cell in row:
                            if cell:
                                bid = cell.get('bead_id', '')
                                if bid and bid in self._stock:
                                    self._stock[bid] -= 1
                except Exception as e:
                    print(f"警告：无法读取 {json_path}: {e}")

        # 确保库存不为负数
        for bid in self._stock:
            self._stock[bid] = max(0, self._stock[bid])

    def _find_grid_json(self, file_path):
        """
        检查文件路径是否为有效的grid.json文件。
        现在传入的就是JSON文件路径，直接验证即可。
        """
        if file_path and os.path.isfile(file_path):
            return file_path
        return None

    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(12, 12, 12, 12)

        # ── 左侧：色号库存面板 ──────────────────────────────────────────────
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(8)

        # 标题和操作行
        title_row = QHBoxLayout()
        lbl = QLabel("🧺 色号库存")
        lbl.setStyleSheet("font-size:16px; font-weight:bold; color:#cba6f7;")
        title_row.addWidget(lbl)
        title_row.addStretch()

        # 统计标签
        self.total_lbl = QLabel()
        self.total_lbl.setStyleSheet("color:#89b4fa; font-size:11px;")
        title_row.addWidget(self.total_lbl)
        left_layout.addLayout(title_row)

        # 搜索框
        self.search_box = QTextEdit()
        self.search_box.setPlaceholderText("搜索色号...")
        self.search_box.setMaximumHeight(30)
        self.search_box.textChanged.connect(self._on_search)
        left_layout.addWidget(self.search_box)

        # 色号网格（可滚动）
        self.color_scroll = QScrollArea()
        self.color_scroll.setWidgetResizable(True)
        self.color_widget = QWidget()
        self.color_grid = QGridLayout(self.color_widget)
        self.color_grid.setSpacing(4)
        self.color_scroll.setWidget(self.color_widget)
        left_layout.addWidget(self.color_scroll)

        # 重建色号网格
        self._build_color_grid()

        # 刷新按钮
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        btn_refresh = QPushButton("🔄 刷新计算")
        btn_refresh.clicked.connect(self._on_refresh)
        btn_row.addWidget(btn_refresh)

        btn_reset = QPushButton("🔁 重置为1000")
        btn_reset.setObjectName("btn_sec")
        btn_reset.clicked.connect(self._on_reset_stock)
        btn_row.addWidget(btn_reset)
        left_layout.addLayout(btn_row)

        left_widget.setMinimumWidth(600)
        layout.addWidget(left_widget, stretch=2)

        # ── 右侧：已出库图纸列表 ──────────────────────────────────────────
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(8)

        # 标题
        lbl2 = QLabel("📋 已出库图纸")
        lbl2.setStyleSheet("font-size:14px; font-weight:bold; color:#cba6f7;")
        right_layout.addWidget(lbl2)

        count_lbl = QLabel(f"共 {len(self._shipped)} 个图纸")
        count_lbl.setStyleSheet("color:#89b4fa; font-size:11px;")
        right_layout.addWidget(count_lbl)

        # 图纸列表
        self.shipped_list = QListWidget()
        for path in self._shipped:
            name = os.path.basename(path)
            self.shipped_list.addItem(name)
        right_layout.addWidget(self.shipped_list)

        # 操作按钮
        op_row = QVBoxLayout()
        op_row.setSpacing(6)

        btn_add = QPushButton("➕ 添加图纸到已出库")
        btn_add.setObjectName("btn_sec")
        btn_add.clicked.connect(self._on_add_shipped)
        op_row.addWidget(btn_add)

        btn_remove = QPushButton("➖ 移除选中图纸")
        btn_remove.setObjectName("btn_danger")
        btn_remove.clicked.connect(self._on_remove_shipped)
        op_row.addWidget(btn_remove)

        btn_clear = QPushButton("🗑 清空已出库")
        btn_clear.setObjectName("btn_danger")
        btn_clear.clicked.connect(self._on_clear_shipped)
        op_row.addWidget(btn_clear)

        right_layout.addLayout(op_row)

        # 关闭按钮
        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.accept)
        right_layout.addWidget(btn_close)

        right_widget.setMinimumWidth(280)
        layout.addWidget(right_widget)

    def _build_color_grid(self):
        """构建色号网格"""
        # 清除旧内容
        while self.color_grid.count():
            child = self.color_grid.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        search_text = self.search_box.toPlainText().strip().lower()
        cols_per_row = 8

        row, col = 0, 0
        for bid in self.FULL_BEAD_IDS:
            # 搜索过滤
            if search_text and search_text not in bid.lower():
                continue

            rgb = self.beads.get(bid, (200, 200, 200))
            if not isinstance(rgb, (tuple, list)) or len(rgb) != 3:
                rgb = (200, 200, 200)

            count = self._stock.get(bid, 1000)

            # 卡片
            card = QWidget()
            card.setObjectName("color_card")

            # 库存颜色：充足=绿色，警告=黄色，短缺=红色
            if count > 100:
                stock_color = "#2ecc71"  # 绿色
            elif count > 20:
                stock_color = "#f39c12"  # 黄色
            else:
                stock_color = "#e74c3c"  # 红色

            card.setStyleSheet(f"""
                QWidget#color_card {{
                    background: rgb({rgb[0]},{rgb[1]},{rgb[2]});
                    border: 1px solid #45475a;
                    border-radius: 4px;
                    padding: 4px;
                }}
            """)

            vbox = QVBoxLayout(card)
            vbox.setContentsMargins(2, 2, 2, 2)
            vbox.setSpacing(0)

            # 色号
            brightness = (rgb[0] * 299 + rgb[1] * 587 + rgb[2] * 114) / 1000
            text_color = "white" if brightness < 128 else "black"

            code_lbl = QLabel(f"<b>{bid}</b>")
            code_lbl.setStyleSheet(f"color:{text_color}; font-size:9px; background:transparent;")
            code_lbl.setAlignment(Qt.AlignCenter)
            vbox.addWidget(code_lbl)

            # 数量（可点击修改）
            count_lbl = QLabel(f"{count}")
            count_lbl.setStyleSheet(
                f"color:{text_color}; font-size:10px; font-weight:bold; "
                f"background:transparent;")
            count_lbl.setAlignment(Qt.AlignCenter)
            count_lbl.setCursor(QCursor(Qt.PointingHandCursor))
            count_lbl.mousePressEvent = lambda _, b=bid: self._on_set_stock(b)
            vbox.addWidget(count_lbl)

            # ±按钮行（支持快速加减）
            btn_row = QHBoxLayout()
            btn_row.setSpacing(2)

            btn_minus = QPushButton("-")
            btn_minus.setFixedSize(20, 18)
            btn_minus.clicked.connect(lambda _, b=bid, d=-1: self._on_adjust_stock(b, d))
            btn_row.addWidget(btn_minus)

            btn_plus = QPushButton("+")
            btn_plus.setFixedSize(20, 18)
            btn_plus.clicked.connect(lambda _, b=bid, d=1: self._on_adjust_stock(b, d))
            btn_row.addWidget(btn_plus)

            vbox.addLayout(btn_row)

            self.color_grid.addWidget(card, row, col)
            col += 1
            if col >= cols_per_row:
                col = 0
                row += 1

        # 更新统计
        total = sum(self._stock.values())
        low_stock = sum(1 for c in self._stock.values() if c < 100)
        self.total_lbl.setText(f"总库存: {total:,} | 低库存: {low_stock} 种")

    def _on_search(self):
        """搜索色号"""
        self._build_color_grid()

    def _on_adjust_stock(self, bid, delta):
        """调整单个色号库存（支持正负增量）"""
        self._stock[bid] = max(0, self._stock.get(bid, 1000) + delta)
        self._save_data()
        self._build_color_grid()

    def _on_set_stock(self, bid):
        """
        弹出对话框让用户自定义设置库存数量。
        支持直接输入新数量，或输入加减幅度（如 +50、-30）。
        """
        current = self._stock.get(bid, 1000)

        # 创建自定义输入对话框
        dlg = QDialog(self)
        dlg.setWindowTitle(f"设置 {bid} 库存数量")
        dlg.setMinimumWidth(320)
        dlg.setStyleSheet(self.styleSheet())

        layout = QVBoxLayout(dlg)
        layout.setSpacing(12)

        # 当前库存提示
        tip = QLabel(f"当前库存：<b>{current}</b>")
        tip.setStyleSheet("color:#89b4fa; font-size:13px; padding:4px;")
        layout.addWidget(tip)

        # 输入说明
        hint = QLabel("输入新数量，或输入加减幅度（如 +50、-30）")
        hint.setStyleSheet("color:#cdd6f4; font-size:11px; padding:4px;")
        layout.addWidget(hint)

        # 输入框
        input_box = QTextEdit()
        input_box.setPlaceholderText(f"{current}")
        input_box.setMaximumHeight(40)
        input_box.setFont(QFont("Consolas", 12))
        layout.addWidget(input_box)

        # 快速调整按钮
        quick_row = QHBoxLayout()
        quick_row.setSpacing(6)
        for delta in [-100, -50, -10, +10, +50, +100]:
            btn = QPushButton(f"{delta:+d}")
            btn.setObjectName("btn_sec")
            btn.clicked.connect(lambda _, d=delta, i=input_box: i.setPlainText(str(d)))
            quick_row.addWidget(btn)
        layout.addLayout(quick_row)

        # 确定/取消按钮
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        btn_ok = QPushButton("确定")
        btn_ok.setObjectName("btn_save")
        btn_ok.clicked.connect(dlg.accept)
        btn_row.addWidget(btn_ok)

        btn_cancel = QPushButton("取消")
        btn_cancel.setObjectName("btn_sec")
        btn_cancel.clicked.connect(dlg.reject)
        btn_row.addWidget(btn_cancel)

        layout.addLayout(btn_row)

        # 显示对话框
        if dlg.exec_() == QDialog.Accepted:
            text = input_box.toPlainText().strip()
            if not text:
                return  # 空输入，不做任何修改

            try:
                # 判断是绝对数量还是增量
                if text.startswith(('+', '-')) and len(text) > 1:
                    # 增量模式
                    delta_val = int(text)
                    new_val = max(0, current + delta_val)
                else:
                    # 绝对数量模式
                    new_val = max(0, int(text))

                self._stock[bid] = new_val
                self._save_data()
                self._build_color_grid()
            except ValueError:
                QMessageBox.warning(self, "输入错误", f"无法解析输入：{text}\n请输入整数（如 500、+50、-30）")

    def _on_refresh(self):
        """重新计算库存"""
        self._recalc_stock()
        self._save_data()
        self._build_color_grid()
        QMessageBox.information(self, "刷新完成", "库存已根据已出库图纸重新计算！")

    def _on_reset_stock(self):
        """重置所有库存为1000"""
        reply = QMessageBox.question(self, "确认重置",
            "确定要将所有色号库存重置为1000吗？\n这将清空已出库图纸的消耗记录！",
            QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self._stock = {bid: 1000 for bid in self.FULL_BEAD_IDS}
            self._shipped = []
            self._save_data()
            self._build_color_grid()
            self.shipped_list.clear()
            QMessageBox.information(self, "重置完成", "所有库存已重置为1000！")

    def _on_add_shipped(self):
        """添加图纸到已出库列表（严格选择JSON文件）"""
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择图纸文件（JSON）", "",
            "图纸文件 (*_grid.json);;JSON文件 (*.json);;所有文件 (*)"
        )
        if not files:
            return

        added = 0
        for f in files:
            if f not in self._shipped:
                self._shipped.append(f)
                added += 1
                name = os.path.basename(f)
                self.shipped_list.addItem(name)

        if added > 0:
            # 重新计算库存
            self._recalc_stock()
            self._save_data()
            self._build_color_grid()
            QMessageBox.information(self, "添加完成", f"已添加 {added} 个图纸到已出库列表！")
        else:
            QMessageBox.information(self, "提示", "所选图纸已在列表中。")

    def _on_remove_shipped(self):
        """移除选中的已出库图纸"""
        selected = self.shipped_list.currentRow()
        if selected < 0:
            QMessageBox.information(self, "提示", "请先选择要移除的图纸。")
            return

        name = self.shipped_list.currentItem().text()
        reply = QMessageBox.question(self, "确认移除",
            f"确定要从已出库列表中移除「{name}」吗？\n移除后将恢复其库存。",
            QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            # 找到完整路径
            removed_path = None
            for p in self._shipped:
                if os.path.basename(p) == name:
                    removed_path = p
                    break

            if removed_path:
                self._shipped.remove(removed_path)

            self.shipped_list.takeItem(selected)
            self._recalc_stock()
            self._save_data()
            self._build_color_grid()

    def _on_clear_shipped(self):
        """清空已出库列表"""
        if not self._shipped:
            QMessageBox.information(self, "提示", "列表已经是空的。")
            return

        reply = QMessageBox.question(self, "确认清空",
            "确定要清空所有已出库图纸吗？\n所有库存将恢复到1000。",
            QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self._shipped = []
            self._stock = {bid: 1000 for bid in self.FULL_BEAD_IDS}
            self._save_data()
            self.shipped_list.clear()
            self._build_color_grid()
            QMessageBox.information(self, "清空完成", "已出库列表已清空，库存已恢复！")


if __name__ == "__main__":
    main()
