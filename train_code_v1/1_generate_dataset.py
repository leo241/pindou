# -*- coding: utf-8 -*-
"""
1_generate_dataset.py
─────────────────────────────────────────────────────────────────────────
合成生成拼豆格子训练数据集。

设计思路
────────
不再从真实图纸裁切（同一色号永远长得一样），而是按以下方式为每个色号
生成大量多样化的合成图片：

普通色号（221个：A01~M15）每种生成 SAMPLES_PER_CLASS 张（默认800张）：
  ┌─────────────────────────────────────────────────────────┐
  │  背景：目标色号的 RGB ± 极小扰动（±8，保持本色）          │
  │  前景：色号文字，字体/大小/颜色完全随机                  │
  │         · 22种字体随机切换                               │
  │         · 文字颜色：黑35%/白20%/对比25%/深浅灰17%/随机3% │
  │  干扰：10%样本带少量其他色方块                            │
  │  后处理：模糊分级                                        │
  │         · 清晰（60%）：sigma 0~0.3                       │
  │         · 轻微模糊（25%）：sigma 0.3~1.0                │
  │         · 中等模糊（10%）：sigma 1.0~2.0                │
  │         · 极模糊（ 5%）：sigma 2.0~4.5（极少数退化）    │
  │         + 高斯噪声 / 轻微旋转 / JPEG压缩                  │
  └─────────────────────────────────────────────────────────┘

"空"类（1个）生成 SAMPLES_PER_EMPTY 张：
  纯白格、随机色格、棋盘格、随机噪声、写非色号文字、
  写了色号但极模糊、有色背景、渐变色 等各式各样的"空"样本。

数据集结构
──────────
grid_category/train_dataset/
  A01/  0.jpg  1.jpg ... (SAMPLES_PER_CLASS 张)
  A02/
  ...
  M15/
  空/
  split/
    train.txt   ← "bead_id/index.jpg\t类别索引"
    val.txt
    test.txt
  label_map.txt  ← "类别索引\t色号名"

用法
────
  python 1_generate_dataset.py                    # 全默认参数
  python 1_generate_dataset.py --overwrite        # 强制重新生成
  python 1_generate_dataset.py --per-class 300     # 每类300张
  python 1_generate_dataset.py --dry-run         # 仅预览，不写文件
"""

import os
import sys
import json
import random
import argparse
import shutil
import io

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as CFG


# ──────────────────────────────────────────────────────────────────────────
#  0. 工具函数
# ──────────────────────────────────────────────────────────────────────────

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def clamp_rgb(val):
    return max(0, min(255, int(val)))


def contrast_color(rgb):
    brightness = (rgb[0] * 299 + rgb[1] * 587 + rgb[2] * 114) / 1000
    return (0, 0, 0) if brightness >= 128 else (255, 255, 255)


def get_random_font(font_size_px, rng, cfg=None):
    """随机从字体列表中选取一个可用字体（每次都随机选择，增加多样性）"""
    cfg = cfg or CFG.GEN_CONFIG
    fonts = cfg.get("fonts", [])
    # 打乱顺序后尝试加载第一个成功的
    shuffled = list(fonts)
    rng.shuffle(shuffled)
    for font_path in shuffled:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, font_size_px)
            except Exception:
                pass
    return ImageFont.load_default()


def get_text_color(bg_color, rng, cfg=None):
    """
    多样化文字颜色，模拟真实图纸中各异的字色：
    - 黑25% / 白15% / 对比色15% / 灰5% / 随机40%
    """
    r = rng.random()
    if r < 0.25:
        return (0, 0, 0)                                    # 黑色
    elif r < 0.40:
        return (255, 255, 255)                              # 白色
    elif r < 0.55:
        # 对比色（亮度反转）
        brightness = (bg_color[0] * 299 + bg_color[1] * 587 + bg_color[2] * 114) / 1000
        return (255, 255, 255) if brightness < 128 else (0, 0, 0)
    elif r < 0.60:
        # 深浅灰
        gray = rng.integers(30, 230)
        return (gray, gray, gray)
    else:
        # 纯随机彩色
        return (rng.integers(0, 256), rng.integers(0, 256), rng.integers(0, 256))


def sample_blur_sigma(rng, cfg=None):
    """根据模糊分级分布随机选取 sigma 值"""
    cfg = cfg or CFG.GEN_CONFIG
    tiers = cfg.get("blur_tiers", {})
    tier_keys = list(tiers.keys())
    tier_probs = [tiers[k]["prob"] for k in tier_keys]
    tier_probs = [p / sum(tier_probs) for p in tier_probs]
    chosen_tier = rng.choice(tier_keys, p=tier_probs)
    s_min, s_max = tiers[chosen_tier]["sigma_range"]
    return rng.uniform(s_min, s_max)


def apply_post_process(img, rng, cfg=None):
    """后处理：旋转 + 模糊 + JPEG压缩"""
    cfg = cfg or CFG.GEN_CONFIG
    rot_deg = rng.uniform(-cfg.get("rotation_degrees", 5), cfg.get("rotation_degrees", 5))
    if abs(rot_deg) > 0.5:
        img = img.rotate(rot_deg, fillcolor=(255, 255, 255), resample=Image.BILINEAR)

    blur_sigma = sample_blur_sigma(rng, cfg)
    if blur_sigma > 0.05:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_sigma))

    q_min, q_max = cfg.get("jpeg_quality_range", [75, 100])
    if rng.random() > 0.3:
        quality = int(rng.integers(q_min, q_max))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    return img


def add_gaussian_noise(img, rng, std=8):
    arr = np.array(img, dtype=np.float32)
    noise = rng.standard_normal(size=arr.shape) * std
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def add_interference(img, rng, noise_range=8, interference_ratio=0.20):
    size = CFG.CELL_IMAGE_SIZE
    draw = ImageDraw.Draw(img)
    n_blocks = rng.integers(1, 4)
    for _ in range(n_blocks):
        bw = rng.integers(3, max(4, size // 6))
        bx = rng.integers(0, size - bw)
        by = rng.integers(0, size - bw)
        base = img.getpixel((size // 2, size // 2))
        ir = clamp_rgb(base[0] + rng.integers(-noise_range, noise_range))
        ig = clamp_rgb(base[1] + rng.integers(-noise_range, noise_range))
        ib = clamp_rgb(base[2] + rng.integers(-noise_range, noise_range))
        draw.rectangle([bx, by, bx + bw, by + bw], fill=(ir, ig, ib))


def add_grid_lines(img, rng, line_color=(180, 180, 180), line_width=1):
    size = CFG.CELL_IMAGE_SIZE
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, size - 1, size - 1], outline=line_color, width=line_width)


# ──────────────────────────────────────────────────────────────────────────
#  1. 普通色号生成器
# ──────────────────────────────────────────────────────────────────────────

class NormalBeadGenerator:
    def __init__(self, bead_id, rgb, rng, cfg=None):
        self.bead_id = bead_id
        self.rgb = rgb
        self.rng = rng
        self.cfg = cfg or CFG.GEN_CONFIG
        self.size = CFG.CELL_IMAGE_SIZE

    def generate_one(self):
        rng = self.rng
        size = self.size
        rgb = self.rgb
        cfg = self.cfg

        # ── 1. 背景色 ± 极小扰动（保持本色） ──────────────────────────
        noise_range = cfg.get("rgb_noise_range", 4)
        r = clamp_rgb(rgb[0] + rng.integers(-noise_range, noise_range))
        g = clamp_rgb(rgb[1] + rng.integers(-noise_range, noise_range))
        b = clamp_rgb(rgb[2] + rng.integers(-noise_range, noise_range))
        bg_color = (r, g, b)
        img = Image.new("RGB", (size, size), color=bg_color)

        # ── 2. 干扰色块（10% 样本） ───────────────────────────────────
        if rng.random() < cfg.get("interference_prob", 0.10):
            add_interference(img, rng, noise_range, cfg.get("interference_ratio", 0.20))

        # ── 3. 文字（参考真实图纸风格） ───────────────────────────────
        # 字体大小：格子尺寸的 30%~45%（真实图纸约35%）
        fs_ratio = rng.uniform(0.30, 0.45)
        font = get_random_font(max(8, int(size * fs_ratio)), rng, cfg)

        # 文字颜色：根据亮度自动选择（与真实图纸一致）
        text_color = get_text_color(bg_color, rng, cfg)

        code_display = self.bead_id
        draw = ImageDraw.Draw(img)

        # 文字居中对齐，稍微偏移模拟真实情况
        offset_range = cfg.get("text_offset_range", 1)
        ox = rng.integers(-offset_range, offset_range)
        oy = rng.integers(-offset_range, offset_range)
        cx, cy = size // 2 + ox, size // 2 + oy

        # 添加描边增加可读性（但比原来轻）
        if cfg.get("add_outline", True):
            thick = cfg.get("outline_thickness", 1)
            for dx in range(-thick, thick + 1):
                for dy in range(-thick, thick + 1):
                    if abs(dx) == thick or abs(dy) == thick:
                        draw.text((cx + dx, cy + dy), code_display,
                                  font=font, fill=(0, 0, 0), anchor="mm")

        draw.text((cx, cy), code_display, font=font, fill=text_color, anchor="mm")

        # ── 4. 网格线（30% 样本，模拟真实图纸） ───────────────────────
        if rng.random() < cfg.get("grid_line_prob", 0.3):
            add_grid_lines(img, rng,
                          cfg.get("grid_line_color", (180, 180, 180)),
                          cfg.get("grid_line_width", 1))

        # ── 5. 后处理（噪声 + 旋转 + 模糊分级 + JPEG压缩）────────────
        if rng.random() < 0.5 and cfg.get("gaussian_noise_std", 3) > 0:
            img = add_gaussian_noise(img, rng, cfg.get("gaussian_noise_std", 3))
        img = apply_post_process(img, rng, cfg)
        return img


# ──────────────────────────────────────────────────────────────────────────
#  2. "空"类别生成器
# ──────────────────────────────────────────────────────────────────────────

class EmptyBeadGenerator:
    """
    "空"类生成器（已修复，严禁放入任何真实色号！）

    用途：模拟无法识别色号的情况，如：
    - 纯色格（无文字）
    - 棋盘格/噪声格（干扰太强）
    - 模糊的非色号文字（数字、符号等）
    - 渐变色格

    注意：绝对不能生成任何 A01~M15 的真实色号！
    """

    def __init__(self, rng, cfg=None):
        self.rng = rng
        self.cfg = cfg or CFG.GEN_CONFIG
        self.size = CFG.CELL_IMAGE_SIZE

    def generate_one(self):
        rng = self.rng
        cfg = self.cfg

        # 空类的类型分布：纯色85%，噪声/文字干扰15%（去除棋盘、渐变）
        types = [
            "solid_color",      # 纯色 - 85%（最核心：无文字的空格子就是纯色）
            "random_noise",     # 随机噪声 - 5%
            "blur_non_bead",    # 模糊的非色号文字 - 6%
            "stripe",           # 条纹干扰 - 3%
            "dots",             # 点状干扰 - 1%
        ]
        weights = [0.85, 0.05, 0.06, 0.03, 0.01]
        sample_type = rng.choice(types, p=np.array(weights, dtype=float) / sum(weights))

        if sample_type == "solid_color":
            # 纯色块：白/灰/彩色，无任何文字或噪声
            # 偶尔加极轻微渐变（模拟斜射光），但整体是纯净单色
            color_choice = rng.random()
            if color_choice < 0.3:
                # 白色系
                v = rng.integers(220, 255)
                img = Image.new("RGB", (self.size, self.size), color=(v, v, v))
            elif color_choice < 0.6:
                # 灰色系
                v = rng.integers(100, 200)
                img = Image.new("RGB", (self.size, self.size), color=(v, v, v))
            else:
                # 彩色系
                img = Image.new("RGB", (self.size, self.size),
                               color=(rng.integers(50, 255), rng.integers(50, 255), rng.integers(50, 255)))
            # 10%概率加极轻微渐变（模拟斜射光，±3亮度）
            if rng.random() < 0.10:
                direction = rng.choice(["horizontal", "vertical"])
                arr = np.array(img, dtype=np.float32)
                for i in range(self.size):
                    delta = int(3.0 * (i / self.size - 0.5))
                    if direction == "horizontal":
                        arr[:, i] = np.clip(arr[:, i] + delta, 0, 255)
                    else:
                        arr[i, :] = np.clip(arr[i, :] + delta, 0, 255)
                img = Image.fromarray(arr.astype(np.uint8))

        elif sample_type == "random_noise":
            arr = rng.integers(0, 256, (self.size, self.size, 3), dtype=np.uint8)
            img = Image.fromarray(arr)

        elif sample_type == "blur_non_bead":
            img = self._gen_blurry_non_bead_text(rng)

        elif sample_type == "stripe":
            img = self._gen_stripes(rng)

        elif sample_type == "dots":
            img = self._gen_dots(rng)

        else:
            img = Image.new("RGB", (self.size, self.size), color=(200, 200, 200))

        # 轻微后处理（模拟真实图纸的退化）
        img = apply_post_process(img, rng, cfg)
        return img

    def _gen_blurry_non_bead_text(self, rng):
        """
        生成模糊的非色号文字（严禁使用真实色号！）
        只能用数字、字母（非A/B/C/D/E/F开头）、符号等
        """
        size = self.size

        # 背景色
        bg_brightness = rng.integers(150, 250)
        img = Image.new("RGB", (size, size), color=(bg_brightness, bg_brightness, bg_brightness))
        draw = ImageDraw.Draw(img)

        # 随机生成非色号文字（绝对不能是A01~M15格式）
        # 只能用：数字、随机字母组合、符号
        text_type = rng.choice(["numbers", "symbols", "garbage"])

        if text_type == "numbers":
            # 只有数字
            n_chars = rng.integers(2, 5)
            chars = "0123456789"
            text = "".join(rng.choice(list(chars)) for _ in range(n_chars))
        elif text_type == "symbols":
            # 只有符号
            n_chars = rng.integers(2, 4)
            chars = "!@#$%^&*+-=?/~"
            text = "".join(rng.choice(list(chars)) for _ in range(n_chars))
        else:
            # 乱码文字（看似像色号但不是）
            n_chars = rng.integers(2, 4)
            chars = "0123456789GHJKLMNPQRSTUVWXYZghjkmnpqrstuvwxyz"
            text = "".join(rng.choice(list(chars)) for _ in range(n_chars))

        font = get_random_font(max(8, int(size * rng.uniform(0.25, 0.45))), rng, self.cfg)

        # 文字颜色与背景对比
        text_color = (0, 0, 0) if bg_brightness > 140 else (255, 255, 255)

        ox = rng.integers(-1, 1)
        oy = rng.integers(-1, 1)
        draw.text((size // 2 + ox, size // 2 + oy), text, font=font, fill=text_color, anchor="mm")

        # 极模糊处理
        blur_radius = rng.uniform(2.5, 4.5)
        return img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    def _gen_stripes(self, rng):
        """生成条纹干扰"""
        size = self.size
        arr = np.zeros((size, size, 3), dtype=np.uint8)

        # 随机颜色
        c1 = (rng.integers(80, 200), rng.integers(80, 200), rng.integers(80, 200))
        c2 = (rng.integers(80, 200), rng.integers(80, 200), rng.integers(80, 200))

        stripe_width = rng.integers(2, 6)
        is_horizontal = rng.random() < 0.5

        for y in range(size):
            for x in range(size):
                if is_horizontal:
                    idx = y // stripe_width
                else:
                    idx = x // stripe_width
                arr[y, x] = c1 if idx % 2 == 0 else c2

        return Image.fromarray(arr)

    def _gen_dots(self, rng):
        """生成点状干扰"""
        size = self.size
        img = Image.new("RGB", (size, size), color=(200, 200, 200))
        draw = ImageDraw.Draw(img)

        n_dots = rng.integers(3, 10)
        dot_color = (rng.integers(0, 100), rng.integers(0, 100), rng.integers(0, 100))

        for _ in range(n_dots):
            x = rng.integers(5, size - 5)
            y = rng.integers(5, size - 5)
            r = rng.integers(2, 6)
            draw.ellipse([x-r, y-r, x+r, y+r], fill=dot_color)

        return img


# ──────────────────────────────────────────────────────────────────────────
#  3. 加载色板
# ──────────────────────────────────────────────────────────────────────────

def load_bead_list():
    with open(CFG.PALETTE_JSON, encoding="utf-8") as f:
        beads = json.load(f)
    sorted_ids = sorted(beads.keys())
    result = [(bid, tuple(beads[bid])) for bid in sorted_ids]
    assert len(result) == 221, f"色板应有221色，实际{len(result)}色"
    return result


# ──────────────────────────────────────────────────────────────────────────
#  4. 主生成流程
# ──────────────────────────────────────────────────────────────────────────

def generate_all(seed=42, per_class=None, per_empty=None,
                output_root=None, overwrite=False, dry_run=False):
    per_class = per_class if per_class is not None else CFG.SAMPLES_PER_CLASS
    per_empty = per_empty if per_empty is not None else per_class  # 空类数量与普通类同步，保持平衡
    output_root = output_root or CFG.TRAIN_DATASET_ROOT

    rng = random.Random(seed)
    bead_list = load_bead_list()
    all_classes = [(bid, rgb) for bid, rgb in bead_list] + [("空", (255, 255, 255))]
    total_expected = len(bead_list) * per_class + per_empty

    print(f"\n{'='*55}")
    print(f" 合成数据集生成器")
    print(f"{'='*55}")
    print(f" 输出目录   : {output_root}")
    print(f" 每普通色号 : {per_class} 张")
    print(f" 每空类     : {per_empty} 张")
    print(f" 随机种子   : {seed}")
    print(f" 总类别数   : {len(all_classes)}")
    print(f" 总图片数   : {total_expected:,} 张")
    print(f"{'='*55}\n")

    if dry_run:
        print("[Dry Run] 仅预览，不写文件")
        for i, (bid, _) in enumerate(all_classes):
            n = per_empty if bid == "空" else per_class
            print(f"  {i+1:3d}. {bid:<6s} → {n} 张")
        return

    if overwrite and os.path.exists(output_root):
        print(f"清空旧数据集：{output_root}")
        shutil.rmtree(output_root)

    for bid, _ in all_classes:
        ensure_dir(os.path.join(output_root, bid))

    total_generated = 0
    print(f"开始生成 {total_expected:,} 张图片...\n")

    # 先生成"空"类（如果有问题可以及时打断）
    empty_dir = os.path.join(output_root, "空")
    empty_rng = np.random.default_rng(seed + 9999)
    empty_gen = EmptyBeadGenerator(empty_rng)
    print(f"  [空] 类别（纯色85% + 其他干扰15%）:")
    for img_idx in range(per_empty):
        img = empty_gen.generate_one()
        save_path = os.path.join(empty_dir, f"{img_idx}.jpg")
        img.save(save_path, quality=92)
        total_generated += 1
        if (img_idx + 1) % 1000 == 0:
            print(f"    已生成 {img_idx+1:,} / {per_empty:,} 张")
    print(f"  [空] → {per_empty:,} 张 [OK]\n")

    # 再生成221个普通色号类
    for class_idx, (bead_id, rgb) in enumerate(bead_list):
        class_dir = os.path.join(output_root, bead_id)
        class_rng = np.random.default_rng(seed + class_idx)
        gen = NormalBeadGenerator(bead_id, rgb, class_rng)

        for img_idx in range(per_class):
            img = gen.generate_one()
            save_path = os.path.join(class_dir, f"{img_idx}.jpg")
            img.save(save_path, quality=92)
            total_generated += 1
            if total_generated % 2000 == 0:
                print(f"  已生成 {total_generated:,} / {total_expected:,} 张  "
                      f"({total_generated/total_expected*100:.1f}%)")
        print(f"  [{class_idx+1:3d}/221] {bead_id} ({rgb}) → {per_class:,} 张 [OK]")
    print(f"\n[OK] 全部生成完毕！共 {total_generated:,} 张图片\n")

    ensure_dir(os.path.dirname(CFG.TRAIN_LABEL_MAP_PATH))
    with open(CFG.TRAIN_LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        for idx, (bid, _) in enumerate(all_classes):
            f.write(f"{idx}\t{bid}\n")
    print(f"标签映射已保存：{CFG.TRAIN_LABEL_MAP_PATH}")

    split_dataset(output_root, seed=seed)
    print_summary(output_root)


# ──────────────────────────────────────────────────────────────────────────
#  5. 划分 train/val/test
# ──────────────────────────────────────────────────────────────────────────

def split_dataset(dataset_root, seed=42):
    rng = random.Random(seed)
    split_dir = os.path.join(dataset_root, "split")
    ensure_dir(split_dir)

    label_map = {}
    with open(CFG.TRAIN_LABEL_MAP_PATH, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                label_map[parts[1]] = int(parts[0])

    train_lines, val_lines, test_lines = [], [], []

    for bead_id in sorted(os.listdir(dataset_root)):
        class_dir = os.path.join(dataset_root, bead_id)
        if not os.path.isdir(class_dir) or bead_id == "split":
            continue
        if bead_id not in label_map:
            continue

        label_idx = label_map[bead_id]
        files = sorted([
            f for f in os.listdir(class_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        rng.shuffle(files)

        n = len(files)
        n_train = max(1, int(n * CFG.TRAIN_RATIO))
        n_val   = max(1, int(n * CFG.VAL_RATIO))
        n_test  = n - n_train - n_val

        for f in files[:n_train]:
            train_lines.append(f"{bead_id}/{f}\t{label_idx}")
        for f in files[n_train:n_train + n_val]:
            val_lines.append(f"{bead_id}/{f}\t{label_idx}")
        for f in files[n_train + n_val:]:
            test_lines.append(f"{bead_id}/{f}\t{label_idx}")

    for split_name, lines in [("train", train_lines), ("val", val_lines), ("test", test_lines)]:
        path = os.path.join(split_dir, f"{split_name}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"  {split_name}.txt: {len(lines):,} 条")

    print(f"\n数据集划分完成 → {split_dir}")


# ──────────────────────────────────────────────────────────────────────────
#  6. 统计摘要
# ──────────────────────────────────────────────────────────────────────────

def print_summary(dataset_root):
    if not os.path.exists(dataset_root):
        return
    total = 0
    class_stats = {}
    for bead_id in sorted(os.listdir(dataset_root)):
        class_dir = os.path.join(dataset_root, bead_id)
        if not os.path.isdir(class_dir) or bead_id == "split":
            continue
        n = len([
            f for f in os.listdir(class_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        class_stats[bead_id] = n
        total += n
    print(f"\n{'─'*45}")
    print(f"类别数  ：{len(class_stats)}")
    print(f"总样本数：{total:,}")
    if class_stats:
        counts = list(class_stats.values())
        print(f"每类    ：最少={min(counts)}  最多={max(counts)}  平均={np.mean(counts):.0f}")
    print(f"{'─'*45}")


# ──────────────────────────────────────────────────────────────────────────
#  7. CLI
# ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="合成生成拼豆格子训练数据集")
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--per-class", type=int)
    parser.add_argument("--per-empty", type=int)
    parser.add_argument("--output",    type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run",   action="store_true")
    args = parser.parse_args()

    print("=" * 55)
    print(" Step 1 - 合成生成训练数据集")
    print("=" * 55)

    generate_all(
        seed       = args.seed,
        per_class  = args.per_class,
        per_empty  = args.per_empty,
        output_root = args.output,
        overwrite  = args.overwrite,
        dry_run    = args.dry_run,
    )

    if not args.dry_run:
        print("\n下一步：python 2_build_dataset.py  （检查数据集）")
        print("        python 3_train.py           （开始训练）")


if __name__ == "__main__":
    main()
