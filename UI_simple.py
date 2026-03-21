import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QSpinBox, QCheckBox, \
    QMessageBox
from PyQt5.QtGui import QPixmap
import os
from time import time
from PIL import Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
from sklearn.cluster import KMeans
from collections import Counter
from PIL import ImageFilter
import numpy as np
import json
import traceback

def load_lut_npy(npy_file):
    rgb_lut_array = np.load(npy_file, allow_pickle=True)
    return rgb_lut_array

def nearest_bead(rgb, rgb_lut_array):
    r, g, b = rgb
    return str(rgb_lut_array[r, g, b])  # np.object_ → str

def color_quantization(img, k=32):
    arr = np.array(img)
    h, w, c = arr.shape
    pixels = arr.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, n_init=5)
    labels = kmeans.fit_predict(pixels)
    new_pixels = kmeans.cluster_centers_[labels]
    new_img = new_pixels.reshape(h, w, 3)
    return Image.fromarray(new_img.astype(np.uint8))

def quantize(image, rgb_lut_array):
    arr = np.array(image)
    h, w, _ = arr.shape
    grid = [[None]*w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            rgb = tuple(arr[y, x])
            bead = nearest_bead(rgb, rgb_lut_array)
            grid[y][x] = bead
    return grid

def generate_pattern(path, rgb_lut_array, N=52, compress_colors=None, use_sharp=False):
    img = Image.open(path).convert("RGB")
    original_width, original_height = img.size

    # 根据长宽比计算目标尺寸
    if original_width >= original_height:
        # 横图或正方形：固定宽度为 N
        new_width = N
        new_height = round(N * original_height / original_width)
    else:
        # 竖图：固定高度为 N
        new_height = N
        new_width = round(N * original_width / original_height)

    print(f'width：{new_width} , height：{new_height}')
    # 缩放图像（保持比例）
    img = img.resize((new_width, new_height), Image.Resampling.BILINEAR)

    # 锐化
    if use_sharp:
        img = img.filter(ImageFilter.SHARPEN)
        # 方法2：自定义高通滤波器（更激进）
        # 使用自定义高通滤波器进行锐化
        kernel = [
            -1, -1, -1,
            -1, 9, -1,
            -1, -1, -1
        ]
        img.filter(ImageFilter.Kernel((3, 3), kernel))
    if compress_colors:
        img = color_quantization(img, compress_colors)
    grid = quantize(img, rgb_lut_array)
    return grid

def render_pattern(grid, beads, cell=28):
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0

    # 增加2行2列作为边缘标记
    new_height, new_width = height + 2, width + 2

    img = Image.new("RGB", (new_width * cell, new_height * cell), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("Arial.ttf", int(cell * 0.35))
    except:
        font = ImageFont.load_default()

    # --- 第一步：填充每个格子的颜色 ---
    for y in range(new_height):
        for x in range(new_width):
            # 判断是否是边缘部分
            if 1 <= x < new_width - 1 and 1 <= y < new_height - 1:
                code = grid[y - 1][x - 1]
                rgb = beads.get(code, (255, 255, 255))  # 安全获取颜色
            else:
                rgb = (200, 200, 200)  # 灰色边缘

            x0 = x * cell
            y0 = y * cell
            x1 = x0 + cell
            y1 = y0 + cell
            draw.rectangle([x0, y0, x1, y1], fill=rgb)

            # 如果是在边缘上，则绘制编号
            if 1 <= x < new_width - 1 and 1 <= y < new_height - 1:
                continue  # 跳过中间部分的文字处理
            elif y == 0:  # 上方边缘
                text = str(x)
            elif y == new_height - 1:  # 下方边缘
                text = str(x)
            elif x == 0:  # 左侧边缘
                text = str(y)
            elif x == new_width - 1:  # 右侧边缘
                text = str(y)
            else:
                continue

            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            tx = x0 + (cell - tw) / 2
            ty = y0 + (cell - th) / 2
            draw.text((tx, ty), text, fill=(0, 0, 0), font=font)

    # --- 第二步：画网格线（基于新尺寸） ---
    # 水平线（每行）
    for y in range(new_height + 1):  # 新高度+1
        line_y = y * cell
        is_5_line = (y % 5 == 0)
        color = (0, 180, 255) if is_5_line else (180, 180, 180)  # 天蓝 / 浅灰
        width_line = 2 if is_5_line else 1
        draw.line([0, line_y, new_width * cell, line_y], fill=color, width=width_line)

    # 垂直线（每列）
    for x in range(new_width + 1):  # 新宽度+1
        line_x = x * cell
        is_5_line = (x % 5 == 0)
        color = (0, 180, 255) if is_5_line else (180, 180, 180)
        width_line = 2 if is_5_line else 1
        draw.line([line_x, 0, line_x, new_height * cell], fill=color, width=width_line)

    # --- 第三步：写入色号（仅限主图区域） ---
    for y in range(height):
        for x in range(width):
            code = grid[y][x]
            rgb = beads.get(code, (255, 255, 255))
            if not isinstance(rgb, (tuple, list)) or len(rgb) != 3:
                rgb = (255, 255, 255)
            x0 = (x + 1) * cell  # 主图从第2列开始
            y0 = (y + 1) * cell  # 主图从第2行开始
            text = code
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            tx = x0 + (cell - tw) / 2
            ty = y0 + (cell - th) / 2
            brightness = sum(rgb) / 3
            color = (255, 255, 255) if brightness < 140 else (0, 0, 0)
            draw.text((tx, ty), text, fill=color, font=font)

    # --- 第四步：添加图例（放在新图底部） ---
    final_img = add_color_legend(img, beads, grid, cell)
    return final_img

def add_color_legend(img, beads, grid, cell=28):
    """
    在图像底部添加宽松、大格子的色号图例卡片。
    - 每个色块较大（默认 48x48）
    - 色块内显示色号（如 'A1'）
    - 色块下方显示 '×数量'
    - 宽松布局，自动换行，底部留白充足
    """
    # 1. 统计并排序
    code_counts = Counter(code for row in grid for code in row)
    sorted_items = sorted(code_counts.items(), key=lambda x: x[1], reverse=True)

    if not sorted_items:
        return img  # 无颜色，直接返回原图

    # 2. 图例样式参数（可调！）
    block_size = max(40, int(cell * 1.2))      # 色块尺寸，至少 40px
    margin_between_blocks = max(16, int(block_size * 0.4))  # 块间水平/垂直间距
    text_gap = max(6, int(block_size * 0.15))   # 色块与下方文字的间隙
    bottom_padding = max(30, block_size)        # 底部额外留白

    # 字体大小
    font_size_code = max(14, int(block_size * 0.35))   # 色号字体
    font_size_count = max(12, int(block_size * 0.3))   # ×数量字体

    try:
        font_code = ImageFont.truetype("Arial.ttf", font_size_code)
        font_count = ImageFont.truetype("Arial.ttf", font_size_count)
    except:
        font_code = ImageFont.load_default()
        font_count = ImageFont.load_default()

    # 3. 计算每行最多放几个
    item_total_width = block_size + margin_between_blocks
    max_per_line = max(1, (img.width + margin_between_blocks) // item_total_width)

    # 4. 分行
    lines = []
    current_line = []
    for item in sorted_items:
        current_line.append(item)
        if len(current_line) >= max_per_line:
            lines.append(current_line)
            current_line = []
    if current_line:
        lines.append(current_line)

    # 5. 计算图例总高度
    legend_content_height = len(lines) * (block_size + text_gap + font_size_count) + \
                            (len(lines) - 1) * margin_between_blocks
    legend_total_height = legend_content_height + bottom_padding

    # 6. 创建新图像
    total_height = img.height + legend_total_height
    combined_img = Image.new("RGB", (img.width, total_height), (255, 255, 255))
    combined_img.paste(img, (0, 0))
    draw = ImageDraw.Draw(combined_img)

    # 7. 绘制图例
    y_start = img.height
    for line_idx, line in enumerate(lines):
        # 当前行起始 Y
        y_line = y_start + line_idx * (block_size + text_gap + font_size_count + margin_between_blocks)

        # 计算当前行总宽度，用于居中
        line_width = len(line) * item_total_width - margin_between_blocks
        start_x = (img.width - line_width) // 2

        for col_idx, (code, count) in enumerate(line):
            x = start_x + col_idx * item_total_width
            y = y_line

            rgb = beads[code]

            # 绘制色块（带边框）
            draw.rectangle([x, y, x + block_size, y + block_size], fill=rgb, outline=(100, 100, 100), width=1)

            # --- 绘制色号（居中在色块内）---
            bbox_code = draw.textbbox((0, 0), code, font=font_code)
            tw_code = bbox_code[2] - bbox_code[0]
            th_code = bbox_code[3] - bbox_code[1]
            tx_code = x + (block_size - tw_code) / 2
            ty_code = y + (block_size - th_code) / 2

            # 自动选择文字颜色（高对比度）
            brightness = sum(rgb) / 3
            text_color = (255, 255, 255) if brightness < 160 else (0, 0, 0)
            draw.text((tx_code, ty_code), code, fill=text_color, font=font_code)

            # --- 绘制数量 "×{count}" ---
            count_text = f"{count}"
            bbox_count = draw.textbbox((0, 0), count_text, font=font_count)
            tw_count = bbox_count[2] - bbox_count[0]
            th_count = bbox_count[3] - bbox_count[1]
            tx_count = x + (block_size - tw_count) / 2
            ty_count = y + block_size + text_gap

            draw.text((tx_count, ty_count), count_text, fill=(0, 0, 0), font=font_count)

    return combined_img

def bead_stats(grid):
    flat = [c for row in grid for c in row]
    return Counter(flat)


class AppDemo(QWidget):
    def __init__(self):
        super().__init__()
        self.title = '拼豆图纸生成器'
        self.selectedImagePath = None  # 👈 显式初始化为 None
        self.output_dir = None  # 用户选择的输出目录
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        layout = QVBoxLayout()

        # 选择图片按钮
        self.btnSelectImg = QPushButton('选择图片', self)
        self.btnSelectImg.clicked.connect(self.load_image)
        layout.addWidget(self.btnSelectImg)

        # 显示选择的图片
        self.lblImage = QLabel(self)
        layout.addWidget(self.lblImage)

        # 压色数设置
        self.compressColorsLabel = QLabel("压缩颜色数:")
        layout.addWidget(self.compressColorsLabel)
        self.compressColorsSpin = QSpinBox(self)
        self.compressColorsSpin.setMinimum(1)
        self.compressColorsSpin.setMaximum(256)
        self.compressColorsSpin.setValue(30)  # 默认值
        layout.addWidget(self.compressColorsSpin)

        # 图纸大小N设置
        self.nSizeLabel = QLabel("图纸大小N*N:")
        layout.addWidget(self.nSizeLabel)
        self.nSizeSpin = QSpinBox(self)
        self.nSizeSpin.setMinimum(1)
        self.nSizeSpin.setMaximum(1000)
        self.nSizeSpin.setValue(100)  # 默认值
        layout.addWidget(self.nSizeSpin)

        # 使用锐化选项
        self.useSharpCheck = QCheckBox('使用锐化', self)
        self.useSharpCheck.setChecked(True)
        layout.addWidget(self.useSharpCheck)

        # 输出镜像选项
        self.open_mirror = QCheckBox('输出镜像图纸', self)
        self.open_mirror.setChecked(True)
        layout.addWidget(self.open_mirror)

        # 选择输出文件夹按钮
        self.btnSelectOutputDir = QPushButton('选择输出文件夹', self)
        self.btnSelectOutputDir.clicked.connect(self.select_output_dir)
        layout.addWidget(self.btnSelectOutputDir)

        # 👇 新增：显示当前输出目录的标签
        self.lblOutputDir = QLabel("未选择输出目录")
        self.lblOutputDir.setWordWrap(True)  # 防止路径太长溢出
        layout.addWidget(self.lblOutputDir)

        # 开始处理按钮
        self.btnStart = QPushButton('开始处理', self)
        self.btnStart.clicked.connect(self.process_image)
        layout.addWidget(self.btnStart)

        self.setLayout(layout)

    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, '选择图片', '', "Image files (*.jpg *.png)")
        if fname:
            pixmap = QPixmap(fname)
            self.lblImage.setPixmap(pixmap.scaled(400, 400))
            self.selectedImagePath = fname
        else:
            # 用户取消了选择，可以清空（可选）
            self.selectedImagePath = None
            self.lblImage.clear()  # 清除预览图

    def select_output_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if folder:
            self.output_dir = folder
            self.lblOutputDir.setText(f"输出目录: {folder}")
        else:
            self.output_dir = None
            self.lblOutputDir.setText("未选择输出目录")

    def process_image(self):
        # 校验：是否选择了图片
        if not self.selectedImagePath or not os.path.isfile(self.selectedImagePath):
            QMessageBox.warning(self, "⚠️ 警告", "请先选择一张有效的图片！")
            return

        # 校验：是否选择了输出目录
        if not self.output_dir or not os.path.isdir(self.output_dir):
            QMessageBox.warning(self, "⚠️ 警告", "请先选择一个输出文件夹！")
            return

        compress_colors = self.compressColorsSpin.value()
        N = self.nSizeSpin.value()
        use_sharp = self.useSharpCheck.isChecked()
        open_mirror = self.open_mirror.isChecked()

        # 在这里调用你现有的逻辑进行处理，并保存结果到output文件夹中。
        output_path = os.path.join(self.output_dir,
                                   f"{os.path.splitext(os.path.basename(self.selectedImagePath))[0]}_processed.png")
        output_path_mirror = os.path.join(self.output_dir,
                                          f"{os.path.splitext(os.path.basename(self.selectedImagePath))[0]}_processed_mirror.png")

        try:
            # 1.读取拼豆色卡
            with open("load/beads_palette_221_correct.json") as f:
                beads = json.load(f)
            for k in beads:
                beads[k] = tuple(beads[k])
            t0 = time()
            npy_file = "load/rgb_to_bead_lut.npy"
            rgb_lut_array = load_lut_npy(npy_file)
            t1 = time()
            # 3.颜色查找
            grid = generate_pattern(self.selectedImagePath,
                                    rgb_lut_array,
                                    N=N,  # 图纸大小 N*N
                                    compress_colors=compress_colors,  # 压缩色数
                                    use_sharp=use_sharp,  # 使用锐化
                                    )
            mirrored_grid = [row[::-1] for row in grid]
            t2 = time()
            # 4.渲染图纸
            img = render_pattern(grid, beads)
            if open_mirror:
                img_mirror = render_pattern(mirrored_grid, beads)
            t3 = time()
            img.save(output_path)
            if open_mirror:
                img_mirror.save(output_path_mirror)
            # 5.日志打印，用户交互
            stats = bead_stats(grid)
            print(f'\nt0 加载npy：{round(t1 - t0, 2)}s')
            print(f't1 颜色查找：{round(t2 - t1, 2)}s')
            print(f't2 图纸渲染：{round(t3 - t2, 2)}s')
            print("\n 总色数:", len(stats))

            QMessageBox.information(self, "完成", f"处理完成！\n输出路径: {output_path}")
        except Exception as err:
            QMessageBox.critical(self, "❌ 错误", f"处理过程中出错：\n{str(e)}")
            traceback.print_exc()

if __name__ == '__main__':
    # 设置工作目录为脚本所在目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    app = QApplication(sys.argv)
    demo = AppDemo()
    demo.show()
    sys.exit(app.exec_())