import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QSpinBox, QCheckBox, \
    QMessageBox
from PyQt5.QtGui import QPixmap
import os
import json
from time import time
from utils.tools import load_lut_npy, generate_pattern, render_pattern, bead_stats


class AppDemo(QWidget):
    def __init__(self):
        super().__init__()
        self.title = '拼豆图案生成器'
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

    def process_image(self):
        if not hasattr(self, 'selectedImagePath'):
            QMessageBox.warning(self, "警告", "请先选择一张图片")
            return

        compress_colors = self.compressColorsSpin.value()
        N = self.nSizeSpin.value()
        use_sharp = self.useSharpCheck.isChecked()
        open_mirror = self.open_mirror.isChecked()

        # 在这里调用你现有的逻辑进行处理，并保存结果到output文件夹中。
        output_path = f"output/{os.path.splitext(os.path.basename(self.selectedImagePath))[0]}_processed.png"
        output_path_mirror = f"output/{os.path.splitext(os.path.basename(self.selectedImagePath))[0]}_processed_mirror.png"

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


app = QApplication(sys.argv)
demo = AppDemo()
demo.show()
sys.exit(app.exec_())