"""
拼豆统一工作台 - pindou_UI.py
集图纸生成、编辑、辅助拼豆于一体的 PyQt5 应用

功能：
- 图案生成：从图片生成拼豆图纸
- 图纸编辑：单格编辑、区域批量修改
- 撤销/重做：支持操作回退
- 镜像生成：保存图纸的镜像版本
- 读取图纸：从output文件夹加载图纸图片，自动映射JSON文件
- 辅助拼豆：按颜色分组高亮，区域标号，完成标记
- 现代紫色主题UI（蓝紫色系、渐变按钮、圆角阴影、无衬线字体）
"""

# 在导入任何其他模块之前设置环境变量，避免OpenMP等触发CMD窗口
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import sys
import json
import copy
from time import time
from collections import Counter, defaultdict


def _preload_heavy_modules():
    """
    预加载重量级模块，避免在首次使用时触发CMD闪跳。
    只预加载numpy和cv2，避免sklearn/scipy在导入时触发子进程。
    """
    import warnings
    warnings.filterwarnings('ignore')

    try:
        # 初始化numpy（可能触发OpenMP配置）
        import numpy as np
        _ = np.zeros((10, 10))
        _ = np.dot(_, _.T)
        del _
    except:
        pass

    try:
        # 初始化cv2
        import cv2
        _ = cv2.__version__
        _ = np.zeros((10, 10, 3), dtype=np.uint8)
        _ = cv2.resize(_, (5, 5))
        del _
    except:
        pass


# 立即预加载numpy和cv2（在导入其他模块之前）
_preload_heavy_modules()


def get_app_root():
    """
    获取应用程序根目录。
    兼容PyInstaller打包后的运行方式。
    """
    if getattr(sys, 'frozen', False):
        # PyInstaller打包后的exe运行
        # onedir模式：exe在根目录，_internal子目录存放依赖
        exe_dir = os.path.dirname(sys.executable)
        return exe_dir
    else:
        # 开发环境下的Python脚本运行
        return os.path.dirname(os.path.abspath(__file__))


def get_internal_path():
    """
    获取_internal目录路径（存放依赖文件）。
    在打包环境下返回 <exe_dir>/_internal，在开发环境下返回根目录。
    """
    root = get_app_root()
    internal = os.path.join(root, '_internal')
    if os.path.isdir(internal):
        return internal
    return root


# 获取应用根目录和_internal目录
APP_ROOT = get_app_root()
INTERNAL_ROOT = get_internal_path()

# 将utils目录添加到sys.path，以便导入tools模块
# 打包后utils在INTERNAL_ROOT/utils，开发时在APP_ROOT/utils
utils_path = os.path.join(INTERNAL_ROOT, 'utils')
if not os.path.isdir(utils_path):
    utils_path = os.path.join(APP_ROOT, 'utils')
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.tools import (
    load_lut_npy, generate_pattern, render_pattern,
    save_grid_to_json, load_grid_from_json
)


from PyQt5.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
    QPushButton, QLabel, QSpinBox, QCheckBox, QFileDialog,
    QInputDialog, QScrollArea, QTabWidget, QMessageBox,
    QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsItem,
    QToolBar, QAction, QStatusBar, QFrame, QSplitter, QListWidget,
    QListWidgetItem, QAbstractItemView, QProgressDialog, QGroupBox,
    QButtonGroup, QRadioButton, QDialog, QTextEdit, QDesktopWidget,
    QSizePolicy
)
from PyQt5.QtGui import (
    QPixmap, QImage, QColor, QPen, QBrush, QPainter, QCursor,
    QFont, QTransform, QLinearGradient, QPalette, QIcon
)
from PyQt5.QtCore import Qt, QPointF, pyqtSignal, QThread, QSize


# =============================================================================
# 自定义现代风格对话框
# =============================================================================

class ModernDialog(QDialog):
    """自定义现代风格对话框，圆角紫色主题"""

    def __init__(self, title="提示", message="", dialog_type="info", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.setModal(True)
        self.setMinimumWidth(400)

        # 根据类型选择颜色
        colors = {
            "info": ("#9C27B0", "#7B1FA2"),
            "warning": ("#FF9800", "#F57C00"),
            "error": ("#F44336", "#D32F2F"),
            "success": ("#4CAF50", "#388E3C"),
            "question": ("#2196F3", "#1976D2"),
        }
        self.color1, self.color2 = colors.get(dialog_type, colors["info"])

        self._setup_ui(message)

    def _setup_ui(self, message):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)

        # 顶部标题栏
        title_bar = QWidget()
        title_bar.setStyleSheet(f"""
            QWidget {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {self.color1}, stop:1 {self.color2});
                border-top-left-radius: 12px;
                border-top-right-radius: 12px;
                padding: 12px 16px;
            }}
        """)
        title_layout = QHBoxLayout()
        title_layout.setContentsMargins(0, 0, 0, 0)

        self.title_label = QLabel(self.windowTitle())
        self.title_label.setStyleSheet("color: white; font-size: 16px; font-weight: bold;")
        title_layout.addWidget(self.title_label)
        title_layout.addStretch()

        title_bar.setLayout(title_layout)
        main_layout.addWidget(title_bar)

        # 内容区域
        content_widget = QWidget()
        content_widget.setStyleSheet("""
            background: #FAFAFA;
            border-bottom-left-radius: 12px;
            border-bottom-right-radius: 12px;
        """)
        content_layout = QVBoxLayout()

        # 消息文本
        msg_label = QLabel(message)
        msg_label.setStyleSheet("""
            color: #333;
            font-size: 14px;
            font-family: 'Microsoft YaHei', 'Segoe UI', Arial, sans-serif;
            padding: 20px;
            line-height: 1.6;
        """)
        msg_label.setWordWrap(True)
        msg_label.setAlignment(Qt.AlignLeft)
        content_layout.addWidget(msg_label)

        content_widget.setLayout(content_layout)
        main_layout.addWidget(content_widget)

        # 按钮区域
        btn_widget = QWidget()
        btn_widget.setStyleSheet("background: #F5F5F5; padding: 12px 20px;")
        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.addStretch()

        self.btn_ok = QPushButton("确定")
        self.btn_ok.setStyleSheet(self._button_style())
        self.btn_ok.clicked.connect(self.accept)
        btn_layout.addWidget(self.btn_ok)

        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setStyleSheet(self._button_style(False))
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_cancel.hide()
        btn_layout.addWidget(self.btn_cancel)

        btn_widget.setLayout(btn_layout)
        main_layout.addWidget(btn_widget)

        self.setLayout(main_layout)

        # 居中显示
        self._center_on_screen()

    def _button_style(self, primary=True):
        if primary:
            return f"""
                QPushButton {{
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 {self.color1}, stop:1 {self.color2});
                    color: white;
                    border: none;
                    border-radius: 6px;
                    padding: 10px 28px;
                    font-size: 14px;
                    font-weight: bold;
                    font-family: 'Microsoft YaHei', 'Segoe UI', Arial, sans-serif;
                    min-height: 36px;
                }}
                QPushButton:hover {{
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 {self.color2}, stop:1 {self.color1});
                }}
            """
        else:
            return """
                QPushButton {
                    background: white;
                    color: #666;
                    border: 1px solid #DDD;
                    border-radius: 6px;
                    padding: 10px 28px;
                    font-size: 14px;
                    font-family: 'Microsoft YaHei', 'Segoe UI', Arial, sans-serif;
                    min-height: 36px;
                }
                QPushButton:hover {
                    background: #F5F5F5;
                    border-color: #CCC;
                }
            """

    def _center_on_screen(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        x = (screen.width() - size.width()) // 2
        y = (screen.height() - size.height()) // 2
        self.move(x, y)

    def set_message(self, message):
        """设置消息内容"""
        for w in self.findChild(QLabel):
            if "color: #333" in w.styleSheet():
                w.setText(message)
                break

    def add_button(self, text, callback, is_primary=True):
        """添加自定义按钮"""
        btn = QPushButton(text)
        btn.setStyleSheet(self._button_style(is_primary))
        btn.clicked.connect(callback)
        # 在OK按钮前插入
        self.layout().itemAt(2).widget().layout().insertWidget(
            self.layout().itemAt(2).widget().layout().count() - 1, btn
        )
        return btn

    def show_cancel(self):
        """显示取消按钮"""
        self.btn_cancel.show()

    @staticmethod
    def info(title, message, parent=None):
        dlg = ModernDialog(title, message, "info", parent)
        dlg.exec_()
        return True

    @staticmethod
    def warning(title, message, parent=None):
        dlg = ModernDialog(title, message, "warning", parent)
        dlg.exec_()
        return True

    @staticmethod
    def error(title, message, parent=None):
        dlg = ModernDialog(title, message, "error", parent)
        dlg.exec_()
        return True

    @staticmethod
    def success(title, message, parent=None):
        dlg = ModernDialog(title, message, "success", parent)
        dlg.exec_()
        return True

    @staticmethod
    def question(title, message, parent=None):
        dlg = ModernDialog(title, message, "question", parent)
        dlg.show_cancel()
        dlg.btn_ok.setText("是")
        dlg.btn_cancel.setText("否")
        return dlg.exec_() == QDialog.Accepted


# =============================================================================
# 撤销管理器
# =============================================================================

class UndoManager:
    """管理编辑操作的撤销/重做 - 使用栈顶存储最新状态"""

    def __init__(self, max_undo=50):
        self.undo_stack = []  # 栈顶是最新保存的状态
        self.redo_stack = []
        self.max_undo = max_undo

    def save_state(self, grid):
        """保存当前状态到撤销栈（栈顶存储最新状态）"""
        state = copy.deepcopy(grid)
        self.undo_stack.append(state)
        if len(self.undo_stack) > self.max_undo:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def undo(self, current_grid):
        """撤销：弹出一个状态恢复。返回(恢复的状态, 是否成功)"""
        if len(self.undo_stack) <= 1:  # 至少保留初始状态
            return None, False
        # 当前状态存入redo栈
        self.redo_stack.append(copy.deepcopy(current_grid))
        # 弹出栈顶（最新状态），恢复下面的状态
        self.undo_stack.pop()
        prev_state = self.undo_stack[-1]  # 获取新的栈顶（上一个状态）
        return copy.deepcopy(prev_state), True

    def redo(self, current_grid):
        """重做：恢复redo栈顶状态。返回(恢复的状态, 是否成功)"""
        if not self.redo_stack:
            return None, False
        # 当前状态存入undo栈
        self.undo_stack.append(copy.deepcopy(current_grid))
        # 弹出redo栈顶作为当前状态
        redo_state = self.redo_stack.pop()
        return copy.deepcopy(redo_state), True

    def clear(self):
        """清空所有历史"""
        self.undo_stack.clear()
        self.redo_stack.clear()

    def can_undo(self):
        return len(self.undo_stack) > 1  # 需要多于初始状态

    def can_redo(self):
        return len(self.redo_stack) > 0


# =============================================================================
# 后台生成线程
# =============================================================================

class GenerateWorker(QThread):
    finished_signal = pyqtSignal(object, float)
    error_signal = pyqtSignal(str, str)

    def __init__(self, image_path, rgb_lut_array, N, compress_colors, use_sharp):
        super().__init__()
        self.image_path = image_path
        self.rgb_lut_array = rgb_lut_array
        self.N = N
        self.compress_colors = compress_colors
        self.use_sharp = use_sharp

    def run(self):
        try:
            t0 = time()
            grid = generate_pattern(
                self.image_path, self.rgb_lut_array,
                N=self.N, compress_colors=self.compress_colors,
                use_sharp=self.use_sharp,
            )
            self.finished_signal.emit(grid, time() - t0)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.error_signal.emit(str(e), tb)


# =============================================================================
# 单个拼豆格子
# =============================================================================

class BeadCellItem(QGraphicsRectItem):
    def __init__(self, row, col, code, rgb, cell_size):
        super().__init__()
        self.row = row
        self.col = col
        self.code = code
        self.rgb = rgb
        self.cell_size = cell_size
        self.highlighted = False
        self.highlight_number = None
        self.contrast_color = None
        self.original_rgb = rgb
        self.setRect(col * cell_size, row * cell_size, cell_size, cell_size)
        self.setPen(QPen(QColor(180, 180, 180), 0.5))
        self.setBrush(QBrush(QColor(*rgb)))
        self.setAcceptedMouseButtons(Qt.LeftButton | Qt.RightButton)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setCacheMode(QGraphicsItem.DeviceCoordinateCache)

    def update_color(self, new_code, new_rgb):
        self.code = new_code
        self.rgb = new_rgb
        self.original_rgb = new_rgb
        self.setBrush(QBrush(QColor(*new_rgb)))
        self.update()

    def set_highlight(self, highlighted, number=None, contrast_color=None):
        """设置高亮状态"""
        self.highlighted = highlighted
        self.highlight_number = number
        self.contrast_color = contrast_color
        if highlighted:
            self.setPen(QPen(QColor(255, 50, 50), 2))
        else:
            self.setPen(QPen(QColor(180, 180, 180), 0.5))
        self.update()

    def set_dimmed(self, dimmed=True):
        """设置淡化状态"""
        if dimmed:
            self.setBrush(QBrush(QColor(230, 230, 230)))
        else:
            self.setBrush(QBrush(QColor(*self.original_rgb)))
        self.update()

    def restore_original(self):
        """恢复原始颜色"""
        self.setBrush(QBrush(QColor(*self.original_rgb)))
        self.update()

    def paint(self, painter, option, widget=None):
        painter.setBrush(self.brush())
        painter.setPen(self.pen())
        painter.drawRect(self.rect())

        if self.highlighted and self.highlight_number is not None:
            font = QFont("Segoe UI", max(8, int(self.cell_size * 0.4)), QFont.Bold)
            painter.setFont(font)
            text_color = self.contrast_color if self.contrast_color else QColor(255, 255, 0)
            painter.setPen(text_color)
            painter.drawText(self.rect(), Qt.AlignCenter, str(self.highlight_number))
        elif self.code:
            font = QFont("Segoe UI", max(6, int(self.cell_size * 0.35)))
            painter.setFont(font)
            brightness = sum(self.rgb) / 3
            text_color = QColor(255, 255, 255) if brightness < 140 else QColor(0, 0, 0)
            painter.setPen(text_color)
            painter.drawText(self.rect(), Qt.AlignCenter, self.code)


# =============================================================================
# 图纸编辑场景
# =============================================================================

class BeadScene(QGraphicsScene):
    cell_clicked = pyqtSignal(int, int, object)
    area_selected = pyqtSignal(int, int, int, int)
    eyedropper_picked = pyqtSignal(str)
    status_message = pyqtSignal(str)

    def __init__(self, cell_size=28):
        super().__init__()
        self.cell_size = cell_size
        self.tool_mode = 'select'
        self.is_drawing = False
        self.start_pos = QPointF()
        self.rubber_band = None
        self.cells = {}
        self.beads = {}
        self.grid_height = 0
        self.grid_width = 0
        self._dimmed_codes = set()  # 记录当前被淡化的颜色

    def set_grid(self, grid, beads, cell_size=28):
        self.clear()
        self.cells.clear()
        self.beads = beads
        self.cell_size = cell_size
        self.grid_height = len(grid)
        self.grid_width = len(grid[0]) if self.grid_height > 0 else 0
        self._dimmed_codes.clear()
        for r, row in enumerate(grid):
            for c, code in enumerate(row):
                rgb = beads.get(code, (200, 200, 200))
                if not isinstance(rgb, (tuple, list)) or len(rgb) != 3:
                    rgb = (200, 200, 200)
                item = BeadCellItem(r, c, code, rgb, cell_size)
                self.cells[(r, c)] = item
                self.addItem(item)
        self.setSceneRect(0, 0, self.grid_width * cell_size, self.grid_height * cell_size)

    def mousePressEvent(self, event):
        if self.tool_mode == 'eyedropper':
            pos = event.scenePos()
            col = int(pos.x() // self.cell_size)
            row = int(pos.y() // self.cell_size)
            if 0 <= row < self.grid_height and 0 <= col < self.grid_width:
                item = self.cells.get((row, col))
                if item:
                    self.eyedropper_picked.emit(item.code)
            return
        pos = event.scenePos()
        self.start_pos = pos
        col = int(pos.x() // self.cell_size)
        row = int(pos.y() // self.cell_size)
        if 0 <= row < self.grid_height and 0 <= col < self.grid_width:
            if event.modifiers() & Qt.ShiftModifier:
                self.is_drawing = True
                # 框从当前格子左上角开始，精确跟随鼠标
                x = col * self.cell_size
                y = row * self.cell_size
                self.rubber_band = QGraphicsRectItem(x, y, 0, 0)
                self.rubber_band.setPen(QPen(QColor(138, 43, 226), 2))
                self.rubber_band.setBrush(QBrush(QColor(138, 43, 226, 50)))
                self.addItem(self.rubber_band)
                self.status_message.emit("框选模式: Shift+拖动选择区域")
            else:
                # 普通点击：只在mouseReleaseEvent中处理，不要在这里发送信号
                pass
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.is_drawing and self.rubber_band:
            pos = event.scenePos()
            sx, sy = self.start_pos.x(), self.start_pos.y()
            ex, ey = pos.x(), pos.y()
            x = min(sx, ex)
            y = min(sy, ey)
            w = abs(ex - sx)
            h = abs(ey - sy)
            # 精确跟随鼠标，不加偏移
            self.rubber_band.setRect(x, y, w, h)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.is_drawing and self.rubber_band:
            self.is_drawing = False
            pos = event.scenePos()
            sx, sy = self.start_pos.x(), self.start_pos.y()
            ex, ey = pos.x(), pos.y()
            c1 = int(min(sx, ex) // self.cell_size)
            r1 = int(min(sy, ey) // self.cell_size)
            c2 = int(max(sx, ex) // self.cell_size)
            r2 = int(max(sy, ey) // self.cell_size)
            r1, c1 = max(0, r1), max(0, c1)
            r2, c2 = min(self.grid_height - 1, r2), min(self.grid_width - 1, c2)
            if r2 >= r1 and c2 >= c1:
                self.area_selected.emit(r1, c1, r2, c2)
            self.removeItem(self.rubber_band)
            self.rubber_band = None
        else:
            # 单格点击：在mouseReleaseEvent中处理
            pos = event.scenePos()
            col = int(pos.x() // self.cell_size)
            row = int(pos.y() // self.cell_size)
            if 0 <= row < self.grid_height and 0 <= col < self.grid_width:
                self.cell_clicked.emit(row, col, self.cells.get((row, col)))
        super().mouseReleaseEvent(event)

    def clear_highlights(self):
        """清除所有高亮和淡化效果"""
        self._dimmed_codes.clear()
        for item in self.cells.values():
            item.set_highlight(False)
            item.restore_original()

    def dim_all_except(self, codes_to_keep):
        """淡化除指定颜色外的所有颜色"""
        self._dimmed_codes = set()
        for item in self.cells.values():
            if item.code not in codes_to_keep:
                item.set_dimmed(True)
                self._dimmed_codes.add(item.code)
            else:
                item.restore_original()

    def restore_dimmed(self):
        """恢复被淡化的颜色"""
        for item in self.cells.values():
            if item.code in self._dimmed_codes:
                item.restore_original()
        self._dimmed_codes.clear()

    def highlight_color(self, code, mode='row'):
        """
        高亮显示指定颜色的所有格子，并返回分组信息
        序号在每个连续段内从1开始递增
        """
        # 先清除现有高亮，但保留淡化效果
        for item in self.cells.values():
            item.set_highlight(False)

        if mode == 'row':
            return self._highlight_row_mode(code)
        else:
            return self._highlight_col_mode(code)

    def _highlight_row_mode(self, code):
        """行聚集模式高亮"""
        groups = []
        for r in range(self.grid_height):
            c = 0
            while c < self.grid_width:
                if (r, c) in self.cells and self.cells[(r, c)].code == code:
                    start_c = c
                    while c < self.grid_width and (r, c) in self.cells and self.cells[(r, c)].code == code:
                        c += 1
                    end_c = c - 1
                    groups.append((r, start_c, end_c))
                else:
                    c += 1

        if not groups:
            return f"颜色 {code} 未找到"

        code_rgb = self.beads.get(code, (128, 128, 128))
        if not isinstance(code_rgb, (tuple, list)) or len(code_rgb) != 3:
            code_rgb = (128, 128, 128)
        contrast_color = self._get_contrast_color(code_rgb)

        # 每个连续段内从1开始编号
        for r, start_c, end_c in groups:
            segment_num = 1  # 每段从1开始
            for c in range(start_c, end_c + 1):
                if (r, c) in self.cells:
                    self.cells[(r, c)].set_highlight(True, segment_num, contrast_color)
                    segment_num += 1

        region_text = f"颜色 {code} 共有 {len(groups)} 个行段:"
        for idx, (row, start_c, end_c) in enumerate(groups, 1):
            count = end_c - start_c + 1
            region_text += f"\n  段{idx}: 行{row}, 列{start_c}-{end_c}, 共{count}格"
        return region_text

    def _highlight_col_mode(self, code):
        """列聚集模式高亮"""
        groups = []
        for c in range(self.grid_width):
            r = 0
            while r < self.grid_height:
                if (r, c) in self.cells and self.cells[(r, c)].code == code:
                    start_r = r
                    while r < self.grid_height and (r, c) in self.cells and self.cells[(r, c)].code == code:
                        r += 1
                    end_r = r - 1
                    groups.append((c, start_r, end_r))
                else:
                    r += 1

        if not groups:
            return f"颜色 {code} 未找到"

        code_rgb = self.beads.get(code, (128, 128, 128))
        if not isinstance(code_rgb, (tuple, list)) or len(code_rgb) != 3:
            code_rgb = (128, 128, 128)
        contrast_color = self._get_contrast_color(code_rgb)

        # 每个连续段内从1开始编号
        for c, start_r, end_r in groups:
            segment_num = 1  # 每段从1开始
            for r in range(start_r, end_r + 1):
                if (r, c) in self.cells:
                    self.cells[(r, c)].set_highlight(True, segment_num, contrast_color)
                    segment_num += 1

        region_text = f"颜色 {code} 共有 {len(groups)} 个列段:"
        for idx, (col, start_r, end_r) in enumerate(groups, 1):
            count = end_r - start_r + 1
            region_text += f"\n  段{idx}: 列{col}, 行{start_r}-{end_r}, 共{count}格"
        return region_text

    def _get_contrast_color(self, rgb):
        """获取对比色"""
        brightness = rgb[0] * 0.299 + rgb[1] * 0.587 + rgb[2] * 0.114
        if brightness > 128:
            return QColor(0, 0, 0)
        else:
            return QColor(255, 255, 0)


# =============================================================================
# 自定义图形视图（处理右键菜单）
# =============================================================================

class ClickableBeadView(QGraphicsView):
    def __init__(self, scene, window):
        super().__init__(scene)
        self.bead_scene = scene
        self.window = window
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setRenderHint(QPainter.Antialiasing)

    def contextMenuEvent(self, event):
        if self.bead_scene.grid_height == 0:
            return
        pos = self.mapToScene(event.pos())
        col = int(pos.x() // self.bead_scene.cell_size)
        row = int(pos.y() // self.bead_scene.cell_size)
        if 0 <= row < self.bead_scene.grid_height and 0 <= col < self.bead_scene.grid_width:
            item = self.bead_scene.cells.get((row, col))
            if item:
                code, ok = QInputDialog.getItem(
                    self, "快速改色",
                    f"格子 ({row},{col}) 当前: {item.code}\n选择新色号:",
                    sorted(self.bead_scene.beads.keys()), 0, False
                )
                if ok and code:
                    self.window._change_cell(row, col, code)
                return
        super().contextMenuEvent(event)


# =============================================================================
# 辅助拼豆面板
# =============================================================================

class BeadAssistDialog(QDialog):
    """辅助拼豆独立弹窗"""
    color_selected = pyqtSignal(str)
    completed_changed = pyqtSignal(str)
    cell_clicked_in_assist = pyqtSignal(str)  # 在辅助模式下点击格子时发送色号
    exit_requested = pyqtSignal()  # 退出辅助模式信号

    def __init__(self, scene, beads, parent=None):
        super().__init__(parent)
        self.setWindowTitle("辅助拼豆")
        self.setModal(False)  # 非模态对话框，不阻塞主窗口
        self.resize(320, 600)  # 默认尺寸，拉长放大
        
        self.scene = scene
        self.beads = beads
        self.completed_colors = set()
        self.current_mode = 'row'
        self.selected_code = None
        self.is_active = False
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(12, 12, 12, 12)
        
        # 设置窗口属性
        self.setWindowFlags(Qt.Dialog | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)

        # 标题栏
        title_layout = QHBoxLayout()
        title = QLabel("辅助拼豆")
        title.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            color: #7B1FA2;
            font-family: 'Microsoft YaHei', 'Segoe UI', Arial, sans-serif;
        """)
        title_layout.addWidget(title)

        # 退出按钮
        self.btn_exit = QPushButton("退出")
        self.btn_exit.setStyleSheet("""
            QPushButton {
                background: #E0E0E0;
                color: #666;
                border: none;
                border-radius: 4px;
                padding: 4px 12px;
                font-size: 11px;
                font-family: 'Microsoft YaHei', 'Segoe UI', Arial, sans-serif;
            }
            QPushButton:hover {
                background: #BDBDBD;
            }
        """)
        self.btn_exit.clicked.connect(self._on_exit)
        # 始终显示退出按钮
        title_layout.addWidget(self.btn_exit)
        layout.addLayout(title_layout)

        # 聚集模式选择
        mode_group = QGroupBox("聚集模式")
        mode_group.setStyleSheet("""
            QGroupBox {
                font-size: 12px;
                font-weight: bold;
                color: #6A1B9A;
                border: 1px solid #E1BEE7;
                border-radius: 6px;
                margin-top: 8px;
                padding: 8px;
                background: #FAFAFA;
                font-family: 'Microsoft YaHei', 'Segoe UI', Arial, sans-serif;
            }
            QRadioButton {
                color: #4A148C;
                font-size: 12px;
                font-family: 'Microsoft YaHei', 'Segoe UI', Arial, sans-serif;
                padding: 2px;
            }
            QRadioButton::indicator {
                width: 14px;
                height: 14px;
                border-radius: 7px;
            }
            QRadioButton::indicator:checked {
                background: #9C27B0;
                border: 2px solid white;
            }
        """)
        mode_layout = QHBoxLayout()
        mode_layout.setSpacing(15)

        self.btn_row_mode = QRadioButton("行聚集")
        self.btn_row_mode.setChecked(True)
        self.btn_row_mode.toggled.connect(lambda: self._set_mode('row'))
        self.btn_col_mode = QRadioButton("列聚集")
        self.btn_col_mode.toggled.connect(lambda: self._set_mode('col'))
        mode_layout.addWidget(self.btn_row_mode)
        mode_layout.addWidget(self.btn_col_mode)
        mode_layout.addStretch()

        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # 颜色列表
        list_label = QLabel("颜色列表 (按数量排序)")
        list_label.setStyleSheet("""
            color: #666;
            font-size: 12px;
            font-family: 'Microsoft YaHei', 'Segoe UI', Arial, sans-serif;
        """)
        layout.addWidget(list_label)

        self.color_list = QListWidget()
        self.color_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.color_list.itemClicked.connect(self._on_color_clicked)
        self.color_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #E1BEE7;
                border-radius: 6px;
                background: #F5F5F5;
                padding: 4px;
                font-family: 'Microsoft YaHei', 'Segoe UI', Arial, sans-serif;
            }
            QListWidget::item {
                padding: 8px;
                border-radius: 4px;
                margin: 2px 0;
                border: 1px solid rgba(0,0,0,0.1);
            }
            QListWidget::item:selected {
                background: #9C27B0;
                color: white;
                border: 2px solid #4A148C;
            }
            QListWidget::item:hover {
                background: #E1BEE7;
                border: 1px solid #9C27B0;
            }
        """)
        layout.addWidget(self.color_list, 1)

        # 区域信息
        self.region_info = QTextEdit()
        self.region_info.setReadOnly(True)
        self.region_info.setMaximumHeight(100)
        self.region_info.setStyleSheet("""
            QTextEdit {
                background: #F3E5F5;
                border: 1px solid #E1BEE7;
                border-radius: 6px;
                color: #4A148C;
                font-size: 11px;
                padding: 8px;
                font-family: 'Microsoft YaHei', 'Segoe UI', Arial, sans-serif;
                line-height: 1.4;
            }
        """)
        layout.addWidget(self.region_info)

        # 按钮区
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)

        self.btn_complete = QPushButton("标记完成")
        self.btn_complete.setStyleSheet(self._gradient_btn("#7B1FA2", "#4A148C"))
        self.btn_complete.clicked.connect(self._on_complete)
        self.btn_complete.setEnabled(False)
        self.btn_complete.setMinimumHeight(40)
        btn_layout.addWidget(self.btn_complete)

        self.btn_reset = QPushButton("重置")
        self.btn_reset.setStyleSheet(self._gradient_btn("#FF9800", "#F57C00"))
        self.btn_reset.clicked.connect(self._on_reset)
        self.btn_reset.setMinimumHeight(40)
        btn_layout.addWidget(self.btn_reset)

        layout.addLayout(btn_layout)

        # 已完成颜色
        completed_label = QLabel("已完成")
        completed_label.setStyleSheet("""
            color: #888;
            font-size: 12px;
            font-family: 'Microsoft YaHei', 'Segoe UI', Arial, sans-serif;
        """)
        layout.addWidget(completed_label)

        self.completed_label = QLabel("无")
        self.completed_label.setStyleSheet("""
            color: #4CAF50;
            font-size: 12px;
            font-weight: bold;
            padding: 4px;
            background: #E8F5E9;
            border-radius: 4px;
            font-family: 'Microsoft YaHei', 'Segoe UI', Arial, sans-serif;
        """)
        self.completed_label.setWordWrap(True)
        layout.addWidget(self.completed_label)

        self.setLayout(layout)

    def _gradient_btn(self, color1, color2):
        return f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 {color1}, stop:1 {color2});
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 14px;
                font-family: 'Microsoft YaHei', 'Segoe UI', Arial, sans-serif;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 {color2}, stop:1 {color1});
            }}
            QPushButton:disabled {{
                background: #E0E0E0;
                color: #9E9E9E;
            }}
        """

    def activate(self):
        """激活辅助模式，显示弹窗"""
        self.is_active = True
        self.show()
        self.raise_()
        self.activateWindow()


    def _on_exit(self):
        """退出辅助模式"""
        self.is_active = False
        self.scene.clear_highlights()
        self.selected_code = None
        self.exit_requested.emit()
        self.close()  # 关闭弹窗

    def _set_mode(self, mode):
        self.current_mode = mode
        if self.selected_code and self.is_active:
            self._apply_highlight(self.selected_code)

    def refresh_colors(self, grid):
        self.color_list.clear()
        if not grid:
            return

        counts = Counter(code for row in grid for code in row)
        sorted_colors = sorted(counts.items(), key=lambda x: -x[1])

        for code, count in sorted_colors:
            rgb = self.beads.get(code, (200, 200, 200))
            if not isinstance(rgb, (tuple, list)) or len(rgb) != 3:
                rgb = (200, 200, 200)

            item = QListWidgetItem(f"{code} ({count}个)")
            item.setData(Qt.UserRole, code)

            color = QColor(*rgb)
            item.setBackground(color)
            # 统一用深灰色文字，确保在各种背景色下都清晰可读
            text_color = QColor(60, 60, 60)  # 深灰色
            item.setForeground(text_color)

            if code in self.completed_colors:
                item.setText(f"{code} ({count}个) ✓")

            self.color_list.addItem(item)

    def _on_color_clicked(self, item):
        if not self.is_active:
            self.activate()
        code = item.data(Qt.UserRole)
        if not code:
            return

        self.selected_code = code
        self._apply_highlight(code)
        self.btn_complete.setEnabled(True)
        self.color_selected.emit(code)

    def _apply_highlight(self, code):
        if not code:
            return
        codes_to_keep = {code} | self.completed_colors
        self.scene.dim_all_except(codes_to_keep)
        region_text = self.scene.highlight_color(code, mode=self.current_mode)
        self.region_info.setPlainText(region_text if region_text else f"颜色 {code} 未找到")
        self.update()

    def _on_complete(self):
        current_item = self.color_list.currentItem()
        if not current_item:
            return
        code = current_item.data(Qt.UserRole)
        if code and code not in self.completed_colors:
            self.completed_colors.add(code)
            self.completed_changed.emit(code)

            current_item.setText(f"{code} ({self._get_count(code)}个) ✓")
            completed = ", ".join(sorted(self.completed_colors)) if self.completed_colors else "无"
            self.completed_label.setText(completed)

            # 自动选择下一个未完成的颜色
            self._select_next_incomplete_color()

            # 强制刷新场景
            self.scene.update()
            if hasattr(self.color_list, 'viewport'):
                self.color_list.viewport().update()

    def _select_next_incomplete_color(self):
        """自动选择下一个未标记完成的颜色"""
        for i in range(self.color_list.count()):
            item = self.color_list.item(i)
            item_code = item.data(Qt.UserRole)
            if item_code and item_code not in self.completed_colors:
                # 找到下一个未完成的颜色，自动选中它
                self.color_list.setCurrentItem(item)
                self._on_color_clicked(item)
                break
        else:
            # 所有颜色都已完成
            self.selected_code = None
            self.btn_complete.setEnabled(False)
            self.scene.clear_highlights()


    def _get_count(self, code):
        for i in range(self.color_list.count()):
            item = self.color_list.item(i)
            if item.data(Qt.UserRole) == code:
                text = item.text()
                import re
                match = re.search(r'\((\d+)个\)', text)
                if match:
                    return int(match.group(1))
        return 0

    def jump_to_color(self, code):
        """跳转到指定颜色，在列表中选中并高亮"""
        if not code:
            return
        # 在列表中找到对应颜色
        for i in range(self.color_list.count()):
            item = self.color_list.item(i)
            if item.data(Qt.UserRole) == code:
                # 选中该项
                self.color_list.setCurrentItem(item)
                # 滚动到该项
                self.color_list.scrollToItem(item)
                # 自动激活并应用高亮
                if not self.is_active:
                    self.activate()
                self._on_color_clicked(item)
                # 发送信号让主窗口更新 selected_label
                self.cell_clicked_in_assist.emit(code)
                break

    def _on_reset(self):
        """重置完成状态"""
        self.completed_colors.clear()
        self.completed_label.setText("无")
        self.scene.clear_highlights()
        # 保存当前的 grid 引用用于刷新（必须在清空前保存）
        grid_to_refresh = getattr(self, '_grid', None)
        if self.selected_code:
            self._apply_highlight(self.selected_code)
        # 刷新列表
        if grid_to_refresh is not None:
            self.refresh_colors(grid_to_refresh)

    def reset_completed(self):
        """重置已完成状态"""
        self.completed_colors.clear()
        self.completed_label.setText("无")
        self.scene.clear_highlights()
        # 保存 grid 引用用于刷新（先保存再操作）
        grid_to_refresh = getattr(self, '_grid', None)
        # 刷新列表，取消所有对号
        if grid_to_refresh is not None:
            self.refresh_colors(grid_to_refresh)
        if self.selected_code:
            self._apply_highlight(self.selected_code)


# =============================================================================
# 色号板面板
# =============================================================================

class ColorPalettePanel(QWidget):
    color_selected = pyqtSignal(str)

    def __init__(self, beads):
        super().__init__()
        self.beads = beads
        self.selected_code = None
        self.code_buttons = {}
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # 输出目录标签（上方）
        self.output_dir_label = QLabel("输出: output")
        self.output_dir_label.setStyleSheet("""
            font-size: 11px;
            color: #666;
            font-family: 'Microsoft YaHei', 'Segoe UI', Arial, sans-serif;
            padding: 2px 4px;
            background: #F5F5F5;
            border-radius: 4px;
        """)
        self.output_dir_label.setWordWrap(True)
        layout.addWidget(self.output_dir_label)

        # 色号板标题（下方）
        title = QLabel("色号板")
        title.setStyleSheet("""
            font-weight: bold;
            font-size: 15px;
            color: #7B1FA2;
            font-family: 'Microsoft YaHei', 'Segoe UI', Arial, sans-serif;
            padding: 4px;
        """)
        layout.addWidget(title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        container = QWidget()
        grid = QGridLayout()
        grid.setSpacing(4)
        container.setLayout(grid)

        sorted_codes = sorted(self.beads.keys())
        cols = 4
        for idx, code in enumerate(sorted_codes):
            r, c = divmod(idx, cols)
            btn = QPushButton(code)
            rgb = self.beads.get(code, (200, 200, 200))
            if not isinstance(rgb, (tuple, list)) or len(rgb) != 3:
                rgb = (200, 200, 200)
            brightness = (rgb[0] * 299 + rgb[1] * 587 + rgb[2] * 114) / 1000
            text_color = "white" if brightness < 128 else "black"
            btn.setStyleSheet(
                f"QPushButton {{ background: rgb({rgb[0]},{rgb[1]},{rgb[2]}); "
                f"border: 1px solid #BDBDBD; border-radius: 6px; "
                f"color: {text_color}; font-size: 10px; font-family: 'Segoe UI', Arial; }}"
                f"QPushButton:hover {{ border: 2px solid #9C27B0; }}"
            )
            btn.setFixedSize(44, 40)
            btn.setToolTip(code)
            btn.clicked.connect(lambda checked, code=code: self._on_select(code))
            grid.addWidget(btn, r, c)
            self.code_buttons[code] = btn

        scroll.setWidget(container)
        # 限制滚动区域的最大高度（约显示10行按钮）
        scroll.setMaximumHeight(500)
        layout.addWidget(scroll)

        self.selected_label = QLabel("未选择")
        self.selected_label.setStyleSheet(
            "QLabel { background: #F3E5F5; border: 2px solid #9C27B0; "
            "border-radius: 8px; padding: 8px; font-weight: bold; color: #4A148C; "
            "font-family: 'Microsoft YaHei', 'Segoe UI', Arial, sans-serif; "
            "min-height: 36px; }"
        )
        self.selected_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.selected_label)
        self.setLayout(layout)
        self.setMinimumWidth(190)
        self.setMaximumWidth(210)

    def _on_select(self, code):
        self.selected_code = code
        rgb = self.beads.get(code, (200, 200, 200))
        self.selected_label.setText(f"选中: {code}")
        self.selected_label.setStyleSheet(
            f"QLabel {{ background: rgb({rgb[0]},{rgb[1]},{rgb[2]}); "
            f"border: 2px solid #9C27B0; border-radius: 8px; padding: 8px; font-weight: bold; "
            f"font-family: 'Microsoft YaHei', 'Segoe UI', Arial, sans-serif; "
            f"min-height: 36px; }}"
        )
        self.color_selected.emit(code)

    def get_selected(self):
        return self.selected_code


# =============================================================================
# 主窗口
# =============================================================================

class PindouWindow(QWidget):
    def __init__(self):
        super().__init__()

        # 使用全局的APP_ROOT，兼容打包后的路径
        self.work_dir = APP_ROOT
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(os.path.join(self.work_dir, "output"), exist_ok=True)
        os.chdir(self.work_dir)

        # 输出文件夹（用户可自定义）
        self.output_dir = os.path.join(self.work_dir, "output")  # 默认使用同目录output
        self._load_settings()  # 加载用户保存的设置
        self.grid = None
        self.beads = {}
        self.rgb_lut_array = None
        self.current_path = None
        self.current_name = None
        self.has_unsaved = False
        self.cell_size = 28
        # 资源文件路径：优先使用INTERNAL_ROOT（打包后），其次APP_ROOT（开发环境）
        self.current_beads_path = os.path.join(INTERNAL_ROOT, "load/beads_palette_221_correct.json")
        if not os.path.exists(self.current_beads_path):
            self.current_beads_path = os.path.join(APP_ROOT, "load/beads_palette_221_correct.json")
        self.current_npy_path = os.path.join(INTERNAL_ROOT, "load/rgb_to_bead_lut.npy")
        if not os.path.exists(self.current_npy_path):
            self.current_npy_path = os.path.join(APP_ROOT, "load/rgb_to_bead_lut.npy")
        self.current_grid_path = None
        self.current_output_path = None
        self.undo_manager = UndoManager()
        self._init_data()
        self._init_ui()
        self._update_output_dir_label()  # 初始化后更新输出目录标签

    def _init_data(self):
        with open(self.current_beads_path, encoding='utf-8') as f:
            self.beads = json.load(f)
        for k in self.beads:
            self.beads[k] = tuple(self.beads[k])
        self.rgb_lut_array = load_lut_npy(self.current_npy_path)

    def _init_ui(self):
        self.setWindowTitle("拼豆工作台")
        self.resize(1400, 900)
        # 限制窗口最大高度为屏幕高度，确保窗口可以正常显示
        screen = QDesktopWidget().screenGeometry()
        self.setMaximumHeight(screen.height())

        # 设置全局无衬线字体
        font = QFont("Microsoft YaHei", 10)
        font.setStyleHint(QFont.SansSerif)
        QApplication.setFont(font)

        # 全局样式表
        self.setStyleSheet("""
            QWidget {
                background: #F5F3FF;
                font-family: 'Microsoft YaHei', 'Segoe UI', Arial, sans-serif;
            }
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                color: #6A1B9A;
                border: 2px solid #E1BEE7;
                border-radius: 10px;
                margin-top: 12px;
                padding: 12px 8px 8px 8px;
                background: #FAFAFA;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }
            QLabel {
                font-family: 'Microsoft YaHei', 'Segoe UI', Arial, sans-serif;
            }
            QSpinBox {
                border: 1px solid #E1BEE7;
                border-radius: 6px;
                padding: 6px;
                background: white;
                font-family: 'Microsoft YaHei', 'Segoe UI', Arial, sans-serif;
            }
            QSpinBox:focus {
                border: 2px solid #9C27B0;
            }
            QCheckBox {
                color: #4A148C;
                font-size: 13px;
                font-family: 'Microsoft YaHei', 'Segoe UI', Arial, sans-serif;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 4px;
            }
            QCheckBox::indicator:checked {
                background: #9C27B0;
            }
        """)

        main_layout = QHBoxLayout()
        main_layout.setSpacing(12)

        # ========== 左侧面板 ==========
        left_panel = QVBoxLayout()
        left_panel.setSpacing(10)

        # ---- 生成参数区 ----
        gen_group = QGroupBox("图案生成")
        gen_layout = QGridLayout()
        gen_layout.setSpacing(12)

        self.btn_select = QPushButton("选择图片")
        self.btn_select.setStyleSheet(self._main_btn_style("#9C27B0", "#7B1FA2"))
        self.btn_select.clicked.connect(self.on_select_image)
        self.btn_select.setMinimumHeight(44)

        self.btn_load_grid = QPushButton("加载图纸")
        self.btn_load_grid.setStyleSheet(self._main_btn_style("#7B1FA2", "#512DA8"))
        self.btn_load_grid.clicked.connect(self.on_load_grid)
        self.btn_load_grid.setMinimumHeight(44)

        gen_layout.addWidget(self.btn_select, 0, 0)
        gen_layout.addWidget(self.btn_load_grid, 0, 1)

        gen_layout.addWidget(QLabel("压缩颜色数:"), 1, 0)
        self.spin_colors = QSpinBox()
        self.spin_colors.setRange(1, 256)
        self.spin_colors.setValue(30)
        gen_layout.addWidget(self.spin_colors, 1, 1)

        gen_layout.addWidget(QLabel("图纸大小 N×N:"), 2, 0)
        self.spin_n = QSpinBox()
        self.spin_n.setRange(10, 1000)
        self.spin_n.setValue(100)
        gen_layout.addWidget(self.spin_n, 2, 1)

        self.chk_sharp = QCheckBox("使用锐化")
        self.chk_sharp.setChecked(True)
        gen_layout.addWidget(self.chk_sharp, 3, 0, 1, 2)

        self.btn_generate = QPushButton("生成图纸")
        self.btn_generate.setStyleSheet(self._main_btn_style("#4CAF50", "#388E3C"))
        self.btn_generate.clicked.connect(self.on_generate)
        self.btn_generate.setEnabled(False)
        self.btn_generate.setMinimumHeight(48)

        # 输出目录按钮（和生成图纸并排，靠右，稍窄）
        self.btn_output_dir = QPushButton("输出目录")
        self.btn_output_dir.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #9C27B0, stop:1 #7B1FA2);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 16px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #AB47BC, stop:1 #8E24AA);
            }
        """)
        self.btn_output_dir.clicked.connect(self._on_select_output_dir)
        self.btn_output_dir.setMinimumHeight(48)

        gen_layout.addWidget(self.btn_generate, 4, 0)
        gen_layout.addWidget(self.btn_output_dir, 4, 1)

        gen_group.setLayout(gen_layout)
        left_panel.addWidget(gen_group)

        # ---- 工具栏 ----
        self.toolbar = QToolBar("工具")
        self.toolbar.setMovable(False)
        self.toolbar.setStyleSheet("""
            QToolBar {
                background: #F3E5F5;
                border: 1px solid #E1BEE7;
                border-radius: 8px;
                padding: 6px;
                spacing: 8px;
            }
            QToolBar::separator {
                background: #E1BEE7;
                width: 1px;
                margin: 4px 8px;
            }
        """)

        self.action_select = QAction("单选", self)
        self.action_select.setCheckable(True)
        self.action_select.setChecked(True)
        self.action_select.triggered.connect(lambda: self._set_tool('select'))

        self.action_area = QAction("框选", self)
        self.action_area.setCheckable(True)
        self.action_area.triggered.connect(lambda: self._set_tool('area'))

        self.action_eyedropper = QAction("吸取", self)
        self.action_eyedropper.setCheckable(True)
        self.action_eyedropper.triggered.connect(lambda: self._set_tool('eyedropper'))

        self.action_undo = QAction("撤销", self)
        self.action_undo.triggered.connect(self.on_undo)
        self.action_undo.setEnabled(False)
        self.action_undo.setShortcut("Ctrl+Z")

        self.action_redo = QAction("重做", self)
        self.action_redo.triggered.connect(self.on_redo)
        self.action_redo.setEnabled(False)
        self.action_redo.setShortcut("Ctrl+Y")

        self.action_save = QAction("保存", self)
        self.action_save.triggered.connect(self.on_save)
        self.action_save.setEnabled(False)

        self.action_mirror = QAction("保存镜像", self)
        self.action_mirror.triggered.connect(self.on_save_mirror)
        self.action_mirror.setEnabled(False)

        self.toolbar.addAction(self.action_select)
        self.toolbar.addAction(self.action_area)
        self.toolbar.addAction(self.action_eyedropper)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.action_undo)
        self.toolbar.addAction(self.action_redo)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.action_save)
        self.toolbar.addAction(self.action_mirror)
        self.toolbar.addSeparator()

        # 全屏动作
        self.action_fullscreen = QAction("全屏", self)
        self.action_fullscreen.setCheckable(True)
        self.action_fullscreen.setShortcut("F11")
        self.action_fullscreen.triggered.connect(self._toggle_fullscreen)
        self.toolbar.addAction(self.action_fullscreen)

        left_panel.addWidget(self.toolbar)

        # ---- 画布 ----
        self.scene = BeadScene(cell_size=self.cell_size)
        self.scene.cell_clicked.connect(self.on_cell_clicked)
        self.scene.area_selected.connect(self.on_area_selected)
        self.scene.eyedropper_picked.connect(self.on_eyedropper_color_picked)
        self.scene.status_message.connect(self._on_status_message)
        self.view = ClickableBeadView(self.scene, self)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.view.wheelEvent = self._wheel_zoom
        self.view.setStyleSheet("""
            QGraphicsView {
                border: 2px solid #E1BEE7;
                border-radius: 10px;
                background: white;
            }
        """)

        # 占位提示
        self.placeholder = QLabel("请选择图片生成图纸\n或加载已有图纸")
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder.setStyleSheet("""
            color: #BDBDBD;
            font-size: 20px;
            font-weight: bold;
            border: 3px dashed #E1BEE7;
            border-radius: 16px;
            background: #FAFAFA;
            padding: 80px;
            font-family: 'Microsoft YaHei', 'Segoe UI', Arial, sans-serif;
        """)
        self.placeholder.setFixedSize(900, 700)
        self.placeholder.setParent(self.view)
        self.placeholder.move(0, 0)
        self.placeholder.show()

        left_panel.addWidget(self.view, 1)

        # ---- 状态栏 ----
        self.statusbar = QLabel("就绪")
        self.statusbar.setStyleSheet("""
            color: #6A1B9A;
            padding: 8px 12px;
            background: #F3E5F5;
            border-radius: 6px;
            font-weight: bold;
            font-size: 13px;
        """)
        left_panel.addWidget(self.statusbar)

        # ========== 右侧面板 ==========
        right_panel = QVBoxLayout()
        right_panel.setSpacing(10)

        # 辅助拼豆按钮（打开独立弹窗）
        self.btn_assist = QPushButton("辅助拼豆")
        self.btn_assist.setStyleSheet(self._main_btn_style("#9C27B0", "#7B1FA2"))
        self.btn_assist.clicked.connect(self._on_open_assist)
        self.btn_assist.setMinimumHeight(45)
        right_panel.addWidget(self.btn_assist)

        # 色号板
        self.palette = ColorPalettePanel(self.beads)
        self.palette.color_selected.connect(self.on_palette_color_selected)
        self.palette.setStyleSheet("""
            QWidget {
                background: white;
                border: 1px solid #E1BEE7;
                border-radius: 10px;
                padding: 4px;
            }
        """)
        right_panel.addWidget(self.palette)
        
        # 辅助拼豆弹窗（按需创建）
        self.assist_dialog = None

        # 组装
        splitter = QSplitter(Qt.Horizontal)
        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        left_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        right_widget.setMaximumWidth(250)
        right_widget.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.MinimumExpanding)

        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

    def _main_btn_style(self, color1, color2):
        return f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 {color1}, stop:1 {color2});
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 14px;
                font-family: 'Microsoft YaHei', 'Segoe UI', Arial, sans-serif;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 {color2}, stop:1 {color1});
            }}
            QPushButton:disabled {{
                background: #E0E0E0;
                color: #9E9E9E;
            }}
        """

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'placeholder') and self.placeholder.isVisible():
            self.placeholder.setFixedSize(self.view.viewport().size())

    def _set_tool(self, tool):
        self.scene.tool_mode = tool
        if tool == 'select':
            self.action_select.setChecked(True)
            self.action_area.setChecked(False)
            self.action_eyedropper.setChecked(False)
            self.view.setDragMode(QGraphicsView.ScrollHandDrag)
            self.view.setCursor(QCursor(Qt.ArrowCursor))
            self.statusbar.setText("工具: 单选模式")
        elif tool == 'area':
            self.action_select.setChecked(False)
            self.action_area.setChecked(True)
            self.action_eyedropper.setChecked(False)
            self.view.setDragMode(QGraphicsView.NoDrag)
            self.view.setCursor(QCursor(Qt.CrossCursor))
            self.statusbar.setText("工具: 框选模式 (Shift+拖拽)")
        elif tool == 'eyedropper':
            self.action_select.setChecked(False)
            self.action_area.setChecked(False)
            self.action_eyedropper.setChecked(True)
            self.view.setDragMode(QGraphicsView.NoDrag)
            self.view.setCursor(QCursor(Qt.CrossCursor))
            self.statusbar.setText("工具: 吸取模式")

    def _wheel_zoom(self, event):
        modifiers = QApplication.keyboardModifiers()
        delta = event.angleDelta()

        if modifiers & Qt.ControlModifier:
            # Ctrl+滚轮 = 缩放
            factor = 1.15 if delta.y() > 0 else 1 / 1.15
            self.view.scale(factor, factor)
        elif modifiers & Qt.ShiftModifier:
            # Shift+滚轮 = 左右滚动（Windows风格：往左滚向左移）
            scroll = -delta.y() * 2  # 往上滚向左
            self.view.horizontalScrollBar().setValue(
                self.view.horizontalScrollBar().value() + scroll
            )
        else:
            # 纯滚轮 = 上下滚动（Windows风格：往上滚向上移）
            scroll = -delta.y() * 2
            self.view.verticalScrollBar().setValue(
                self.view.verticalScrollBar().value() + scroll
            )

    def _on_open_assist(self):
        """打开辅助拼豆弹窗"""
        if self.grid is None:
            ModernDialog.warning("提示", "请先加载或生成图纸", self)
            return
        # 创建弹窗（如果还没有）
        if self.assist_dialog is None:
            self.assist_dialog = BeadAssistDialog(self.scene, self.beads, self)
            self.assist_dialog.exit_requested.connect(self._on_assist_exit)
            self.assist_dialog.cell_clicked_in_assist.connect(self._on_assist_cell_clicked)
        # 更新颜色数据
        self.assist_dialog._grid = self.grid
        self.assist_dialog.refresh_colors(self.grid)
        self.assist_dialog.reset_completed()
        # 显示弹窗
        self.assist_dialog.activate()
        self.statusbar.setText("已进入辅助拼豆模式")

    def _on_assist_cell_clicked(self, code):
        """辅助模式下点击格子，更新选中颜色显示"""
        self.palette._on_select(code)
        self.palette.selected_label.setText(f"选中: {code}")
        rgb = self.beads.get(code, (200, 200, 200))
        self.palette.selected_label.setStyleSheet(
            f"QLabel {{ background: rgb({rgb[0]},{rgb[1]},{rgb[2]}); "
            f"border: 2px solid #9C27B0; border-radius: 8px; padding: 8px; font-weight: bold; "
            f"font-family: 'Microsoft YaHei', 'Segoe UI', Arial, sans-serif; "
            f"min-height: 36px; }}"
        )

    def _on_assist_exit(self):
        """辅助模式退出回调"""
        self.statusbar.setText("已退出辅助拼豆模式")

    # ---- 事件处理 ----

    def on_select_image(self):
        if self.has_unsaved:
            if not ModernDialog.question("警告", "当前有未保存的修改，确定要放弃吗？", self):
                return

        fname, _ = QFileDialog.getOpenFileName(
            self, "选择图片", self.work_dir, "图片文件 (*.jpg *.png *.jpeg *.bmp)"
        )
        if not fname:
            return

        self.current_path = fname
        self.current_name = os.path.splitext(os.path.basename(fname))[0]
        self.btn_generate.setEnabled(True)
        self.action_mirror.setEnabled(False)
        self._clear_canvas()
        self.statusbar.setText(f"已选择图片: {fname}")
        ModernDialog.info("提示", f"图片已加载：{os.path.basename(fname)}\n点击「生成图纸」继续。", self)

    def on_load_grid(self):
        """从output文件夹加载图纸图片，自动映射JSON文件"""
        if self.has_unsaved:
            if not ModernDialog.question("警告", "当前有未保存的修改，确定要放弃吗？", self):
                return

        # 使用用户选择的输出目录（self.output_dir），而非硬编码的"output"
        output_dir = self.output_dir

        # 筛选processed.png文件
        fname, _ = QFileDialog.getOpenFileName(
            self, "选择图纸图片", output_dir,
            "图纸图片 (*_processed.png);;所有图片 (*.png *.jpg *.jpeg)"
        )
        if not fname:
            return

        # 从文件名提取基础名称
        basename = os.path.basename(fname)
        name_without_ext = os.path.splitext(basename)[0]

        # 处理镜像文件
        if '_mirror_processed' in name_without_ext:
            grid_name = name_without_ext.replace('_mirror_processed', '_mirror_grid')
        elif '_processed' in name_without_ext:
            grid_name = name_without_ext.replace('_processed', '_grid')
        else:
            grid_name = name_without_ext + '_grid'

        # 查找对应的JSON文件
        grid_path = os.path.join(output_dir, f"{grid_name}.json")

        if not os.path.exists(grid_path):
            ModernDialog.error("加载失败", f"未找到对应的图纸文件：\n{grid_path}", self)
            return

        try:
            grid = load_grid_from_json(grid_path)

            # 设置当前名称
            self.current_name = name_without_ext.replace('_processed', '').replace('_mirror_processed', '_mirror')
            self.current_path = None

            self.grid = grid
            self.current_grid_path = grid_path
            self.current_output_path = fname

            # 渲染到场景
            self._render_grid_to_scene()
            self._save_undo_state()

            self.has_unsaved = False
            self.action_save.setEnabled(True)
            self.action_mirror.setEnabled(True)
            self._update_undo_buttons()

            # 更新辅助弹窗
            if self.assist_dialog:
                self.assist_dialog._grid = self.grid
                self.assist_dialog.refresh_colors(self.grid)
                self.assist_dialog.reset_completed()

            self.statusbar.setText(f"已加载图纸: {basename} ({len(grid[0])}×{len(grid)})")
            self.placeholder.hide()

            ModernDialog.success("加载成功",
                f"图纸已加载：{basename}\n尺寸: {len(grid[0])}×{len(grid)}\n可以开始编辑了。", self)

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            ModernDialog.error("加载失败", f"加载图纸时出错:\n\n{e}\n\n详细信息:\n{tb}", self)

    def _clear_canvas(self):
        self.scene.clear()
        self.scene.cells.clear()
        self.grid = None
        self.current_grid_path = None
        self.current_output_path = None
        self.undo_manager.clear()
        self._update_undo_buttons()
        self.placeholder.setFixedSize(self.view.viewport().size())
        self.placeholder.show()
        self.view.resetTransform()
        self.has_unsaved = False
        self.action_save.setEnabled(False)
        self.action_mirror.setEnabled(False)

    def on_generate(self):
        if not self.current_path:
            ModernDialog.warning("警告", "请先选择一张图片", self)
            return

        compress_colors = self.spin_colors.value()
        N = self.spin_n.value()
        use_sharp = self.chk_sharp.isChecked()
        self.statusbar.setText("正在生成图纸，请稍候...")
        self.btn_generate.setEnabled(False)
        self.btn_select.setEnabled(False)
        self.btn_load_grid.setEnabled(False)
        QApplication.processEvents()

        self.worker = GenerateWorker(
            self.current_path, self.rgb_lut_array, N, compress_colors, use_sharp
        )
        self.worker.finished_signal.connect(self._on_generate_done)
        self.worker.error_signal.connect(self._on_generate_error)
        self.worker.start()

    def _on_generate_done(self, grid, t_gen):
        self.grid = grid
        self.current_grid_path = os.path.join(self.output_dir, f"{self.current_name}_grid.json")
        self.current_output_path = os.path.join(self.output_dir, f"{self.current_name}_processed.png")

        t0 = time()
        self._render_grid_to_scene()
        t_render = time() - t0

        self._save_undo_state()
        self.has_unsaved = True
        self.action_save.setEnabled(True)
        self.action_mirror.setEnabled(True)
        self._update_undo_buttons()

        self.btn_generate.setEnabled(False)
        self.btn_select.setEnabled(True)
        self.btn_load_grid.setEnabled(True)
        self.placeholder.hide()
        self._set_tool('select')

        # 更新辅助弹窗
        if self.assist_dialog:
            self.assist_dialog._grid = self.grid
            self.assist_dialog.refresh_colors(self.grid)
            self.assist_dialog.reset_completed()

        self.statusbar.setText(
            f"生成完成！尺寸: {len(self.grid[0])}×{len(self.grid)} "
            f"| 快捷键: Ctrl+滚轮=缩放 | 滚轮=上下滚动 | Shift+滚轮=左右滚动"
        )
        ModernDialog.success("完成",
            f"图纸生成完成！\n尺寸: {len(self.grid[0])}×{len(self.grid)}\n"
            f"颜色查找: {t_gen:.2f}s 渲染: {t_render:.2f}s\n"
            f"保存路径: output/\n可以开始编辑了。", self)

    def _on_generate_error(self, error_msg, tb):
        self.btn_generate.setEnabled(True)
        self.btn_select.setEnabled(True)
        self.btn_load_grid.setEnabled(True)
        self.statusbar.setText(f"生成失败: {error_msg}")
        ModernDialog.error("生成失败", f"生成图纸时出错:\n\n{error_msg}\n\n详细信息:\n{tb}", self)

    def _toggle_fullscreen(self, checked):
        """切换全屏模式"""
        if checked:
            self.showFullScreen()
            self.action_fullscreen.setText("退出全屏")
        else:
            self.showNormal()
            self.action_fullscreen.setText("全屏")

    def _on_select_output_dir(self):
        """选择自定义输出文件夹"""
        # 如果当前output目录有内容，先询问是否复制
        default_output = os.path.join(self.work_dir, "output")
        has_content = os.path.exists(default_output) and os.listdir(default_output)

        # 选择新目录
        new_dir = QFileDialog.getExistingDirectory(
            self, "选择输出文件夹", self.work_dir,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )

        if not new_dir:
            return  # 用户取消

        # 如果新目录不是默认的output，需要复制内容
        if new_dir != default_output and has_content:
            reply = QMessageBox.question(
                self, "复制文件",
                f"是否将默认output文件夹的内容复制到:\n{new_dir}",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                self._copy_output_to_dir(default_output, new_dir)

        self.output_dir = new_dir
        self._update_output_dir_label()  # 更新输出目录标签
        self._save_settings()  # 保存设置
        self.statusbar.setText(f"输出文件夹已设置为: {new_dir}")

    def _update_output_dir_label(self):
        """更新右侧面板的输出目录标签"""
        if hasattr(self, 'palette') and hasattr(self.palette, 'output_dir_label'):
            # 显示简短的路径（取最后一级目录名）
            dir_name = os.path.basename(self.output_dir)
            if not dir_name:  # 如果是根目录
                dir_name = self.output_dir
            self.palette.output_dir_label.setText(f"输出: {dir_name}")
            self.palette.output_dir_label.setToolTip(self.output_dir)  # 鼠标悬停显示完整路径

    def _get_settings_path(self):
        """获取设置文件路径"""
        return os.path.join(APP_ROOT, "settings.json")

    def _save_settings(self):
        """保存用户设置到文件"""
        try:
            settings = {
                "output_dir": self.output_dir
            }
            with open(self._get_settings_path(), 'w', encoding='utf-8') as f:
                json.dump(settings, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存设置失败: {e}")

    def _load_settings(self):
        """加载用户设置"""
        settings_path = self._get_settings_path()
        if os.path.exists(settings_path):
            try:
                with open(settings_path, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                if "output_dir" in settings:
                    saved_dir = settings["output_dir"]
                    # 验证目录是否存在
                    if os.path.isdir(saved_dir):
                        self.output_dir = saved_dir
            except Exception as e:
                print(f"加载设置失败: {e}")

    def _copy_output_to_dir(self, src_dir, dst_dir):
        """复制output目录内容到目标目录"""
        import shutil
        try:
            os.makedirs(dst_dir, exist_ok=True)
            for item in os.listdir(src_dir):
                src_path = os.path.join(src_dir, item)
                dst_path = os.path.join(dst_dir, item)
                if os.path.isfile(src_path):
                    shutil.copy2(src_path, dst_path)
                elif os.path.isdir(src_path):
                    if os.path.exists(dst_path):
                        shutil.rmtree(dst_path)
                    shutil.copytree(src_path, dst_path)
            return True
        except Exception as e:
            ModernDialog.error("复制失败", f"复制文件时出错:\n{e}")
            return False

    def _on_status_message(self, msg):
        """接收场景的状态消息并显示"""
        self.statusbar.setText(msg)

    def _render_grid_to_scene(self):
        if not self.grid or not self.grid[0]:
            raise ValueError("网格为空，无法显示")
        self.scene.clear()
        self.scene.cells.clear()
        self.scene.beads = self.beads
        self.scene.grid_height = len(self.grid)
        self.scene.grid_width = len(self.grid[0]) if self.grid else 0
        for r, row in enumerate(self.grid):
            for c, code in enumerate(row):
                rgb = self.beads.get(code, (200, 200, 200))
                if not isinstance(rgb, (tuple, list)) or len(rgb) != 3:
                    rgb = (200, 200, 200)
                item = BeadCellItem(r, c, code, rgb, self.cell_size)
                self.scene.cells[(r, c)] = item
                self.scene.addItem(item)
        w = self.scene.grid_width * self.cell_size
        h = self.scene.grid_height * self.cell_size
        self.scene.setSceneRect(0, 0, w, h)

    def _save_undo_state(self):
        if self.grid is not None:
            self.undo_manager.save_state(self.grid)
            self._update_undo_buttons()

    def _update_undo_buttons(self):
        self.action_undo.setEnabled(self.undo_manager.can_undo())
        self.action_redo.setEnabled(self.undo_manager.can_redo())

    def on_undo(self):
        old_grid, success = self.undo_manager.undo(self.grid)
        if success:
            self.grid = old_grid
            self._render_grid_to_scene()
            self.has_unsaved = True
            self._update_undo_buttons()
            self.statusbar.setText("已撤销 1 步")
        else:
            self.statusbar.setText("没有可撤销的操作")

    def on_redo(self):
        redo_grid, success = self.undo_manager.redo(self.grid)
        if success:
            self.grid = redo_grid
            self._render_grid_to_scene()
            self.has_unsaved = True
            self._update_undo_buttons()
            self.statusbar.setText("已重做 1 步")
        else:
            self.statusbar.setText("没有可重做的操作")

    def on_cell_clicked(self, row, col, item):
        if item is None:
            return

        # 如果辅助模式激活，则跳转到该颜色
        if self.assist_dialog and self.assist_dialog.is_active:
            self.assist_dialog.jump_to_color(item.code)
            return

        selected_code = self.palette.get_selected()
        if not selected_code:
            code, ok = QInputDialog.getText(
                self, "修改色号",
                f"格子 ({row},{col}) 当前色号: {item.code}\n输入新色号:",
                text=item.code
            )
            if not ok or not code:
                return
            if code not in self.beads:
                ModernDialog.warning("无效色号", f"色号 '{code}' 不存在于色卡中。", self)
                return
        else:
            code = selected_code

        self._change_cell(row, col, code)

    def on_area_selected(self, r1, c1, r2, c2):
        # 如果辅助模式激活，则跳转到该区域第一个格子的颜色
        if self.assist_dialog and self.assist_dialog.is_active:
            item = self.scene.cells.get((r1, c1))
            if item:
                self.assist_dialog.jump_to_color(item.code)
            return
        
        selected_code = self.palette.get_selected()
        if not selected_code:
            code, ok = QInputDialog.getItem(
                self, "批量修改色号",
                f"选中区域: ({r1},{c1}) ~ ({r2},{c2}) "
                f"共 {(r2-r1+1)*(c2-c1+1)} 格\n选择新色号:",
                sorted(self.beads.keys()), 0, False
            )
            if not ok or not code:
                return
        else:
            code = selected_code
        self._change_area(r1, c1, r2, c2, code)

    def _change_cell(self, row, col, code):
        # 在修改之前保存状态，确保只保存一次
        self._save_undo_state()
        
        rgb = self.beads.get(code, (200, 200, 200))
        if not isinstance(rgb, (tuple, list)) or len(rgb) != 3:
            rgb = (200, 200, 200)
        item = self.scene.cells.get((row, col))
        if item:
            item.update_color(code, rgb)
            self.grid[row][col] = code
            self.has_unsaved = True
            self.statusbar.setText(f"已修改 ({row},{col}) → {code}")
            # 更新辅助弹窗
            if self.assist_dialog:
                self.assist_dialog._grid = self.grid
                self.assist_dialog.refresh_colors(self.grid)

    def _change_area(self, r1, c1, r2, c2, code):
        # 在修改之前保存状态，确保只保存一次
        self._save_undo_state()
        
        rgb = self.beads.get(code, (200, 200, 200))
        if not isinstance(rgb, (tuple, list)) or len(rgb) != 3:
            rgb = (200, 200, 200)

        count = 0
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                item = self.scene.cells.get((r, c))
                if item:
                    item.update_color(code, rgb)
                    self.grid[r][c] = code
                    count += 1

        self.has_unsaved = True
        self.statusbar.setText(f"批量修改完成: {count} 格 → {code}")
        # 更新辅助弹窗
        if self.assist_dialog:
            self.assist_dialog._grid = self.grid
            self.assist_dialog.refresh_colors(self.grid)

    def on_palette_color_selected(self, code):
        self.statusbar.setText(f"当前色号: {code}，点击格子即可修改")

    def on_eyedropper_color_picked(self, code):
        self.palette._on_select(code)
        self._set_tool('select')
        self.statusbar.setText(f"已吸取色号: {code}")

    def on_save(self):
        if self.grid is None:
            ModernDialog.warning("提示", "没有可保存的图纸", self)
            return
        try:
            # 确保输出目录存在
            os.makedirs(self.output_dir, exist_ok=True)
            save_grid_to_json(self.grid, self.current_grid_path)
            img = render_pattern(self.grid, self.beads, self.cell_size)
            img.save(self.current_output_path)
            self.has_unsaved = False
            self.statusbar.setText(f"已保存: {self.current_grid_path}")
            ModernDialog.success("保存成功",
                f"grid 已保存:\n{self.current_grid_path}\n\n"
                f"图纸已保存:\n{self.current_output_path}", self)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            ModernDialog.error("保存失败", f"保存时出错:\n\n{e}\n\n详细信息:\n{tb}", self)

    def on_save_mirror(self):
        if self.grid is None:
            ModernDialog.warning("提示", "没有可保存的图纸", self)
            return

        mirror_grid = [row[::-1] for row in self.grid]

        name_part = self.current_name or "pattern"
        mirror_grid_path = os.path.join(self.output_dir, f"{name_part}_mirror_grid.json")
        mirror_output_path = os.path.join(self.output_dir, f"{name_part}_mirror_processed.png")

        try:
            os.makedirs(self.output_dir, exist_ok=True)
            save_grid_to_json(mirror_grid, mirror_grid_path)
            img = render_pattern(mirror_grid, self.beads, self.cell_size)
            img.save(mirror_output_path)
            self.statusbar.setText(f"镜像已保存: {mirror_grid_path}")
            ModernDialog.success("保存成功",
                f"镜像图纸已保存！\n\n"
                f"镜像 grid:\n{mirror_grid_path}\n\n"
                f"镜像图纸:\n{mirror_output_path}", self)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            ModernDialog.error("保存失败", f"保存镜像时出错:\n\n{e}\n\n详细信息:\n{tb}", self)

    def closeEvent(self, event):
        if self.has_unsaved:
            if ModernDialog.question("退出确认", "有未保存的修改，是否保存？", self):
                self.on_save()
                event.accept()
            else:
                event.accept()
        else:
            event.accept()


# =============================================================================
# 程序入口
# =============================================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("fusion")

    # 设置窗口图标（运行时图标）
    icon_path = os.path.join(APP_ROOT, "icon.png")
    if os.path.exists(icon_path):
        app_icon = QIcon(icon_path)
        app.setWindowIcon(app_icon)

    # 设置全局字体
    font = QFont("Microsoft YaHei", 10)
    font.setStyleHint(QFont.SansSerif)
    app.setFont(font)

    # 设置调色板
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(245, 243, 255))
    palette.setColor(QPalette.WindowText, QColor(74, 20, 140))
    palette.setColor(QPalette.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.AlternateBase, QColor(245, 243, 255))
    palette.setColor(QPalette.ToolTipBase, QColor(156, 39, 176))
    palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
    palette.setColor(QPalette.Text, QColor(74, 20, 140))
    palette.setColor(QPalette.Button, QColor(225, 190, 231))
    palette.setColor(QPalette.ButtonText, QColor(74, 20, 140))
    palette.setColor(QPalette.BrightText, QColor(255, 255, 255))
    palette.setColor(QPalette.Highlight, QColor(156, 39, 176))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    window = PindouWindow()
    window.show()
    sys.exit(app.exec_())
