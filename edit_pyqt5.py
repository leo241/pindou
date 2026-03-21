import sys
import json
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QPushButton, QMessageBox, QInputDialog
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from utils.tools import render_pattern, load_grid_from_json


def save_grid_to_json(grid, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(grid, f, ensure_ascii=False, indent=2)


class BeadPatternEditor(QMainWindow):
    def __init__(self, grid, beads, cell=28, grid_filename=None):
        super().__init__()
        self.grid = grid
        self.beads = beads
        self.cell = cell
        self.grid_filename = grid_filename
        self.height = len(grid)
        self.width = len(grid[0]) if self.height > 0 else 0

        self.setWindowTitle("🧶 拼豆图纸编辑器 (PyQt5)")
        self.resize(1200, 900)

        # 中央 widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # GraphicsView + Scene（用于显示图像 + 滚动）
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        # self.view.setRenderHint(self.view.renderHints())  # 抗锯齿（可选）
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)  # 拖拽平移
        self.view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        layout.addWidget(self.view)

        # 保存按钮
        self.save_btn = QPushButton("💾 保存修改")
        self.save_btn.clicked.connect(self.save_grid)
        layout.addWidget(self.save_btn)

        # 初始渲染
        self.redraw()

    def redraw(self):
        img_pil = render_pattern(self.grid, self.beads, self.cell)  # PIL Image
        # 转为 QImage
        img_qt = QImage(
            img_pil.tobytes(), img_pil.width, img_pil.height, QImage.Format_RGB888
        )
        pixmap = QPixmap.fromImage(img_qt)

        self.scene.clear()
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.pixmap_item)
        self.scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())
        self.view.setScene(self.scene)

    def mousePressEvent(self, event):
        # 不在这里处理！因为 view 会拦截
        super().mousePressEvent(event)

    def save_grid(self):
        if not self.grid_filename:
            QMessageBox.critical(self, "错误", "未指定保存文件路径！")
            return
        try:
            save_grid_to_json(self.grid, self.grid_filename)
            QMessageBox.information(self, "成功", f"已保存到:\n{self.grid_filename}")
        except Exception as e:
            QMessageBox.critical(self, "保存失败", str(e))


# --- 自定义 View 以捕获点击 ---
class ClickableGraphicsView(QGraphicsView):
    def __init__(self, editor_instance):
        super().__init__()
        self.editor = editor_instance

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # 获取 scene 坐标
            pos = self.mapToScene(event.pos())
            x, y = pos.x(), pos.y()

            cell = self.editor.cell
            col = int(x // cell)
            row = int(y // cell)

            width = self.editor.width
            height = self.editor.height

            # 只处理主图区域（跳过边缘）
            if 1 <= col <= width and 1 <= row <= height:
                grid_x = col - 1
                grid_y = row - 1
                current_code = self.editor.grid[grid_y][grid_x]

                new_code, ok = QInputDialog.getText(
                    self,
                    "替换色号",
                    f"当前色号: {current_code}\n输入新色号:",
                    text=current_code
                )

                if ok and new_code:
                    if new_code in self.editor.beads:
                        self.editor.grid[grid_y][grid_x] = new_code
                        self.editor.redraw()
                    else:
                        QMessageBox.warning(self, "无效色号", f"色号 '{new_code}' 不存在于色卡中。")
            event.accept()
        else:
            super().mousePressEvent(event)


# ----------------------------
# 启动入口
# ----------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 加载数据
    grid_file = "grid/fox_mirror.json"
    grid = load_grid_from_json(grid_file)

    with open("load/beads_palette_221_correct.json", encoding='utf-8') as f:
        beads = json.load(f)
    for k in beads:
        beads[k] = tuple(beads[k])

    # 创建窗口
    window = BeadPatternEditor(grid, beads, cell=28, grid_filename=grid_file)

    # 替换 view 为可点击版本
    clickable_view = ClickableGraphicsView(window)
    clickable_view.setScene(window.scene)
    # clickable_view.setRenderHint(clickable_view.renderHints())
    clickable_view.setDragMode(QGraphicsView.ScrollHandDrag)
    clickable_view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

    # 替换布局中的 view
    central_layout = window.centralWidget().layout()
    central_layout.replaceWidget(window.view, clickable_view)
    window.view.deleteLater()
    window.view = clickable_view

    window.show()
    sys.exit(app.exec_())