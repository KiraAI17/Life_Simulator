import numpy as np
from PyQt6.QtWidgets import QMainWindow, QLabel, QStatusBar, QToolBar
from PyQt6.QtGui import QPixmap, QImage, QAction
from PyQt6.QtCore import Qt, QTimer

from core.world import World


class MapViewer(QMainWindow):
    MODES = ['composite', 'elevation', 'moisture', 'temperature', 'plants']

    def __init__(self, world: World):
        super().__init__()
        self.world = world
        self.current_mode = 'composite'
        self.scale = 4  # масштаб: 1 пиксель карты = 2x2 пикселя экрана

        self.setWindowTitle("Life Simulator")
        
        # Подгоняем размер окна под карту
        map_width = world.width * self.scale
        map_height = world.height * self.scale
        self.resize(820, 680)

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)  # центрирование
        self.label.setStyleSheet("background-color: black;")
        self.setCentralWidget(self.label)

        view_menu = self.menuBar().addMenu("Вид")
        self.mode_actions = {}
        for mode in self.MODES:
            action = QAction(mode.capitalize(), self)
            action.setCheckable(True)
            action.setChecked(mode == self.current_mode)
            action.triggered.connect(lambda checked, m=mode: self.set_mode(m))
            view_menu.addAction(action)
            self.mode_actions[mode] = action

        toolbar = QToolBar("Tools", self)
        self.addToolBar(toolbar)

        new_map_action = QAction("🎲 Новая карта", self)
        new_map_action.triggered.connect(self.new_map)
        toolbar.addAction(new_map_action)

        self.status = QStatusBar()
        self.setStatusBar(self.status)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(100)

        self.render()

    def set_mode(self, mode: str):
        self.current_mode = mode
        for m, action in self.mode_actions.items():
            action.setChecked(m == mode)
        self.render()

    def new_map(self):
        new_seed = np.random.randint(0, 2**31)
        self.world = World(width=self.world.width, height=self.world.height, seed=new_seed)
        self.render()

    def tick(self):
        self.world.step()
        self.render()

    def render(self):
        rgb = self.world.render(self.current_mode)
        rgb = np.ascontiguousarray(rgb)
        h, w, _ = rgb.shape
        bytes_per_line = 3 * w
        image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        pixmap = QPixmap.fromImage(image)
        
        # Масштабируем с коэффициентом self.scale
        scaled = pixmap.scaled(
            w * self.scale, h * self.scale,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation
        )
        
        self.label.setPixmap(scaled)
        self.status.showMessage(
            f"Tick: {self.world.tick} | Mode: {self.current_mode} | "
            f"Avg plants: {self.world.plants.mean():.3f}"
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.render()