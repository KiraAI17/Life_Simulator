import numpy as np
from PyQt6.QtWidgets import QMainWindow, QStatusBar, QToolBar
from PyQt6.QtGui import QAction
from PyQt6.QtCore import QTimer

from core.world import World
from ui.map_canvas import MapCanvas


class MainWindow(QMainWindow):
    MODES = ['composite', 'elevation', 'moisture', 'temperature', 'plants']

    def __init__(self, world: World):
        super().__init__()
        self.world = world
        self.is_paused = False

        self.setWindowTitle("Life Simulator")
        self.resize(1280, 850)

        self.canvas = MapCanvas(self.world)
        self.setCentralWidget(self.canvas)

        view_menu = self.menuBar().addMenu("Вид")
        self.mode_actions = {}
        for mode in self.MODES:
            action = QAction(mode.capitalize(), self)
            action.setCheckable(True)
            action.setChecked(mode == 'composite')
            action.triggered.connect(lambda checked, m=mode: self.set_mode(m))
            view_menu.addAction(action)
            self.mode_actions[mode] = action

        toolbar = QToolBar("Tools", self)
        self.addToolBar(toolbar)

        self.pause_action = QAction("⏸ Пауза", self)
        self.pause_action.triggered.connect(self.toggle_pause)
        toolbar.addAction(self.pause_action)

        new_map_action = QAction("🎲 Новая карта", self)
        new_map_action.triggered.connect(self.new_map)
        toolbar.addAction(new_map_action)

        self.status = QStatusBar()
        self.setStatusBar(self.status)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(100)

        self.update_status()

    def set_mode(self, mode: str):
        for m, action in self.mode_actions.items():
            action.setChecked(m == mode)
        self.canvas.set_mode(mode)
        self.update_status()

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        self.pause_action.setText("▶ Старт" if self.is_paused else "⏸ Пауза")

    def new_map(self):
        new_seed = int(np.random.randint(0, 2**31))
        self.world = World(width=self.world.width, height=self.world.height, seed=new_seed)
        self.canvas.set_world(self.world)
        self.update_status()

    def tick(self):
        if not self.is_paused:
            self.world.step()
            self.canvas.update_texture()
            self.canvas.update()
            self.update_status()

    def update_status(self):
        self.status.showMessage(
            f"Tick: {self.world.tick} | Mode: {self.canvas.mode} | "
            f"Zoom: {self.canvas.zoom:.1f}x | Avg plants: {self.world.plants.mean():.3f}"
        )
