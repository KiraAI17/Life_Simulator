import numpy as np
from PyQt6.QtWidgets import QMainWindow, QStatusBar, QToolBar, QComboBox
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

        self.speed_multiplier = 1
        for speed in [1, 2, 5, 10]:
            btn = QAction(f"x{speed}", self)
            btn.triggered.connect(lambda checked, s=speed: self.set_speed(s))
            toolbar.addAction(btn)

        new_map_action = QAction("🎲 Новая карта", self)
        new_map_action.triggered.connect(self.new_map)
        toolbar.addAction(new_map_action)

        self.preset_box = QComboBox(self)
        self.preset_box.addItems(World.PRESETS)
        self.preset_box.setCurrentText(self.world.preset)
        toolbar.addWidget(self.preset_box)

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
        preset = self.preset_box.currentText()
        self.world = World(width=self.world.width, height=self.world.height, seed=new_seed, preset=preset)
        self.canvas.set_world(self.world)
        self.update_status()

    def tick(self):
        if not self.is_paused:
            self.world.step()
            self.canvas.update_texture()
            self.canvas.update()
            self.update_status()

    def update_status(self):
        time_label = "🌙 Ночь" if self.world.is_night else "☀️ День"
        herbs = sum(1 for a in self.world.agents if not a.is_predator)
        preds = len(self.world.agents) - herbs
        max_crab = max((a.crab_score for a in self.world.agents), default=0.0)
        carns = len(self.world.agents) - herbs
        self.status.showMessage(
            f"{time_label} | Сезон: {self.world.season_name} | Tick: {self.world.tick} | "
            f"Mode: {self.canvas.mode} | Zoom: {self.canvas.zoom:.1f}x | "
            f"Avg plants: {self.world.plants.mean():.3f} "
            f"Травоядные {herbs} | хищники {carns} | Max Crab: {max_crab:.0%}"
        )

    def set_speed(self, speed: int):
        self.speed_multiplier = speed
        self.update_status()

    def tick(self):
        if not self.is_paused:
            for _ in range(self.speed_multiplier):
                self.world.step()
            # Пересобираем пиксмап ландшафта только при изменении климата/травы, а не на каждый шаг зверей!
            if self.world.tick % 5 == 0 or self.speed_multiplier >= 5:
                self.canvas.update_texture()
            self.canvas.update()
            self.update_status()