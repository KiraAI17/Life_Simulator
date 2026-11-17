import time
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QInputDialog,
)

from core.analytics import EcosystemTracker
from core.storage import ExperimentStorage
from ui.map_canvas import MapCanvas


class PopulationChartWidget(QWidget):
    """Лёгкий живой график численности видов во времени (задача максимум кейса)."""

    def __init__(self, tracker: EcosystemTracker, parent=None):
        super().__init__(parent)
        self.tracker = tracker
        self.setMinimumHeight(155)

    def paintEvent(self, event):
        p = QPainter(self)
        try:
            p.fillRect(self.rect(), QColor(24, 28, 36))
            p.setPen(QColor(55, 62, 75))
            p.drawRect(0, 0, self.width() - 1, self.height() - 1)

            data = list(self.tracker.history)
            if len(data) < 2:
                p.setPen(QColor(160, 170, 185))
                p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Накопление данных графика...")
                return

            series = [
                ("plants", QColor(75, 195, 95), "Растения"),
                ("herbivores", QColor(235, 215, 145), "Травоядн."),
                ("aquatic", QColor(65, 200, 235), "Рыбы"),
                ("wolves", QColor(150, 160, 175), "Волки"),
                ("foxes", QColor(240, 115, 40), "Лисы"),
                ("crabs", QColor(225, 55, 75), "Крабы"),
            ]

            max_val = 10.0
            for row in data:
                for key, _, _ in series:
                    if row[key] > max_val:
                        max_val = float(row[key])

            pad_l, pad_r, pad_t, pad_b = 8, 8, 22, 8
            w = max(1, self.width() - pad_l - pad_r)
            h = max(1, self.height() - pad_t - pad_b)
            n = len(data)

            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            for key, color, _ in series:
                p.setPen(QPen(color, 1.6))
                prev_x, prev_y = None, None
                for i, row in enumerate(data):
                    x = pad_l + (i / (n - 1)) * w
                    y = pad_t + h - (row[key] / max_val) * h
                    if prev_x is not None:
                        p.drawLine(int(prev_x), int(prev_y), int(x), int(y))
                    prev_x, prev_y = x, y

            # Компактная легенда сверху
            lx = 8
            for _, color, label in series:
                p.setPen(color)
                p.drawText(lx, 14, label)
                lx += 48
        finally:
            p.end()


class MainWindow(QMainWindow):
    def __init__(self, world):
        super().__init__()
        self.world = world
        self.paused = False
        self.speed_mult = 1

        self.tracker = EcosystemTracker(max_points=300, sample_interval=10)
        self.storage = ExperimentStorage()
        self.tracker.record(self.world, force=True)

        self.setWindowTitle("Жизнь в миниатюре — Симулятор эволюционной экосистемы")
        self.resize(1380, 820)

        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QHBoxLayout(root)
        main_layout.setContentsMargins(6, 6, 6, 6)

        # Левая часть: карта мира
        self.canvas = MapCanvas(self.world)
        main_layout.addWidget(self.canvas, stretch=1)

        # Правая панель: управление экспериментом, графики и БД
        side_panel = QWidget()
        side_panel.setFixedWidth(315)
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(4, 0, 4, 0)
        main_layout.addWidget(side_panel)

        self.cb_view_mode = QComboBox()
        self.cb_view_mode.addItems(["composite", "elevation", "moisture", "temperature", "plants"])
        self.cb_view_mode.currentTextChanged.connect(self.canvas.set_mode)
        side_layout.addWidget(self.cb_view_mode)

        # 1. Управление временем
        ctrl_group = QGroupBox("Симуляция")
        ctrl_layout = QHBoxLayout(ctrl_group)
        self.btn_pause = QPushButton("Пауза")
        self.btn_pause.clicked.connect(self.toggle_pause)
        ctrl_layout.addWidget(self.btn_pause)

        for spd in (1, 2, 5, 10):
            btn = QPushButton(f"x{spd}")
            btn.clicked.connect(lambda _, s=spd: self.set_speed(s))
            ctrl_layout.addWidget(btn)
        side_layout.addWidget(ctrl_group)

        # 2. Живой график численности
        chart_group = QGroupBox("Динамика популяций во времени")
        chart_layout = QVBoxLayout(chart_group)
        self.chart = PopulationChartWidget(self.tracker)
        chart_layout.addWidget(self.chart)
        side_layout.addWidget(chart_group)

        # 3. Параметры в реальном времени (Задача максимум)
        live_group = QGroupBox("Параметры среды (в реальном времени)")
        live_form = QFormLayout(live_group)

        self.sl_plants = self._make_slider(10, 300, 100, self._on_live_params_changed)
        self.lbl_plants = QLabel("100%")
        live_form.addRow("Рост растений:", self._wrap_slider(self.sl_plants, self.lbl_plants))

        self.sl_repro = self._make_slider(40, 220, 100, self._on_live_params_changed)
        self.lbl_repro = QLabel("100%")
        live_form.addRow("Размножение:", self._wrap_slider(self.sl_repro, self.lbl_repro))

        self.sl_life = self._make_slider(40, 220, 100, self._on_live_params_changed)
        self.lbl_life = QLabel("100%")
        live_form.addRow("Срок жизни:", self._wrap_slider(self.sl_life, self.lbl_life))

        self.sl_temp = self._make_slider(-15, 15, 0, self._on_live_params_changed)
        self.lbl_temp = QLabel("0°C")
        live_form.addRow("Сдвиг темп.:", self._wrap_slider(self.sl_temp, self.lbl_temp))

        side_layout.addWidget(live_group)

        # 4. Начальные условия перед запуском (Задача минимум)
        init_group = QGroupBox("Начальные условия нового мира")
        init_form = QFormLayout(init_group)

        self.cb_preset = QComboBox()
        self.cb_preset.addItems(["archipelago", "continent", "pangaea", "oasis"])
        self.cb_preset.setCurrentText(getattr(self.world, "preset", "archipelago"))
        init_form.addRow("Тип карты:", self.cb_preset)

        self.sp_herbs = QSpinBox()
        self.sp_herbs.setRange(0, 400)
        self.sp_herbs.setValue(120)
        init_form.addRow("Травоядные:", self.sp_herbs)

        self.sp_wolves = QSpinBox()
        self.sp_wolves.setRange(0, 150)
        self.sp_wolves.setValue(20)
        init_form.addRow("Волки:", self.sp_wolves)

        self.sp_foxes = QSpinBox()
        self.sp_foxes.setRange(0, 150)
        self.sp_foxes.setValue(24)
        init_form.addRow("Лисы:", self.sp_foxes)

        self.sp_worms = QSpinBox()
        self.sp_worms.setRange(0, 300)
        self.sp_worms.setValue(90)
        init_form.addRow("Водные виды:", self.sp_worms)

        btn_restart = QPushButton("Перезапустить с этими условиями")
        btn_restart.clicked.connect(self.restart_world)
        init_form.addRow(btn_restart)
        side_layout.addWidget(init_group)

        # 5. Сохранение эксперимента в БД
        btn_save = QPushButton("Сохранить эксперимент (SQLite + JSON)")
        btn_save.clicked.connect(self.save_experiment)
        side_layout.addWidget(btn_save)

        btn_load = QPushButton("Загрузить эксперимент (SQLite / JSON)")
        btn_load.clicked.connect(self.load_experiment)
        side_layout.addWidget(btn_load)

        side_layout.addStretch(1)

        self.status_label = QLabel()
        self.statusBar().addWidget(self.status_label)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(16)
        self.update_status()

    def _make_slider(self, mn: int, mx: int, val: int, cb) -> QSlider:
        sl = QSlider(Qt.Orientation.Horizontal)
        sl.setRange(mn, mx)
        sl.setValue(val)
        sl.valueChanged.connect(cb)
        return sl

    def _wrap_slider(self, slider: QSlider, label: QLabel) -> QWidget:
        box = QWidget()
        lay = QHBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        label.setFixedWidth(38)
        lay.addWidget(slider)
        lay.addWidget(label)
        return box

    def _on_live_params_changed(self):
        self.world.plant_growth_mult = self.sl_plants.value() / 100.0
        self.world.repro_rate_mult = self.sl_repro.value() / 100.0
        self.world.lifespan_mult = self.sl_life.value() / 100.0
        self.world.temp_offset = float(self.sl_temp.value())

        self.lbl_plants.setText(f"{self.sl_plants.value()}%")
        self.lbl_repro.setText(f"{self.sl_repro.value()}%")
        self.lbl_life.setText(f"{self.sl_life.value()}%")
        self.lbl_temp.setText(f"{self.sl_temp.value():+d}°C")

    def toggle_pause(self):
        self.paused = not self.paused
        self.btn_pause.setText("Продолжить" if self.paused else "Пауза")

    def set_speed(self, spd: int):
        self.speed_mult = spd

    def restart_world(self):
        self.world.preset = self.cb_preset.currentText()
        self.world.generate_map()
        self.world.spawn_initial_life(
            land_herbs=self.sp_herbs.value(),
            land_preds=self.sp_wolves.value(),
            foxes=self.sp_foxes.value(),
            aquatic_worms=self.sp_worms.value(),
        )
        self._on_live_params_changed()
        self.tracker.reset()
        self.tracker.record(self.world, force=True)
        self.canvas._sprite_cache.clear()
        self.canvas.set_world(self.world)
        self.canvas.update()
        self.chart.update()
        self.update_status()

    def save_experiment(self):
        try:
            params = {
                "preset": self.cb_preset.currentText(),
                "init_herbivores": self.sp_herbs.value(),
                "init_wolves": self.sp_wolves.value(),
                "init_foxes": self.sp_foxes.value(),
                "init_aquatic": self.sp_worms.value(),
                "plant_growth_mult": self.world.plant_growth_mult,
                "repro_rate_mult": self.world.repro_rate_mult,
                "lifespan_mult": self.world.lifespan_mult,
                "temp_offset": self.world.temp_offset,
            }
            exp_id, json_path = self.storage.save_experiment(self.world, self.tracker, params)
            QMessageBox.information(
                self,
                "Эксперимент сохранён",
                f"Запись #{exp_id} сохранена в SQLite (data/experiments.db)\nи экспортирована в {json_path}",
            )
        except Exception as e:
            QMessageBox.critical(self, "Ошибка сохранения", str(e))

    def load_experiment(self):
        try:
            exps = self.storage.list_experiments()
            items = [f"#{e['id']} | {e['created_at']} | {e['preset']} (тиков: {e['total_ticks']}, краб: {e['max_crab_score']:.0%})" for e in exps]
            items.append("Выбрать JSON-файл с диска...")

            choice, ok = QInputDialog.getItem(self, "Загрузка эксперимента", "Выберите сохранённый прогон:", items, 0, False)
            if not ok or not choice:
                return

            if choice.startswith("Выбрать JSON"):
                path, _ = QFileDialog.getOpenFileName(self, "Открыть JSON эксперимента", "data/exports", "JSON (*.json)")
                if not path:
                    return
                data = self.storage.load_from_json(path)
            else:
                exp_id = int(choice.split("|")[0].replace("#", "").strip())
                data = self.storage.load_from_db(exp_id)
                if not data:
                    return

            # Восстанавливаем параметры и график из сохранённого эксперимента
            p = data.get("parameters", {})
            self.paused = True
            self.btn_pause.setText("Продолжить")

            self.cb_preset.setCurrentText(p.get("preset", "archipelago"))
            self.sp_herbs.setValue(int(p.get("init_herbivores", 120)))
            self.sp_wolves.setValue(int(p.get("init_wolves", 20)))
            self.sp_foxes.setValue(int(p.get("init_foxes", 24)))
            self.sp_worms.setValue(int(p.get("init_aquatic", 90)))

            self.sl_plants.setValue(int(round(p.get("plant_growth_mult", 1.0) * 100)))
            self.sl_repro.setValue(int(round(p.get("repro_rate_mult", 1.0) * 100)))
            self.sl_life.setValue(int(round(p.get("lifespan_mult", 1.0) * 100)))
            self.sl_temp.setValue(int(round(p.get("temp_offset", 0.0))))

            self.restart_world()

            # Загружаем сохранённую историю популяций на график
            self.tracker.history.clear()
            for row in data.get("timeline", []):
                self.tracker.history.append(row)
            self.chart.update()
            self.update_status()

            QMessageBox.information(
                self,
                "Эксперимент загружен",
                f"Параметры и график прогона #{data.get('experiment_id')} восстановлены.\nНажмите «Продолжить» для запуска.",
            )
        except Exception as e:
            QMessageBox.critical(self, "Ошибка загрузки", str(e))

    def tick(self):
        if self.paused:
            return
        t0 = time.perf_counter()
        for _ in range(self.speed_mult):
            self.world.step()
            self.tracker.record(self.world)
            if (time.perf_counter() - t0) > 0.014:
                break

        self.canvas.update()
        if self.world.tick % 10 == 0:
            self.chart.update()
        if self.world.tick % 60 == 0:
            self.canvas.update_texture()
        self.update_status()

    def update_status(self):
        last = self.tracker.history[-1] if self.tracker.history else {}
        self.status_label.setText(
            f"Тик: {self.world.tick} | Скорость: x{self.speed_mult} | "
            f"Всего: {len(self.world.agents)} (Травоядн.: {last.get('herbivores', 0)}, "
            f"Волки: {last.get('wolves', 0)}, Лисы: {last.get('foxes', 0)}, "
            f"Рыбы: {last.get('aquatic', 0)}, Крабы: {last.get('crabs', 0)}) | "
            f"Max Crab: {last.get('max_crab_score', 0.0):.0%}"
        )