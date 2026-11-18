import time
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QRectF
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QMainWindow,
    QMessageBox, QPushButton, QSlider, QSpinBox, QVBoxLayout, QWidget,
    QFileDialog, QInputDialog, QCheckBox, QScrollArea, QLineEdit, QGridLayout,
)
from core.analytics import EcosystemTracker
from core.storage import ExperimentStorage
from core.world import World
from ui.map_canvas import MapCanvas
from core.morphology import (
    G_WIDTH, G_LENGTH, G_SHELL, G_CLAWS, G_LEGS, G_LATERAL, G_GILLS, G_LUNGS,
    G_DIET, G_SPEED, G_SENSE, G_SIZE, G_FUR, G_OPT_TEMP, G_NOCTURNAL,
    G_HUE_SHIFT, G_SATURATION, G_BRIGHTNESS,
)


from PyQt6.QtCore import pyqtSignal, QRectF 

class PopulationChartWidget(QWidget):
    """Живой график с кликабельной легендой: клик по подписи -> метки на существах."""

    species_clicked = pyqtSignal(str)  #сигнал "пользователь кликнул по подписи вида"

    def __init__(self, tracker: EcosystemTracker, parent=None):
        super().__init__(parent)
        self.tracker = tracker
        self.setMinimumHeight(155)
        self._legend_rects: list[tuple[QRectF, str]] = []  # (прямоугольник, ключ вида)
        self.setMouseTracking(True)  # для подсветки при наведении (опционально)

    def paintEvent(self, event):
        p = QPainter(self)
        self._legend_rects.clear()  #сбрасываем перед каждой перерисовкой
        try:
            p.fillRect(self.rect(), QColor(24, 28, 36))
            p.setPen(QColor(55, 62, 75))
            p.drawRect(0, 0, self.width() - 1, self.height() - 1)

            data = list(self.tracker.history)
            if len(data) < 2:
                p.setPen(QColor(160, 170, 185))
                p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Накопление данных графика...")
                return

            animal_series = [
                ("herbivores", QColor(235, 215, 145), "Трав"),
                ("aquatic", QColor(65, 200, 235), "Рыбы"),
                ("sharks", QColor(70, 115, 195), "Акулы"),
                ("wolves", QColor(150, 160, 175), "Волки"),
                ("foxes", QColor(240, 115, 40), "Лисы"),
                ("crabs", QColor(225, 55, 75), "Крабы"),
            ]

            max_animals = max(15.0, max(float(row[k]) for row in data for k, _, _ in animal_series))
            max_plants = max(100.0, max(float(row["plants"]) for row in data))

            pad_l, pad_r, pad_t, pad_b = 8, 8, 28, 8
            w = max(1, self.width() - pad_l - pad_r)
            h = max(1, self.height() - pad_t - pad_b)
            n = len(data)

            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

            pen_plant = QPen(QColor(75, 195, 95, 170), 1.3, Qt.PenStyle.DashLine)
            p.setPen(pen_plant)
            prev_x, prev_y = None, None
            for i, row in enumerate(data):
                x = pad_l + (i / (n - 1)) * w
                y = pad_t + h - (row["plants"] / max_plants) * h
                if prev_x is not None:
                    p.drawLine(int(prev_x), int(prev_y), int(x), int(y))
                prev_x, prev_y = x, y

            for key, color, _ in animal_series:
                p.setPen(QPen(color, 1.7))
                prev_x, prev_y = None, None
                for i, row in enumerate(data):
                    x = pad_l + (i / (n - 1)) * w
                    y = pad_t + h - (row[key] / max_animals) * h
                    if prev_x is not None:
                        p.drawLine(int(prev_x), int(prev_y), int(x), int(y))
                    prev_x, prev_y = x, y

            #Компактная легенда с запоминанием прямоугольников для клика
            last = data[-1]
            fm = p.fontMetrics()
            lx = 8
            p.setPen(QColor(75, 195, 95))
            text = f"Флора:{int(last['plants'])}"
            p.drawText(lx, 13, text)
            self._legend_rects.append(
                (QRectF(lx - 2, 2, fm.horizontalAdvance(text) + 4, 16), "plants")
            )
            lx += fm.horizontalAdvance(text) + 12

            for key, color, label in animal_series:
                p.setPen(color)
                text = f"{label}:{last[key]}"
                p.drawText(lx, 13, text)
                self._legend_rects.append(
                    (QRectF(lx - 2, 2, fm.horizontalAdvance(text) + 4, 16), key)
                )
                lx += fm.horizontalAdvance(text) + 10
        finally:
            p.end()

    def mousePressEvent(self, event):
        """Клик по подписи в легенде -> сигнал для маркировки вида на карте."""
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            for rect, key in self._legend_rects:
                if rect.contains(pos):
                    self.species_clicked.emit(key)
                    return
        super().mousePressEvent(event)


class MainWindow(QMainWindow):
    def __init__(self, world):
        super().__init__()
        self.world = world
        self.paused = False
        self.speed_mult = 1
        self._marked_species: str | None = None

        self.tracker = EcosystemTracker(max_points=300, sample_interval=10)
        self.storage = ExperimentStorage()
        self.tracker.record(self.world, force=True)

        self.setWindowTitle("Жизнь в миниатюре — Симулятор эволюционной экосистемы")
        self.resize(1380, 820)

        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QHBoxLayout(root)
        main_layout.setContentsMargins(6, 6, 6, 6)

        self.canvas = MapCanvas(self.world)
        main_layout.addWidget(self.canvas, stretch=1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(500)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        main_layout.addWidget(scroll)

        side_panel = QWidget()
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(6, 6, 6, 6)
        side_layout.setSpacing(8)
        scroll.setWidget(side_panel)

        self.cb_view_mode = QComboBox()
        self.cb_view_mode.addItems(["composite", "elevation", "moisture", "temperature", "plants"])
        self.cb_view_mode.currentTextChanged.connect(self.canvas.set_mode)
        side_layout.addWidget(self.cb_view_mode)

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

        chart_group = QGroupBox("Динамика популяций во времени")
        chart_layout = QVBoxLayout(chart_group)
        self.chart = PopulationChartWidget(self.tracker)
        chart_layout.addWidget(self.chart)
        side_layout.addWidget(chart_group)

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

        init_group = QGroupBox("Начальные условия нового мира")
        init_form = QFormLayout(init_group)
        self.cb_preset = QComboBox()
        self.cb_preset.addItems(World.PRESETS)
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
        init_form.addRow("Рыбы:", self.sp_worms)
        self.sp_sharks = QSpinBox()
        self.sp_sharks.setRange(0, 100)
        self.sp_sharks.setValue(12)
        init_form.addRow("Акулы:", self.sp_sharks)
        btn_restart = QPushButton("Перезапустить мир")
        btn_restart.clicked.connect(self.restart_world)
        init_form.addRow(btn_restart)
        side_layout.addWidget(init_group)

        btn_save = QPushButton("Сохранить (SQLite + JSON)")
        btn_save.clicked.connect(self.save_experiment)
        side_layout.addWidget(btn_save)
        btn_load = QPushButton("Загрузить (SQLite / JSON)")
        btn_load.clicked.connect(self.load_experiment)
        side_layout.addWidget(btn_load)
        side_layout.addStretch(1)

        self.status_label = QLabel()
        self.statusBar().addWidget(self.status_label)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(33)

        insp_group = QGroupBox("Инспектор и Редактор ДНК")
        insp_layout = QVBoxLayout(insp_group)

        spawn_row = QHBoxLayout()
        self.chk_spawn_mode = QCheckBox("Спавн кликом:")
        self.cb_spawn_arch = QComboBox()
        self.cb_spawn_arch.addItems(["fox", "land_grazer", "land_stalker", "aquatic_worm", "aquatic_predator"])
        spawn_row.addWidget(self.chk_spawn_mode)
        spawn_row.addWidget(self.cb_spawn_arch)
        insp_layout.addLayout(spawn_row)

        self.lbl_insp_info = QLabel("Кликните на любое животное на карте")
        self.lbl_insp_info.setWordWrap(True)
        insp_layout.addWidget(self.lbl_insp_info)

        self.ed_agent_name = QLineEdit()
        self.ed_agent_name.setPlaceholderText("Имя особи (необязательно)...")
        self.ed_agent_name.setEnabled(False)
        self.ed_agent_name.textChanged.connect(self._on_name_edited)
        insp_layout.addWidget(self.ed_agent_name)

        act_grid = QGridLayout()
        act_grid.setSpacing(4)
        self.btn_mark = QPushButton("Маячок")
        self.btn_heal = QPushButton("Исцелить")
        self.btn_clone = QPushButton("Клонировать")
        self.btn_kill = QPushButton("Удалить")
        for idx, b in enumerate((self.btn_mark, self.btn_heal, self.btn_clone, self.btn_kill)):
            b.setEnabled(False)
            act_grid.addWidget(b, idx // 2, idx % 2)
        self.btn_mark.clicked.connect(self._toggle_agent_mark)
        self.btn_heal.clicked.connect(self._heal_agent)
        self.btn_clone.clicked.connect(self._clone_agent)
        self.btn_kill.clicked.connect(self._kill_agent)
        insp_layout.addLayout(act_grid)

        morph_row = QHBoxLayout()
        self.cb_morph_arch = QComboBox()
        self.cb_morph_arch.addItems(["fox", "land_grazer", "land_stalker", "aquatic_worm", "aquatic_predator"])
        self.cb_morph_arch.setEnabled(False)
        self.btn_apply_arch = QPushButton("Превратить")
        self.btn_apply_arch.setEnabled(False)
        self.btn_apply_arch.clicked.connect(self._apply_archetype_to_selected)
        morph_row.addWidget(self.cb_morph_arch)
        morph_row.addWidget(self.btn_apply_arch)
        insp_layout.addLayout(morph_row)

        self._gene_sliders = {}
        self._gene_labels = {}
        self._gene_scales = {}
        gene_defs = [
            ("Ширина тела", G_WIDTH, 15, 100, 100.0),
            ("Длина тела", G_LENGTH, 20, 100, 100.0),
            ("Панцирь", G_SHELL, 0, 100, 100.0),
            ("Клешни", G_CLAWS, 0, 100, 100.0),
            ("Кол-во ног", G_LEGS, 0, 8, 1.0),
            ("Бок. походка", G_LATERAL, 0, 100, 100.0),
            ("Жабры", G_GILLS, 0, 100, 100.0),
            ("Лёгкие", G_LUNGS, 0, 100, 100.0),
            ("Хищность", G_DIET, -100, 100, 100.0),
            ("Скорость", G_SPEED, 10, 90, 100.0),
            ("Чутьё (px)", G_SENSE, 15, 120, 1.0),
            ("Размер", G_SIZE, 40, 220, 100.0),
            ("Мех", G_FUR, 0, 100, 100.0),
            ("Опт. темп °C", G_OPT_TEMP, -15, 40, 1.0),
            ("Ночной режим", G_NOCTURNAL, 0, 100, 100.0),
            ("Сдвиг оттенка", G_HUE_SHIFT, -15, 15, 100.0),
            ("Насыщенность", G_SATURATION, 75, 125, 100.0),
            ("Яркость", G_BRIGHTNESS, 85, 115, 100.0),
        ]
        gene_form = QFormLayout()
        gene_form.setHorizontalSpacing(6)
        gene_form.setVerticalSpacing(4)
        gene_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
        for title, g_idx, mn, mx, scale in gene_defs:
            sl = QSlider(Qt.Orientation.Horizontal)
            sl.setRange(mn, mx)
            sl.setEnabled(False)
            lbl = QLabel("-")
            sl.valueChanged.connect(lambda val, idx=g_idx, sc=scale: self._on_gene_edited(idx, val, sc))
            self._gene_sliders[g_idx] = sl
            self._gene_labels[g_idx] = lbl
            self._gene_scales[g_idx] = scale
            gene_form.addRow(f"{title}:", self._wrap_slider(sl, lbl))
        insp_layout.addLayout(gene_form)
        side_layout.addWidget(insp_group)

        self.canvas.agent_selected.connect(self._on_agent_selected)
        self.canvas.map_clicked.connect(self._on_map_clicked)
        self.chart.species_clicked.connect(self._mark_species)
        
        self.showMaximized()
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
        label.setFixedWidth(34)
        slider.setMinimumWidth(95)
        slider.setMaximumWidth(180)  # Защита от наезда на скроллбар
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

    def _on_map_clicked(self, wx: float, wy: float):
        if not self.chk_spawn_mode.isChecked(): return
        from core.agent import Agent
        arch = self.cb_spawn_arch.currentText()
        genes = self.world.evo_config.spawn_genome(arch, self.world.rng)
        if Agent.is_tile_passable(self.world, int(wx), int(wy), genes):
            new_agent = Agent(wx, wy, genes=genes, config=self.world.evo_config, rng=self.world.rng)
            new_agent.marked = True
            self.world.agents.append(new_agent)
            self.canvas.selected_agent = new_agent
            self._on_agent_selected(new_agent)
            self.canvas.update()
            self.update_status()

    def _on_agent_selected(self, agent):
        self._selected_agent = agent
        has_agent = agent is not None and agent.alive
        for w in (self.ed_agent_name, self.btn_mark, self.btn_heal, self.btn_clone, self.btn_kill, self.cb_morph_arch, self.btn_apply_arch):
            w.setEnabled(has_agent)
        if has_agent:
            self.ed_agent_name.blockSignals(True)
            self.ed_agent_name.setText(getattr(agent, "custom_name", ""))
            self.ed_agent_name.blockSignals(False)
        for g_idx, sl in self._gene_sliders.items():
            sl.setEnabled(has_agent)
            scale = self._gene_scales[g_idx]
            if has_agent:
                raw_val = float(agent.genes[g_idx])
                sl.blockSignals(True)
                sl.setValue(int(round(raw_val * scale)))
                sl.blockSignals(False)
                self._gene_labels[g_idx].setText(f"{raw_val:.2f}" if scale > 1 else f"{raw_val:.0f}")
            else:
                self._gene_labels[g_idx].setText("-")
        self._refresh_inspector_text()

    def _on_gene_edited(self, g_idx: int, val: int, scale: float):
        agent = getattr(self, "_selected_agent", None)
        if agent is None or not agent.alive: return
        real_val = val / scale
        agent.genes[g_idx] = real_val
        self._gene_labels[g_idx].setText(f"{real_val:.2f}" if scale > 1 else f"{real_val:.0f}")
        agent.rebuild_phenotype()
        self.canvas._sprite_cache.pop(id(agent), None)
        self.canvas.update()
        self._refresh_inspector_text()

    def _on_name_edited(self, text: str):
        agent = getattr(self, "_selected_agent", None)
        if agent is not None and agent.alive:
            agent.custom_name = text

    def _toggle_agent_mark(self):
        agent = getattr(self, "_selected_agent", None)
        if agent is not None and agent.alive:
            agent.marked = not agent.marked
            self.canvas.update()

    def _heal_agent(self):
        agent = getattr(self, "_selected_agent", None)
        if agent is not None and agent.alive:
            agent.hp = agent.max_hp
            agent.energy = 100.0
            agent.hydration = 100.0
            agent.oxygen = 100.0
            agent.stamina = 100.0
            self._refresh_inspector_text()

    def _clone_agent(self):
        agent = getattr(self, "_selected_agent", None)
        if agent is not None and agent.alive:
            from core.agent import Agent
            clone = Agent(agent.x + 3.0, agent.y + 3.0, genes=agent.genes.copy(), config=self.world.evo_config, rng=self.world.rng)
            clone.marked = agent.marked
            self.world.agents.append(clone)
            self.canvas.update()
            self.update_status()

    def _kill_agent(self):
        agent = getattr(self, "_selected_agent", None)
        if agent is not None and agent.alive:
            agent.alive = False
            if agent in self.world.agents:
                self.world.agents.remove(agent)
            self.canvas.selected_agent = None
            self._on_agent_selected(None)
            self.canvas.update()
            self.update_status()

    def _apply_archetype_to_selected(self):
        agent = getattr(self, "_selected_agent", None)
        if agent is not None and agent.alive:
            agent.apply_archetype(self.cb_morph_arch.currentText())
            self.canvas._sprite_cache.pop(id(agent), None)
            self._on_agent_selected(agent)
            self.canvas.update()

    def _mark_species(self, key: str):
        """Маркирует всех существ выбранного вида (или снимает метки при повторном клике)."""
        from core.morphology import G_DIET, G_FUR, G_GILLS, G_LEGS, G_LUNGS, G_SHELL

        # клик по "Флора" переключает режим карты на plants
        if key == "plants":
            if self.canvas.mode == "plants":
                self.canvas.set_mode("composite")
                self.cb_view_mode.setCurrentText("composite")
            else:
                self.canvas.set_mode("plants")
                self.cb_view_mode.setCurrentText("plants")
            return

        # Если этот вид уже помечен — снимаем метки (toggle)
        if self._marked_species == key:
            for a in self.world.agents:
                a.marked = False
            self._marked_species = None
            self.canvas.update()
            return

        # Иначе — сначала снимаем старые метки, потом ставим новые
        for a in self.world.agents:
            a.marked = False

        g = None
        for a in self.world.agents:
            g = a.genes
            is_fox = (g[G_FUR] > 0.55) and (g[G_SHELL] < 0.30) and (0.25 < g[G_DIET] <= 0.72) and (g[G_LEGS] >= 4.0)
            if a.crab_score > 0.55:
                kind = "crabs"
            elif g[G_GILLS] > 0.55 and g[G_LUNGS] < 0.45:
                if g[G_DIET] > 0.35:
                    kind = "sharks"
                else:
                    kind = "aquatic"
            elif is_fox:
                kind = "foxes"
            elif a.is_predator:
                kind = "wolves"
            else:
                kind = "herbivores"
            if kind == key:
                a.marked = True

        self._marked_species = key
        self.canvas.update()

    def _refresh_inspector_text(self):
        if not hasattr(self, "lbl_insp_info"): return
        agent = getattr(self, "_selected_agent", None)
        if agent is None or not agent.alive:
            self.lbl_insp_info.setText("Кликните на любое животное на карте")
            return
        name_prefix = f"[{agent.custom_name}] " if getattr(agent, "custom_name", "") else ""
        self.lbl_insp_info.setText(
            f"{name_prefix}Состояние: {agent.state} | Возраст: {agent.age}\n"
            f"HP: {agent.hp:.0f}/{agent.max_hp:.0f} | Энергия: {agent.energy:.0f}% | Вода: {agent.hydration:.0f}% | O2: {agent.oxygen:.0f}%\n"
            f"Индекс краба: {agent.crab_score:.0%}"
        )

    def toggle_pause(self):
        self.paused = not self.paused
        self.btn_pause.setText("Продолжить" if self.paused else "Пауза")

    def set_speed(self, spd: int):
        self.speed_mult = spd
        if spd == 1:
            self.timer.setInterval(33)
        elif spd == 2:
            self.timer.setInterval(22)
        else:
            self.timer.setInterval(16)

    def restart_world(self):
        self.world.preset = self.cb_preset.currentText()
        self.world.tick = 0
        self.world.generate_map()
        self.world.spawn_initial_life(
            land_herbs=self.sp_herbs.value(),
            land_preds=self.sp_wolves.value(),
            foxes=self.sp_foxes.value(),
            aquatic_worms=self.sp_worms.value(),
            aquatic_preds=self.sp_sharks.value(),
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
            QMessageBox.information(self, "Эксперимент сохранён", f"Запись #{exp_id} сохранена в SQLite (data/experiments.db)\nи экспортирована в {json_path}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка сохранения", str(e))

    def load_experiment(self):
        try:
            exps = self.storage.list_experiments()
            items = [f"#{e['id']} | {e['created_at']} | {e['preset']} (тиков: {e['total_ticks']}, краб: {e['max_crab_score']:.0%})" for e in exps]
            items.append("Выбрать JSON-файл с диска...")
            choice, ok = QInputDialog.getItem(self, "Загрузка эксперимента", "Выберите сохранённый прогон:", items, 0, False)
            if not ok or not choice: return

            if choice.startswith("Выбрать JSON"):
                path, _ = QFileDialog.getOpenFileName(self, "Открыть JSON эксперимента", "data/exports", "JSON (*.json)")
                if not path: return
                data = self.storage.load_from_json(path)
            else:
                exp_id = int(choice.split("|")[0].replace("#", "").strip())
                data = self.storage.load_from_db(exp_id)
                if not data: return

            p = data.get("parameters", {})
            p.get("init_sharks", 12)
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

            self.tracker.history.clear()
            for row in data.get("timeline", []):
                self.tracker.history.append(row)
            self.chart.update()
            self.update_status()
            QMessageBox.information(self, "Эксперимент загружен", f"Параметры и график прогона #{data.get('experiment_id')} восстановлены.\nНажмите «Продолжить» для запуска.")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка загрузки", str(e))

    def tick(self):
        if self.paused:
            return
        t0 = time.perf_counter()
        steps_done = 0
        # Разрешаем до 28 мс на симуляцию (из 33 мс кадра), чтобы x5 и x10 реально ускоряли мир
        time_budget = 0.014 if self.speed_mult == 1 else 0.028
        for _ in range(self.speed_mult):
            self.world.step()
            self.tracker.record(self.world)
            steps_done += 1
            if (time.perf_counter() - t0) > time_budget:
                break
        self.canvas.update()
        if self.world.tick % 10 == 0:
            self.chart.update()
        if self.world.tick % 60 == 0:
            self.canvas.update_texture()
        self.update_status()

    def update_status(self):
        self._refresh_inspector_text()
        last = self.tracker.history[-1] if self.tracker.history else {}
        self.status_label.setText(
            f"Тик: {self.world.tick} | Скорость: x{self.speed_mult} | "
            f"Всего: {len(self.world.agents)} (Травоядн.: {last.get('herbivores', 0)}, "
            f"Волки: {last.get('wolves', 0)}, Лисы: {last.get('foxes', 0)}, "
            f"Рыбы: {last.get('aquatic', 0)}, Крабы: {last.get('crabs', 0)}) | "
            f"Max Crab: {last.get('max_crab_score', 0.0):.0%}"
        )