import numpy as np
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QImage, QPixmap, QColor
from PyQt6.QtCore import Qt, QPointF

from core.world import World


class MapCanvas(QWidget):
    def __init__(self, world: World):
        super().__init__()
        self.world = world
        self.mode = 'composite'

        self.zoom = 4.0
        self.offset = QPointF(10.0, 10.0)
        self.last_mouse_pos = None

        self.pixmap = QPixmap()
        self.update_texture()

    def set_world(self, world: World):
        self.world = world
        self.update_texture()
        self.update()

    def set_mode(self, mode: str):
        self.mode = mode
        self.update_texture()
        self.update()

    def _build_rgb(self) -> np.ndarray:
        h, w = self.world.height, self.world.width
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        elev = self.world.elevation
        moist = self.world.moisture

        if self.mode == 'composite':
            deep_ocean = np.array([12, 30, 75])
            shallow_ocean = np.array([38, 100, 160])
            shelf_ocean = np.array([65, 140, 190])
            beach = np.array([220, 200, 145])
            desert = np.array([210, 185, 120])
            savanna = np.array([145, 160, 75])
            grass_base = np.array([75, 138, 55])
            forest = np.array([28, 85, 32])
            rock = np.array([115, 110, 105])
            snow = np.array([235, 242, 250])
            freshwater_c = np.array([38, 115, 180])
            ice_color = np.array([180, 222, 245])

            is_ocean = elev < self.world.sea_level
            ocean_depth = np.clip(elev / max(0.01, self.world.sea_level), 0, 1)
            is_shelf = is_ocean & (ocean_depth > 0.85)

            is_beach = (~is_ocean) & (elev < self.world.sea_level + 0.022)
            is_mountain = elev >= self.world.mountain_level
            is_lowland = (~is_ocean) & (~is_beach) & (elev < self.world.mountain_level - 0.12)
            is_land = (~is_ocean) & (~is_beach) & (~is_mountain)

            # 1. Океан и шельф
            for c in range(3):
                ocean_col = (shallow_ocean[c] * ocean_depth + deep_ocean[c] * (1 - ocean_depth)).astype(np.uint8)
                ocean_col = np.where(is_shelf, shelf_ocean[c], ocean_col)
                rgb[..., c] = np.where(is_ocean, ocean_col, rgb[..., c])

            # 2. Биомы суши (пустыни только на низменностях, без жёлтой сыпи в горах!)
            plant_factor = np.clip(self.world.plants, 0, 1)
            is_desert = is_lowland & (moist < 0.28)
            is_savanna = is_lowland & (moist >= 0.28) & (moist < 0.42)

            for c in range(3):
                veg_col = (grass_base[c] * (1 - plant_factor) + forest[c] * plant_factor).astype(np.uint8)
                land_col = np.where(is_desert, desert[c], np.where(is_savanna, savanna[c], veg_col))

                rgb[..., c] = np.where(is_land, land_col, rgb[..., c])
                rgb[..., c] = np.where(is_beach, beach[c], rgb[..., c])
                rgb[..., c] = np.where(is_mountain, rock[c], rgb[..., c])
                rgb[..., c] = np.where(self.world.freshwater, freshwater_c[c], rgb[..., c])

                # Снег и лёд накладываем ДО теней, чтобы на горах были объёмные хребты!
                s = self.world.snow
                rgb[..., c] = (rgb[..., c] * (1.0 - s) + snow[c] * s).astype(np.uint8)
                ice = self.world.ice
                rgb[..., c] = (rgb[..., c] * (1.0 - ice) + ice_color[c] * ice).astype(np.uint8)

            # 3. 3D-рельеф (тени от солнца поверх всего, включая снежные вершины)
            dy, dx = np.gradient(elev)
            slope_light = 1.0 - (dx * 0.75 + dy * 0.75) * 42.0
            shade = np.clip(slope_light, 0.58, 1.25)

            apply_shade = (~is_ocean) & (~self.world.freshwater)
            rgb = np.where(apply_shade[..., None], np.clip(rgb * shade[..., None], 0, 255).astype(np.uint8), rgb)

        elif self.mode == 'elevation':
            rgb[..., 0] = (60 + 195 * elev).astype(np.uint8)
            rgb[..., 1] = (40 + 210 * elev).astype(np.uint8)
            rgb[..., 2] = (25 + 225 * elev).astype(np.uint8)

        elif self.mode == 'moisture':
            rgb[..., 0] = (220 * (1 - moist)).astype(np.uint8)
            rgb[..., 1] = (180 * moist + 80 * (1 - moist)).astype(np.uint8)
            rgb[..., 2] = (220 * moist).astype(np.uint8)

        elif self.mode == 'temperature':
            # Честная температурная шкала: -20°C (синий) -> +12°C (бирюзово-зелёный) -> +42°C (красный)
            t = np.clip((self.world.temperature + 20.0) / 62.0, 0.0, 1.0)
            rgb[..., 0] = np.clip(255 * (t * 1.6 - 0.5), 0, 255).astype(np.uint8)
            rgb[..., 1] = (220 * (1.0 - np.abs(t - 0.52) * 1.9).clip(0, 1)).astype(np.uint8)
            rgb[..., 2] = np.clip(255 * (1.0 - t * 1.5), 0, 255).astype(np.uint8)

        elif self.mode == 'plants':
            p = self.world.plants
            rgb[..., 0] = (120 * (1 - p)).astype(np.uint8)
            rgb[..., 1] = (80 + 170 * p).astype(np.uint8)
            rgb[..., 2] = (40 * (1 - p)).astype(np.uint8)

        # Смена дня и ночи (приглушаем яркость и добавляем лёгкий лунный оттенок ночью)
        light = self.world.light_level
        if light < 0.99:
            rgb[..., 0] = (rgb[..., 0] * light * 0.92).astype(np.uint8)
            rgb[..., 1] = (rgb[..., 1] * light * 0.96).astype(np.uint8)
            rgb[..., 2] = np.clip(rgb[..., 2] * (0.45 + 0.55 * light), 0, 255).astype(np.uint8)

        return rgb

    def update_texture(self):
        rgb = np.ascontiguousarray(self._build_rgb())
        h, w, _ = rgb.shape
        image = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self.pixmap = QPixmap.fromImage(image)

    def paintEvent(self, event):
        self._clamp_camera()
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(12, 30, 75))

        painter.translate(self.offset)
        painter.scale(self.zoom, self.zoom)
        painter.drawPixmap(0, 0, self.pixmap)

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        new_zoom = np.clip(self.zoom * factor, 1.0, 20.0)

        mouse_pos = event.position()
        self.offset = mouse_pos - (mouse_pos - self.offset) * (new_zoom / self.zoom)
        self.zoom = float(new_zoom)
        self.update()

        self._clamp_camera()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_mouse_pos = event.position()

    def mouseMoveEvent(self, event):
        if self.last_mouse_pos is not None:
            delta = event.position() - self.last_mouse_pos
            self.offset += delta
            self.last_mouse_pos = event.position()
            self.update()
        self._clamp_camera()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_mouse_pos = None
        self._clamp_camera()

    def _clamp_camera(self):
        min_zoom_x = self.width() / self.world.width
        min_zoom_y = self.height() / self.world.height
        min_zoom = min(min_zoom_x, min_zoom_y)

        if self.zoom < min_zoom:
            self.zoom = min_zoom

        map_w = self.world.width * self.zoom
        map_h = self.world.height * self.zoom

        if map_w <= self.width():
            self.offset.setX((self.width() - map_w) / 2.0)
        else:
            self.offset.setX(float(np.clip(self.offset.x(), self.width() - map_w, 0.0)))

        if map_h <= self.height():
            self.offset.setY((self.height() - map_h) / 2.0)
        else:
            self.offset.setY(float(np.clip(self.offset.y(), self.height() - map_h, 0.0)))
