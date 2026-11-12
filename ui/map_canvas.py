import numpy as np
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QImage, QPixmap
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

        if self.mode == 'composite':
            deep_ocean = np.array([15, 30, 80])
            shallow_ocean = np.array([40, 90, 150])
            beach = np.array([220, 200, 150])
            grass_base = np.array([80, 140, 60])
            forest = np.array([30, 90, 30])
            rock = np.array([110, 100, 90])
            snow = np.array([240, 245, 250])
            freshwater_c = np.array([45, 115, 175])

            ocean_depth = np.clip(elev / self.world.sea_level, 0, 1)
            is_ocean = elev < self.world.sea_level
            is_beach = (~is_ocean) & (elev < self.world.sea_level + 0.04)
            is_mountain = elev >= self.world.mountain_level
            is_land = (~is_ocean) & (~is_beach) & (~is_mountain)

            for c in range(3):
                rgb[..., c] = np.where(
                    is_ocean,
                    (shallow_ocean[c] * ocean_depth + deep_ocean[c] * (1 - ocean_depth)).astype(np.uint8),
                    rgb[..., c]
                )
                rgb[..., c] = np.where(self.world.freshwater, freshwater_c[c], rgb[..., c])
                rgb[..., c] = np.where(is_beach & (~self.world.freshwater), beach[c], rgb[..., c])
                rgb[..., c] = np.where(is_mountain, rock[c], rgb[..., c])

            plant_factor = np.clip(self.world.plants, 0, 1)

            for c in range(3):
                land_color = (grass_base[c] * (1 - plant_factor) + forest[c] * plant_factor).astype(np.uint8)
                rgb[..., c] = np.where(is_land & (~self.world.freshwater), land_color, rgb[..., c])

            snow_mask = (elev > 0.62) & (self.world.temperature < 0)
            for c in range(3):
                rgb[..., c] = np.where(snow_mask, snow[c], rgb[..., c])

        elif self.mode == 'elevation':
            rgb[..., 0] = (80 + 175 * elev).astype(np.uint8)
            rgb[..., 1] = (50 + 200 * elev).astype(np.uint8)
            rgb[..., 2] = (30 + 220 * elev).astype(np.uint8)

        elif self.mode == 'moisture':
            m = self.world.moisture
            rgb[..., 0] = (220 * (1 - m)).astype(np.uint8)
            rgb[..., 1] = (180 * m + 80 * (1 - m)).astype(np.uint8)
            rgb[..., 2] = (220 * m).astype(np.uint8)

        elif self.mode == 'temperature':
            t = np.clip((self.world.temperature + 12) / 40, 0, 1)
            rgb[..., 0] = (255 * t).astype(np.uint8)
            rgb[..., 1] = (200 * (1 - np.abs(t - 0.5) * 2)).astype(np.uint8)
            rgb[..., 2] = (255 * (1 - t)).astype(np.uint8)

        elif self.mode == 'plants':
            p = self.world.plants
            rgb[..., 0] = (120 * (1 - p)).astype(np.uint8)
            rgb[..., 1] = (80 + 170 * p).astype(np.uint8)
            rgb[..., 2] = (40 * (1 - p)).astype(np.uint8)

        return rgb

    def update_texture(self):
        rgb = np.ascontiguousarray(self._build_rgb())
        h, w, _ = rgb.shape
        image = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self.pixmap = QPixmap.fromImage(image)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)

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

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_mouse_pos = event.position()

    def mouseMoveEvent(self, event):
        if self.last_mouse_pos is not None:
            delta = event.position() - self.last_mouse_pos
            self.offset += delta
            self.last_mouse_pos = event.position()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_mouse_pos = None