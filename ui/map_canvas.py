import numpy as np
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QImage, QPixmap, QColor, QPolygonF
from PyQt6.QtCore import Qt, QPointF

from core.world import World


class MapCanvas(QWidget):
    def __init__(self, world: World):
        super().__init__()
        self.world = world
        self.mode = 'composite'

        self.zoom = 1.0
        self.offset = QPointF(0.0, 0.0)
        self.last_mouse_pos = None

        self.pixmap = QPixmap()
        self._build_static_terrain()
        self.update_texture()

    def set_world(self, world: World):
        self.world = world
        self._build_static_terrain()
        self.update_texture()
        self.fit_to_window()
        self.update()

    def _build_static_terrain(self):
        h, w = self.world.height, self.world.width
        elev = self.world.elevation
        moist = self.world._upscale_bilinear(self.world.moisture, h, w)

        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        deep_ocean = np.array([12, 30, 75])
        shallow_ocean = np.array([38, 100, 160])
        shelf_ocean = np.array([65, 140, 190])
        beach = np.array([220, 200, 145])
        desert = np.array([210, 185, 120])
        savanna = np.array([145, 160, 75])
        grass_base = np.array([75, 138, 55])
        rock = np.array([115, 110, 105])
        freshwater_c = np.array([38, 115, 180])

        is_ocean = elev < self.world.sea_level
        ocean_depth = np.clip(elev / max(0.01, self.world.sea_level), 0, 1)
        is_shelf = is_ocean & (ocean_depth > 0.85)

        is_beach = (~is_ocean) & (elev < self.world.sea_level + 0.020)
        is_mountain = elev >= self.world.mountain_level
        is_lowland = (~is_ocean) & (~is_beach) & (elev < self.world.mountain_level - 0.12)
        is_land = (~is_ocean) & (~is_beach) & (~is_mountain)

        is_desert = is_lowland & (moist < 0.28)
        is_savanna = is_lowland & (moist >= 0.28) & (moist < 0.42)

        for c in range(3):
            ocean_col = (shallow_ocean[c] * ocean_depth + deep_ocean[c] * (1 - ocean_depth)).astype(np.uint8)
            ocean_col = np.where(is_shelf, shelf_ocean[c], ocean_col)
            rgb[..., c] = np.where(is_ocean, ocean_col, rgb[..., c])

            land_col = np.where(is_desert, desert[c], np.where(is_savanna, savanna[c], grass_base[c]))
            rgb[..., c] = np.where(is_land, land_col, rgb[..., c])
            rgb[..., c] = np.where(is_beach, beach[c], rgb[..., c])
            rgb[..., c] = np.where(is_mountain, rock[c], rgb[..., c])
            rgb[..., c] = np.where(self.world.freshwater, freshwater_c[c], rgb[..., c])

        land_norm = np.clip((elev - self.world.sea_level) / max(0.01, self.world.mountain_level - self.world.sea_level), 0.0, 1.0)
        dy, dx = np.gradient(elev)
        slope_light = 1.0 - (dx * 0.75 + dy * 0.75) * 240.0
        elev_tint = 1.08 - 0.22 * land_norm

        self._shade = np.clip(slope_light * elev_tint, 0.45, 1.35)
        self._apply_shade = (~is_ocean) & (~self.world.freshwater)
        self._static_base_rgb = rgb
        self._is_veg_zone = is_land & (~self.world.freshwater) & (~is_desert) & (~self.world.freshwater) & (~is_savanna)

    def _build_rgb(self) -> np.ndarray:
        s = self.world.SCALE

        if self.mode == 'composite':
            rgb = self._static_base_rgb.copy()
            forest = np.array([28, 85, 32])
            snow = np.array([235, 242, 250])
            ice_color = np.array([180, 222, 245])

            # Растягиваем быстрые слои (250x400) -> (1000x1600)
            h, w = self.world.height, self.world.width
            p = self.world._upscale_bilinear(self.world.plants, h, w)
            sn = self.world._upscale_bilinear(self.world.snow, h, w)
            ic = self.world._upscale_bilinear(self.world.ice, h, w)

            for c in range(3):
                rgb[..., c] = np.where(
                    self._is_veg_zone,
                    (rgb[..., c] * (1.0 - p * 0.65) + forest[c] * (p * 0.65)).astype(np.uint8),
                    rgb[..., c]
                )
                rgb[..., c] = (rgb[..., c] * (1.0 - sn) + snow[c] * sn).astype(np.uint8)
                rgb[..., c] = (rgb[..., c] * (1.0 - ic) + ice_color[c] * ic).astype(np.uint8)

            # Накладываем 3D-тени ПОСЛЕ снега — горы обретают объём и хребты!
            rgb = np.where(self._apply_shade[..., None], np.clip(rgb * self._shade[..., None], 0, 255).astype(np.uint8),
                           rgb)


        elif self.mode == 'elevation':
            h, w = self.world.height, self.world.width
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
            e = self.world.elevation
            rgb[..., 0] = (60 + 195 * e).astype(np.uint8)
            rgb[..., 1] = (40 + 210 * e).astype(np.uint8)
            rgb[..., 2] = (25 + 225 * e).astype(np.uint8)

        elif self.mode == 'moisture':
            h, w = self.world.height, self.world.width
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
            m = np.repeat(np.repeat(self.world.moisture, s, axis=0), s, axis=1)
            rgb[..., 0] = (220 * (1 - m)).astype(np.uint8)
            rgb[..., 1] = (180 * m + 80 * (1 - m)).astype(np.uint8)
            rgb[..., 2] = (220 * m).astype(np.uint8)

        elif self.mode == 'temperature':
            h, w = self.world.height, self.world.width
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
            temp = np.repeat(np.repeat(self.world.temperature, s, axis=0), s, axis=1)
            t = np.clip((temp + 20.0) / 62.0, 0.0, 1.0)
            rgb[..., 0] = np.clip(255 * (t * 1.6 - 0.5), 0, 255).astype(np.uint8)
            rgb[..., 1] = (220 * (1.0 - np.abs(t - 0.52) * 1.9).clip(0, 1)).astype(np.uint8)
            rgb[..., 2] = np.clip(255 * (1.0 - t * 1.5), 0, 255).astype(np.uint8)

        else:  # plants
            h, w = self.world.height, self.world.width
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
            p = np.repeat(np.repeat(self.world.plants, s, axis=0), s, axis=1)
            rgb[..., 0] = (120 * (1 - p)).astype(np.uint8)
            rgb[..., 1] = (80 + 170 * p).astype(np.uint8)
            rgb[..., 2] = (40 * (1 - p)).astype(np.uint8)

        # Ночное освещение
        light = self.world.light_level
        if light < 0.99:
            rgb[..., 0] = (rgb[..., 0] * light * 0.92).astype(np.uint8)
            rgb[..., 1] = (rgb[..., 1] * light * 0.96).astype(np.uint8)
            rgb[..., 2] = np.clip(rgb[..., 2] * (0.45 + 0.55 * light), 0, 255).astype(np.uint8)

        return rgb

    def set_mode(self, mode: str):
        self.mode = mode
        self.update_texture()
        self.update()

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

        # 1. Пиксель-арт падаль на земле
        painter.setPen(Qt.PenStyle.NoPen)
        for c in self.world.carcasses:
            cx, cy = int(c["x"]), int(c["y"])
            painter.fillRect(cx, cy, 2, 1, QColor(145, 35, 35))
            painter.fillRect(cx, cy + 1, 1, 1, QColor(210, 195, 175))  # пиксельная косточка

            # 2. Пиксель-арт животные (без овалов и сглаживания — чистые пиксели в сетке мира!)
            # Видимая область карты в координатах мира (не тратим время на рисование зверей за экраном!)
            vis_x0 = max(0, int(-self.offset.x() / self.zoom) - 6)
            vis_y0 = max(0, int(-self.offset.y() / self.zoom) - 6)
            vis_x1 = min(self.world.width, int((self.width() - self.offset.x()) / self.zoom) + 6)
            vis_y1 = min(self.world.height, int((self.height() - self.offset.y()) / self.zoom) + 6)

            # При отдалении камеры чуть увеличиваем спрайты, чтобы звери не превращались в 1 невидимый пиксель
            sprite_scale = float(np.clip(1.25 / max(self.zoom, 0.35), 1.0, 2.5))

            for agent in self.world.agents:
                painter.save()
                painter.translate(agent.x, agent.y)
                painter.scale(sprite_scale, sprite_scale)
                painter.rotate(np.degrees(agent.heading) + 90.0)

                h_s, w_s, _ = agent.sprite_rgba.shape
                raw_bytes = agent.sprite_rgba.tobytes()
                qimg = QImage(raw_bytes, w_s, h_s, w_s * 4, QImage.Format.Format_RGBA8888)
                painter.drawImage(-w_s // 2, -h_s // 2, qimg)
                painter.restore()

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        min_zoom = min(self.width() / self.world.width, self.height() / self.world.height)
        new_zoom = float(np.clip(self.zoom * factor, min_zoom, 20.0))

        mouse_pos = event.position()
        self.offset = mouse_pos - (mouse_pos - self.offset) * (new_zoom / self.zoom)
        self.zoom = new_zoom
        self._clamp_camera()
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

    def fit_to_window(self):
        if self.width() > 0 and self.height() > 0:
            self.zoom = min(self.width() / self.world.width, self.height() / self.world.height)
            self._clamp_camera()

    def showEvent(self, event):
        super().showEvent(event)

        if self.width() > 0 and self.height() > 0:
            fit_zoom = min(self.width() / self.world.width, self.height() / self.world.height)
            self.zoom = max(0.35, fit_zoom)
            self.offset_x = (self.width() - self.world.width * self.zoom) / 2.0
            self.offset_y = (self.height() - self.world.height * self.zoom) / 2.0
