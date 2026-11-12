import numpy as np


class World:
    def __init__(self, width: int = 200, height: int = 150, seed: int = 42):
        # Размеры карты в условных координатах (int)
        self.width = width
        self.height = height
        self.rng = np.random.default_rng(seed)

        # Пороги ландшафта и настройки климата (float) — можно менять из настроек UI
        self.sea_level = 0.30 # всё, что ниже — океан
        self.mountain_level = 0.75 # всё, что выше — горы (трава не растёт)
        self.beach_temp = 26.0 # базовая температура на уровне моря (°C)
        self.peak_temp = -7.0 # базовая температура на максимальной высоте (°C)
        self.ideal_plant_temp = 20.0
        self.growth_speed = 0.002

        # Слои карты для отрисовки и инспектора — 2D-массивы numpy.ndarray формы (height, width)
        self.elevation = np.zeros((height, width)) # высота рельефа: float от 0.0 до 1.0
        self.moisture = np.zeros((height, width)) # влажность почвы: float от 0.0 до 1.0
        self.temperature = np.zeros((height, width)) # текущая температура с учётом сезона: float (°C)
        self.plants = np.zeros((height, width)) # плотность травы в точке: float от 0.0 до 1.0
        self.freshwater = np.zeros((height, width), dtype=bool) # маска пресных озёр: bool (True — пресная вода)

        # Счётчик прошедших тиков симуляции (int)
        self.tick = 0

        self.generate_map()

    def _make_noise(self, waves: int = 6) -> np.ndarray:
        y, x = np.mgrid[0:self.height, 0:self.width]
        noise = np.zeros((self.height, self.width))

        for _ in range(waves):
            freq = self.rng.uniform(0.02, 0.08)
            angle = self.rng.uniform(0, 2 * np.pi)
            phase = self.rng.uniform(0, 2 * np.pi)
            noise += np.sin((x * np.cos(angle) + y * np.sin(angle)) * freq + phase)

        return (noise - noise.min()) / (noise.max() - noise.min())

    def generate_map(self):
        raw_height = self._make_noise(waves=8)
        y, x = np.mgrid[0:self.height, 0:self.width]

        center_x = (x / self.width) * 2 - 1
        center_y = (y / self.height) * 2 - 1
        dist_from_center = np.sqrt(center_x**2 + center_y**2)

        self.elevation = np.clip(raw_height - 0.25 * dist_from_center, 0.0, 1.0)
        self.moisture = self._make_noise(waves=5)

        is_land = self.elevation >= self.sea_level
        lowland = self.elevation < (self.sea_level + 0.08)
        self.freshwater = is_land & lowland & (self.moisture > 0.65)

        max_land_height = 1.0 - self.sea_level
        temp_drop = self.beach_temp - self.peak_temp
        height_above_sea = np.maximum(0.0, self.elevation - self.sea_level)

        self.base_temp = self.beach_temp - (height_above_sea / max_land_height) * temp_drop
        self.temperature = self.base_temp.copy()

        can_grow = is_land & (~self.freshwater) & (self.elevation < self.mountain_level) & (self.moisture > 0.3)
        start_grass = self.rng.uniform(0.1, 0.5, size=(self.height, self.width))
        self.plants = np.where(can_grow, start_grass, 0.0)

    def step(self):
        # Вызывать каждый тик симуляции: обновляет сезон, температуру, рост и споры травы
        self.tick += 1

        season_shift = np.sin(self.tick * 0.01) * 7.0
        self.temperature = self.base_temp + season_shift

        is_land = (self.elevation >= self.sea_level) & (~self.freshwater) & (self.elevation < self.mountain_level)
        temp_diff = np.abs(self.temperature - self.ideal_plant_temp)
        comfort = np.clip(1.0 - temp_diff / 25.0, 0.0, 1.0)

        growth = self.growth_speed * self.moisture * comfort
        self.plants = np.where(is_land, np.clip(self.plants + growth, 0.0, 1.0), 0.0)

        if self.tick % 10 == 0:
            dy = self.rng.integers(-2, 3)
            dx = self.rng.integers(-2, 3)
            spores = np.roll(self.plants > 0.7, shift=(dy, dx), axis=(0, 1))
            new_sprouts = spores & is_land & (self.plants == 0.0) & (self.moisture > 0.3)
            self.plants = np.where(new_sprouts, 0.15, self.plants)