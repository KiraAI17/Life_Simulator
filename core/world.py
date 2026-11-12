import numpy as np


class World:
    PRESETS = ["continent", "archipelago", "island", "pangaea"]

    def __init__(
        self,
        width: int = 500,
        height: int = 350,
        seed: int | None = None,
        preset: str = "continent",
        water_ratio: float = 0.45,
    ):
        self.width = width
        self.height = height
        self.preset = preset
        self.water_ratio = float(np.clip(water_ratio, 0.15, 0.80))
        self.rng = np.random.default_rng(seed)

        self.sea_level = 0.35
        self.mountain_level = 0.76
        self.beach_temp = 20.0
        self.peak_temp = -16.0
        self.ocean_temp = 5.0
        self.ideal_plant_temp = 20.0
        self.growth_speed = 0.003

        self.elevation = np.zeros((height, width))
        self.moisture = np.zeros((height, width))
        self.temperature = np.zeros((height, width))
        self.plants = np.zeros((height, width))
        self.snow = np.zeros((height, width))
        self.ice = np.zeros((height, width))
        self.freshwater = np.zeros((height, width), dtype=bool)

        self.tick = 0
        self.season_name = "Весна"
        self.day_time = 0.25 # 0.0..1.0 (0.25 = утро, 0.5 = полдень, 0.75 = вечер, 0.0 = полночь)
        self.light_level = 1.0 # яркость освещения: 0.35 ночью .. 1.0 днём
        self.is_night = False

        self.generate_map()

    def _fbm_noise(self, octaves: int = 6, base_freq: float = 0.008) -> np.ndarray:
        y, x = np.mgrid[0:self.height, 0:self.width].astype(float)

        warp_x = np.sin(x * 0.015 + y * 0.01) * 18.0 + np.cos(y * 0.03) * 8.0
        warp_y = np.cos(x * 0.01 - y * 0.015) * 18.0 + np.sin(x * 0.03) * 8.0
        wx = x + warp_x
        wy = y + warp_y

        total = np.zeros((self.height, self.width))
        amplitude = 1.0
        freq = base_freq
        max_amp = 0.0

        for _ in range(octaves):
            layer = np.zeros((self.height, self.width))
            for _ in range(4):
                angle = self.rng.uniform(0, 2 * np.pi)
                phase = self.rng.uniform(0, 2 * np.pi)
                f = freq * self.rng.uniform(0.85, 1.15)
                layer += np.sin((wx * np.cos(angle) + wy * np.sin(angle)) * f + phase)

            total += (layer / 4.0) * amplitude
            max_amp += amplitude
            amplitude *= 0.58
            freq *= 2.1

        total = total / max_amp
        return (total - total.min()) / (total.max() - total.min())

    def generate_map(self):
        y, x = np.mgrid[0:self.height, 0:self.width]
        nx = (x / self.width) * 2 - 1
        ny = (y / self.height) * 2 - 1
        dist = np.sqrt(nx**2 + ny**2)

        if self.preset == "continent":
            raw = self._fbm_noise(octaves=6, base_freq=0.007)
            edge_drop = np.maximum(0.0, dist - 0.65) * 0.75
            terrain = raw - edge_drop
        elif self.preset == "archipelago":
            raw = self._fbm_noise(octaves=6, base_freq=0.014)
            terrain = raw - 0.12 * dist
        elif self.preset == "island":
            raw = self._fbm_noise(octaves=6, base_freq=0.009)
            terrain = raw - 0.42 * (dist**1.3)
        else: # pangaea
            raw = self._fbm_noise(octaves=6, base_freq=0.006)
            terrain = raw - 0.25 * (dist**2)

        terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min())
        self.elevation = terrain
        self.sea_level = float(np.quantile(self.elevation, self.water_ratio))
        self.mountain_level = self.sea_level + (1.0 - self.sea_level) * 0.66

        base_moisture = self._fbm_noise(octaves=4, base_freq=0.009)
        coast_proximity = np.clip(1.0 - (self.elevation - self.sea_level) * 2.2, 0.0, 1.0)
        self.moisture = np.clip(base_moisture * 0.75 + coast_proximity * 0.35, 0.0, 1.0)

        is_land = self.elevation >= self.sea_level

        inland = self.elevation > (self.sea_level + 0.05)
        basin_noise = self._fbm_noise(octaves=4, base_freq=0.014)
        valley = (self.elevation > self.sea_level + 0.05) & (self.elevation < self.sea_level + (self.mountain_level - self.sea_level) * 0.55)
        self.freshwater = inland & valley & (basin_noise > 0.66)

        max_land_height = max(0.01, 1.0 - self.sea_level)
        temp_drop = self.beach_temp - self.peak_temp
        height_above_sea = np.maximum(0.0, self.elevation - self.sea_level)

        latitude_shift = ((y / self.height) - 0.6) * 11.0
        land_temp = self.beach_temp - (height_above_sea / max_land_height) * temp_drop + latitude_shift

        ocean_grad = np.clip(self.elevation / max(0.01, self.sea_level), 0.0, 1.0)
        water_temp = self.ocean_temp + 3.0 * ocean_grad + latitude_shift * 0.5

        self.base_temp = np.where(is_land, land_temp, water_temp)
        self.temperature = self.base_temp.copy()

        self.snow = np.where(self.elevation >= self.mountain_level, 1.0, 0.0)

        can_grow = is_land & (~self.freshwater) & (self.elevation < self.mountain_level) & (self.moisture > 0.32)
        start_grass = self.rng.uniform(0.15, 0.50, size=(self.height, self.width))
        self.plants = np.where(can_grow, start_grass, 0.0)

    def step(self):
        self.tick += 1

        # Суточный цикл: 1 сутки = 240 тиков (плавное освещение от 0.38 ночью до 1.0 днём)
        self.day_time = ((self.tick + 60) % 240) / 240.0
        sun_angle = (self.day_time - 0.25) * 2 * np.pi
        sun_height = np.sin(sun_angle) # -1 в полночь, +1 в полдень
        self.light_level = float(np.clip(0.68 + 0.35 * sun_height, 0.38, 1.0))
        self.is_night = sun_height < -0.15

        if self.tick % 5 != 0:
            return

        dt = 5.0
        # Длинные сезоны: полный год ~ 3900 тиков (в 3 раза длиннее прежнего)
        phase = (self.tick * 0.0016) % (2 * np.pi)
        if phase < np.pi / 2:
            self.season_name = "Весна"
        elif phase < np.pi:
            self.season_name = "Лето"
        elif phase < 3 * np.pi / 2:
            self.season_name = "Осень"
        else:
            self.season_name = "Зима"

        season_shift = np.sin(phase) * 10.5
        night_chill = sun_height * 2.5 # ночью прохладнее на 2.5°C, днём теплее на 2.5°C
        self.temperature = self.base_temp + season_shift + night_chill

        is_ocean = self.elevation < self.sea_level
        is_land = (~is_ocean) & (~self.freshwater) & (self.elevation < self.mountain_level)

        # Быстрое и плотное накопление снега зимой с низин и севера
        valley_humidity = np.clip(1.0 - (self.elevation - self.sea_level) * 2.0, 0.2, 1.0)
        cold_factor = np.clip((-self.temperature + 2.0 * valley_humidity) / 6.0, 0.0, 1.0)
        warm_factor = np.clip(self.temperature / 6.0, 0.0, 1.0)

        snow_delta = (0.006 * cold_factor * (0.4 + 0.6 * self.moisture) - 0.005 * warm_factor) * dt
        land_snow = np.clip(self.snow + snow_delta, 0.0, 1.0)
        self.snow = np.where(self.elevation >= self.mountain_level, 1.0, np.where(is_land, land_snow, 0.0))

        # Уверенное замерзание озёр (при < +0.5°C) и прибрежного моря (при < -0.5°C)
        is_water = is_ocean | self.freshwater
        freeze_threshold = np.where(self.freshwater, 0.5, -0.5 - (self.sea_level - self.elevation) * 22.0)
        freeze_delta = np.where(
            self.temperature < freeze_threshold,
            0.005 * dt * np.clip((freeze_threshold - self.temperature) / 4.0, 0.2, 1.0),
            -0.004 * dt * np.clip((self.temperature - freeze_threshold) / 4.0, 0.2, 1.0)
        )
        self.ice = np.where(is_water, np.clip(self.ice + freeze_delta, 0.0, 1.0), 0.0)

        # Рост травы (ночью и под снегом не растёт)
        temp_diff = np.abs(self.temperature - self.ideal_plant_temp)
        comfort = np.clip(1.0 - temp_diff / 22.0, 0.0, 1.0)
        wet_enough = np.maximum(0.0, self.moisture - 0.32)
        daylight_growth = max(0.0, sun_height)
        growth = (self.growth_speed * wet_enough * comfort * daylight_growth * (1.0 - self.snow) - 0.0003 * (1.0 - self.moisture)) * dt

        self.plants = np.where(is_land, np.clip(self.plants + growth, 0.0, 1.0), 0.0)

        if self.tick % 10 == 0:
            dy = self.rng.integers(-2, 3)
            dx = self.rng.integers(-2, 3)
            spores = np.roll(self.plants > 0.7, shift=(dy, dx), axis=(0, 1))
            new_sprouts = spores & is_land & (self.plants == 0.0) & (self.moisture > 0.35) & (self.snow < 0.2)
            self.plants = np.where(new_sprouts, 0.15, self.plants)
