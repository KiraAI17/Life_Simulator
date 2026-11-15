import numpy as np
from core.agent import Agent
from core.morphology import EvolutionConfig


class World:
    PRESETS = ["continents", "pangaea", "archipelago", "island"]
    SCALE = 4  # 1 клетка погоды/травы = 4x4 пикселя детального рельефа (ускорение в 16 раз!)

    def __init__(
            self,
            width: int = 1600,
            height: int = 1000,
            seed: int | None = None,
            preset: str = "continents",
            water_ratio: float = 0.48,
    ):
        self.width = width
        self.height = height
        self.sim_w = width // self.SCALE
        self.sim_h = height // self.SCALE
        self.preset = preset
        self.water_ratio = float(np.clip(water_ratio, 0.15, 0.80))
        self.rng = np.random.default_rng(seed)

        self.sea_level = 0.35
        self.mountain_level = 0.76
        self.beach_temp = 21.0
        self.peak_temp = -16.0
        self.ocean_temp = 5.0
        self.ideal_plant_temp = 20.0
        self.growth_speed = 0.0035

        # Детальные массивы полного разрешения (height, width) для точной навигации и рельефа
        self.elevation = np.zeros((height, width), dtype=np.float32)
        self.freshwater = np.zeros((height, width), dtype=bool)

        # Быстрые массивы климата и биомассы (sim_h, sim_w)
        self.sim_elev = np.zeros((self.sim_h, self.sim_w), dtype=np.float32)
        self.sim_freshwater = np.zeros((self.sim_h, self.sim_w), dtype=bool)
        self.moisture = np.zeros((self.sim_h, self.sim_w), dtype=np.float32)
        self.temperature = np.zeros((self.sim_h, self.sim_w), dtype=np.float32)
        self.plants = np.zeros((self.sim_h, self.sim_w), dtype=np.float32)
        self.snow = np.zeros((self.sim_h, self.sim_w), dtype=np.float32)
        self.ice = np.zeros((self.sim_h, self.sim_w), dtype=np.float32)

        self.tick = 0
        self.season_name = "Весна"
        self.day_time = 0.25
        self.light_level = 1.0
        self.is_night = False

        self.agents: list[Agent] = []
        self.carcasses: list[dict] = []

        self.evo_config = EvolutionConfig()

        self.generate_map()

    def _upscale_bilinear(self, arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
        """Быстрая векторная билинейная интерполяция на чистом NumPy (без лишних библиотек)."""
        in_h, in_w = arr.shape
        y = np.linspace(0, in_h - 1, out_h, dtype=np.float32)
        x = np.linspace(0, in_w - 1, out_w, dtype=np.float32)

        y0 = np.floor(y).astype(np.int32)
        x0 = np.floor(x).astype(np.int32)
        y1 = np.minimum(y0 + 1, in_h - 1)
        x1 = np.minimum(x0 + 1, in_w - 1)

        wy = (y - y0)[:, None]
        wx = (x - x0)[None, :]
        wx = (x - x0)[None, :]

        top = arr[y0[:, None], x0[None, :]] * (1.0 - wx) + arr[y0[:, None], x1[None, :]] * wx
        bot = arr[y1[:, None], x0[None, :]] * (1.0 - wx) + arr[y1[:, None], x1[None, :]] * wx
        return top * (1.0 - wy) + bot * wy

    def _value_noise_layer(self, grid_h: int, grid_w: int, out_h: int, out_w: int) -> np.ndarray:
        """Генерирует слой истинного случайного шума (без синусоид и периодичности)."""
        base = self.rng.random((grid_h, grid_w), dtype=np.float32)
        return self._upscale_bilinear(base, out_h, out_w)

    def _fbm_noise(self, h: int, w: int, octaves: int = 6, base_grid: tuple[int, int] = (4, 6),
                   ridged: bool = False) -> np.ndarray:
        total = np.zeros((h, w), dtype=np.float32)
        amplitude = 1.0
        max_amp = 0.0
        gh, gw = base_grid

        for _ in range(octaves):
            layer = self._value_noise_layer(gh, gw, h, w)
            if ridged:
                # Превращает гладкие холмы в острые горные хребты
                layer = 1.0 - np.abs(layer * 2.0 - 1.0)
                layer = layer ** 1.5

            total += layer * amplitude
            max_amp += amplitude
            amplitude *= 0.52
            gh = min(h, gh * 2)
            gw = min(w, gw * 2)

        total /= max_amp
        return (total - total.min()) / (total.max() - total.min() + 1e-6)

    def generate_map(self):
        h, w = self.height, self.width
        sh, sw = self.sim_h, self.sim_w
        sy, sx = np.mgrid[0:sh, 0:sw].astype(np.float32)
        nx = (sx / sw) * 2.0 - 1.0
        ny = (sy / sh) * 2.0 - 1.0

        # Базовый рельеф + острые горные цепи (ridged noise)
        base_relief = self._fbm_noise(sh, sw, octaves=6, base_grid=(4, 6), ridged=False)
        mountain_ridges = self._fbm_noise(sh, sw, octaves=5, base_grid=(5, 8), ridged=True)
        raw = base_relief * 0.72 + mountain_ridges * 0.28

        if self.preset == "continents":
            # 3 случайных центра тектонических плит -> сложные материки, проливы и заливы
            mask = np.zeros((sh, sw), dtype=np.float32)
            for _ in range(3):
                cx = self.rng.uniform(-0.55, 0.55)
                cy = self.rng.uniform(-0.45, 0.45)
                rx = self.rng.uniform(0.35, 0.55)
                ry = self.rng.uniform(0.40, 0.65)
                d = np.sqrt(((nx - cx) / rx) ** 2 + ((ny - cy) / ry) ** 2)
                mask = np.maximum(mask, np.clip(1.2 - d, 0.0, 1.0))
            low_terrain = raw * 0.65 + mask * 0.45
        elif self.preset == "pangaea":
            # Асимметричный суперматерик из 2 сросшихся плит с внутренними морями
            mask = np.zeros((sh, sw), dtype=np.float32)
            for cx, cy in [(-0.25, self.rng.uniform(-0.2, 0.2)), (0.28, self.rng.uniform(-0.2, 0.2))]:
                d = np.sqrt(((nx - cx) / 0.75) ** 2 + ((ny - cy) / 0.72) ** 2)
                mask = np.maximum(mask, np.clip(1.25 - d, 0.0, 1.0))
            low_terrain = raw * 0.60 + mask * 0.50
        elif self.preset == "archipelago":
            # Тот самый удачный архипелаг, но на чистом value-шуме
            dist = np.sqrt(nx ** 2 + ny ** 2)
            fine = self._fbm_noise(sh, sw, octaves=6, base_grid=(8, 12), ridged=False)
            low_terrain = fine * 0.85 + mountain_ridges * 0.15 - 0.15 * dist
        else:  # island
            dist = np.sqrt((nx * 1.1) ** 2 + (ny * 1.1) ** 2)
            low_terrain = raw * 0.75 - 0.55 * (dist ** 1.4) + 0.35

        terrain = self._upscale_bilinear(low_terrain, h, w)
        terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min() + 1e-6)
        self.elevation = terrain
        self.sea_level = float(np.quantile(self.elevation, self.water_ratio))
        self.mountain_level = self.sea_level + (1.0 - self.sea_level) * 0.68

        # Естественные гладкие озёра в низинах внутри континентов
        low_basin = self._fbm_noise(sh, sw, octaves=3, base_grid=(5, 8), ridged=False)
        basin_noise = self._upscale_bilinear(low_basin, h, w)
        valley = (self.elevation > self.sea_level + 0.07) & (
                self.elevation < self.sea_level + (self.mountain_level - self.sea_level) * 0.45)
        self.freshwater = valley & (basin_noise > 0.70)

        s = self.SCALE
        self.sim_elev = self.elevation.reshape(self.sim_h, s, self.sim_w, s).mean(axis=(1, 3))
        self.sim_freshwater = self.freshwater.reshape(self.sim_h, s, self.sim_w, s).any(axis=(1, 3))

        base_moisture = self._fbm_noise(self.sim_h, self.sim_w, octaves=4, base_grid=(4, 6), ridged=False)
        coast_proximity = np.clip(1.0 - (self.sim_elev - self.sea_level) * 2.2, 0.0, 1.0)
        self.moisture = np.clip(base_moisture * 0.75 + coast_proximity * 0.35, 0.0, 1.0)

        is_land = self.sim_elev >= self.sea_level
        max_land_height = max(0.01, 1.0 - self.sea_level)
        temp_drop = self.beach_temp - self.peak_temp
        height_above_sea = np.maximum(0.0, self.sim_elev - self.sea_level)

        latitude_shift = ((sy / self.sim_h) - 0.6) * 11.0
        land_temp = self.beach_temp - (height_above_sea / max_land_height) * temp_drop + latitude_shift
        ocean_grad = np.clip(self.sim_elev / max(0.01, self.sea_level), 0.0, 1.0)
        water_temp = self.ocean_temp + 3.0 * ocean_grad + latitude_shift * 0.5

        self.base_temp = np.where(is_land, land_temp, water_temp)
        self.temperature = self.base_temp.copy()

        self.snow = np.where(self.sim_elev >= self.mountain_level, 1.0, 0.0).astype(np.float32)
        self.ice = np.zeros((self.sim_h, self.sim_w), dtype=np.float32)

        # Плодородная суша + прибрежные отмели (водоросли и детрит)
        is_shallow_water = (self.sim_elev >= self.sea_level - 0.14) & (self.sim_elev < self.sea_level)
        is_fertile_land = is_land & (~self.sim_freshwater) & (self.sim_elev < self.mountain_level) & (
                self.moisture > 0.32)
        can_grow_biome = is_fertile_land | is_shallow_water

        start_grass = self.rng.uniform(0.20, 0.55, size=(self.sim_h, self.sim_w)).astype(np.float32)
        self.plants = np.where(can_grow_biome, start_grass, 0.0)

        self.spawn_initial_life(land_herbs=120, land_preds=30, aquatic_worms=90)

    # Быстрый доступ к климату/траве по координатам полного мира (x, y)
    def get_temp_at(self, ix: int, iy: int) -> float:
        return float(self.temperature[iy // self.SCALE, ix // self.SCALE])

    def get_snow_at(self, ix: int, iy: int) -> float:
        return float(self.snow[iy // self.SCALE, ix // self.SCALE])

    def get_ice_at(self, ix: int, iy: int) -> float:
        return float(self.ice[iy // self.SCALE, ix // self.SCALE])

    def get_plant_at(self, ix: int, iy: int) -> float:
        return float(self.plants[iy // self.SCALE, ix // self.SCALE])

    def eat_plant_at(self, ix: int, iy: int, amount: float) -> float:
        sx, sy = ix // self.SCALE, iy // self.SCALE
        avail = float(self.plants[sy, sx])
        eaten = min(avail, amount)
        self.plants[sy, sx] = avail - eaten
        return eaten

    def spawn_initial_life(
            self,
            land_herbs: int = 120,
            land_preds: int = 20,
            foxes: int = 24,
            aquatic_worms: int = 90,
    ):
        self.agents.clear()
        self.carcasses.clear()

        # 1. Сухопутные очаги (травоядные, волки и лисы)
        land_mask = (self.elevation >= self.sea_level) & (~self.freshwater) & (self.elevation < self.mountain_level)
        land_coords = np.argwhere(land_mask)
        if len(land_coords) > 0:
            herds = 8
            centers = land_coords[self.rng.choice(len(land_coords), size=min(herds, len(land_coords)), replace=False)]
            for cy, cx in centers:
                h_count = land_herbs // herds
                c_count = land_preds // herds
                f_count = foxes // herds

                spawns = (
                        ["land_grazer"] * h_count
                        + ["land_stalker"] * c_count
                        + ["fox"] * f_count
                )
                for arch in spawns:
                    genes = self.evo_config.spawn_genome(arch, self.rng)
                    for _ in range(20):
                        ax = float(cx + self.rng.uniform(-45, 45))
                        ay = float(cy + self.rng.uniform(-45, 45))
                        if Agent.is_tile_passable(self, int(ax), int(ay), genes):
                            self.agents.append(Agent(ax, ay, genes=genes, config=self.evo_config, rng=self.rng))
                            break

        # 2. Морской шельф и лагуны (водные черви / рыбы)
        shallow_mask = (self.elevation >= self.sea_level - 0.12) & (self.elevation < self.sea_level - 0.005)
        shallow_coords = np.argwhere(shallow_mask)
        if len(shallow_coords) > 0:
            chosen = shallow_coords[
                self.rng.choice(len(shallow_coords), size=min(aquatic_worms, len(shallow_coords)), replace=False)]
            for cy, cx in chosen:
                genes = self.evo_config.spawn_genome("aquatic_worm", self.rng)
                if Agent.is_tile_passable(self, int(cx), int(cy), genes):
                    self.agents.append(Agent(float(cx), float(cy), genes=genes, config=self.evo_config, rng=self.rng))

    def _update_agents(self):
        if not self.agents:
            return

        cell_size = 65  # радиус сектора под увеличенный обзор хищников
        grid: dict[tuple[int, int], list[Agent]] = {}
        for a in self.agents:
            key = (int(a.x) // cell_size, int(a.y) // cell_size)
            grid.setdefault(key, []).append(a)

        herb_count = sum(1 for a in self.agents if not a.is_predator)
        carn_count = len(self.agents) - herb_count
        herb_pressure = float(np.clip((herb_count - 240) / 320.0, 0.0, 1.0))
        carn_pressure = float(np.clip((carn_count - 60) / 90.0, 0.0, 1.0))

        newborns = []
        for a in self.agents:
            gx, gy = int(a.x) // cell_size, int(a.y) // cell_size
            neighbors = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    cell_list = grid.get((gx + dx, gy + dy))
                    if cell_list:
                        neighbors.extend(cell_list)

            pressure = carn_pressure if a.is_predator else herb_pressure
            child = a.update(self, neighbors, self.carcasses, pressure)
            if child is not None:
                newborns.append(child)

        if newborns:
            self.agents.extend(newborns)

        self.agents = [a for a in self.agents if a.alive]
        for c in self.carcasses:
            c["meat"] -= 0.04
        self.carcasses = [c for c in self.carcasses if c["meat"] > 0]

    def step(self):
        self.tick += 1

        self.day_time = ((self.tick + 150) % 600) / 600.0
        sun_angle = (self.day_time - 0.25) * 2 * np.pi
        sun_height = np.sin(sun_angle)
        self.light_level = float(np.clip(0.68 + 0.35 * sun_height, 0.38, 1.0))
        self.is_night = sun_height < -0.15

        self._update_agents()

        if self.tick % 5 != 0:
            return

        dt = 5.0
        phase = (self.tick * (2 * np.pi / 7200.0)) % (2 * np.pi)
        if phase < np.pi / 2:
            self.season_name = "Весна"
        elif phase < np.pi:
            self.season_name = "Лето"
        elif phase < 3 * np.pi / 2:
            self.season_name = "Осень"
        else:
            self.season_name = "Зима"

        season_shift = np.sin(phase) * 10.5
        night_chill = sun_height * 2.5
        self.temperature = self.base_temp + season_shift + night_chill

        is_ocean = self.sim_elev < self.sea_level
        is_land = (~is_ocean) & (~self.sim_freshwater) & (self.sim_elev < self.mountain_level)

        valley_humidity = np.clip(1.0 - (self.sim_elev - self.sea_level) * 2.0, 0.2, 1.0)
        cold_factor = np.clip((-self.temperature + 2.0 * valley_humidity) / 6.0, 0.0, 1.0)
        warm_factor = np.clip(self.temperature / 6.0, 0.0, 1.0)

        snow_delta = (0.006 * cold_factor * (0.4 + 0.6 * self.moisture) - 0.005 * warm_factor) * dt
        land_snow = np.clip(self.snow + snow_delta, 0.0, 1.0)
        self.snow = np.where(self.sim_elev >= self.mountain_level, 1.0, np.where(is_land, land_snow, 0.0))

        is_water = is_ocean | self.sim_freshwater
        freeze_threshold = np.where(self.sim_freshwater, 0.5, -0.5 - (self.sea_level - self.sim_elev) * 22.0)
        freeze_delta = np.where(
            self.temperature < freeze_threshold,
            0.005 * dt * np.clip((freeze_threshold - self.temperature) / 4.0, 0.2, 1.0),
            -0.004 * dt * np.clip((self.temperature - freeze_threshold) / 4.0, 0.2, 1.0)
        )
        self.ice = np.where(is_water, np.clip(self.ice + freeze_delta, 0.0, 1.0), 0.0)

        temp_diff = np.abs(self.temperature - self.ideal_plant_temp)
        comfort = np.clip(1.0 - temp_diff / 22.0, 0.0, 1.0)
        wet_enough = np.maximum(0.0, self.moisture - 0.32)
        daylight_growth = max(0.0, sun_height)
        growth = (self.growth_speed * wet_enough * comfort * daylight_growth * (1.0 - self.snow) - 0.00025 * (
                1.0 - self.moisture)) * dt

        is_shallow_water = (self.sim_elev >= self.sea_level - 0.035) & (self.sim_elev < self.sea_level)
        vegetated_zone = is_land | is_shallow_water
        self.plants = np.where(vegetated_zone, np.clip(self.plants + growth, 0.0, 1.0), 0.0)

        if self.tick % 10 == 0:
            dy = self.rng.integers(-2, 3)
            dx = self.rng.integers(-2, 3)
            spores = np.roll(self.plants > 0.7, shift=(dy, dx), axis=(0, 1))
            new_sprouts = spores & is_land & (self.plants == 0.0) & (self.moisture > 0.35) & (self.snow < 0.2)
            self.plants = np.where(new_sprouts, 0.15, self.plants)
