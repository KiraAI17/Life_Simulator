from typing import TYPE_CHECKING
import numpy as np
from core.morphology import (
    EvolutionConfig,
    G_CLAWS, G_DIET, G_FUR, G_GILLS, G_LATERAL, G_LEGS, G_LENGTH,
    G_LUNGS, G_NOCTURNAL, G_OPT_TEMP, G_SENSE, G_SHELL, G_SIZE, G_SPEED, G_WIDTH,
    G_HUE_SHIFT, G_SATURATION, G_BRIGHTNESS,
    build_sprite_rgba, calculate_crab_score,
)

if TYPE_CHECKING:
    from core.world import World


class Agent:
    def __init__(self, x: float, y: float, genes: np.ndarray, config: EvolutionConfig,
                 rng: np.random.Generator | None = None):
        self.rng = rng or np.random.default_rng()
        self.config = config
        self.genes = genes.astype(np.float32)

        self.x = float(x)
        self.y = float(y)
        self.vx = float(self.rng.uniform(-0.15, 0.15))
        self.vy = float(self.rng.uniform(-0.15, 0.15))
        self.heading = float(np.arctan2(self.vy, self.vx))

        self.radius = float(0.85 + 0.75 * self.genes[G_SIZE] + 0.45 * self.genes[G_WIDTH])
        self.mass = float(self.genes[G_SIZE] * (1.0 + 0.6 * self.genes[G_SHELL]))
        self.max_hp = float(45.0 * self.genes[G_SIZE] + 55.0 * self.genes[G_SHELL])
        self.hp = self.max_hp

        self.energy = 75.0
        self.hydration = 85.0
        self.oxygen = 100.0
        self.stamina = 100.0
        self.turn_dir = float(self.rng.choice([-1.0, 1.0]))
        self.stuck_counter = 0          # Счётчик тиков без движения (для детектора застревания)
        self.last_x, self.last_y = self.x, self.y  # Координаты для детектора движения
        self.ticks_without_food = 0  # Счётчик тиков без еды (для механики "нюха")
        self.sniff_target = None  # Целевая точка "нюха" (x, y)
        self.age = 0
        self.age = 0
        self.max_age = int(self.rng.integers(9500, 14500))
        self.reproduce_cooldown = 360
        self.attack_cooldown = 0
        self.state = "wandering"
        self.alive = True
        self.marked = False
        self.custom_name = ""

        self.crab_score = calculate_crab_score(self.genes)
        self.sprite_rgba = build_sprite_rgba(self.genes)

    @property
    def is_predator(self) -> bool:
        return bool(self.genes[G_DIET] > 0.35)

    @property
    def is_scavenger(self) -> bool:
        return bool(-0.35 <= self.genes[G_DIET] <= 0.45)

    @property
    def is_herbivore(self) -> bool:
        return bool(self.genes[G_DIET] < 0.25)

    def can_breathe_at(self, is_water: bool) -> bool:
        return bool(self.genes[G_GILLS] >= 0.42) if is_water else bool(self.genes[G_LUNGS] >= 0.42)

    @staticmethod
    def is_tile_passable(world: "World", ix: int, iy: int, g: np.ndarray) -> bool:
        if ix < 4 or ix >= world.width - 4 or iy < 4 or iy >= world.height - 4:
            return False
        elev = world.elevation[iy, ix]
        if elev >= world.mountain_level - 0.11:
            return False
        is_water = (elev < world.sea_level) or world.freshwater[iy, ix]
        if is_water:
            is_frozen = (world.get_ice_at(ix, iy) > 0.65) and (world.get_temp_at(ix, iy) < -0.5)
            if is_frozen:
                return bool(g[G_LUNGS] > 0.35 or g[G_LEGS] >= 4.0)
            return bool(g[G_GILLS] > 0.25 or g[G_LEGS] >= 4.0)
        return bool(g[G_LUNGS] > 0.25 or g[G_LEGS] >= 2.0)

    def _steer(self, tx: float, ty: float, weight: float = 1.0):
        dx, dy = tx - self.x, ty - self.y
        dist = np.hypot(dx, dy)
        if dist > 1e-4:
            self.vx += (dx / dist) * 0.085 * weight
            self.vy += (dy / dist) * 0.085 * weight

    def _steer_lateral(self, threat_x: float, threat_y: float, weight: float = 1.0):
        dx, dy = self.x - threat_x, self.y - threat_y
        dist = np.hypot(dx, dy)
        if dist > 1e-4:
            nx, ny = dx / dist, dy / dist
            side = 1.0 if (self.age // 50) % 2 == 0 else -1.0
            lat = self.genes[G_LATERAL]
            self.vx += (nx * (1.0 - 0.5 * lat) + (-ny * side) * lat) * 0.065 * weight
            self.vy += (ny * (1.0 - 0.5 * lat) + (nx * side) * lat) * 0.065 * weight

    def _seek_safe_medium(self, world: "World", ix: int, iy: int, want_water: bool):
        r = 36
        x0, x1 = max(4, ix - r), min(world.width - 4, ix + r + 1)
        y0, y1 = max(4, iy - r), min(world.height - 4, iy + r + 1)
        sub_elev = world.elevation[y0:y1, x0:x1]
        sub_fw = world.freshwater[y0:y1, x0:x1]
        mask = (sub_elev < world.sea_level) | sub_fw if want_water else (sub_elev >= world.sea_level) & (
                    sub_elev < world.mountain_level) & (~sub_fw)
        coords = np.argwhere(mask)
        if len(coords) > 0:
            dists = (coords[:, 0] + y0 - iy) ** 2 + (coords[:, 1] + x0 - ix) ** 2
            by, bx = coords[np.argmin(dists)]
            self._steer(x0 + bx + 0.5, y0 + by + 0.5, weight=1.6)
        else:
            self._steer(world.width / 2.0, world.height / 2.0, weight=1.0)

    def _attack(self, target: "Agent", carcasses: list[dict]):
        if self.attack_cooldown > 0 or not target.alive:
            return
        self.attack_cooldown = 14
        raw_dmg = 8.0 + 18.0 * self.genes[G_CLAWS] + 6.0 * self.genes[G_SIZE]
        final_dmg = max(1.5, raw_dmg * (1.0 - 0.68 * target.genes[G_SHELL]))
        target.hp -= final_dmg
        if target.genes[G_CLAWS] > 0.25 or target.genes[G_SHELL] > 0.45:
            self.hp -= max(0.0, (4.0 * target.genes[G_CLAWS] + 2.5 * target.genes[G_SHELL]) * (
                        1.0 - 0.5 * self.genes[G_SHELL]))
        if target.hp <= 0:
            target.alive = False
            meat_total = 45.0 * target.genes[G_SIZE] * (1.0 - 0.35 * target.genes[G_SHELL])
            self.energy = min(100.0, self.energy + meat_total * 0.55)
            self.ticks_without_food = 0
            carcasses.append({"x": target.x, "y": target.y, "meat": meat_total * 0.45})

    def rebuild_phenotype(self):
        self.radius = float(0.85 + 0.75 * self.genes[G_SIZE] + 0.45 * self.genes[G_WIDTH])
        self.mass = float(self.genes[G_SIZE] * (1.0 + 0.6 * self.genes[G_SHELL]))
        self.max_hp = float(45.0 * self.genes[G_SIZE] + 55.0 * self.genes[G_SHELL])
        self.hp = min(self.hp, self.max_hp)
        self.crab_score = calculate_crab_score(self.genes)
        self.sprite_rgba = build_sprite_rgba(self.genes)

    def apply_archetype(self, archetype: str):
        self.genes = self.config.spawn_genome(archetype, self.rng)
        self.rebuild_phenotype()
        self.hp = self.max_hp
        self.energy = 85.0
        self.hydration = 90.0
        self.oxygen = 100.0

    def _is_native_medium(self, world: "World", ix: int, iy: int) -> bool:
        """Проверяет, может ли существо нормально дышать в клетке (ix, iy)."""
        ix = int(np.clip(ix, 0, world.width - 1))
        iy = int(np.clip(iy, 0, world.height - 1))
        is_water = (world.elevation[iy, ix] < world.sea_level) or world.freshwater[iy, ix]
        if is_water and (world.get_ice_at(ix, iy) > 0.78) and (world.get_temp_at(ix, iy) < -0.5):
            is_water = False
        return self.can_breathe_at(is_water)

    def _has_safe_path(self, world: "World", tx: float, ty: float, steps: int = 6) -> bool:
        """Проверяет по точкам отрезка, не придётся ли плыть/идти через среду без кислорода."""
        # Чисто водные (акулы, рыбы) — проверяем только конечную точку,
        if self.genes[G_GILLS] > 0.7 and self.genes[G_LEGS] < 2.0:
            px = int(np.clip(tx, 0, world.width - 1))
            py = int(np.clip(ty, 0, world.height - 1))
            return self.is_tile_passable(world, px, py, self.genes) and self._is_native_medium(world, px, py)
        
        # Для остальных — полная проверка по точкам отрезка
        for i in range(1, steps + 1):
            t = i / steps
            px = int(self.x + (tx - self.x) * t)
            py = int(self.y + (ty - self.y) * t)
            if not self.is_tile_passable(world, px, py, self.genes):
                return False
            if not self._is_native_medium(world, px, py):
                return False
        return True

    def update(self, world: "World", neighbors: list["Agent"], carcasses: list[dict],
               population_pressure: float) -> "Agent | None":
        if not self.alive:
            return None
        self.age += 1
        
        moved_dist = np.hypot(self.x - self.last_x, self.y - self.last_y)
        if moved_dist < 0.5:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        self.last_x, self.last_y = self.x, self.y
        
        if self.reproduce_cooldown > 0: self.reproduce_cooldown -= 1
        if self.attack_cooldown > 0: self.attack_cooldown -= 1
        
        if self.is_predator:
            self.ticks_without_food += 1

        ix, iy = int(np.clip(self.x, 0, world.width - 1)), int(np.clip(self.y, 0, world.height - 1))
        elev = world.elevation[iy, ix]
        is_water = (elev < world.sea_level) or world.freshwater[iy, ix]
        on_ice = is_water and (world.get_ice_at(ix, iy) > 0.78) and (world.get_temp_at(ix, iy) < -0.5)
        in_liquid_water = is_water and not on_ice
        local_temp = world.get_temp_at(ix, iy)
        s = world.SCALE
        costs = self.config.costs

        if self.can_breathe_at(in_liquid_water):
            self.oxygen = min(100.0, self.oxygen + 4.0)
        else:
            self.oxygen -= 1.8
            if self.oxygen <= 0:
                self.oxygen = 0.0
                self.hp -= costs["suffocation_damage"]
                self.state = "suffocating"
                self._seek_safe_medium(world, ix, iy, want_water=not in_liquid_water)

        if self.hp <= 0:
            self.alive = False
            carcasses.append({"x": self.x, "y": self.y, "meat": 30.0 * self.genes[G_SIZE]})
            return None
        if self.hp < self.max_hp and self.energy > 65.0:
            self.hp = min(self.max_hp, self.hp + 0.08)

        nearest_threat, nearest_prey = None, None
        nearest_threat_dist, nearest_prey_dist = self.genes[G_SENSE], self.genes[G_SENSE]
        flock_x, flock_y, flock_count = 0.0, 0.0, 0

        am_in_native = self._is_native_medium(world, ix, iy)

        for other in neighbors:
            if other is self or not other.alive:
                continue
            dx, dy = other.x - self.x, other.y - self.y
            d = np.hypot(dx, dy)
            min_dist = (self.radius + other.radius) * 1.15
            is_hunting_target = self.is_predator and (not other.is_predator)
            if (not is_hunting_target) and 1e-4 < d < min_dist:
                push = ((min_dist - d) / min_dist) * 0.65 * (other.mass / (self.mass + other.mass))
                self.vx -= (dx / d) * push
                self.vy -= (dy / d) * push
            if d > self.genes[G_SENSE]:
                continue

            threat_power = other.genes[G_DIET] + 0.4 * other.genes[G_CLAWS] + 0.3 * other.genes[G_SIZE]
            my_power = self.genes[G_DIET] + 0.4 * self.genes[G_CLAWS] + 0.3 * self.genes[G_SIZE]

            # 1. Угроза: спящий хищник не пугает (если не наступить на него < 4 px), и угроза из чужой среды не пугает
            other_sleeping = getattr(other, "state", "") == "sleeping"
            if other.is_predator and threat_power > my_power + 0.30 and d < nearest_threat_dist:
                if (not other_sleeping or d < 4.5) and other._is_native_medium(world, ix, iy):
                    nearest_threat_dist, nearest_threat = d, other

            # 2. Добыча: охотимся только если путь до жертвы лежит в нашей родной среде
            elif (
                    self.is_predator
                    and am_in_native
                    and (not other.is_predator or self.energy < 22.0)
                    and my_power > threat_power - 0.15
                    and d < nearest_prey_dist
                    and self._has_safe_path(world, other.x, other.y)
            ):
                nearest_prey_dist, nearest_prey = d, other

            elif abs(other.genes[G_DIET] - self.genes[G_DIET]) < 0.35 and 7.0 < d < 22.0:
                flock_x += other.x
                flock_y += other.y
                flock_count += 1

        effective_temp = local_temp + self.genes[G_FUR] * 16.0
        organ_cost = self.genes[G_SHELL] * costs["shell_maintenance"] + self.genes[G_CLAWS] * costs[
            "claw_maintenance"] + self.genes[G_FUR] * costs["fur_maintenance"]
        move_mult = 0.5 if (self.is_predator and self.state == "wandering") else 1.0
        metabolism = costs["base_metabolism"] * (self.genes[G_SIZE] ** costs["size_exponent"]) + costs[
            "speed_cost"] * move_mult * (self.genes[G_SPEED] ** costs["speed_exponent"]) + organ_cost + abs(
            effective_temp - self.genes[G_OPT_TEMP]) * costs["temp_stress_factor"]

        self.vx *= 0.91
        self.vy *= 0.91

        wants_sleep = world.is_night != (self.genes[G_NOCTURNAL] > 0.5)
        sprinting = False
        is_thirsty = (not in_liquid_water) and (
                    (self.hydration < 50.0) or (self.state == "drinking" and self.hydration < 88.0))
        is_hungry = (self.energy < 72.0) or (self.state in ("foraging", "hunting") and self.energy < 82.0)

        if self.oxygen < 65.0 or (self.state == "suffocating" and self.oxygen < 85.0):
            self.state = "suffocating"
            self._seek_safe_medium(world, ix, iy, want_water=not in_liquid_water)
            sprinting = True
        elif nearest_threat is not None:
            self.state = "fleeing"
            self._steer_lateral(nearest_threat.x, nearest_threat.y, weight=1.25)
            if self.stamina > 15.0:
                sprinting, self.stamina, self.energy = True, self.stamina - 0.45, self.energy - 0.012
        elif is_thirsty:
            self.state = "drinking"
            if world.get_snow_at(ix, iy) > 0.35:
                self.hydration = min(100.0, self.hydration + 10.0)
            else:
                sr = 45  # Дальний поиск водоёмов (до 180 px)
                six, siy = ix // s, iy // s
                y_start, y_end = max(0, siy - sr), min(world.sim_h, siy + sr + 1)
                x_start, x_end = max(0, six - sr), min(world.sim_w, six + sr + 1)
                water_cells = np.argwhere(world.sim_freshwater[y_start:y_end, x_start:x_end])
                if len(water_cells) > 0:
                    # Берём БЛИЖАЙШУЮ клетку пресной воды, а не первую в массиве
                    dists_sq = (water_cells[:, 0] + y_start - siy) ** 2 + (water_cells[:, 1] + x_start - six) ** 2
                    wy, wx = water_cells[int(np.argmin(dists_sq))]
                    tx = (x_start + wx + 0.5) * s
                    ty = (y_start + wy + 0.5) * s
                    self._steer(tx, ty, weight=0.95)
                    if np.hypot(tx - self.x, ty - self.y) < 6.0:
                        self.hydration = min(100.0, self.hydration + 30.0)
                        self.vx = -self.vx * 1.1 + self.rng.uniform(-0.12, 0.12)
                        self.vy = -self.vy * 1.1 + self.rng.uniform(-0.12, 0.12)
                else:
                    # Если озёр рядом нет — идём в сторону максимальной влажности (к низинам/рекам)
                    moist_patch = world.moisture[y_start:y_end, x_start:x_end]
                    if moist_patch.size > 0:
                        my, mx = np.unravel_index(int(np.argmax(moist_patch)), moist_patch.shape)
                        self._steer((x_start + mx + 0.5) * s, (y_start + my + 0.5) * s, weight=0.75)
        elif is_hungry:
            ate_carcass = False
            if self.is_predator or self.is_scavenger:
                for c in carcasses:
                    if (
                            c["meat"] > 0
                            and np.hypot(c["x"] - self.x, c["y"] - self.y) < self.genes[G_SENSE]
                            and self._has_safe_path(world, c["x"], c["y"])
                    ):
                        self.state = "foraging"
                        self._steer(c["x"], c["y"], weight=0.75)
                        if np.hypot(c["x"] - self.x, c["y"] - self.y) <= self.radius + 1.5:
                            bite = min(c["meat"], 14.0 + 8.0 * self.genes[G_CLAWS])
                            c["meat"] -= bite
                            self.energy = min(100.0, self.energy + bite * 1.4)
                            self.ticks_without_food = 0
                        ate_carcass = True
                        break
            if not ate_carcass:
                if self.is_herbivore and world.get_plant_at(ix, iy) > 0.16:
                    self.state = "foraging"
                    eat_radius = max(1, int(self.genes[G_SIZE]))
                    eaten = world.eat_plant_area(ix, iy, 0.8, radius=eat_radius)
                    self.energy = min(100.0,
                                      self.energy + eaten * 18.0 * np.clip(1.0 - max(0.0, self.genes[G_DIET]), 0.3,
                                                                           1.0))
                    self.vx *= 0.55
                    self.vy *= 0.55
                elif self.is_predator and nearest_prey is not None:
                    self.state = "hunting"
                    self._steer(nearest_prey.x, nearest_prey.y, weight=0.9)
                    if nearest_prey_dist < 18.0 and self.stamina > 20.0:
                        sprinting, self.stamina, self.energy = True, self.stamina - 0.4, self.energy - 0.006
                    if nearest_prey_dist <= self.radius + nearest_prey.radius + 1.7:
                        self._attack(nearest_prey, carcasses)
                elif self.is_herbivore:
                    self.state = "foraging"
                    sr = max(8, int(min(self.genes[G_SENSE], 72) // s))
                    six, siy = ix // s, iy // s
                    patch = world.plants[max(0, siy - sr):min(world.sim_h, siy + sr + 1),
                            max(0, six - sr):min(world.sim_w, six + sr + 1)]
                    if patch.size > 0 and patch.max() > 0.18:
                        by, bx = np.unravel_index(np.argmax(patch), patch.shape)
                        self._steer((max(0, six - sr) + bx + 0.5) * s, (max(0, siy - sr) + by + 0.5) * s, weight=0.55)
                    else:
                        self.vx += self.rng.uniform(-0.025, 0.025)
                        self.vy += self.rng.uniform(-0.025, 0.025)
                else:
                    self.state = "wandering"
                    self.vx += self.rng.uniform(-0.025, 0.025)
                    self.vy += self.rng.uniform(-0.025, 0.025)
        elif (wants_sleep and am_in_native and nearest_prey is None 
                and self.energy > (55.0 if self.is_predator else 35.0)  #Хищники не спят сытыми
                and self.hydration > 35.0 and self.oxygen > 75.0):
            self.state = "sleeping"
            if not in_liquid_water:
                self.hydration -= 0.012
            self.stamina = min(100.0, self.stamina + 0.5)
            return None
        elif local_temp < self.genes[G_OPT_TEMP] - 11.0 and self.energy < 75.0:
            # Сезонная миграция в более тёплые широты (к центру/югу карты)
            self.state = "migrating"
            self._steer(self.x + self.rng.uniform(-15, 15), min(world.height - 30.0, self.y + 45.0), weight=0.65)
        else:
            self.stamina = min(100.0, self.stamina + 0.25)
            if 0 < flock_count < 5:
                self._steer(flock_x / flock_count, flock_y / flock_count, weight=0.10)
            
                #ВОДНЫЕ ХИЩНИКИ (акулы) — активное патрулирование широкими дугами
                if self.is_predator and self.genes[G_GILLS] > 0.7 and self.genes[G_LEGS] < 2.0:
                    self.state = "patrolling"
                    # Увеличенная скорость патруля (0.085 вместо 0.065)
                    sweep = 0.12 * self.turn_dir  # Увеличенный угол поворота (0.12 вместо 0.09)
                    self.vx += np.cos(self.heading + sweep) * 0.085 + self.rng.uniform(-0.015, 0.015)
                    self.vy += np.sin(self.heading + sweep) * 0.085 + self.rng.uniform(-0.015, 0.015)
                    # Более частая смена направления (1.8% вместо 1.2%)
                    if self.rng.random() < 0.018:
                        self.turn_dir = -self.turn_dir
                    # Более частые рывки (0.8% вместо 0.4%)
                    if self.rng.random() < 0.008:
                        self.vx *= 2.0  # Усиленный рывок (×2.0 вместо ×1.8)
                        self.vy *= 2.0
                
                else:
                    self.stamina = min(100.0, self.stamina + 0.25)
                    if 0 < flock_count < 5:
                        self._steer(flock_x / flock_count, flock_y / flock_count, weight=0.10)
                        
                    is_wolf = (self.is_predator and 
                        self.genes[G_FUR] > 0.20 and 
                        self.genes[G_FUR] < 0.55 and 
                        self.genes[G_LEGS] >= 4.0 and
                        self.genes[G_GILLS] < 0.4)

                    #МЕХАНИКА "НЮХА": волк долго без еды -> с шансом 2.5% чует добычу
                    if is_wolf and self.ticks_without_food >= 250:
                        self.state = "sniffing"
                        
                        # Каждый тик с шансом 2.5% ищем ближайшего травоядного
                        if self.rng.random() < 0.025 or self.sniff_target is None:
                            nearest_herb = None
                            nearest_herb_dist = 600.0  # Радиус "нюха" (большой)
                            for other in world.agents:
                                if not other.alive or other.is_predator:
                                    continue
                                d = np.hypot(other.x - self.x, other.y - self.y)
                                if d < nearest_herb_dist:
                                    nearest_herb_dist = d
                                    nearest_herb = other
                            
                            if nearest_herb is not None:
                                # Случайная точка в радиусе 100 px вокруг травоядного
                                angle = self.rng.uniform(0, 2 * np.pi)
                                radius = self.rng.uniform(0, 100.0)
                                self.sniff_target = (
                                    nearest_herb.x + radius * np.cos(angle),
                                    nearest_herb.y + radius * np.sin(angle)
                                )
                        
                        # Идём к цели "нюха"
                        if self.sniff_target is not None:
                            self._steer(self.sniff_target[0], self.sniff_target[1], weight=0.8)
                            # Если дошли — сбрасываем цель
                            if np.hypot(self.sniff_target[0] - self.x, self.sniff_target[1] - self.y) < 10.0:
                                self.sniff_target = None
                        else:
                            # Патрулируем, если цели нет
                            self.vx += np.cos(self.heading) * 0.065 + self.rng.uniform(-0.025, 0.025)
                            self.vy += np.sin(self.heading) * 0.065 + self.rng.uniform(-0.025, 0.025)

    
                    elif self.is_predator:
                        self.state = "patrolling"
                        # Увеличенная скорость патруля (0.065 вместо 0.022)
                        self.vx += np.cos(self.heading) * 0.065 + self.rng.uniform(-0.025, 0.025)
                        self.vy += np.sin(self.heading) * 0.065 + self.rng.uniform(-0.025, 0.025)
                        
                        # Периодические развороты (имитация поиска в разных направлениях)
                        if self.rng.random() < 0.015:
                            self.turn_dir = -self.turn_dir
                            self.heading += self.rng.uniform(-0.5, 0.5)  # Случайное отклонение курса
                        
                        # Рывки вперёд (бросок на замеченное движение)
                        if self.rng.random() < 0.008:
                            self.vx *= 1.9
                            self.vy *= 1.9
                    
                    else:
                        self.state = "wandering"
                        self.vx += self.rng.uniform(-0.025, 0.025)
                        self.vy += self.rng.uniform(-0.025, 0.025)
                
        margin = 24.0
        if self.x < margin:
            self.vx += 0.08
        elif self.x > world.width - margin:
            self.vx -= 0.08
        if self.y < margin:
            self.vy += 0.08
        elif self.y > world.height - margin:
            self.vy -= 0.08

        self.energy -= metabolism
        if in_liquid_water:
            self.hydration = min(100.0, self.hydration + 1.5)
        else:
            self.hydration -= (0.024 + max(0.0, (local_temp - 20.0) * 0.0015)) * (1.0 - 0.55 * self.genes[G_SHELL])

        if self.energy <= 0 or self.hydration <= 0 or self.age >= (self.max_age * world.lifespan_mult):
            self.alive = False
            carcasses.append({"x": self.x, "y": self.y, "meat": 35.0 * self.genes[G_SIZE]})
            return None

        shell_penalty = 1.0 - costs["shell_weight_speed_penalty"] * self.genes[G_SHELL]
        medium_mult = np.clip(0.65 + 0.45 * self.genes[G_GILLS] - 0.25 * self.genes[G_WIDTH], 0.45,
                              1.25) if in_liquid_water else (0.45 + 0.55 * np.clip(self.genes[G_LEGS] / 4.0, 0.0, 1.2))
        slope_penalty = 0.75 if elev > world.mountain_level - 0.10 else 1.0
        snow_penalty = 0.72 if world.get_snow_at(ix, iy) > 0.5 else 1.0
        sprint_mult = 1.55 if sprinting else (0.72 if (self.is_predator and self.state == "wandering") else 1.0)
        max_spd = self.genes[G_SPEED] * shell_penalty * medium_mult * slope_penalty * snow_penalty * sprint_mult

        v_norm = np.hypot(self.vx, self.vy)
        if v_norm > max_spd and v_norm > 1e-6:
            self.vx, self.vy = (self.vx / v_norm) * max_spd, (self.vy / v_norm) * max_spd

        if v_norm > 0.04:
            target_angle = float(np.arctan2(self.vy, self.vx))
            diff = (target_angle - self.heading + np.pi) % (2 * np.pi) - np.pi
            self.heading += diff * 0.18

        nx, ny = self.x + self.vx, self.y + self.vy
        next_passable = self.is_tile_passable(world, int(nx), int(ny), self.genes)

        is_water_ahead = (world.elevation[int(np.clip(ny, 0, world.height - 1)), int(np.clip(nx, 0, world.width - 1))] < world.sea_level)
        avoid_water = (not self.is_predator) and (self.genes[G_GILLS] < 0.42) and is_water_ahead

        if next_passable and not avoid_water:
            self.x, self.y = nx, ny
            self.stuck_counter = 0  # Сброс счётчика
        else:
            self.stuck_counter += 1
            
            # Пробуем поворот в обе стороны на 60°
            for angle_offset in (0.9, -0.9, 1.5, -1.5, 2.2, -2.2):
                test_heading = self.heading + angle_offset
                test_vx = np.cos(test_heading) * max_spd
                test_vy = np.sin(test_heading) * max_spd
                test_x, test_y = self.x + test_vx, self.y + test_vy
                test_passable = self.is_tile_passable(world, int(test_x), int(test_y), self.genes)
                test_water = (world.elevation[int(np.clip(test_y, 0, world.height - 1)), int(np.clip(test_x, 0, world.width - 1))] < world.sea_level)
                test_avoid = (not self.is_predator) and (self.genes[G_GILLS] < 0.42) and test_water
                
                if test_passable and not test_avoid:
                    self.heading = test_heading
                    self.vx = test_vx
                    self.vy = test_vy
                    break
            else:
                # Если ни одно направление не подошло — разворот на 180°
                self.heading += np.pi
                self.vx = np.cos(self.heading) * max_spd * 0.5
                self.vy = np.sin(self.heading) * max_spd * 0.5
            
            if self.stuck_counter >= 8:
                self.heading += np.pi + self.rng.uniform(-0.3, 0.3)
                self.vx = np.cos(self.heading) * max_spd
                self.vy = np.sin(self.heading) * max_spd
                self.stuck_counter = 0

        base_repro = 84.0 if self.is_herbivore else 86.0
        repro_threshold = (base_repro + population_pressure * 14.0) / max(0.4, world.repro_rate_mult)
        if self.energy > repro_threshold and self.hydration > 65.0 and self.oxygen > 70.0 and self.reproduce_cooldown == 0:
            for _ in range(6):
                cx, cy = self.x + self.rng.uniform(-3.0, 3.0), self.y + self.rng.uniform(-3.0, 3.0)
                if self.is_tile_passable(world, int(cx), int(cy), self.genes):
                    self.energy -= 42.0 if self.is_herbivore else 45.0
                    base_cd = 500 if self.is_herbivore else 650
                    self.reproduce_cooldown = int(base_cd / max(0.4, world.repro_rate_mult))
                    return Agent(cx, cy, genes=self.config.mutate(self.genes, self.rng), config=self.config,
                                 rng=self.rng)
        return None
