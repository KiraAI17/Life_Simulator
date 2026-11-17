from typing import TYPE_CHECKING
import numpy as np

from core.morphology import (
    EvolutionConfig,
    G_CLAWS,
    G_DIET,
    G_FUR,
    G_GILLS,
    G_LATERAL,
    G_LEGS,
    G_LENGTH,
    G_LUNGS,
    G_NOCTURNAL,
    G_OPT_TEMP,
    G_SENSE,
    G_SHELL,
    G_SIZE,
    G_SPEED,
    G_WIDTH,
    build_sprite_rgba,
    calculate_crab_score,
)

if TYPE_CHECKING:
    from core.world import World


class Agent:
    def __init__(
        self,
        x: float,
        y: float,
        genes: np.ndarray,
        config: EvolutionConfig,
        rng: np.random.Generator | None = None,
    ):
        self.rng = rng or np.random.default_rng()
        self.config = config
        self.genes = genes.astype(np.float32)

        self.x = float(x)
        self.y = float(y)
        self.vx = float(self.rng.uniform(-0.15, 0.15))
        self.vy = float(self.rng.uniform(-0.15, 0.15))
        self.heading = float(np.arctan2(self.vy, self.vx))

        # Физический радиус для мягкой коллизии и масса
        self.radius = float(0.85 + 0.75 * self.genes[G_SIZE] + 0.45 * self.genes[G_WIDTH])
        self.mass = float(self.genes[G_SIZE] * (1.0 + 0.6 * self.genes[G_SHELL]))

        # Здоровье зависит от размера и прочности панциря
        self.max_hp = float(45.0 * self.genes[G_SIZE] + 55.0 * self.genes[G_SHELL])
        self.hp = self.max_hp

        self.energy = 75.0
        self.hydration = 85.0
        self.oxygen = 100.0
        self.stamina = 100.0
        self.turn_dir = float(self.rng.choice([-1.0, 1.0]))

        self.age = 0
        self.max_age = int(self.rng.integers(9500, 14500))
        self.reproduce_cooldown = 360
        self.attack_cooldown = 0
        self.state = "wandering" # wandering | foraging | hunting | fleeing | drinking | suffocating | sleeping
        self.alive = True

        # Предвычисленные данные для аналитики и рендера (PyQt / PyGame)
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
        if is_water:
            return bool(self.genes[G_GILLS] >= 0.42)
        return bool(self.genes[G_LUNGS] >= 0.42)

    @staticmethod
    def is_tile_passable(world: "World", ix: int, iy: int, g: np.ndarray) -> bool:
        if ix < 4 or ix >= world.width - 4 or iy < 4 or iy >= world.height - 4:
            return False
        elev = world.elevation[iy, ix]
        if elev >= world.mountain_level:
            return False

        is_water = (elev < world.sea_level) or world.freshwater[iy, ix]
        if is_water:
            is_frozen = (world.get_ice_at(ix, iy) > 0.65) and (world.get_temp_at(ix, iy) < -0.5)
            if is_frozen:
                # По льду могут ходить только те, кто умеет ходить по суше/льду (не чисто жаберные рыбы)
                return bool(g[G_LUNGS] > 0.35 or g[G_LEGS] >= 4.0)
            return bool(g[G_GILLS] > 0.25 or g[G_LEGS] >= 4.0)
        else:
            return bool(g[G_LUNGS] > 0.25 or g[G_LEGS] >= 2.0)

    def _steer(self, tx: float, ty: float, weight: float = 1.0):
        dx = tx - self.x
        dy = ty - self.y
        dist = np.hypot(dx, dy)
        if dist > 1e-4:
            self.vx += (dx / dist) * weight
            self.vy += (dy / dist) * weight

    def _steer_lateral(self, threat_x: float, threat_y: float, weight: float = 1.0):
        """Уклонение боком (крабья походка) перпендикулярно угрозе."""
        dx = self.x - threat_x
        dy = self.y - threat_y
        dist = np.hypot(dx, dy)
        if dist > 1e-4:
            nx, ny = dx / dist, dy / dist
            side = 1.0 if (self.age // 40) % 2 == 0 else -1.0
            lat = self.genes[G_LATERAL]
            # Смесь отступления назад и рывка вбок
            steer_x = nx * (1.0 - 0.5 * lat) + (-ny * side) * lat
            steer_y = ny * (1.0 - 0.5 * lat) + (nx * side) * lat
            self.vx += steer_x * weight
            self.vy += steer_y * weight

    def _seek_safe_medium(self, world: "World", ix: int, iy: int, want_water: bool):
        """Если существо задыхается, ищет ближайшую подходящую среду (берег или воду)."""
        r = 36
        x0, x1 = max(4, ix - r), min(world.width - 4, ix + r + 1)
        y0, y1 = max(4, iy - r), min(world.height - 4, iy + r + 1)
        sub_elev = world.elevation[y0:y1, x0:x1]
        sub_fw = world.freshwater[y0:y1, x0:x1]

        if want_water:
            mask = (sub_elev < world.sea_level) | sub_fw
        else:
            mask = (sub_elev >= world.sea_level) & (sub_elev < world.mountain_level) & (~sub_fw)

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
        # Урон зависит от клешней, размера и хищности
        raw_dmg = 8.0 + 18.0 * self.genes[G_CLAWS] + 6.0 * self.genes[G_SIZE]
        armor_mitigation = 1.0 - 0.68 * target.genes[G_SHELL]
        final_dmg = max(1.5, raw_dmg * armor_mitigation)
        target.hp -= final_dmg

        # Ответный урон от клешней или шипастого панциря жертвы
        if target.genes[G_CLAWS] > 0.25 or target.genes[G_SHELL] > 0.45:
            retaliation = (4.0 * target.genes[G_CLAWS] + 2.5 * target.genes[G_SHELL]) * (1.0 - 0.5 * self.genes[G_SHELL])
            self.hp -= max(0.0, retaliation)

        if target.hp <= 0:
            target.alive = False
            meat_total = 45.0 * target.genes[G_SIZE] * (1.0 - 0.35 * target.genes[G_SHELL])
            self.energy = min(100.0, self.energy + meat_total * 0.55)
            carcasses.append({"x": target.x, "y": target.y, "meat": meat_total * 0.45})

    def update(self, world: "World", neighbors: list["Agent"], carcasses: list[dict], population_pressure: float) -> "Agent | None":
        if not self.alive:
            return None

        self.age += 1
        if self.reproduce_cooldown > 0:
            self.reproduce_cooldown -= 1
        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1

        ix = int(np.clip(self.x, 0, world.width - 1))
        iy = int(np.clip(self.y, 0, world.height - 1))
        elev = world.elevation[iy, ix]
        is_water = (elev < world.sea_level) or world.freshwater[iy, ix]
        on_ice = is_water and (world.get_ice_at(ix, iy) > 0.78) and (world.get_temp_at(ix, iy) < -0.5)
        in_liquid_water = is_water and not on_ice

        local_temp = world.get_temp_at(ix, iy)
        s = world.SCALE
        costs = self.config.costs

        # 1. Дыхание (жабры vs лёгкие)
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

        # Медленная регенерация здоровья при хорошем питании
        if self.hp < self.max_hp and self.energy > 65.0:
            self.hp = min(self.max_hp, self.hp + 0.08)

        # 2. Мягкая коллизия между существами + поиск врагов/добычи/стаи
        nearest_threat = None
        nearest_threat_dist = self.genes[G_SENSE]
        nearest_prey = None
        nearest_prey_dist = self.genes[G_SENSE]
        flock_x, flock_y, flock_count = 0.0, 0.0, 0

        for other in neighbors:
            if other is self or not other.alive:
                continue
            dx = other.x - self.x
            dy = other.y - self.y
            d = np.hypot(dx, dy)

            # Физическое расталкивание при пересечении радиусов тел
            min_dist = (self.radius + other.radius) * 1.45
            if 1e-4 < d < min_dist:
                overlap = (min_dist - d) / min_dist
                mass_ratio = other.mass / (self.mass + other.mass)
                push = overlap * 1.35 * mass_ratio
                self.vx -= (dx / d) * push
                self.vy -= (dy / d) * push

            if d > self.genes[G_SENSE]:
                continue

            # Оценка угрозы и добычи по разнице в хищности, размере и клешнях
            threat_power = other.genes[G_DIET] + 0.4 * other.genes[G_CLAWS] + 0.3 * other.genes[G_SIZE]
            my_power = self.genes[G_DIET] + 0.4 * self.genes[G_CLAWS] + 0.3 * self.genes[G_SIZE]

            if other.is_predator and threat_power > my_power + 0.30:
                if d < nearest_threat_dist:
                    nearest_threat_dist = d
                    nearest_threat = other
            elif self.is_predator and (not other.is_predator or self.energy < 22.0) and my_power > threat_power + 0.10:
                if d < nearest_prey_dist and self.is_tile_passable(world, int(other.x), int(other.y), self.genes):
                    nearest_prey_dist = d
                    nearest_prey = other
            elif abs(other.genes[G_DIET] - self.genes[G_DIET]) < 0.35 and 7.0 < d < 22.0:
                flock_x += other.x
                flock_y += other.y
                flock_count += 1

        # 3. Аллометрический метаболизм и стоимость органов
        effective_temp = local_temp + self.genes[G_FUR] * 16.0
        temp_stress = abs(effective_temp - self.genes[G_OPT_TEMP]) * costs["temp_stress_factor"]
        organ_cost = (
            self.genes[G_SHELL] * costs["shell_maintenance"]
            + self.genes[G_CLAWS] * costs["claw_maintenance"]
            + self.genes[G_FUR] * costs["fur_maintenance"]
        )
        move_mult = 0.5 if (self.is_predator and self.state == "wandering") else 1.0
        metabolism = (
            costs["base_metabolism"] * (self.genes[G_SIZE] ** costs["size_exponent"])
            + costs["speed_cost"] * move_mult * (self.genes[G_SPEED] ** costs["speed_exponent"])
            + organ_cost
            + temp_stress
        )

        # Плавное затухание скорости, чтобы существа не дёргались на месте
        self.vx *= 0.80
        self.vy *= 0.80

        # 4. Поведенческий автомат (FSM)
        wants_sleep = world.is_night != (self.genes[G_NOCTURNAL] > 0.5)
        sprinting = False
        is_thirsty = (not in_liquid_water) and ((self.hydration < 50.0) or (self.state == "drinking" and self.hydration < 88.0))
        is_hungry = (self.energy < 68.0) or (self.state in ("foraging", "hunting") and self.energy < 82.0)

        if self.oxygen < 45.0:
            self.state = "suffocating"
            self._seek_safe_medium(world, ix, iy, want_water=not in_liquid_water)
            sprinting = True

        elif nearest_threat is not None:
            self.state = "fleeing"
            self._steer_lateral(nearest_threat.x, nearest_threat.y, weight=1.25)
            if self.stamina > 15.0:
                sprinting = True
                self.stamina -= 0.45
                self.energy -= 0.012

        elif is_thirsty:
            self.state = "drinking"
            if world.get_snow_at(ix, iy) > 0.4:
                self.hydration = min(100.0, self.hydration + 8.0)
            else:
                sr = max(4, int(min(self.genes[G_SENSE], 48) // s))
                six, siy = ix // s, iy // s
                sx0, sx1 = max(0, six - sr), min(world.sim_w, six + sr + 1)
                sy0, sy1 = max(0, siy - sr), min(world.sim_h, siy + sr + 1)
                water_cells = np.argwhere(world.sim_freshwater[sy0:sy1, sx0:sx1])
                if len(water_cells) > 0:
                    wy, wx = water_cells[0]
                    tx, ty = (sx0 + wx + 0.5) * s, (sy0 + wy + 0.5) * s
                    self._steer(tx, ty, weight=0.65)
                    if np.hypot(tx - self.x, ty - self.y) < 4.5:
                        self.hydration = min(100.0, self.hydration + 25.0)
                else:
                    self.vx += self.rng.uniform(-0.08, 0.08)
                    self.vy += self.rng.uniform(-0.08, 0.08)

        elif is_hungry:
            ate_carcass = False
            if self.is_predator or self.is_scavenger:
                for c in carcasses:
                    if c["meat"] > 0 and np.hypot(c["x"] - self.x, c["y"] - self.y) < self.genes[G_SENSE]:
                        self.state = "foraging"
                        self._steer(c["x"], c["y"], weight=0.75)
                        if np.hypot(c["x"] - self.x, c["y"] - self.y) <= self.radius + 1.5:
                            bite = min(c["meat"], 14.0 + 8.0 * self.genes[G_CLAWS])
                            c["meat"] -= bite
                            self.energy = min(100.0, self.energy + bite * 1.4)
                        ate_carcass = True
                        break

            if not ate_carcass:
                if self.is_herbivore and world.get_plant_at(ix, iy) > 0.14:
                    self.state = "foraging"
                    eaten = world.eat_plant_at(ix, iy, 0.18)
                    # У травоядных усвоение растений выше, чем у хищников
                    plant_eff = 48.0 * np.clip(1.0 - max(0.0, self.genes[G_DIET]), 0.3, 1.0)
                    self.energy = min(100.0, self.energy + eaten * plant_eff)
                    self.vx *= 0.4
                    self.vy *= 0.4
                elif self.is_predator and nearest_prey is not None:
                    self.state = "hunting"
                    self._steer(nearest_prey.x, nearest_prey.y, weight=0.9)
                    if nearest_prey_dist < 18.0 and self.stamina > 20.0:
                        sprinting = True
                        self.stamina -= 0.4
                        self.energy -= 0.012
                    if nearest_prey_dist <= self.radius + nearest_prey.radius + 0.7:
                        self._attack(nearest_prey, carcasses)
                elif self.is_herbivore:
                    self.state = "foraging"
                    sr = max(4, int(min(self.genes[G_SENSE], 42) // s))
                    six, siy = ix // s, iy // s
                    sx0, sx1 = max(0, six - sr), min(world.sim_w, six + sr + 1)
                    sy0, sy1 = max(0, siy - sr), min(world.sim_h, siy + sr + 1)
                    patch = world.plants[sy0:sy1, sx0:sx1]
                    if patch.size > 0 and patch.max() > 0.18:
                        by, bx = np.unravel_index(np.argmax(patch), patch.shape)
                        self._steer((sx0 + bx + 0.5) * s, (sy0 + by + 0.5) * s, weight=0.55)
                    else:
                        self.vx += self.rng.uniform(-0.08, 0.08)
                        self.vy += self.rng.uniform(-0.08, 0.08)
                else:
                    self.state = "wandering"
                    self.vx += self.rng.uniform(-0.06, 0.06)
                    self.vy += self.rng.uniform(-0.06, 0.06)

        elif wants_sleep and self.energy > 35.0 and self.hydration > 35.0 and self.oxygen > 70.0:
            self.state = "sleeping"
            self.energy -= metabolism * 0.35
            if not in_liquid_water:
                self.hydration -= 0.012
            self.stamina = min(100.0, self.stamina + 0.5)
            return None

        else:
            self.state = "wandering"
            self.stamina = min(100.0, self.stamina + 0.25)
            if 0 < flock_count < 5:
                self._steer(flock_x / flock_count, flock_y / flock_count, weight=0.10)
            self.vx += self.rng.uniform(-0.06, 0.06)
            self.vy += self.rng.uniform(-0.06, 0.06)

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
            # Панцирь защищает от высыхания на суше!
            dry_resist = 1.0 - 0.55 * self.genes[G_SHELL]
            self.hydration -= (0.024 + max(0.0, (local_temp - 20.0) * 0.0015)) * dry_resist

        if self.energy <= 0 or self.hydration <= 0 or self.age >= (self.max_age * world.lifespan_mult):
            self.alive = False
            carcasses.append({"x": self.x, "y": self.y, "meat": 35.0 * self.genes[G_SIZE]})
            return None

        # 5. Физика движения с учётом тяжести панциря, ног и среды
        shell_penalty = 1.0 - costs["shell_weight_speed_penalty"] * self.genes[G_SHELL]
        if in_liquid_water:
            # В воде быстры обтекаемые пловцы (низкая ширина, много жабр, мало ног)
            hydro = 0.65 + 0.45 * self.genes[G_GILLS] - 0.25 * self.genes[G_WIDTH]
            medium_mult = np.clip(hydro, 0.45, 1.25)
        else:
            # На суше важны ноги! Без ног ползти медленно
            leg_factor = 0.45 + 0.55 * np.clip(self.genes[G_LEGS] / 4.0, 0.0, 1.2)
            medium_mult = leg_factor

        slope_penalty = 0.75 if elev > world.mountain_level - 0.10 else 1.0
        snow_penalty = 0.72 if world.get_snow_at(ix, iy) > 0.5 else 1.0
        sprint_mult = 1.55 if sprinting else (0.45 if (self.is_predator and self.state == "wandering") else 1.0)

        max_spd = self.genes[G_SPEED] * shell_penalty * medium_mult * slope_penalty * snow_penalty * sprint_mult

        v_norm = np.hypot(self.vx, self.vy)
        if v_norm > max_spd and v_norm > 1e-6:
            self.vx = (self.vx / v_norm) * max_spd
            self.vy = (self.vy / v_norm) * max_spd

        # Поворачиваем корпус только при уверенном движении (без кручения волчком на месте)
        if v_norm > 0.05:
            target_angle = float(np.arctan2(self.vy, self.vx))
            diff = (target_angle - self.heading + np.pi) % (2 * np.pi) - np.pi
            self.heading += diff * 0.35

        nx, ny = self.x + self.vx, self.y + self.vy
        if self.is_tile_passable(world, int(nx), int(ny), self.genes):
            self.x, self.y = nx, ny
        else:
            # Плавный обход берега/горы в выбранную сторону вместо случайного дёргания туда-сюда
            self.heading += 0.45 * self.turn_dir
            self.vx = np.cos(self.heading) * max_spd * 0.65
            self.vy = np.sin(self.heading) * max_spd * 0.65

        # 6. Размножение с мутацией вектора генов
        repro_threshold = (84.0 + population_pressure * 14.0) / max(0.4, world.repro_rate_mult)
        if self.energy > repro_threshold and self.hydration > 65.0 and self.oxygen > 70.0 and self.reproduce_cooldown == 0:
            for _ in range(6):
                cx = self.x + self.rng.uniform(-3.0, 3.0)
                cy = self.y + self.rng.uniform(-3.0, 3.0)
                if self.is_tile_passable(world, int(cx), int(cy), self.genes):
                    self.energy -= 38.0
                    self.reproduce_cooldown = int(420 / max(0.4, world.repro_rate_mult))
                    child_genes = self.config.mutate(self.genes, self.rng)
                    return Agent(cx, cy, genes=child_genes, config=self.config, rng=self.rng)

        return None