import numpy as np


class Agent:
    def __init__(
        self,
        x: float,
        y: float,
        diet: str = "herbivore",
        genes: dict | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.rng = rng or np.random.default_rng()

        self.x = float(x)
        self.y = float(y)
        self.vx = float(self.rng.uniform(-0.15, 0.15))
        self.vy = float(self.rng.uniform(-0.15, 0.15))
        self.heading = float(np.arctan2(self.vy, self.vx))

        self.diet = diet # "herbivore" | "carnivore"

        if genes is not None:
            self.genes = genes.copy()
        else:
            is_pred = diet == "carnivore"
            self.genes = {
                "speed": 0.36 if is_pred else 0.25,
                "sense": 62.0 if is_pred else 28.0, # хищник чует добычу далеко на 1600x1000
                "size": 1.2 if is_pred else 1.0,
                "fur": float(self.rng.uniform(0.25, 0.55)),
                "optimal_temp": 18.0,
                "social": 0.5 if not is_pred else 0.2,
                "nocturnal": False,
                "amphibian": False,
                "scavenger": False,
            }

        self.energy = 72.0
        self.hydration = 82.0
        self.stamina = 100.0
        self.age = 0
        self.max_age = int(self.rng.integers(9000, 14000))
        self.reproduce_cooldown = 350
        self.state = "wandering"
        self.alive = True

    def mutate(self, x: float, y: float) -> "Agent":
        child_genes = self.genes.copy()

        for key, lo, hi in [("speed", 0.12, 0.65), ("sense", 10.0, 45.0), ("size", 0.6, 1.8)]:
            if self.rng.random() < 0.30:
                factor = self.rng.uniform(0.92, 1.09)
                child_genes[key] = float(np.clip(child_genes[key] * factor, lo, hi))

        for key in ["fur", "social"]:
            if self.rng.random() < 0.25:
                child_genes[key] = float(np.clip(child_genes[key] + self.rng.uniform(-0.1, 0.1), 0.0, 1.0))

        if self.rng.random() < 0.25:
            child_genes["optimal_temp"] = float(
                np.clip(child_genes["optimal_temp"] + self.rng.uniform(-2.5, 2.5), -10.0, 34.0))

        for perk in ["nocturnal", "amphibian", "scavenger"]:
            if self.rng.random() < 0.01:
                child_genes[perk] = not child_genes[perk]

        return Agent(x, y, diet=self.diet, genes=child_genes, rng=self.rng)

    @staticmethod
    def is_tile_land(world, ix: int, iy: int) -> bool:
        if ix < 4 or ix >= world.width - 4 or iy < 4 or iy >= world.height - 4:
            return False
        elev = world.elevation[iy, ix]
        return (world.sea_level <= elev < world.mountain_level) and (not world.freshwater[iy, ix])

    @staticmethod
    def is_tile_walkable(world, ix: int, iy: int, amphibian: bool = False, allow_ice: bool = True) -> bool:
        if ix < 4 or ix >= world.width - 4 or iy < 4 or iy >= world.height - 4:
            return False
        elev = world.elevation[iy, ix]
        if elev >= world.mountain_level:
            return False
        is_water = (elev < world.sea_level) or world.freshwater[iy, ix]
        if not is_water:
            return True
        if allow_ice and world.get_ice_at(ix, iy) > 0.78 and world.get_temp_at(ix, iy) < -0.5:
            return True
        return amphibian and (elev > world.sea_level * 0.55)

    def _steer(self, tx: float, ty: float, weight: float = 1.0):
        dx = tx - self.x
        dy = ty - self.y
        dist = np.hypot(dx, dy)
        if dist > 1e-4:
            self.vx += (dx / dist) * weight
            self.vy += (dy / dist) * weight

    def update(self, world, neighbors: list["Agent"], carcasses: list[dict],
               population_pressure: float) -> "Agent | None":
        if not self.alive:
            return None

        self.age += 1
        if self.reproduce_cooldown > 0:
            self.reproduce_cooldown -= 1

        ix = int(np.clip(self.x, 0, world.width - 1))
        iy = int(np.clip(self.y, 0, world.height - 1))
        local_temp = world.get_temp_at(ix, iy)
        s = world.SCALE  # масштаб сетки климата/травы (4)

        # 0. Спасение к берегу, если лёд растаял под ногами
        in_deep_trouble = not self.is_tile_walkable(world, ix, iy, self.genes["amphibian"], allow_ice=True)
        if in_deep_trouble:
            self.state = "swimming"
            self.energy -= 0.08
            self.stamina -= 0.4
            if self.energy <= 0:
                self.alive = False
                return None

            r = 40
            x0, x1 = max(4, ix - r), min(world.width - 4, ix + r + 1)
            y0, y1 = max(4, iy - r), min(world.height - 4, iy + r + 1)
            sub_elev = world.elevation[y0:y1, x0:x1]
            sub_fw = world.freshwater[y0:y1, x0:x1]
            dry = (sub_elev >= world.sea_level) & (sub_elev < world.mountain_level) & (~sub_fw)
            dry_coords = np.argwhere(dry)
            if len(dry_coords) > 0:
                dists = (dry_coords[:, 0] + y0 - iy) ** 2 + (dry_coords[:, 1] + x0 - ix) ** 2
                by, bx = dry_coords[np.argmin(dists)]
                self._steer(x0 + bx + 0.5, y0 + by + 0.5, weight=1.5)
            else:
                self._steer(world.width / 2.0, world.height / 2.0, weight=1.0)

            swim_spd = max(0.22, self.genes["speed"] * 0.85)
            v_norm = np.hypot(self.vx, self.vy)
            if v_norm > swim_spd:
                self.vx = (self.vx / v_norm) * swim_spd
                self.vy = (self.vy / v_norm) * swim_spd
            self.heading = float(np.arctan2(self.vy, self.vx))
            self.x = float(np.clip(self.x + self.vx, 4, world.width - 5))
            self.y = float(np.clip(self.y + self.vy, 4, world.height - 5))
            return None

        # 1. Локальные соседи
        nearest_pred = None
        nearest_pred_dist = self.genes["sense"]
        nearest_prey = None
        nearest_prey_dist = self.genes["sense"]
        flock_x, flock_y, flock_count = 0.0, 0.0, 0

        for other in neighbors:
            if other is self or not other.alive:
                continue
            dx = other.x - self.x
            dy = other.y - self.y
            d = np.hypot(dx, dy)
            if d > self.genes["sense"]:
                continue

            if self.diet == "herbivore" and other.diet == "carnivore":
                if d < nearest_pred_dist:
                    nearest_pred_dist = d
                    nearest_pred = other
            elif self.diet == "carnivore" and other.diet == "herbivore":
                if d < nearest_prey_dist:
                    nearest_prey_dist = d
                    nearest_prey = other
            elif other.diet == self.diet and d < 14.0:
                flock_x += other.x
                flock_y += other.y
                flock_count += 1
                if d < 2.8 and d > 1e-4:
                    self.vx -= (dx / d) * 0.25
                    self.vy -= (dy / d) * 0.25

        # 2. Метаболизм (в спокойном патруле хищник тратит меньше сил)
        effective_temp = local_temp + self.genes["fur"] * 15.0
        temp_stress = abs(effective_temp - self.genes["optimal_temp"]) * 0.0015
        move_cost = 0.012 if (self.diet == "carnivore" and self.state == "wandering") else 0.024
        metabolism = 0.015 * (self.genes["size"] ** 1.35) + move_cost * (self.genes["speed"] ** 1.4) + temp_stress

        # 3. Выбор поведения
        wants_sleep = world.is_night != self.genes["nocturnal"]
        sprinting = False
        is_thirsty = (self.hydration < 50.0) or (self.state == "drinking" and self.hydration < 88.0)
        is_hungry = (self.energy < 68.0) or (self.state in ("foraging", "hunting") and self.energy < 82.0)

        if nearest_pred is not None:
            self.state = "fleeing"
            self._steer(self.x - (nearest_pred.x - self.x), self.y - (nearest_pred.y - self.y), weight=1.2)
            if self.stamina > 15.0:
                sprinting = True
                self.stamina -= 0.45
                self.energy -= 0.012

        elif is_thirsty:
            self.state = "drinking"
            if world.get_snow_at(ix, iy) > 0.4:
                self.hydration = min(100.0, self.hydration + 8.0)
            else:
                # Ищем воду в уменьшенной сетке sim_freshwater и переводим координаты обратно * s
                sr = max(4, int(min(self.genes["sense"], 48) // s))
                six, siy = ix // s, iy // s
                sx0, sx1 = max(0, six - sr), min(world.sim_w, six + sr + 1)
                sy0, sy1 = max(0, siy - sr), min(world.sim_h, siy + sr + 1)
                water_cells = np.argwhere(world.sim_freshwater[sy0:sy1, sx0:sx1])
                if len(water_cells) > 0:
                    wy, wx = water_cells[0]
                    tx = (sx0 + wx + 0.5) * s
                    ty = (sy0 + wy + 0.5) * s
                    self._steer(tx, ty, weight=0.6)
                    if np.hypot(tx - self.x, ty - self.y) < 4.5:
                        self.hydration = min(100.0, self.hydration + 25.0)
                else:
                    self.vx += self.rng.uniform(-0.08, 0.08)
                    self.vy += self.rng.uniform(-0.08, 0.08)

        elif is_hungry:
            ate_carcass = False
            if self.diet == "carnivore" or self.genes["scavenger"]:
                for c in carcasses:
                    if c["meat"] > 0 and np.hypot(c["x"] - self.x, c["y"] - self.y) < self.genes["sense"]:
                        self.state = "foraging"
                        self._steer(c["x"], c["y"], weight=0.7)
                        if np.hypot(c["x"] - self.x, c["y"] - self.y) < 2.5:
                            bite = min(c["meat"], 15.0)
                            c["meat"] -= bite
                            self.energy = min(100.0, self.energy + bite * 1.4)
                        ate_carcass = True
                        break

            if not ate_carcass:
                if self.diet == "herbivore":
                    self.state = "foraging"
                    if world.get_plant_at(ix, iy) > 0.14:
                        eaten = world.eat_plant_at(ix, iy, 0.18)
                        self.energy = min(100.0, self.energy + eaten * 45.0)
                        self.vx *= 0.4
                        self.vy *= 0.4
                    else:
                        # Поиск лучшей травы в сетке world.plants с переводом координат * s
                        sr = max(4, int(min(self.genes["sense"], 40) // s))
                        six, siy = ix // s, iy // s
                        sx0, sx1 = max(0, six - sr), min(world.sim_w, six + sr + 1)
                        sy0, sy1 = max(0, siy - sr), min(world.sim_h, siy + sr + 1)
                        patch = world.plants[sy0:sy1, sx0:sx1]
                        if patch.size > 0 and patch.max() > 0.18:
                            by, bx = np.unravel_index(np.argmax(patch), patch.shape)
                            tx = (sx0 + bx + 0.5) * s
                            ty = (sy0 + by + 0.5) * s
                            self._steer(tx, ty, weight=0.5)
                        else:
                            self.vx += self.rng.uniform(-0.08, 0.08)
                            self.vy += self.rng.uniform(-0.08, 0.08)
                elif nearest_prey is not None:
                    self.state = "hunting"
                    self._steer(nearest_prey.x, nearest_prey.y, weight=0.85)
                    if nearest_prey_dist < 18.0 and self.stamina > 20.0:
                        sprinting = True
                        self.stamina -= 0.4
                        self.energy -= 0.012
                    if nearest_prey_dist < 2.2:
                        nearest_prey.alive = False
                        meat_total = 48.0 * nearest_prey.genes["size"]
                        self.energy = min(100.0, self.energy + meat_total * 0.55)
                        carcasses.append({"x": nearest_prey.x, "y": nearest_prey.y, "meat": meat_total * 0.45})
                else:
                    self.state = "wandering"
                    self.vx += self.rng.uniform(-0.06, 0.06)
                    self.vy += self.rng.uniform(-0.06, 0.06)

        elif wants_sleep and self.energy > 35.0 and self.hydration > 35.0:
            self.state = "sleeping"
            self.energy -= metabolism * 0.35
            self.hydration -= 0.012
            self.stamina = min(100.0, self.stamina + 0.5)
            return None

        else:
            self.state = "wandering"
            self.stamina = min(100.0, self.stamina + 0.25)
            if 0 < flock_count < 4 and self.genes["social"] > 0.2:
                self._steer(flock_x / flock_count, flock_y / flock_count, weight=0.12 * self.genes["social"])
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
        self.hydration -= 0.024 + max(0.0, (local_temp - 20.0) * 0.0015)

        if self.energy <= 0 or self.hydration <= 0 or self.age >= self.max_age:
            self.alive = False
            carcasses.append({"x": self.x, "y": self.y, "meat": 35.0 * self.genes["size"]})
            return None

        # 4. Физика движения (в wandering хищники патрулируют шагом 0.45x, чтобы не сжигать энергию)
        slope_penalty = 0.75 if world.elevation[iy, ix] > world.mountain_level - 0.10 else 1.0
        snow_penalty = 0.72 if world.get_snow_at(ix, iy) > 0.5 else 1.0
        sprint_mult = 1.55 if sprinting else (0.45 if (self.diet == "carnivore" and self.state == "wandering") else 1.0)
        max_spd = self.genes["speed"] * slope_penalty * snow_penalty * sprint_mult

        v_norm = np.hypot(self.vx, self.vy)
        if v_norm > max_spd:
            self.vx = (self.vx / v_norm) * max_spd
            self.vy = (self.vy / v_norm) * max_spd

        if v_norm > 0.02:
            self.heading = float(np.arctan2(self.vy, self.vx))

        nx, ny = self.x + self.vx, self.y + self.vy
        if self.is_tile_walkable(world, int(nx), int(ny), self.genes["amphibian"]):
            self.x, self.y = nx, ny
        else:
            angle = self.heading + self.rng.choice([-1.4, 1.4])
            self.vx = np.cos(angle) * max_spd * 0.6
            self.vy = np.sin(angle) * max_spd * 0.6

        # 5. Размножение
        repro_threshold = 84.0 + population_pressure * 14.0
        if self.energy > repro_threshold and self.hydration > 65.0 and self.reproduce_cooldown == 0:
            if self.is_tile_land(world, int(self.x), int(self.y)):
                for _ in range(6):
                    cx = self.x + self.rng.uniform(-3.0, 3.0)
                    cy = self.y + self.rng.uniform(-3.0, 3.0)
                    if self.is_tile_land(world, int(cx), int(cy)):
                        self.energy -= 38.0
                        self.reproduce_cooldown = 420
                        return self.mutate(cx, cy)

        return None
