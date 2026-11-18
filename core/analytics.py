from collections import deque
from typing import TYPE_CHECKING
import numpy as np

from core.morphology import G_DIET, G_FUR, G_GILLS, G_LEGS, G_LUNGS, G_SHELL, G_SPEED

if TYPE_CHECKING:
    from core.world import World


class EcosystemTracker:
    """Сбор временных рядов численности видов и эволюционных метрик для графиков и БД."""

    def __init__(self, max_points: int = 400, sample_interval: int = 10):
        self.sample_interval = sample_interval
        self.history: deque[dict] = deque(maxlen=max_points)

    def reset(self):
        self.history.clear()

    def record(self, world: "World", force: bool = False):
        if not force and (world.tick % self.sample_interval != 0):
            return

        herbs = 0
        wolves = 0
        foxes = 0
        aquatic = 0
        sharks = 0
        crabs = 0

        sum_speed = 0.0
        sum_shell = 0.0
        max_crab = 0.0

        for a in world.agents:
            g = a.genes
            sum_speed += float(g[G_SPEED])
            sum_shell += float(g[G_SHELL])
            if a.crab_score > max_crab:
                max_crab = a.crab_score

            is_fox = (g[G_FUR] > 0.55) and (g[G_SHELL] < 0.30) and (0.25 < g[G_DIET] <= 0.72) and (g[G_LEGS] >= 4.0)
            if a.crab_score > 0.55:
                crabs += 1
            elif g[G_GILLS] > 0.55 and g[G_LUNGS] < 0.45:
                if g[G_DIET] > 0.35:
                    sharks += 1
                else:
                    aquatic += 1
            elif is_fox:
                foxes += 1
            elif a.is_predator:
                wolves += 1
            else:
                herbs += 1

        total = len(world.agents)
        plant_biomass = float(np.sum(world.plants) / 100.0) # Нормированная шкала для графика

        self.history.append({
            "tick": int(world.tick),
            "total": total,
            "plants": round(plant_biomass, 1),
            "herbivores": herbs,
            "wolves": wolves,
            "foxes": foxes,
            "aquatic": aquatic,
            "sharks": sharks,
            "crabs": crabs,
            "avg_speed": round(sum_speed / total, 3) if total else 0.0,
            "avg_shell": round(sum_shell / total, 3) if total else 0.0,
            "max_crab_score": round(max_crab, 3),
            "avg_temp": round(float(np.mean(world.temperature)), 2),
        })
