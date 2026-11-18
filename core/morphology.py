import json
import colorsys
from pathlib import Path
import numpy as np

# Индексы генов в float32-векторе
G_WIDTH = 0
G_LENGTH = 1
G_SHELL = 2
G_CLAWS = 3
G_LEGS = 4
G_LATERAL = 5
G_GILLS = 6
G_LUNGS = 7
G_DIET = 8
G_SPEED = 9
G_SENSE = 10
G_SIZE = 11
G_FUR = 12
G_OPT_TEMP = 13
G_NOCTURNAL = 14
G_HUE_SHIFT = 15
G_SATURATION = 16
G_BRIGHTNESS = 17

GENE_NAMES = [
    "body_width", "body_length", "shell", "claws", "legs",
    "lateral_gait", "gills", "lungs", "diet_bias", "speed",
    "sense", "size", "fur", "optimal_temp", "nocturnal",
    "hue_shift", "saturation", "brightness",
]


class EvolutionConfig:
    def __init__(self, json_path: str | Path = "data/evolution_rules.json"):
        if not Path(json_path).exists():
            json_path = Path(__file__).parent.parent / "data" / "evolution_rules.json"

        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.costs: dict = raw["energy_costs"]
        self.archetypes_raw: dict = raw["archetypes"]

        self.mins = np.array([raw["genes"][k]["min"] for k in GENE_NAMES], dtype=np.float32)
        self.maxs = np.array([raw["genes"][k]["max"] for k in GENE_NAMES], dtype=np.float32)
        self.sigmas = np.array([raw["genes"][k]["sigma"] for k in GENE_NAMES], dtype=np.float32)
        self.probs = np.array([raw["genes"][k]["prob"] for k in GENE_NAMES], dtype=np.float32)

    def spawn_genome(self, archetype: str, rng: np.random.Generator) -> np.ndarray:
        base = self.archetypes_raw[archetype]
        vec = np.array([base[k] for k in GENE_NAMES], dtype=np.float32)
        jitter = rng.normal(0.0, self.sigmas * 0.35).astype(np.float32)
        jitter[G_LEGS] = 0.0
        jitter[G_NOCTURNAL] = 0.0
        return np.clip(vec + jitter, self.mins, self.maxs)

    def mutate(self, parent: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        child = parent.copy()
        mask = rng.random(len(GENE_NAMES)) < self.probs
        noise = rng.normal(0.0, self.sigmas).astype(np.float32)

        if mask[G_LEGS]:
            noise[G_LEGS] = float(rng.choice([-2.0, 2.0]))
        if mask[G_NOCTURNAL]:
            child[G_NOCTURNAL] = 1.0 - child[G_NOCTURNAL]
            mask[G_NOCTURNAL] = False

        child = np.where(mask, child + noise, child)
        return np.clip(child, self.mins, self.maxs)


def calculate_crab_score(g: np.ndarray) -> float:
    wide_flat = np.clip((g[G_WIDTH] - g[G_LENGTH] + 0.4) / 0.9, 0.0, 1.0)
    armored = np.clip(g[G_SHELL] / 0.7, 0.0, 1.0)
    has_claws = np.clip(g[G_CLAWS] / 0.6, 0.0, 1.0)
    many_legs = np.clip((g[G_LEGS] - 2.0) / 6.0, 0.0, 1.0)
    sideways = np.clip(g[G_LATERAL] / 0.6, 0.0, 1.0)
    amphibious = np.clip(min(g[G_GILLS], g[G_LUNGS]) / 0.45, 0.0, 1.0)
    scavenger = 1.0 - np.clip(abs(g[G_DIET] - 0.1) / 0.9, 0.0, 1.0)

    score = (
        0.22 * wide_flat + 0.20 * armored + 0.18 * has_claws +
        0.15 * many_legs + 0.10 * sideways + 0.10 * amphibious + 0.05 * scavenger
    )
    return float(np.clip(score, 0.0, 1.0))


def _shift_color(col: tuple[int, int, int, int], hue: float, sat_mult: float, bright_mult: float) -> tuple[int, int, int, int]:
    r, g, b, a = col
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    h = (h + hue) % 1.0
    s = max(0.0, min(1.0, s * sat_mult))
    v = max(0.0, min(1.0, v * bright_mult))
    r2, g2, b2 = colorsys.hsv_to_rgb(h, s, v)
    return (int(r2 * 255), int(g2 * 255), int(b2 * 255), a)


def build_sprite_rgba(g: np.ndarray) -> np.ndarray:
    sprite = np.zeros((9, 9, 4), dtype=np.uint8)
    cx, cy = 4, 4
    outline = (18, 16, 20, 235)

    is_fox = (g[G_FUR] > 0.55) and (g[G_SHELL] < 0.30) and (0.25 < g[G_DIET] <= 0.72) and (g[G_LEGS] >= 4.0)

    # 1. Базовая палитра
    if is_fox:
        body_col = (238, 105, 28, 255)
        accent_col = (248, 245, 240, 255)
        limb_col = (30, 25, 28, 255)
    elif g[G_SHELL] > 0.42:
        s = np.clip((g[G_SHELL] - 0.42) / 0.58, 0.0, 1.0)
        body_col = (int(165 + 35 * s), int(58 - 22 * s), int(68 - 15 * s), 255)
        accent_col = (115, 26, 36, 255)
        limb_col = outline
    elif g[G_GILLS] > 0.55 and g[G_LUNGS] < 0.45:
        body_col = (45, 75, 115, 255) if g[G_DIET] > 0.35 else (50, 195, 235, 255)
        accent_col = (28, 135, 185, 255)
        limb_col = outline
    elif g[G_DIET] > 0.35:
        f = np.clip(g[G_FUR], 0.0, 1.0)
        body_col = (int(95 + 45 * f), int(100 + 48 * f), int(110 + 52 * f), 255)
        accent_col = (max(0, body_col[0] - 42), max(0, body_col[1] - 42), max(0, body_col[2] - 40), 255)
        limb_col = outline
    else:
        f = np.clip(g[G_FUR], 0.0, 1.0)
        body_col = (int(195 + 42 * f), int(190 + 46 * f), int(125 + 110 * f), 255)
        accent_col = (max(0, body_col[0] - 45), max(0, body_col[1] - 45), max(0, body_col[2] - 45), 255)
        limb_col = outline

    base_claw_col = (245, 175, 45, 255)

    # 2. Применяем эволюционный сдвиг палитры ДО отрисовки
    hue = float(g[G_HUE_SHIFT]) if len(g) > G_HUE_SHIFT else 0.0
    sat = float(g[G_SATURATION]) if len(g) > G_SATURATION else 1.0
    bri = float(g[G_BRIGHTNESS]) if len(g) > G_BRIGHTNESS else 1.0

    if abs(hue) > 0.005 or abs(sat - 1.0) > 0.01 or abs(bri - 1.0) > 0.01:
        body_col = _shift_color(body_col, hue, sat, bri)
        accent_col = _shift_color(accent_col, hue, sat, bri)
        limb_col = _shift_color(limb_col, hue, sat, bri)
        base_claw_col = _shift_color(base_claw_col, hue, sat, bri)

    # 3. Геометрия и отрисовка
    half_w = 2 if g[G_WIDTH] >= 0.56 else 1
    half_l = 2 if g[G_LENGTH] >= 0.62 else 1
    if g[G_LEGS] < 1.5 and g[G_WIDTH] < 0.45:
        half_w, half_l = 1, 3

    y0, y1 = max(2, cy - half_l), min(7, cy + half_l + 1)
    x0, x1 = max(2, cx - half_w), min(7, cx + half_w + 1)
    front_y = max(0, y0 - 1)

    if g[G_LENGTH] > 0.55 and g[G_SHELL] < 0.45:
        tail_y = min(8, y1)
        sprite[tail_y, cx] = body_col if is_fox else accent_col
        if g[G_LENGTH] > 0.72 and tail_y + 1 < 9:
            sprite[tail_y + 1, cx] = accent_col

    pairs = min(4, int(round(g[G_LEGS])) // 2)
    leg_rows = [cy, cy - 1, cy + 1, cy - 2][:pairs]
    for ly in leg_rows:
        if 0 <= ly < 9:
            sprite[ly, max(0, x0 - 1)] = limb_col
            sprite[ly, min(8, x1)] = limb_col

    sprite[y0:y1, x0:x1] = body_col
    if g[G_SHELL] > 0.25:
        sprite[y0, x0:x1] = accent_col
        if g[G_SHELL] > 0.55:
            sprite[y1 - 1, x0:x1] = accent_col

    if g[G_CLAWS] > 0.28:
        sprite[front_y, max(0, x0)] = base_claw_col
        sprite[front_y, min(8, x1 - 1)] = base_claw_col
        if g[G_CLAWS] > 0.55 and front_y - 1 >= 0:
            sprite[front_y - 1, max(0, x0)] = base_claw_col
            sprite[front_y - 1, min(8, x1 - 1)] = base_claw_col
    elif g[G_FUR] > 0.45 and g[G_SHELL] < 0.30:
        sprite[front_y, max(0, x0)] = limb_col if is_fox else accent_col
        sprite[front_y, min(8, x1 - 1)] = limb_col if is_fox else accent_col
    elif g[G_DIET] > 0.35:
        sprite[front_y, cx] = body_col

    sprite[y0, x0] = outline
    sprite[y0, x1 - 1] = outline
    return sprite