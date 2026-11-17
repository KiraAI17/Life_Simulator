import json
from pathlib import Path
import numpy as np

# Индексы генов в float32-векторе (быстрее и компактнее словарей/JSON в рантайме)
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

GENE_NAMES = [
    "body_width", "body_length", "shell", "claws", "legs",
    "lateral_gait", "gills", "lungs", "diet_bias", "speed",
    "sense", "size", "fur", "optimal_temp", "nocturnal",
]


class EvolutionConfig:
    def __init__(self, json_path: str | Path = "data/evolution_rules.json"):
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
        # Лёгкое стартовое разнообразие популяции
        jitter = rng.normal(0.0, self.sigmas * 0.35).astype(np.float32)
        jitter[G_LEGS] = 0.0
        jitter[G_NOCTURNAL] = 0.0
        return np.clip(vec + jitter, self.mins, self.maxs)

    def mutate(self, parent: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        child = parent.copy()
        mask = rng.random(len(GENE_NAMES)) < self.probs
        noise = rng.normal(0.0, self.sigmas).astype(np.float32)

        # Ноги мутируют дискретно с шагом +-2
        if mask[G_LEGS]:
            noise[G_LEGS] = float(rng.choice([-2.0, 2.0]))

        # Ночной образ жизни переключается бинарно
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
            0.22 * wide_flat
            + 0.20 * armored
            + 0.18 * has_claws
            + 0.15 * many_legs
            + 0.10 * sideways
            + 0.10 * amphibious
            + 0.05 * scavenger
    )
    return float(np.clip(score, 0.0, 1.0))


def build_sprite_rgba(g: np.ndarray) -> np.ndarray:
    """
    Послойный анатомический генератор 9x9 RGBA.
    Органы (панцирь, ноги, клешни, уши, пушистый хвост) появляются и растут постепенно!
    """
    sprite = np.zeros((9, 9, 4), dtype=np.uint8)
    cx, cy = 4, 4
    outline = (18, 16, 20, 235)

    # Признак лисы: пушистый некрупный хищник без панциря
    is_fox = (g[G_FUR] > 0.55) and (g[G_SHELL] < 0.30) and (0.25 < g[G_DIET] <= 0.72) and (g[G_LEGS] >= 4.0)

    # 1. Разводим палитру по экологическим нишам (без смешивания всех в один рыжий оттенок)
    if is_fox:
        body_col = (238, 105, 28, 255)  # Ярко-рыжая лиса
        accent_col = (248, 245, 240, 255)  # Белый кончик хвоста и ушки
        limb_col = (30, 25, 28, 255)  # Чёрные «чулочки» на лапках
    elif g[G_SHELL] > 0.42:
        # Хитиновый панцирь (крабы, жуки, броненосцы): глубокий бордово-вишнёвый хитин
        s = np.clip((g[G_SHELL] - 0.42) / 0.58, 0.0, 1.0)
        body_col = (int(165 + 35 * s), int(58 - 22 * s), int(68 - 15 * s), 255)
        accent_col = (115, 26, 36, 255)
        limb_col = outline
    elif g[G_GILLS] > 0.55 and g[G_LUNGS] < 0.45:
        if g[G_DIET] > 0.35:
            # Акулы тёмно-синие с алыми жабрами
            body_col = (45, 75, 115, 255)
        else:
            # Водные рыбы и черви: бирюзово-лазурные
            body_col = (50, 195, 235, 255)
        accent_col = (28, 135, 185, 255)
        limb_col = outline
    elif g[G_DIET] > 0.35:
        # Волки и крупные сухопутные хищники: пепельно-серый / графитовый мех
        f = np.clip(g[G_FUR], 0.0, 1.0)
        r = int(95 + 45 * f)
        gr = int(100 + 48 * f)
        b = int(110 + 52 * f)
        body_col = (r, gr, b, 255)
        accent_col = (max(0, r - 42), max(0, gr - 42), max(0, b - 40), 255)
        limb_col = outline
    else:
        # Травоядные: тёплый бежево-оливковый, при высоком мехе — снежно-кремовый
        f = np.clip(g[G_FUR], 0.0, 1.0)
        r = int(195 + 42 * f)
        gr = int(190 + 46 * f)
        b = int(125 + 110 * f)
        body_col = (r, gr, b, 255)
        accent_col = (max(0, r - 45), max(0, gr - 45), max(0, b - 45), 255)
        limb_col = outline

    # 2. Геометрия корпуса (плавно зависит от body_width и body_length)
    half_w = 2 if g[G_WIDTH] >= 0.56 else 1
    half_l = 2 if g[G_LENGTH] >= 0.62 else 1
    # Если безногий пловец — вытягиваем в торпеду
    if g[G_LEGS] < 1.5 and g[G_WIDTH] < 0.45:
        half_w = 1
        half_l = 3

    y0, y1 = max(2, cy - half_l), min(7, cy + half_l + 1)
    x0, x1 = max(2, cx - half_w), min(7, cx + half_w + 1)

    # 3. Хвост (исчезает по мере укорочения body_length или роста панциря/карцинизации)
    if g[G_LENGTH] > 0.55 and g[G_SHELL] < 0.45:
        tail_y = min(8, y1)
        sprite[tail_y, cx] = body_col if is_fox else accent_col
        if g[G_LENGTH] > 0.72 and tail_y + 1 < 9:
            sprite[tail_y + 1, cx] = accent_col  # У лисы здесь белый кончик хвоста!

    # 4. Ноги (от 0 до 4 пар, появляются постепенно по 2 штуки)
    pairs = min(4, int(round(g[G_LEGS])) // 2)
    leg_rows = [cy, cy - 1, cy + 1, cy - 2][:pairs]
    for ly in leg_rows:
        if 0 <= ly < 9:
            sprite[ly, max(0, x0 - 1)] = limb_col
            sprite[ly, min(8, x1)] = limb_col

    # 5. Основное тело и постепенное нарастание хитинового панциря
    sprite[y0:y1, x0:x1] = body_col
    if g[G_SHELL] > 0.25:
        # Сначала тёмная полоса на спине, при shell > 0.55 — полноценный щиток
        sprite[y0, x0:x1] = accent_col
        if g[G_SHELL] > 0.55:
            sprite[y1 - 1, x0:x1] = accent_col

    # 6. Клешни (растут постепенно: при 0.28 — зачатки 1px, при 0.55+ — полноценные клешни 2px)
    front_y = max(0, y0 - 1)
    if g[G_CLAWS] > 0.28:
        claw_col = (245, 175, 45, 255)
        sprite[front_y, max(0, x0)] = claw_col
        sprite[front_y, min(8, x1 - 1)] = claw_col
        if g[G_CLAWS] > 0.55 and front_y - 1 >= 0:
            sprite[front_y - 1, max(0, x0)] = claw_col
            sprite[front_y - 1, min(8, x1 - 1)] = claw_col
    elif g[G_FUR] > 0.45 and g[G_SHELL] < 0.30:
        # Ушки у пушистых зверей (лис, зайцев, волков)
        sprite[front_y, max(0, x0)] = limb_col if is_fox else accent_col
        sprite[front_y, min(8, x1 - 1)] = limb_col if is_fox else accent_col
    elif g[G_DIET] > 0.35:
        # Острая морда хищника
        sprite[front_y, cx] = body_col

    # 7. Глаза
    sprite[y0, x0] = outline
    sprite[y0, x1 - 1] = outline

    return sprite
