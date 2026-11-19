import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from core.analytics import EcosystemTracker
    from core.world import World


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


class ExperimentStorage:
    """Сохранение и загрузка экспериментов в SQLite и JSON."""

    def __init__(self, db_path: str | Path = "data/experiments.db", export_dir: str | Path = "data/exports"):
        self.db_path = Path(db_path)
        self.export_dir = Path(export_dir)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    preset TEXT NOT NULL,
                    seed INTEGER,
                    total_ticks INTEGER NOT NULL,
                    final_population INTEGER NOT NULL,
                    max_crab_score REAL NOT NULL,
                    params_json TEXT NOT NULL,
                    timeline_json TEXT NOT NULL
                )
            """)

    def save_experiment(self, world: "World", tracker: "EcosystemTracker", params: dict) -> tuple[int, Path]:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        timeline = list(tracker.history)
        max_crab = max((float(row.get("max_crab_score", 0.0)) for row in timeline), default=0.0)
        raw_seed = getattr(world, "seed", None)
        seed_val = int(raw_seed) if raw_seed is not None else None
        preset_val = str(getattr(world, "preset", "archipelago"))

        params_str = json.dumps(params, ensure_ascii=False, default=_json_default)
        timeline_str = json.dumps(timeline, ensure_ascii=False, default=_json_default)

        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                """
                INSERT INTO experiments (
                    created_at, preset, seed, total_ticks, final_population,
                    max_crab_score, params_json, timeline_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    preset_val,
                    seed_val,
                    int(world.tick),
                    len(world.agents),
                    max_crab,
                    params_str,
                    timeline_str,
                ),
            )
            exp_id = int(cur.lastrowid)

        json_file = self.export_dir / f"exp_{exp_id}_{timestamp}.json"
        payload = {
            "experiment_id": exp_id,
            "created_at": timestamp,
            "preset": preset_val,
            "seed": seed_val,
            "total_ticks": int(world.tick),
            "parameters": params,
            "timeline": timeline,
        }
        json_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
        return exp_id, json_file

    def list_experiments(self) -> list[dict]:
        """Возвращает список сохранённых экспериментов из SQLite (новые сверху)."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id, created_at, preset, total_ticks, final_population, max_crab_score "
                "FROM experiments ORDER BY id DESC LIMIT 30"
            ).fetchall()
        return [
            {
                "id": r[0],
                "created_at": r[1],
                "preset": r[2],
                "total_ticks": r[3],
                "final_population": r[4],
                "max_crab_score": r[5],
            }
            for r in rows
        ]

    def load_from_db(self, exp_id: int) -> dict | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT id, created_at, preset, seed, total_ticks, params_json, timeline_json "
                "FROM experiments WHERE id = ?",
                (exp_id,),
            ).fetchone()
        if not row:
            return None
        return {
            "experiment_id": row[0],
            "created_at": row[1],
            "preset": row[2],
            "seed": row[3],
            "total_ticks": row[4],
            "parameters": json.loads(row[5]),
            "timeline": json.loads(row[6]),
        }

    def load_from_json(self, json_path: str | Path) -> dict:
        return json.loads(Path(json_path).read_text(encoding="utf-8"))