"""Общие типы данных: кадр, результат детекции и базовый класс детектора."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Рекомендуемый манёвр: none (препятствия нет), left/right (обойти слева/справа),
# stop (препятствие перекрывает кадр; для БПЛА это остановка или набор высоты).
SIDES = ("none", "left", "right", "stop")


@dataclass
class Frame:
    """Один кадр источника."""

    index: int
    t: float                      # время кадра, секунды
    bgr: np.ndarray               # изображение HxWx3, uint8
    depth: np.ndarray | None = None  # карта глубины в метрах (0 = нет данных), только для датасетов
    gravity: np.ndarray | None = None  # единичный вектор силы тяжести в осях камеры (x вправо, y вниз,
                                       # z вперёд), из крена/тангажа автопилота; None = неизвестен
    R_wc: np.ndarray | None = None     # ориентация камеры в мире (3x3, камера -> мир), из автопилота/гироскопа

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.bgr.shape[:2]
        return w, h


@dataclass
class ObstacleResult:
    """Единый выход любого детектора."""

    method: str
    frame_index: int
    t: float
    present: bool = False
    bbox: tuple[int, int, int, int] | None = None   # x1, y1, x2, y2 в координатах исходного кадра
    side: str = "none"
    proximity: float | None = None                  # мера близости, см. proximity_kind
    proximity_kind: str = ""                        # "ttc_s" | "dist_m" | "size_frac"
    score: float = 0.0                              # уверенность 0..1
    timing_ms: dict[str, float] = field(default_factory=dict)
    debug: dict[str, Any] = field(default_factory=dict)   # карты для визуализации, в CSV не пишутся

    def to_row(self) -> dict[str, Any]:
        x1, y1, x2, y2 = self.bbox if self.bbox else (-1, -1, -1, -1)
        prox = ""
        if self.proximity is not None and np.isfinite(self.proximity):
            prox = round(float(self.proximity), 3)
        row: dict[str, Any] = {
            "frame": self.frame_index,
            "t": round(self.t, 4),
            "method": self.method,
            "present": int(self.present),
            "side": self.side,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "proximity": prox,
            "proximity_kind": self.proximity_kind,
            "score": round(self.score, 3),
        }
        for k, v in self.timing_ms.items():
            row[f"ms_{k}"] = round(v, 3)
        return row


class BaseDetector:
    """Интерфейс детектора. Реализации: FlowObstacleDetector, позже depth/yolo."""

    name: str = "base"
    proximity_kind: str = ""

    def reset(self) -> None:
        """Сбросить состояние между последовательностями."""

    def process(self, frame: Frame) -> ObstacleResult:
        raise NotImplementedError
