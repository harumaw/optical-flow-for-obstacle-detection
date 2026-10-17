"""Правило выбора манёвра по области препятствия. Общее для всех методов
и для автоматической разметки, чтобы сравнение было честным."""
from __future__ import annotations

import numpy as np


def decide(bbox: tuple[int, int, int, int] | None, frame_w: int, wide_frac: float = 0.6) -> str:
    """Область слева от центра -> уходим вправо, справа -> влево.
    Если область шире wide_frac кадра -> stop."""
    if bbox is None:
        return "none"
    x1, _, x2, _ = bbox
    if (x2 - x1) / float(frame_w) >= wide_frac:
        return "stop"
    cx = 0.5 * (x1 + x2)
    return "right" if cx < frame_w / 2.0 else "left"


def flow_balance(flow: np.ndarray) -> float:
    """Баланс потока: (правая половина - левая) / сумма, в [-1, 1].
    Положительное значение: справа поток сильнее, объекты справа ближе.
    Классическая стратегия пчёл: поворачивать в сторону меньшего потока."""
    mag = np.linalg.norm(flow, axis=2)
    half = mag.shape[1] // 2
    left = float(mag[:, :half].mean())
    right = float(mag[:, half:].mean())
    return (right - left) / (right + left + 1e-6)
