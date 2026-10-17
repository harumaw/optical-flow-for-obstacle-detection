"""Бэкенды оптического потока.

Каждый принимает два полутоновых кадра (uint8, одинакового размера) и
возвращает плотное поле (H, W, 2) float32: смещение пикселя в пикселях за кадр.
"""
from __future__ import annotations

import cv2
import numpy as np


class FarnebackFlow:
    """Плотный поток Фарнебака (полиномиальное разложение окрестностей)."""

    name = "farneback"

    def __init__(self, pyr_scale: float = 0.5, levels: int = 3, winsize: int = 15,
                 iterations: int = 3, poly_n: int = 5, poly_sigma: float = 1.2) -> None:
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma

    def compute(self, prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
        return cv2.calcOpticalFlowFarneback(
            prev, cur, None, self.pyr_scale, self.levels, self.winsize,
            self.iterations, self.poly_n, self.poly_sigma, 0)


class DISFlow:
    """Dense Inverse Search (Kroeger et al., 2016). Самый быстрый плотный метод в OpenCV."""

    name = "dis"
    PRESETS = {
        "ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
        "fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
        "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
    }

    def __init__(self, preset: str = "medium") -> None:
        # fast: ~7 мс, но на мелких объектах теряет расширение (грубая шкала);
        # medium: ~10 мс при 320 px, качество на уровне Farneback
        if preset not in self.PRESETS:
            raise ValueError(f"DIS preset must be one of {list(self.PRESETS)}, got {preset!r}")
        self.preset = preset
        self._dis = cv2.DISOpticalFlow_create(self.PRESETS[preset])
        self.name = f"dis-{preset}"

    def compute(self, prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
        return self._dis.calc(prev, cur, None)


class LKGridFlow:
    """Разреженный Лукас-Канаде (как в курсовой) на регулярной сетке точек.
    Поток в узлах сетки интерполируется до плотного поля."""

    name = "lk"

    def __init__(self, cell: int = 16, win: int = 21, levels: int = 3, max_err: float = 30.0) -> None:
        self.cell = cell
        self.win = win
        self.levels = levels
        self.max_err = max_err
        self._shape: tuple[int, int] | None = None
        self._pts: np.ndarray | None = None
        self._rows = 0
        self._cols = 0
        self._criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)

    def _build_grid(self, h: int, w: int) -> None:
        xs = np.arange(self.cell // 2, w, self.cell, dtype=np.float32)
        ys = np.arange(self.cell // 2, h, self.cell, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)
        self._rows, self._cols = gx.shape
        self._pts = np.stack([gx.ravel(), gy.ravel()], axis=1).reshape(-1, 1, 2).astype(np.float32)
        self._shape = (h, w)

    def compute(self, prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
        h, w = prev.shape
        if self._shape != (h, w):
            self._build_grid(h, w)
        nxt, st, err = cv2.calcOpticalFlowPyrLK(
            prev, cur, self._pts, None, winSize=(self.win, self.win),
            maxLevel=self.levels, criteria=self._criteria)
        d = (nxt - self._pts).reshape(self._rows, self._cols, 2).astype(np.float32)
        ok = (st.reshape(self._rows, self._cols) == 1) & (err.reshape(self._rows, self._cols) < self.max_err)
        d = _fill_invalid(d, ok)
        return cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)


def _fill_invalid(d: np.ndarray, ok: np.ndarray, passes: int = 3) -> np.ndarray:
    """Заполнить ненадёжные узлы средним по соседям (нормированная свёртка)."""
    if ok.all():
        return d
    d = d.copy()
    d[~ok] = 0.0
    okf = ok.astype(np.float32)
    for _ in range(passes):
        num = cv2.blur(d * okf[..., None], (3, 3))
        den = cv2.blur(okf, (3, 3))
        filled = num / np.maximum(den, 1e-6)[..., None]
        newly = (~ok) & (den > 1e-6)
        d[newly] = filled[newly]
        ok = ok | newly
        okf = ok.astype(np.float32)
        if ok.all():
            break
    return d


BACKENDS = {
    "farneback": FarnebackFlow,
    "dis": DISFlow,
    "lk": LKGridFlow,
}


def make_flow_backend(name: str, **kwargs):
    if name not in BACKENDS:
        raise ValueError(f"Unknown flow backend {name!r}; available: {list(BACKENDS)}")
    return BACKENDS[name](**kwargs)
