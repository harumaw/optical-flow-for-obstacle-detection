"""Датасет TUM RGB-D (https://cvg.cit.tum.de/data/datasets/rgbd-dataset).

Структура папки последовательности:
    rgb.txt      строки "timestamp rgb/xxx.png"
    depth.txt    строки "timestamp depth/xxx.png"   (uint16, метры * 5000)
    rgb/, depth/

Глубина даёт автоматическую разметку: есть ли препятствие ближе near_m
в центральной зоне кадра и с какой оно стороны.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .decision import decide

DEPTH_SCALE = 5000.0


def _read_list(path: Path) -> list[tuple[float, str]]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ts, name = line.split()[:2]
            out.append((float(ts), name))
    return out


def associate(a: list[tuple[float, str]], b: list[tuple[float, str]],
              max_dt: float = 0.02) -> list[tuple[float, str, str]]:
    """Сопоставить два списка по ближайшим временным меткам (как associate.py из TUM)."""
    b_ts = np.array([t for t, _ in b])
    used = np.zeros(len(b), dtype=bool)
    pairs = []
    for t, name in a:
        j = int(np.argmin(np.abs(b_ts - t)))
        if not used[j] and abs(b_ts[j] - t) <= max_dt:
            used[j] = True
            pairs.append((t, name, b[j][1]))
    return pairs


class TumSequence:
    def __init__(self, root: str | Path, max_dt: float = 0.02) -> None:
        self.root = Path(root)
        rgb = _read_list(self.root / "rgb.txt")
        depth = _read_list(self.root / "depth.txt")
        self.pairs = associate(rgb, depth, max_dt)
        if not self.pairs:
            raise RuntimeError(f"No rgb/depth pairs in {self.root}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __iter__(self) -> Iterator[tuple[float, Path, Path]]:
        for t, rgb, depth in self.pairs:
            yield t, self.root / rgb, self.root / depth

    @staticmethod
    def read_depth(path: Path) -> np.ndarray:
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise RuntimeError(f"Cannot read depth {path}")
        return raw.astype(np.float32) / DEPTH_SCALE


def auto_label(depth: np.ndarray, near_m: float = 1.0, center_frac: float = 0.6,
               min_frac: float = 0.05, wide_frac: float = 0.6,
               min_valid_frac: float = 0.2) -> dict:
    """Разметка по глубине.

    present: доля пикселей центральной зоны с 0 < depth < near_m не меньше min_frac.
    bbox:    прямоугольник самой большой связной "ближней" области (в координатах кадра).
    side:    то же правило decide(), что и у детекторов.
    dist_m:  10-й перцентиль валидной глубины в центральной зоне (расстояние до ближайшего).
    Если валидной глубины в зоне меньше min_valid_frac, кадр помечается как unknown.
    """
    h, w = depth.shape
    x0, x1 = int(w * (0.5 - center_frac / 2)), int(w * (0.5 + center_frac / 2))
    y0, y1 = int(h * (0.5 - center_frac / 2)), int(h * (0.5 + center_frac / 2))
    roi = depth[y0:y1, x0:x1]
    valid = roi > 0
    if valid.mean() < min_valid_frac:
        return {"present": False, "side": "none", "bbox": None, "dist_m": None, "unknown": True}
    dist = float(np.percentile(roi[valid], 10))
    near = (valid & (roi < near_m)).astype(np.uint8)
    frac = float(near.mean())
    if frac < min_frac:
        return {"present": False, "side": "none", "bbox": None, "dist_m": dist, "unknown": False}
    near = cv2.morphologyEx(near, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(near, connectivity=8)
    if n <= 1:
        return {"present": False, "side": "none", "bbox": None, "dist_m": dist, "unknown": False}
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, bw, bh = (stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP],
                    stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT])
    bbox = (x0 + int(x), y0 + int(y), x0 + int(x + bw), y0 + int(y + bh))
    return {"present": True, "side": decide(bbox, w, wide_frac), "bbox": bbox,
            "dist_m": dist, "unknown": False}


# ---------------------------------------------------------------------------
# Траектория камеры (groundtruth.txt): "timestamp tx ty tz qx qy qz qw" —
# положение и ориентация оптического центра цветной камеры в мировой системе
# захвата движения. Оси камеры оптические: z вперёд, x вправо, y вниз.

def read_groundtruth(root: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    path = Path(root) / "groundtruth.txt"
    if not path.exists():
        return None
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append([float(v) for v in line.split()[:8]])
    a = np.array(rows, dtype=np.float64)
    return a[:, 0], a[:, 1:4], a[:, 4:8]


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Кватернион (qx, qy, qz, qw) -> матрица поворота камера->мир."""
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def forward_speed(gt, times: np.ndarray, half_window: float = 0.15) -> np.ndarray:
    """Скорость камеры вдоль её оптической оси (м/с) в моменты times.
    Скорость в мире берётся центральной разностью по окну ±half_window,
    затем проецируется на ось z камеры. NaN там, где траектории нет."""
    gt_t, gt_p, gt_q = gt
    out = np.full(len(times), np.nan)
    for i, t in enumerate(times):
        if t - half_window < gt_t[0] or t + half_window > gt_t[-1]:
            continue
        p0 = np.array([np.interp(t - half_window, gt_t, gt_p[:, k]) for k in range(3)])
        p1 = np.array([np.interp(t + half_window, gt_t, gt_p[:, k]) for k in range(3)])
        v_world = (p1 - p0) / (2 * half_window)
        j = int(np.clip(np.searchsorted(gt_t, t), 0, len(gt_t) - 1))
        R = quat_to_rot(gt_q[j])
        out[i] = float((R.T @ v_world)[2])
    return out


def nearest_region(depth: np.ndarray, center_frac: float = 0.6, rel: float = 1.3,
                   min_valid_frac: float = 0.2) -> tuple[float | None, tuple | None]:
    """Расстояние до ближайшего в центральной зоне (10-й перцентиль) и
    прямоугольник поверхностей, лежащих не дальше rel * этого расстояния."""
    h, w = depth.shape
    x0, x1 = int(w * (0.5 - center_frac / 2)), int(w * (0.5 + center_frac / 2))
    y0, y1 = int(h * (0.5 - center_frac / 2)), int(h * (0.5 + center_frac / 2))
    roi = depth[y0:y1, x0:x1]
    valid = roi > 0
    if valid.mean() < min_valid_frac:
        return None, None
    dist = float(np.percentile(roi[valid], 10))
    near = (valid & (roi <= dist * rel)).astype(np.uint8)
    near = cv2.morphologyEx(near, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(near, connectivity=8)
    if n <= 1:
        return dist, None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, bw, bh = (stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP],
                    stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT])
    return dist, (x0 + int(x), y0 + int(y), x0 + int(x + bw), y0 + int(y + bh))


# ---------------------------------------------------------------------------
# Пол. Для камеры на колёсном роботе пол лежит на постоянной высоте под камерой.
# Высота точки под оптическим центром: Y = (v - cy) * Z / fy (ось y камеры вниз).
# Пиксели с Y > floor_y - margin считаются полом и из разметки исключаются:
# сближение с полом даёт реальную дивергенцию, но препятствием не является.

KINECT_FR2 = {"fx": 520.9, "fy": 521.0, "cx": 325.1, "cy": 249.7}


def height_below_camera(depth: np.ndarray, fy: float = KINECT_FR2["fy"],
                        cy: float = KINECT_FR2["cy"]) -> np.ndarray:
    v = np.arange(depth.shape[0], dtype=np.float32)[:, None]
    return (v - cy) * depth / fy


def estimate_floor_height(seq: "TumSequence", step: int = 20, bottom_frac: float = 0.15) -> float:
    """Мода высоты под камерой по нижним строкам кадров: там почти всегда пол."""
    vals = []
    for i, (_, _, dp) in enumerate(seq):
        if i % step:
            continue
        Z = seq.read_depth(dp)
        Y = height_below_camera(Z)
        rows = int(Z.shape[0] * (1 - bottom_frac))
        m = Z[rows:] > 0
        vals.append(Y[rows:][m])
    y = np.concatenate(vals)
    y = y[(y > 0.05) & (y < 3.0)]
    hist, edges = np.histogram(y, bins=150)
    k = int(np.argmax(hist))
    return float(0.5 * (edges[k] + edges[k + 1]))


def remove_floor(depth: np.ndarray, floor_y: float, margin: float = 0.10) -> np.ndarray:
    """Копия карты глубины, где пиксели пола обнулены (как невалидные)."""
    out = depth.copy()
    out[height_below_camera(depth) > floor_y - margin] = 0.0
    return out
