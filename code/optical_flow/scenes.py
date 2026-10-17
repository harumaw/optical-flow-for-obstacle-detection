"""Генератор синтетических сцен полёта с препятствиями.

Сцена состоит из плоских текстурированных слоёв, перпендикулярных направлению
полёта: дальний фон и одно препятствие (стена, столб, ящик). Камера-обскура
летит вдоль оси Z мира с постоянной скоростью v, может поворачиваться (рыскание
и тангаж) и смещаться вбок. Каждый пиксель рендерится честно: луч из камеры
пересекается с плоскостями слоёв, цвет берётся из текстуры (cv2.remap). Поэтому
поле движения, включая члены от вращения, физически правильное.

Правильный ответ для кадра:
    ttc_s    = (Z_препятствия - Z_камеры) / v
    present  = препятствие на пути (его поперечный размер перекрывает коридор
               шириной дрона) и ttc_s < ttc_thr
    side     = правило common.decision.decide по проекции препятствия
Препятствие сбоку (passby) не на пути: present всегда False, хотя оно близко.

Последовательность начинается при TTC = ttc_start и заканчивается при TTC = ttc_end,
поэтому при любой скорости она длится одинаковое время.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from common.decision import decide

W, H = 640, 480
HFOV_DEG = 60.0
F_PX = (W / 2) / math.tan(math.radians(HFOV_DEG / 2))
CX, CY = W / 2, H / 2
DRONE_HALF_WIDTH = 0.25        # м: половина ширины дрона, коридор пути


# ---------------------------------------------------------------- текстуры
def fractal_texture(h: int, w: int, rng: np.random.Generator, contrast: float = 1.0,
                    shapes: int = 40) -> np.ndarray:
    """Многомасштабная текстура: сумма октав шума + фигуры. Даёт детали на любой дистанции."""
    acc = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for scale in (64, 24, 8, 3):
        sh, sw = max(2, h // scale), max(2, w // scale)
        n = rng.random((sh, sw), dtype=np.float32)
        acc += amp * cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)
        total += amp
        amp *= 0.6
    acc /= total
    acc = (acc - acc.min()) / (acc.max() - acc.min() + 1e-6)
    img = acc * 255.0
    for _ in range(shapes):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = int(rng.integers(max(3, w // 80), max(6, w // 15)))
        g = float(rng.integers(0, 255))
        if rng.random() < 0.5:
            cv2.circle(img, (x, y), r, g, -1)
        else:
            cv2.rectangle(img, (x - r, y - r // 2), (x + r, y + r // 2), g, -1)
    img = 128.0 + (img - 128.0) * contrast
    return np.clip(img, 0, 255).astype(np.uint8)


@dataclass
class Layer:
    """Плоский прямоугольник в мире на глубине z, центр (x, y), размер w×h метров."""
    z: float
    x: float = 0.0
    y: float = 0.0
    w: float = 1e9           # бесконечная стена
    h: float = 1e9
    ppm: float = 300.0       # пикселей текстуры на метр
    tex: np.ndarray | None = None
    is_obstacle: bool = False


# ---------------------------------------------------------------- сценарий
@dataclass
class Scenario:
    name: str
    kind: str = "wall"              # wall | post | box | passby | none
    speed: float = 1.5              # м/с вперёд
    fps: float = 30.0
    obstacle_x: float = 0.0         # м, поперечное положение центра препятствия
    obstacle_w: float = 0.4
    obstacle_h: float = 2.0
    background_z: float = 25.0      # м до фона в начале
    yaw_rate: float = 0.0           # град/с, постоянный поворот (только для открытых сцен)
    yaw_sweep: float = 0.0          # град/с, максимальная скорость медленного рыскания туда-обратно
    yaw_sweep_hz: float = 0.25
    wobble_deg: float = 0.0         # амплитуда покачивания по рысканию и тангажу, град
    wobble_hz: float = 1.5
    lateral_speed: float = 0.0      # м/с вбок
    noise: float = 2.0              # СКО шума сенсора, уровней
    blur: float = 0.0               # размытие движения: доля межкадрового смещения
    contrast: float = 1.0           # контраст текстуры препятствия
    ttc_start: float = 8.0
    ttc_end: float = 0.6
    ttc_thr: float = 4.0
    seed: int = 0
    tags: list[str] = field(default_factory=list)

    @property
    def positive(self) -> bool:
        return self.kind in ("wall", "post", "box")


def _rot(yaw: float, pitch: float) -> np.ndarray:
    cy_, sy_ = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    Ry = np.array([[cy_, 0, sy_], [0, 1, 0], [-sy_, 0, cy_]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    return Ry @ Rx


_UU, _VV = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
_RAYS = np.stack([(_UU - CX) / F_PX, (_VV - CY) / F_PX, np.ones_like(_UU)], axis=-1)


def render(layers: list[Layer], cam: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Рендер слоёв от дальнего к ближнему. cam = (x, y, z) камеры, R камера->мир."""
    d = _RAYS @ R.T.astype(np.float32)                    # лучи в мире
    out = np.zeros((H, W), np.float32)
    for L in sorted(layers, key=lambda l: -l.z):
        dz = d[..., 2]
        t = (L.z - cam[2]) / np.where(np.abs(dz) < 1e-6, 1e-6, dz)
        X = cam[0] + t * d[..., 0]
        Y = cam[1] + t * d[..., 1]
        th, tw = L.tex.shape
        if L.w > 1e6:                                     # бесконечный слой: текстура повторяется
            mx = ((X - L.x) * L.ppm) % tw
            my = ((Y - L.y) * L.ppm) % th
            vals = cv2.remap(L.tex, mx.astype(np.float32), my.astype(np.float32),
                             cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            mask = t > 0
        else:
            mx = (X - (L.x - L.w / 2)) * L.ppm
            my = (Y - (L.y - L.h / 2)) * L.ppm
            vals = cv2.remap(L.tex, mx.astype(np.float32), my.astype(np.float32),
                             cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            mask = (t > 0) & (mx >= 0) & (mx < tw - 1) & (my >= 0) & (my < th - 1)
        out[mask] = vals[mask]
    return out


def project_box(L: Layer, cam: np.ndarray, R: np.ndarray) -> tuple[int, int, int, int] | None:
    """Прямоугольник проекции слоя-препятствия в кадре (обрезанный по кадру)."""
    xs, ys = [], []
    for dx in (-L.w / 2, L.w / 2):
        for dy in (-L.h / 2, L.h / 2):
            p = R.T @ (np.array([L.x + dx, L.y + dy, L.z]) - cam)
            if p[2] <= 0.05:
                return None
            xs.append(CX + F_PX * p[0] / p[2])
            ys.append(CY + F_PX * p[1] / p[2])
    x1, x2 = max(0, min(xs)), min(W, max(xs))
    y1, y2 = max(0, min(ys)), min(H, max(ys))
    if x2 <= x1 or y2 <= y1:
        return None
    return int(x1), int(y1), int(x2), int(y2)


def generate(sc: Scenario):
    """Итератор (bgr, t, label) по кадрам сценария."""
    rng = np.random.default_rng(sc.seed)
    v = sc.speed
    dt = 1.0 / sc.fps
    # дистанция до препятствия в начале: TTC_start * v (для hover — фиксированная стена в 3 м)
    obs_z = sc.ttc_start * v if v > 0 else 3.0
    bg_z = max(sc.background_z, obs_z + 10.0)
    bg_tex = fractal_texture(1200, 1200, rng, 1.0, 400)
    # текстура фона покрывает поле зрения с запасом на повороты, чтобы не было видно повторов
    bg = Layer(z=bg_z, ppm=1200.0 / (2.6 * bg_z * math.tan(math.radians(HFOV_DEG / 2 + 15))), tex=bg_tex)
    layers = [bg]
    obs = None
    if sc.kind == "wall":
        obs = Layer(z=obs_z, ppm=250.0, tex=fractal_texture(1400, 1400, rng, sc.contrast, 120),
                    is_obstacle=True)
    elif sc.kind in ("post", "box", "passby"):
        ppm = 500.0
        th, tw = int(sc.obstacle_h * ppm), int(sc.obstacle_w * ppm)
        obs = Layer(z=obs_z, x=sc.obstacle_x, w=sc.obstacle_w, h=sc.obstacle_h, ppm=ppm,
                    tex=fractal_texture(th, tw, rng, sc.contrast, 30), is_obstacle=True)
    elif sc.kind == "none" and v == 0:
        obs = Layer(z=obs_z, ppm=250.0, tex=fractal_texture(1400, 1400, rng, sc.contrast, 120))
    if obs is not None:
        layers.append(obs)

    duration = (sc.ttc_start - sc.ttc_end) if v > 0 else 6.0
    n = int(round(duration * sc.fps))
    prev = None
    for k in range(n):
        t = k * dt
        cam = np.array([sc.lateral_speed * t, 0.0, v * t])
        sweep_amp = sc.yaw_sweep / (2 * math.pi * sc.yaw_sweep_hz) if sc.yaw_sweep else 0.0
        yaw = math.radians(sc.yaw_rate * t + sweep_amp * math.sin(2 * math.pi * sc.yaw_sweep_hz * t)
                           + sc.wobble_deg * math.sin(2 * math.pi * sc.wobble_hz * t))
        pitch = math.radians(0.6 * sc.wobble_deg * math.sin(2 * math.pi * sc.wobble_hz * 1.3 * t + 1.0))
        R = _rot(yaw, pitch)
        img = render(layers, cam, R)
        if sc.blur > 0 and prev is not None:
            img = (1 - sc.blur) * img + sc.blur * prev       # смаз: смешение с предыдущим положением
        prev = img
        if sc.noise > 0:
            img = img + rng.normal(0, sc.noise, img.shape).astype(np.float32)
        gray = np.clip(img, 0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        label = {"frame": k, "present": False, "side": "none", "dist_m": None, "ttc_s": None,
                 "on_path": False, "box": None}
        if obs is not None and obs.is_obstacle and v > 0:
            dist = obs.z - cam[2]
            ttc = dist / v
            on_path = (obs.w > 1e6) or (abs(obs.x - cam[0]) < obs.w / 2 + DRONE_HALF_WIDTH)
            box = project_box(obs, cam, R) if obs.w < 1e6 else (0, 0, W, H)
            present = bool(on_path and ttc < sc.ttc_thr and box is not None)
            label.update({"dist_m": dist, "ttc_s": ttc, "present": present,
                          "side": decide(box, W) if present else "none",
                          "on_path": bool(on_path and box is not None), "box": box})
        yield bgr, t, label


# ---------------------------------------------------------------- набор сценариев
def default_suite() -> list[Scenario]:
    S: list[Scenario] = []
    for v in (0.5, 1.0, 2.0, 3.0, 4.5, 6.0):
        S.append(Scenario(f"wall_v{v:g}", "wall", speed=v, tags=["speed"]))
        S.append(Scenario(f"post_v{v:g}", "post", speed=v, obstacle_w=0.4, tags=["speed"]))
    S += [
        Scenario("box_left", "box", obstacle_x=-0.45, obstacle_w=0.8, obstacle_h=0.8, tags=["side"]),
        Scenario("box_right", "box", obstacle_x=0.45, obstacle_w=0.8, obstacle_h=0.8, seed=1, tags=["side"]),
        Scenario("wall_yaw10", "wall", yaw_sweep=10, seed=2, tags=["rotation"]),
        Scenario("wall_yaw25", "wall", yaw_sweep=25, seed=3, tags=["rotation"]),
        Scenario("post_yaw15", "post", yaw_sweep=15, seed=13, tags=["rotation"]),
        Scenario("post_wobble", "post", wobble_deg=4.0, seed=4, tags=["rotation"]),
        Scenario("post_lateral", "post", lateral_speed=0.1, obstacle_x=0.55, seed=5, tags=["rotation"]),
        Scenario("wall_lowtex", "wall", contrast=0.25, seed=6, tags=["quality"]),
        Scenario("post_noisy", "post", noise=8.0, blur=0.5, seed=7, tags=["quality"]),
        Scenario("post_fps20", "post", fps=20.0, seed=8, tags=["quality"]),
        # негативные: препятствия на пути нет
        Scenario("passby", "passby", obstacle_x=1.3, obstacle_w=0.5, seed=9, tags=["negative"]),
        Scenario("open_yaw20", "none", yaw_sweep=20, seed=10, tags=["negative"]),
        Scenario("hover_yaw20", "none", speed=0.0, yaw_sweep=20, seed=11, tags=["negative"]),
        Scenario("open_wobble", "none", wobble_deg=4.0, seed=12, tags=["negative"]),
    ]
    return S


def load_cached(sc: Scenario, cache_dir) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Кадры сценария (N, H, W) uint8 полутон, времена и разметка; рендер один раз, дальше с диска."""
    import json
    from pathlib import Path
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    f_img, f_meta = cache / f"{sc.name}.npy", cache / f"{sc.name}.json"
    if f_img.exists() and f_meta.exists():
        meta = json.loads(f_meta.read_text(encoding="utf-8"))
        if meta.get("scenario") == repr(sc):
            return np.load(f_img), np.array(meta["t"]), meta["labels"]
    frames, ts, labels = [], [], []
    for bgr, t, lab in generate(sc):
        frames.append(bgr[..., 0].copy())
        ts.append(t)
        labels.append(lab)
    arr = np.stack(frames)
    np.save(f_img, arr)
    f_meta.write_text(json.dumps({"scenario": repr(sc), "t": ts, "labels": labels}), encoding="utf-8")
    return arr, np.array(ts), labels


def scenario_by_name(name: str) -> Scenario:
    for s in default_suite():
        if s.name == name:
            return s
    raise KeyError(name)


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common.labels import write_labels
    from common.sources import make_writer

    ap = argparse.ArgumentParser(description="Записать сценарии в mp4 + разметку")
    ap.add_argument("--out", default="data/scenes")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for sc in default_suite():
        if args.only and sc.name not in args.only:
            continue
        wr = make_writer(out / f"{sc.name}.mp4", (W, H), sc.fps)
        labels = []
        for bgr, t, lab in generate(sc):
            wr.write(bgr)
            labels.append(lab)
        wr.release()
        write_labels(out / f"{sc.name}.labels.csv", labels)
        print(f"{sc.name}: {len(labels)} frames, {sum(l['present'] for l in labels)} with obstacle")
