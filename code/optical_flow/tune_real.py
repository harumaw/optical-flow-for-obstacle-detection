"""Подбор параметров детектора на размеченных реальных полётах (ангар UZH-FPV и др.).

    python optical_flow/tune_real.py cache  VIDEO [VIDEO ...]   # один раз: кадры и поток Farneback на диск
    python optical_flow/tune_real.py search --train V1 [V2] [--test V3 ...]
    python optical_flow/tune_real.py eval   --config results/tuning_real/best.json VIDEO [...]

Кэш: data/flow_cache/<видео>_w<ширина>.npz — полутоновые кадры рабочей ширины, поле потока (float16),
время кадров. Поток считается один раз на ширину; параметры обработки перебираются без пересчёта.

Метрики на видео (разметка <видео>.events.json, см. real_videos.py):
    caught     доля опасных эпизодов, где тревога прозвучала не позже чем за LEAD_MIN с до конца эпизода
    lead       среднее упреждение пойманных эпизодов, с
    fa         доля кадров открытого пути (safe) с тревогой
Критерий (порядок): fa ≤ FA_LIMIT; больше caught; больше lead (шаг 0.25 с — точность разметки); меньше fa.
Поиск покоординатный, 3 прохода, с несколькими стартами.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.sources import VideoSource                               # noqa: E402
from common.types import Frame                                       # noqa: E402
from optical_flow.backends import FarnebackFlow                      # noqa: E402
from optical_flow.detector import FlowConfig, FlowObstacleDetector   # noqa: E402

CACHE = ROOT / "data" / "flow_cache"
OUT = ROOT / "results" / "tuning_real"
FA_LIMIT = 0.05
LEAD_MIN = 0.3
MAX_FPS = 30.0

GRID = {
    "work_width": [160, 240],
    "flow_sigma": [2.0, 3.0, 5.0],          # px при ширине 320 (масштабируется)
    "row_baseline": [0.0, 1.0, 1.2, 1.5, 2.0],
    "ground_frac": [0.5, 0.6, 0.7, 0.75, 0.8],
    "corridor_frac": [0.25, 0.3, 0.4, 0.6],
    "path_frac": [0.1, 0.15, 0.3, 1.0],
    "min_cells": [2, 4, 8, 12],
    "ema_alpha": [0.4, 0.7, 1.0],
    "derotate": [True, False],
    "min_persist": [1, 2, 3],
    "ttc_threshold_s": [0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0],
    "horizon_margin_deg": [-1, 0, 5, 10, 15, 20, 30],  # -1 = маска по горизонту выключена (тогда ground_frac)
    "gyro_derotate": [False, True],
    "consistency_tol": [0.0, 0.2, 0.5, 1.0],
}
STARTS = [
    # лучшая с маской горизонта (прошлый круг)
    {"work_width": 160, "flow_sigma": 2.0, "row_baseline": 1.5, "ground_frac": 0.5, "corridor_frac": 0.6,
     "path_frac": 0.1, "min_cells": 2, "ema_alpha": 1.0, "derotate": True, "min_persist": 1, "ttc_threshold_s": 2.0,
     "horizon_margin_deg": 10, "gyro_derotate": False, "consistency_tol": 0.0},
    # то же + поворот по ориентации + проверка надвигания
    {"work_width": 160, "flow_sigma": 2.0, "row_baseline": 1.5, "ground_frac": 0.5, "corridor_frac": 0.6,
     "path_frac": 0.1, "min_cells": 2, "ema_alpha": 1.0, "derotate": False, "min_persist": 2, "ttc_threshold_s": 3.0,
     "horizon_margin_deg": 10, "gyro_derotate": True, "consistency_tol": 0.5},
    # лучшая по синтетике (для сравнения)
    {"work_width": 160, "flow_sigma": 3.0, "row_baseline": 0.0, "ground_frac": 0.2, "corridor_frac": 0.6,
     "path_frac": 0.15, "min_cells": 4, "ema_alpha": 0.7, "derotate": True, "min_persist": 1, "ttc_threshold_s": 4.0,
     "horizon_margin_deg": -1, "gyro_derotate": False, "consistency_tol": 0.0},
    {"work_width": 240, "flow_sigma": 3.0, "row_baseline": 1.2, "ground_frac": 0.5, "corridor_frac": 0.6,
     "path_frac": 0.15, "min_cells": 4, "ema_alpha": 0.7, "derotate": True, "min_persist": 2, "ttc_threshold_s": 3.0,
     "horizon_margin_deg": 15, "gyro_derotate": True, "consistency_tol": 0.2},
]


# ------------------------------------------------------------------ кэш
def cache_path(video: Path, width: int) -> Path:
    return CACHE / f"{video.stem}_w{width}.npz"


def build_cache(video: Path, width: int) -> Path:
    p = cache_path(video, width)
    if p.exists():
        return p
    CACHE.mkdir(parents=True, exist_ok=True)
    src = VideoSource(video)
    step = max(1, int(round(src.fps / MAX_FPS)))      # 60 к/с -> каждый второй кадр
    grays, ts = [], []
    k = 0
    for fr in src:
        if k % step == 0:
            g = cv2.cvtColor(fr.bgr, cv2.COLOR_BGR2GRAY)
            h = max(8, int(round(g.shape[0] * width / g.shape[1])))
            grays.append(cv2.resize(g, (width, h), interpolation=cv2.INTER_AREA))
            ts.append(fr.t)
        k += 1
    src.release()
    fb = FarnebackFlow()
    flows = [np.zeros(grays[0].shape + (2,), np.float16)]
    for i in range(1, len(grays)):
        flows.append(fb.compute(grays[i - 1], grays[i]).astype(np.float16))
    np.savez(p, gray=np.stack(grays), flow=np.stack(flows), t=np.array(ts))
    return p


class CachedFlow:
    name = "farneback"

    def __init__(self, flows):
        self.flows, self.i = flows, 1

    def compute(self, prev, cur):
        f = self.flows[self.i].astype(np.float32)
        self.i += 1
        return f


_DATA: dict = {}


def load(video: str, width: int):
    key = (video, width)
    if key not in _DATA:
        z = np.load(cache_path(Path(video), width))
        lab = json.loads(Path(video).with_name(Path(video).stem + ".events.json").read_text(encoding="utf-8"))
        grav = rot = None
        seq_root = Path(video).with_suffix("")
        if (seq_root / "groundtruth.txt").exists():
            from common.uzh import camera_orientation, gravity_in_camera
            grav = gravity_in_camera(seq_root, z["t"])
            rot = camera_orientation(seq_root, z["t"])
        _DATA[key] = (z["gray"], z["flow"], z["t"], lab, grav, rot)
    return _DATA[key]


# ------------------------------------------------------------------ прогон и метрики
def make_cfg(p: dict) -> FlowConfig:
    kw = dict(p)
    kw["flow_sigma"] = p["flow_sigma"] * p["work_width"] / 320.0
    m = kw.pop("horizon_margin_deg", None)
    from common.uzh import INDOOR_CAM0 as C          # модель объектива (UZH: рыбий глаз, калибровка cam0)
    sc = p["work_width"] / C["width"]
    kw.update(cam_model=C["model"], cam_f=C["f"] * sc, cam_cx=C["cx"] * sc, cam_cy=C["cy"] * sc)
    if m is not None and m >= 0:                     # маска пола по горизонту
        kw.update(horizon_mask=True, horizon_margin_deg=m)
    return FlowConfig(backend="farneback", max_dt=0.2, **kw)


def run(video: str, p: dict) -> dict:
    gray, flow, t, lab, grav, rot = load(video, p["work_width"])
    det = FlowObstacleDetector(make_cfg(p))
    det.backend = CachedFlow(flow)
    alarm = np.zeros(len(t), bool)
    for k in range(len(t)):
        g = None if grav is None or np.isnan(grav[k, 0]) else grav[k]
        R = None if rot is None or np.isnan(rot[k, 0, 0]) else rot[k]
        alarm[k] = det.process(Frame(k, float(t[k]), gray[k], gravity=g, R_wc=R)).present
    inn = lambda spans: np.any([(t >= a) & (t <= b) for a, b in spans], axis=0) if spans else np.zeros(len(t), bool)
    ev_mask = inn([(e["start"], e["end"]) for e in lab["events"]])
    safe = inn([tuple(x) for x in lab.get("safe", [])]) & ~ev_mask & ~inn([tuple(x) for x in lab.get("ignore", [])])
    leads = []
    for e in lab["events"]:
        m = (t >= e["start"]) & (t <= e["end"] - LEAD_MIN) & alarm
        leads.append(e["end"] - t[m][0] if m.any() else None)
    hit = [x for x in leads if x is not None]
    return {"events": len(leads), "caught": len(hit), "lead": float(np.mean(hit)) if hit else 0.0,
            "fa": float(alarm[safe].mean()) if safe.any() else 0.0, "safe_frames": int(safe.sum())}


def combine(rs: list[dict]) -> dict:
    ev = sum(r["events"] for r in rs)
    ca = sum(r["caught"] for r in rs)
    lead = sum(r["lead"] * r["caught"] for r in rs) / ca if ca else 0.0
    sf = sum(r["safe_frames"] for r in rs)
    fa = sum(r["fa"] * r["safe_frames"] for r in rs) / sf if sf else 0.0
    return {"events": ev, "caught": ca, "caught_rate": ca / ev if ev else 0.0, "lead": lead, "fa": fa}


LEAD_STEP = 0.25   # с: разметка сделана с шагом 0.5 с, разница в упреждении меньше 0.25 с не значима


def rank(m: dict):
    return (max(m["fa"] - FA_LIMIT, 0.0), -m["caught_rate"], -math.floor(m["lead"] / LEAD_STEP + 1e-9), m["fa"])


def _job(args):
    videos, p = args
    return combine([run(v, p) for v in videos])


def _init_worker():
    cv2.setNumThreads(1)


def key_of(p):
    return tuple(sorted(p.items()))


def search(train: list[str], test: list[str], workers: int) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    memo: dict = {}
    t0 = time.perf_counter()
    with Pool(workers, initializer=_init_worker) as pool:
        def evaluate(ps):
            todo = [p for p in ps if key_of(p) not in memo]
            for p, m in zip(todo, pool.map(_job, [(train, p) for p in todo])):
                memo[key_of(p)] = m
            return [memo[key_of(p)] for p in ps]

        bests = []
        for si, start in enumerate(STARTS):
            cur = dict(start)
            evaluate([cur])
            for rnd in range(3):
                changed = False
                for name, values in GRID.items():
                    cands = [dict(cur, **{name: v}) for v in values]
                    ms = evaluate(cands)
                    best_i = min(range(len(cands)), key=lambda i: rank(ms[i]))
                    if cands[best_i][name] != cur[name]:
                        changed = True
                    cur = cands[best_i]
                m = memo[key_of(cur)]
                print(f"start {si} round {rnd + 1}: caught {m['caught']}/{m['events']} lead {m['lead']:.2f} "
                      f"fa {100 * m['fa']:.1f}%  ({time.perf_counter() - t0:.0f} s)  {cur}", flush=True)
                if not changed:
                    break
            bests.append(cur)
    allr = [{**dict(k), **v} for k, v in memo.items()]
    allr.sort(key=rank)
    import pandas as pd
    pd.DataFrame(allr).to_csv(OUT / "search_all.csv", index=False)
    best = min(bests, key=lambda p: rank(memo[key_of(p)]))
    report = {"train": train, "test": test, "best": best, "train_metrics": memo[key_of(best)]}
    # проверка: лучшая настройка и исходная (по синтетике) на обучающих и тестовых видео
    base = STARTS[2]   # лучшая по синтетике (без маски горизонта)
    for name, p in (("best", best), ("synthetic_best", base)):
        report[f"{name}_per_video"] = {Path(v).stem: run(v, p) for v in train + test}
    (OUT / "best.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache"); c.add_argument("videos", nargs="+")
    s = sub.add_parser("search"); s.add_argument("--train", nargs="+", required=True)
    s.add_argument("--test", nargs="*", default=[]); s.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.cmd == "cache":
        for v in a.videos:
            for w in GRID["work_width"]:
                print(build_cache(Path(v), w), flush=True)
    else:
        search(a.train, a.test, a.workers)


if __name__ == "__main__":
    main()
