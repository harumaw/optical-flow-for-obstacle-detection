"""Подбор лучшего метода и параметров детектора по оптическому потоку на синтетических сценах.

    python optical_flow/tune.py stage1      # 42 варианта вычисления потока, параметры обработки по умолчанию
    python optical_flow/tune.py stage2      # сетка параметров обработки для 4 лучших вариантов этапа 1
    python optical_flow/tune.py final       # лучшая настройка: пороги TTC 2/3/4/5 с, таблица по сценам
    python optical_flow/tune.py timing      # время на кадр: обычный режим и один поток процессора
    python optical_flow/tune.py all

Сцены берутся из data/scenes_cache (см. render_cache.py). Результаты пишутся в results/tuning/.

Метрики (порог TTC = thr, одинаковый для разметки и детектора):
    gt_present  препятствие на пути и TTC < thr
    safe кадр   препятствия на пути нет или TTC > thr + 2 с; тревога на нём ложная
    промежуток  thr < TTC <= thr + 2 с на пути: ранняя тревога не штрафуется
    missed      сцена с препятствием, где не было тревоги до того, как TTC упало ниже 1 с
    warning     истинное TTC в момент первой тревоги (запас времени), с
    recall      доля кадров gt_present с тревогой
    fa_rate     доля safe кадров с тревогой по всем сценам

Критерий выбора (ответ пользователя: «не пропустить»): fa_rate <= 2 %, затем
минимум missed, затем максимум warning, затем recall, затем меньшее время.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.decision import decide                                   # noqa: E402
from common.types import Frame                                       # noqa: E402
from optical_flow.detector import FlowConfig, FlowObstacleDetector   # noqa: E402
from optical_flow.scenes import W, default_suite, load_cached, scenario_by_name  # noqa: E402

CACHE = ROOT / "data" / "scenes_cache"
OUT = ROOT / "results" / "tuning"
FA_LIMIT = 0.02
SAFE_MARGIN = 2.0
LATE_TTC = 1.0

# ------------------------------------------------------------------ варианты вычисления потока
BACKENDS = {
    "lk16": ("lk", {"cell": 16}),
    "lk12": ("lk", {"cell": 12}),
    "lk24": ("lk", {"cell": 24}),
    "dis-ultrafast": ("dis", {"preset": "ultrafast"}),
    "dis-fast": ("dis", {"preset": "fast"}),
    "dis-medium": ("dis", {"preset": "medium"}),
    "farneback": ("farneback", {}),
}
WIDTHS = (160, 240, 320)
STEPS = (1, 2)


def flow_combos() -> list[dict]:
    return [{"backend": b, "work_width": w, "frame_step": s}
            for b in BACKENDS for w in WIDTHS for s in STEPS]


def combo_name(c: dict) -> str:
    return f"{c['backend']}@{c['work_width']}/s{c['frame_step']}"


# ------------------------------------------------------------------ параметры обработки
DEFAULT_POST = {"flow_sigma": 5.0, "ema_alpha": 0.4, "min_cells": 4, "path_frac": 0.25,
                "derotate": True, "min_persist": 1}
POST_GRID = {
    "flow_sigma": [3.0, 5.0, 8.0],      # в пикселях при ширине 320, масштабируется с шириной
    "ema_alpha": [0.25, 0.4, 0.7],
    "min_cells": [2, 4, 8],
    "path_frac": [1.0, 0.25, 0.15],     # 1.0 = без проверки курса столкновения
    "derotate": [True, False],
    "min_persist": [1, 2],
}


def post_grid() -> list[dict]:
    keys = list(POST_GRID)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(POST_GRID[k] for k in keys))]


def make_cfg(combo: dict, post: dict, thr: float) -> FlowConfig:
    bname, bkw = BACKENDS[combo["backend"]]
    p = dict(post)
    p["flow_sigma"] = p["flow_sigma"] * combo["work_width"] / 320.0
    return FlowConfig(backend=bname, backend_kwargs=dict(bkw), work_width=combo["work_width"],
                      frame_step=combo["frame_step"], ttc_threshold_s=thr, **p)


# ------------------------------------------------------------------ запись и воспроизведение потока
class Recorder:
    def __init__(self, backend):
        self.inner, self.name, self.flows, self.ms = backend, backend.name, [], []

    def compute(self, prev, cur):
        t0 = time.perf_counter()
        f = self.inner.compute(prev, cur)
        self.ms.append((time.perf_counter() - t0) * 1000)
        self.flows.append(f)
        return f


class Replay:
    def __init__(self, name, flows):
        self.name, self.flows, self.i = name, flows, 0

    def compute(self, prev, cur):
        f = self.flows[self.i]
        self.i += 1
        return f


def run_detector(det: FlowObstacleDetector, frames: np.ndarray, ts: np.ndarray) -> tuple[list, float]:
    out, post_ms = [], []
    for k in range(len(frames)):
        r = det.process(Frame(k, float(ts[k]), frames[k]))
        out.append((r.present, r.proximity, r.side))
        if "post" in r.timing_ms:
            post_ms.append(r.timing_ms["post"])
    return out, float(np.mean(post_ms)) if post_ms else 0.0


# ------------------------------------------------------------------ метрики одной сцены
def evaluate(preds: list, labels: list[dict], thr: float) -> dict:
    n = len(labels)
    safe = fa = 0
    pres_idx, hit = [], 0
    first_alarm_ttc = None
    late_ok = False
    side_ok = side_n = 0
    rel_err = []
    for k in range(n):
        lab = labels[k]
        alarm, est, side = preds[k]
        ttc, on_path = lab.get("ttc_s"), lab.get("on_path", False)
        if not on_path or ttc is None or ttc > thr + SAFE_MARGIN:
            safe += 1
            fa += int(alarm)
            continue
        if alarm and first_alarm_ttc is None:
            first_alarm_ttc = ttc
        if alarm and ttc >= LATE_TTC:
            late_ok = True
        if ttc < thr:
            pres_idx.append(k)
            if alarm:
                hit += 1
                if lab.get("box"):
                    # сторона однозначна только для «стоп» или для предмета, заметно смещённого от центра;
                    # у столба точно по центру правильной стороны обхода нет
                    box = tuple(lab["box"])
                    gt_side = decide(box, W)
                    if gt_side == "stop" or abs(0.5 * (box[0] + box[2]) - W / 2) > 0.1 * W:
                        side_n += 1
                        side_ok += int(side == gt_side)
                if est is not None and math.isfinite(est):
                    rel_err.append(abs(est - ttc) / ttc)
    positive = len(pres_idx) > 0
    return {
        "positive": positive,
        "safe_frames": safe, "fa_frames": fa,
        "missed": (not late_ok) if positive else None,
        "warning_ttc": first_alarm_ttc if positive else None,
        "recall": hit / len(pres_idx) if positive else None,
        "side_acc": side_ok / side_n if side_n else None,
        "ttc_rel_err": float(np.median(rel_err)) if rel_err else None,
    }


def aggregate(rows: list[dict], thr: float = 4.0) -> dict:
    pos = [r for r in rows if r["positive"]]
    safe = sum(r["safe_frames"] for r in rows)
    fa = sum(r["fa_frames"] for r in rows)
    hits = [r["warning_ttc"] for r in pos if r["warning_ttc"] is not None]
    mean = lambda xs: float(np.mean(xs)) if xs else None
    return {
        "fa_rate": fa / safe if safe else 0.0,
        "miss_rate": mean([float(r["missed"]) for r in pos]),
        "warning_ttc": mean([r["warning_ttc"] if r["warning_ttc"] is not None else 0.0 for r in pos]),
        "warning_hits": mean(hits),
        # запас времени, ограниченный порогом: тревога раньше порога не даёт дополнительного балла
        "warning_capped": mean([min(r["warning_ttc"], thr) if r["warning_ttc"] is not None else 0.0 for r in pos]),
        "recall": mean([r["recall"] for r in pos]),
        "side_acc": mean([r["side_acc"] for r in pos if r["side_acc"] is not None]),
        "ttc_rel_err": mean([r["ttc_rel_err"] for r in pos if r["ttc_rel_err"] is not None]),
        "scenes": len(rows),
    }


WARN_STEP = 0.1      # с: разница в запасе меньше ~3 кадров считается ничьей
RECALL_STEP = 0.02


def rank_key(a: dict, ms: float = 0.0):
    """Порядок сравнения: ложные тревоги в пределах 2 %, пропуски, запас времени
    (с точностью WARN_STEP), полнота (с точностью RECALL_STEP), затем меньше
    ложных тревог, затем быстрее. Различия меньше точности измерения не решают исход."""
    return (max(a["fa_rate"] - FA_LIMIT, 0.0), a["miss_rate"] or 0.0,
            -math.floor((a["warning_capped"] or 0.0) / WARN_STEP + 1e-9),
            -math.floor((a["recall"] or 0.0) / RECALL_STEP + 1e-9),
            round(a["fa_rate"], 3), ms)


# ------------------------------------------------------------------ задачи для процессов
def _init_worker():
    import cv2
    cv2.setNumThreads(1)        # процессы и так параллельны, внутренние потоки OpenCV только мешают


def _task_stage1(args):
    combo, scene, thr = args
    frames, ts, labels = load_cached(scenario_by_name(scene), CACHE)
    det = FlowObstacleDetector(make_cfg(combo, DEFAULT_POST, thr))
    rec = Recorder(det.backend)
    det.backend = rec
    preds, post_ms = run_detector(det, frames, ts)
    m = evaluate(preds, labels, thr)
    m.update({"combo": combo_name(combo), "scene": scene, "flow_ms": float(np.mean(rec.ms)), "post_ms": post_ms})
    return m


def _task_stage2(args):
    combo, scene, thr, posts = args
    frames, ts, labels = load_cached(scenario_by_name(scene), CACHE)
    det = FlowObstacleDetector(make_cfg(combo, DEFAULT_POST, thr))
    rec = Recorder(det.backend)
    det.backend = rec
    run_detector(det, frames, ts)
    out = []
    for pi, post in enumerate(posts):
        d = FlowObstacleDetector(make_cfg(combo, post, thr))
        d.backend = Replay(rec.name, rec.flows)
        preds, post_ms = run_detector(d, frames, ts)
        m = evaluate(preds, labels, thr)
        m.update({"combo": combo_name(combo), "scene": scene, "post_id": pi, "post_ms": post_ms,
                  "flow_ms": float(np.mean(rec.ms))})
        out.append(m)
    return out


# ------------------------------------------------------------------ этапы
def scenes() -> list[str]:
    return [s.name for s in default_suite()]


def save_csv(path: Path, rows: list[dict]) -> None:
    import pandas as pd
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def stage1(workers: int, thr: float = 4.0) -> list[dict]:
    tasks = [(c, s, thr) for c in flow_combos() for s in scenes()]
    print(f"stage1: {len(flow_combos())} flow variants x {len(scenes())} scenes = {len(tasks)} runs")
    rows, t0 = [], time.perf_counter()
    with Pool(workers, initializer=_init_worker) as pool:
        for i, r in enumerate(pool.imap_unordered(_task_stage1, tasks, chunksize=2)):
            rows.append(r)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(tasks)}  {time.perf_counter() - t0:.0f} s", flush=True)
    save_csv(OUT / "stage1_runs.csv", rows)
    return summarize_stage1(rows)


def summarize_stage1(rows: list[dict]) -> list[dict]:
    summary = []
    for c in flow_combos():
        name = combo_name(c)
        rr = [r for r in rows if r["combo"] == name]
        a = aggregate(rr)
        a.update({"combo": name, **c, "flow_ms_1thread": float(np.mean([r["flow_ms"] for r in rr])),
                  "post_ms_1thread": float(np.mean([r["post_ms"] for r in rr]))})
        summary.append(a)
    summary.sort(key=lambda a: rank_key(a, a["flow_ms_1thread"] + a["post_ms_1thread"]))
    save_csv(OUT / "stage1_summary.csv", summary)
    print_table(summary[:15], "stage1, top 15")
    return summary


def pick_combos(top: int = 4) -> list[dict]:
    """По одному лучшему варианту каждого подхода (Лукас-Канаде, DIS, Farneback)
    плюс самый быстрый среди первых десяти этапа 1: кандидат для слабого процессора."""
    import pandas as pd
    s1 = pd.read_csv(OUT / "stage1_summary.csv")
    family = lambda b: "lk" if b.startswith("lk") else ("dis" if b.startswith("dis") else b)
    as_combo = lambda r: {"backend": r["backend"], "work_width": int(r["work_width"]), "frame_step": int(r["frame_step"])}
    picked, seen = [], set()
    for _, r in s1.iterrows():
        if family(r["backend"]) not in seen:
            seen.add(family(r["backend"]))
            picked.append(as_combo(r))
    top10 = s1.head(10).copy()
    top10["ms"] = top10["flow_ms_1thread"] + top10["post_ms_1thread"]
    fastest = as_combo(top10.sort_values("ms").iloc[0])
    if fastest not in picked:
        picked.append(fastest)
    return picked[:top]


def stage2(workers: int, rounds: int = 2, thr: float = 4.0) -> dict:
    """Покоординатный поиск параметров обработки для каждого варианта потока.

    Старт с DEFAULT_POST. На каждом шаге один параметр перебирается по POST_GRID при
    фиксированных остальных, лучшее значение (по rank_key) закрепляется. Два прохода по
    всем параметрам учитывают их взаимное влияние. Уже проверенные настройки не пересчитываются.
    """
    import pandas as pd
    picked = pick_combos()
    s1 = pd.read_csv(OUT / "stage1_summary.csv")
    s1_ms = {r["combo"]: r["flow_ms_1thread"] for _, r in s1.iterrows()}
    print("stage2 (coordinate search) for:", ", ".join(combo_name(c) for c in picked))
    current = {combo_name(c): dict(DEFAULT_POST) for c in picked}
    memo: dict[tuple, dict] = {}           # (combo, настройка) -> сводка
    key = lambda name, post: (name, tuple(sorted(post.items())))
    t0 = time.perf_counter()
    with Pool(workers, initializer=_init_worker) as pool:
        for rnd in range(rounds):
            for param, values in POST_GRID.items():
                todo = {}                  # имя варианта -> список новых настроек
                for c in picked:
                    name = combo_name(c)
                    for v in values:
                        post = dict(current[name]); post[param] = v
                        if key(name, post) not in memo and post not in todo.setdefault(name, []):
                            todo[name].append(post)
                tasks = [(c, sc, thr, todo[combo_name(c)]) for c in picked if todo.get(combo_name(c))
                         for sc in scenes()]
                rows = []
                for rr in pool.imap_unordered(_task_stage2, tasks):
                    rows.extend(rr)
                for c in picked:
                    name = combo_name(c)
                    for pi, post in enumerate(todo.get(name, [])):
                        rr = [r for r in rows if r["combo"] == name and r["post_id"] == pi]
                        agg = aggregate(rr, thr)
                        agg.update({"combo": name, **c, **post,
                                    "ms_1thread": s1_ms[name] + float(np.mean([r["post_ms"] for r in rr]))})
                        memo[key(name, post)] = agg
                    cands = []
                    for v in values:
                        post = dict(current[name]); post[param] = v
                        cands.append((rank_key(memo[key(name, post)], memo[key(name, post)]["ms_1thread"]), v))
                    current[name][param] = min(cands, key=lambda x: x[0])[1]
                print(f"  round {rnd + 1}, {param:<12} -> " +
                      "; ".join(f"{n.split('@')[0]}={current[n][param]}" for n in current) +
                      f"   ({time.perf_counter() - t0:.0f} s)", flush=True)
    summary = sorted(memo.values(), key=lambda a: rank_key(a, a["ms_1thread"]))
    for i, a in enumerate(summary):
        a["post_id"] = i
    save_csv(OUT / "stage2_summary.csv", summary)
    print_table(summary[:15], "stage2, top 15")
    keys = ["backend", "work_width", "frame_step"] + list(POST_GRID)
    clean = lambda a: {k: (a[k].item() if hasattr(a[k], "item") else a[k]) for k in keys}
    per_method = {}
    for c in picked:
        name = combo_name(c)
        per_method[c["backend"]] = clean({**c, **current[name]})
    best = clean(summary[0])
    (OUT / "best.json").write_text(json.dumps({"best": best, "per_method": per_method}, indent=2,
                                              ensure_ascii=False), encoding="utf-8")
    print("best:", best)
    return best


def reselect() -> dict:
    """Пересчитать выбор лучшей настройки по сохранённым результатам этапа 2."""
    import pandas as pd
    s2 = pd.read_csv(OUT / "stage2_summary.csv").to_dict("records")
    s2.sort(key=lambda a: rank_key(a, a["ms_1thread"]))
    for i, a in enumerate(s2):
        a["post_id"] = i
    save_csv(OUT / "stage2_summary.csv", s2)
    keys = ["backend", "work_width", "frame_step"] + list(POST_GRID)
    clean = lambda a: {k: (a[k].item() if hasattr(a[k], "item") else a[k]) for k in keys}
    per_method = {}
    for a in s2:
        per_method.setdefault(a["backend"], clean(a))
    best = clean(s2[0])
    (OUT / "best.json").write_text(json.dumps({"best": best, "per_method": per_method}, indent=2,
                                              ensure_ascii=False), encoding="utf-8")
    print_table(s2[:10], "stage2 re-ranked, top 10")
    print("best:", best)
    return best


def final(workers: int) -> None:
    info = json.loads((OUT / "best.json").read_text(encoding="utf-8"))
    rows = []
    configs = {"best": info["best"], **{f"best_{m}": c for m, c in info["per_method"].items()}}
    tasks = []
    for tag, cfg in configs.items():
        combo = {k: cfg[k] for k in ("backend", "work_width", "frame_step")}
        post = {k: cfg[k] for k in POST_GRID}
        for thr in (2.0, 3.0, 4.0, 5.0):
            for s in scenes():
                tasks.append((tag, combo, post, s, thr))
    with Pool(workers, initializer=_init_worker) as pool:
        for r in pool.imap_unordered(_task_final, tasks, chunksize=2):
            rows.append(r)
    save_csv(OUT / "final_runs.csv", rows)
    summary = []
    for tag in configs:
        for thr in (2.0, 3.0, 4.0, 5.0):
            rr = [r for r in rows if r["tag"] == tag and r["thr"] == thr]
            a = aggregate(rr, thr)
            a.update({"tag": tag, "thr": thr, "combo": rr[0]["combo"]})
            summary.append(a)
    save_csv(OUT / "final_summary.csv", summary)
    print_table([a for a in summary], "final: thresholds")


def _task_final(args):
    tag, combo, post, scene, thr = args
    frames, ts, labels = load_cached(scenario_by_name(scene), CACHE)
    det = FlowObstacleDetector(make_cfg(combo, post, thr))
    preds, _ = run_detector(det, frames, ts)
    m = evaluate(preds, labels, thr)
    sc = scenario_by_name(scene)
    m.update({"tag": tag, "combo": combo_name(combo), "scene": scene, "thr": thr, "speed": sc.speed,
              "tags": ",".join(sc.tags),
              "warning_dist_m": m["warning_ttc"] * sc.speed if m["warning_ttc"] is not None else None})
    return m


def timing() -> None:
    """Время на кадр без параллельных процессов: обычный режим и один поток OpenCV."""
    import cv2
    frames, ts, labels = load_cached(scenario_by_name("post_v2"), CACHE)
    frames, ts = frames[:120], ts[:120]
    rows = []
    for threads in (0, 1):
        cv2.setNumThreads(threads if threads else -1)
        if threads == 0:
            cv2.setNumThreads(cv2.getNumberOfCPUs())
        for c in flow_combos():
            det = FlowObstacleDetector(make_cfg(c, DEFAULT_POST, 4.0))
            tot, flo = [], []
            for k in range(len(frames)):
                r = det.process(Frame(k, float(ts[k]), frames[k]))
                if k >= 2:
                    tot.append(r.timing_ms["total"])
                    flo.append(r.timing_ms["flow"])
            rows.append({"combo": combo_name(c), **c, "mode": "1 thread" if threads == 1 else "all threads",
                         "total_ms": float(np.mean(tot)), "total_p95_ms": float(np.percentile(tot, 95)),
                         "flow_ms": float(np.mean(flo))})
        print(f"timing done: {'1 thread' if threads == 1 else 'all threads'}")
    save_csv(OUT / "timing.csv", rows)


def print_table(rows: list[dict], title: str) -> None:
    print(f"\n== {title}")
    print(f"{'config':<30}{'FA%':>6}{'miss%':>7}{'warn<=thr':>10}{'recall':>8}{'side':>6}{'ttcErr':>8}{'ms':>7}")
    for a in rows:
        name = a.get("combo", "") + (f" #{a['post_id']}" if "post_id" in a else "") + (f" thr{a['thr']:g}" if "thr" in a else "")
        ms = a.get("ms_1thread", a.get("flow_ms_1thread", 0) + a.get("post_ms_1thread", 0))
        f = lambda v, d=2: "-" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{d}f}"
        print(f"{name:<30}{100 * a['fa_rate']:>6.1f}{100 * (a['miss_rate'] or 0):>7.1f}{f(a.get('warning_capped')):>10}"
              f"{f(a['recall']):>8}{f(a['side_acc']):>6}{f(a['ttc_rel_err']):>8}{ms:>7.1f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["stage1", "stage2", "reselect", "final", "timing", "all"])
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage in ("stage1", "all"):
        stage1(args.workers)
    if args.stage in ("stage2", "all"):
        stage2(args.workers)
    if args.stage == "reselect":
        reselect()
    if args.stage in ("final", "all"):
        final(args.workers)
    if args.stage in ("timing", "all"):
        timing()


if __name__ == "__main__":
    main()
