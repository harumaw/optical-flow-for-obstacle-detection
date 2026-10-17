"""Графики по результатам run_bench.py для диплома.

    python plots.py results/wall_approach_01 [--labels data/raw/wall_approach_01.labels.csv]

Создаёт в results/<имя>/figs/:
    latency.png      среднее время на кадр по методам и стадиям
    timeline.png     наличие препятствия по кадрам: разметка и каждый метод
    ttc.png          время до столкновения по кадрам (если есть в разметке или у методов)
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.labels import read_labels
from common.util import safe_name


def load_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir")
    ap.add_argument("--labels", type=str, default=None)
    args = ap.parse_args()

    rd = Path(args.results_dir)
    figs = rd / "figs"
    figs.mkdir(exist_ok=True)
    summary = json.loads((rd / "summary.json").read_text(encoding="utf-8"))
    methods = list(summary["methods"])
    rows = {m: load_rows(rd / f"{safe_name(m)}.csv") for m in methods}
    labels = read_labels(args.labels) if args.labels else None

    # --- время на кадр по стадиям
    stages = ["pre", "flow", "infer", "post"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bottoms = np.zeros(len(methods))
    for st in stages:
        vals = np.array([summary["methods"][m].get(f"timing_{st}", {}).get("mean_ms", 0) or 0 for m in methods])
        if vals.sum() == 0:
            continue
        ax.bar(methods, vals, bottom=bottoms, label=st)
        bottoms += vals
    p95 = [summary["methods"][m].get("timing_total", {}).get("p95_ms", 0) for m in methods]
    ax.plot(methods, p95, "k_", markersize=25, label="p95 total")
    ax.set_ylabel("мс на кадр")
    ax.set_title("Время обработки кадра")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs / "latency.png", dpi=150)

    # --- таймлайн наличия препятствия
    fig, ax = plt.subplots(figsize=(9, 0.6 * (len(methods) + 1) + 1))
    yt, yl = [], []
    y = 0
    if labels:
        fr = sorted(labels)
        pres = np.array([labels[f]["present"] for f in fr], dtype=float)
        ax.fill_between(fr, y, y + pres * 0.8, step="mid", color="gray", alpha=0.6)
        yt.append(y + 0.4); yl.append("разметка")
        y += 1
    for m in methods:
        fr = [int(r["frame"]) for r in rows[m]]
        pres = np.array([int(r["present"]) for r in rows[m]], dtype=float)
        ax.fill_between(fr, y, y + pres * 0.8, step="mid", alpha=0.8)
        yt.append(y + 0.4); yl.append(m)
        y += 1
    ax.set_yticks(yt); ax.set_yticklabels(yl)
    ax.set_xlabel("кадр")
    ax.set_title("Наличие препятствия по кадрам")
    fig.tight_layout()
    fig.savefig(figs / "timeline.png", dpi=150)

    # --- TTC
    fig, ax = plt.subplots(figsize=(9, 3.6))
    drawn = False
    if labels:
        fr = sorted(f for f in labels if labels[f].get("ttc_s") is not None)
        if fr:
            ax.plot(fr, [labels[f]["ttc_s"] for f in fr], "k--", label="TTC по разметке")
            drawn = True
    for m in methods:
        pts = [(int(r["frame"]), float(r["proximity"])) for r in rows[m]
               if r.get("proximity_kind") == "ttc_s" and r.get("proximity")]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], ".", markersize=3, label=m)
            drawn = True
    if drawn:
        ax.set_ylim(0, 10)
        ax.set_xlabel("кадр"); ax.set_ylabel("TTC, с")
        ax.set_title("Оценка времени до столкновения")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figs / "ttc.png", dpi=150)
    print(f"figures saved to {figs}")


if __name__ == "__main__":
    main()
