"""Графики по результатам tune.py для диплома.

    python optical_flow/tune_plots.py        -> results/tuning/figs/*.png

    stage1_methods.png   этап 1: пропуски, запас времени и ложные тревоги по методам и разрешению
    speed.png            лучшая настройка каждого метода: запас времени и дистанция против скорости
    thresholds.png       лучшая настройка: компромисс пропусков и ложных тревог при порогах 2-5 с
    timing.png           время на кадр: все потоки и один поток (модель Orange Pi)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "tuning"
FIGS = OUT / "figs"


def stage1_methods() -> None:
    s = pd.read_csv(OUT / "stage1_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, col, title in zip(axes, ["miss_rate", "warning_ttc", "fa_rate"],
                              ["доля пропущенных препятствий", "запас времени, с", "доля ложных тревог"]):
        for step, ls in ((1, "-"), (2, "--")):
            for b in s["backend"].unique():
                d = s[(s.backend == b) & (s.frame_step == step)].sort_values("work_width")
                ax.plot(d.work_width, d[col], ls, marker="o", label=f"{b}, шаг {step}" if ax is axes[0] else None)
        ax.set_xlabel("рабочая ширина кадра, px")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[2].axhline(0.02, color="k", lw=0.8, ls=":")
    fig.legend(loc="lower center", ncol=7, fontsize=7)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(FIGS / "stage1_methods.png", dpi=150)


METHOD_NAMES = {"best_farneback": "Farneback, 160 px", "best_lk12": "Лукас-Канаде, сетка 12, 320 px",
                "best_lk16": "Лукас-Канаде, сетка 16, 160 px", "best_dis-ultrafast": "DIS ultrafast, 320 px"}
COLORS = {"best_farneback": "#c0392b", "best_lk12": "#2471a3", "best_lk16": "#7fb3d5", "best_dis-ultrafast": "#7d8b8c"}


def speed() -> None:
    """Столб (трудный случай) против скорости; стены у всех методов дают 5.6-6 с и не показаны."""
    r = pd.read_csv(OUT / "final_runs.csv")
    r = r[(r.thr == 4.0) & r.tags.fillna("").str.contains("speed") & r.scene.str.startswith("post") & (r.tag != "best")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for tag in METHOD_NAMES:
        d = r[r.tag == tag].sort_values("speed")
        if d.empty:
            continue
        lw = 2.6 if tag == "best_farneback" else 1.4
        axes[0].plot(d.speed, d.warning_ttc.fillna(0), marker="o", lw=lw, color=COLORS[tag], label=METHOD_NAMES[tag])
        axes[1].plot(d.speed, d.warning_dist_m.fillna(0), marker="o", lw=lw, color=COLORS[tag])
    axes[0].set_ylabel("TTC при первой тревоге, с")
    axes[1].set_ylabel("дистанция при первой тревоге, м")
    axes[0].axhline(4.0, color="k", lw=0.8, ls="--")
    axes[0].text(0.55, 4.08, "порог 4 с", fontsize=8)
    axes[0].axhline(1.0, color="k", lw=0.8, ls=":")
    axes[0].text(0.55, 1.08, "поздно для манёвра", fontsize=8)
    axes[0].set_ylim(0, 6.5)
    for ax in axes:
        ax.set_xlabel("скорость, м/с")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Столб на пути: когда звучит первая тревога", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "speed.png", dpi=150)


def thresholds() -> None:
    s = pd.read_csv(OUT / "final_summary.csv")
    s = s[s.tag != "best"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for tag in METHOD_NAMES:
        d = s[s.tag == tag].sort_values("thr")
        if d.empty:
            continue
        lw = 2.6 if tag == "best_farneback" else 1.4
        axes[0].plot(d.thr, 100 * d.fa_rate, marker="o", lw=lw, color=COLORS[tag], label=METHOD_NAMES[tag])
        axes[1].plot(d.thr, d.warning_capped, marker="o", lw=lw, color=COLORS[tag])
    axes[0].axhline(2.0, color="k", lw=0.8, ls=":")
    axes[0].text(2.05, 2.2, "допустимо 2 %", fontsize=8)
    axes[0].set_ylabel("ложные тревоги, % кадров")
    axes[1].set_ylabel("запас времени, с (не больше порога)")
    axes[1].plot([2, 5], [2, 5], color="k", lw=0.8, ls="--")
    axes[1].text(4.2, 4.5, "идеал", fontsize=8)
    for ax in axes:
        ax.set_xlabel("порог времени до столкновения, с")
        ax.set_xticks([2, 3, 4, 5])
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Выбор порога TTC: пропусков нет ни при одном пороге", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "thresholds.png", dpi=150)


def timing() -> None:
    t = pd.read_csv(OUT / "timing.csv")
    t = t[t.frame_step == 1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, mode in zip(axes, ["all threads", "1 thread"]):
        d = t[t["mode"] == mode]
        for b in d.backend.unique():
            dd = d[d.backend == b].sort_values("work_width")
            ax.plot(dd.work_width, dd.total_ms, marker="o", label=b)
        ax.axhline(50, color="k", lw=0.8, ls=":")
        ax.text(162, 52, "20 кадров/с", fontsize=7)
        ax.set_title("все потоки" if mode == "all threads" else "один поток")
        ax.set_xlabel("рабочая ширина кадра, px")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("мс на кадр")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / "timing.png", dpi=150)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    for fn, need in ((stage1_methods, "stage1_summary.csv"), (speed, "final_runs.csv"),
                     (thresholds, "final_summary.csv"), (timing, "timing.csv")):
        if (OUT / need).exists():
            fn()
            print(f"{fn.__name__}: ok")
    print(f"figures -> {FIGS}")


if __name__ == "__main__":
    main()
