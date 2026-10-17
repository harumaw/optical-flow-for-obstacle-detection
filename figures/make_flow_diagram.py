"""Структурная схема детекции препятствий по оптическому потоку (в стиле рис. 1 курсовой).

    python figures/make_flow_diagram.py        -> figures/flow_pipeline.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).with_name("flow_pipeline.png")

PURPLE = ("#e6e0fb", "#7a5fd6")
BLUE = ("#dbe8fb", "#3b6fd6")
RED = ("#fadbd8", "#c0392b")
YELLOW = ("#fdebc8", "#d68910")
GREEN = ("#d5f5e3", "#1e8449")
GRAY = ("#e9e6e0", "#555555")
GRAY2 = ("#f3f0ea", "#555555")

W, H = 106, 166
X0, BW, BH = 20, 60, 8.6
STEP = 13.0

fig, ax = plt.subplots(figsize=(10.6, 16.6), dpi=200)
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
fig.patch.set_facecolor("white")


def box(y, title, sub, colors, x=X0, w=BW, h=BH, fs=12.5, fs_sub=10):
    fc, ec = colors
    ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle="round,pad=0.3,rounding_size=1.4",
                                fc=fc, ec=ec, lw=1.6))
    if sub:
        ax.text(x + w / 2, y + 1.7, title, ha="center", va="center", fontsize=fs, fontweight="bold")
        ax.text(x + w / 2, y - 1.9, sub, ha="center", va="center", fontsize=fs_sub,
                style="italic", color="#333333")
    else:
        ax.text(x + w / 2, y, title, ha="center", va="center", fontsize=fs, fontweight="bold")


def down_arrow(y1, y2, label, x=X0 + BW / 2):
    ax.annotate("", xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle="-|>", color="#444444", lw=1.4, shrinkA=0, shrinkB=0))
    ax.text(x + 1.2, (y1 + y2) / 2, label, fontsize=9.5, color="#555555", va="center")


blocks = [
    ("Фронтальная камера", None, PURPLE, "кадр BGR 640×480"),
    ("Предобработка", "полутон, уменьшение до ширины 320 px", BLUE, "gray 320×240"),
    ("Оптический поток", "Farneback  |  DIS  |  Лукас-Канаде на сетке", BLUE, "поле (u, v)  [px/кадр]"),
    ("Взвешенное сглаживание", r"$u_s = G*(w\,u)\;/\;G*w$   (нормированная свёртка)", BLUE,
     "сглаженное поле + карта достоверности"),
    ("Дивергенция поля", r"$\mathrm{div} = \partial u/\partial x + \partial v/\partial y$   (оператор Собеля)",
     YELLOW, "div  [1/кадр]"),
    ("Сетка ячеек 20×15", "усреднение по ячейкам, сглаживание во времени (EMA)", BLUE, "div по ячейкам"),
    ("Порог по времени до столкновения",
     r"$\mathrm{div} > 2\,\Delta t\,/\,\tau_{\max}$,  достоверность > 0.2,  камера движется", RED,
     "маска ячеек"),
    ("Выделение области", r"связные компоненты → bbox,   $\tau = 2\,\Delta t\,/\,\overline{\mathrm{div}}$",
     YELLOW, "bbox, TTC [с]"),
    ("Правило манёвра", "stop  |  left  |  right   (общее для всех методов)", GREEN, "манёвр"),
    ("Автопилот (GUIDED)  /  оператор", None, GRAY, "команда скорости"),
    ("Дрон меняет курс", None, GRAY2, None),
]

ys = []
y = H - 10
for i in range(len(blocks)):
    ys.append(y)
    y -= STEP + (6.0 if i == 0 else 4.0 if i == 8 else 0.0)   # доп. зазор под рамку
for (title, sub, colors, _), y in zip(blocks, ys):
    box(y, title, sub, colors)
for i in range(len(blocks) - 1):
    down_arrow(ys[i] - BH / 2, ys[i + 1] + BH / 2, blocks[i][3])

# --- рамка "бортовой компьютер" вокруг блоков обработки
y_top = ys[1] + BH / 2 + 4.2
y_bot = ys[8] - BH / 2 - 1.6
ax.add_patch(FancyBboxPatch((X0 - 4.5, y_bot), BW + 9, y_top - y_bot,
                            boxstyle="round,pad=0.3,rounding_size=2", fc="none", ec="#444444",
                            lw=1.3, ls="--"))
ax.text(X0 - 3.5, y_top - 1.0, "Бортовой компьютер (ноутбук на испытаниях, Orange Pi на борту)",
        fontsize=10, style="italic", color="#333333", va="top")

# --- боковой блок: вес достоверности по текстуре -> взвешенное сглаживание
sx, sw, sh = X0 + BW + 4.5, 19.5, 11
sy = ys[3] + 1.5
box(sy, "Достоверность\nпо текстуре", None, GREEN, x=sx, w=sw, h=sh, fs=10.5)
ax.text(sx + sw / 2, sy - 3.6, r"$w = \left(\frac{g}{g + t_0}\right)^2,\; g = |\nabla I|$",
        ha="center", va="center", fontsize=9.5, style="italic", color="#1e8449")
ax.annotate("", xy=(X0 + BW, ys[3]), xytext=(sx, sy),
            arrowprops=dict(arrowstyle="-|>", color="#1e8449", lw=1.4, shrinkA=0, shrinkB=0,
                            connectionstyle="arc3,rad=0"))
ax.annotate("", xy=(sx + sw / 2, sy + sh / 2), xytext=(X0 + BW, ys[1]),
            arrowprops=dict(arrowstyle="-|>", color="#1e8449", lw=1.2, shrinkA=0, shrinkB=0,
                            connectionstyle="angle,angleA=0,angleB=90,rad=4"))
ax.text(X0 + BW + 1.5, ys[1] + 1.3, "gray", fontsize=9, color="#1e8449")

# --- красная петля: нет движения -> кадр пропущен
xl = X0 - 2.5
ax.plot([X0, xl, xl, X0], [ys[6], ys[6], ys[2], ys[2]], ls="--", color="#c0392b", lw=1.3)
ax.annotate("", xy=(X0, ys[2]), xytext=(xl, ys[2]),
            arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=1.3, shrinkA=0, shrinkB=0))
ax.text(xl - 0.8, (ys[2] + ys[6]) / 2, "нет движения\nили текстуры:\nрешения нет,\nждём кадр",
        fontsize=8, color="#c0392b", ha="right", va="center")

# --- синяя обратная связь: дрон сместился -> камера видит другое
xf = 3.0
ax.plot([X0, xf, xf], [ys[-1], ys[-1], ys[0]], ls="--", color="#2e86de", lw=1.5)
ax.annotate("", xy=(X0, ys[0]), xytext=(xf, ys[0]),
            arrowprops=dict(arrowstyle="-|>", color="#2e86de", lw=1.5, shrinkA=0, shrinkB=0))
ax.text(xf - 1.5, (ys[0] + ys[-1]) / 2, "ОБРАТНАЯ  СВЯЗЬ", rotation=90, fontsize=9.5,
        color="#2e86de", ha="center", va="center", fontweight="bold")

ax.text(W / 2, H - 2.5, "Структурная схема детекции препятствий по оптическому потоку",
        ha="center", va="center", fontsize=14, fontweight="bold")

fig.savefig(OUT, bbox_inches="tight", pad_inches=0.3)
print(f"saved {OUT}")
