"""Отрисовка результата на кадре: область, стрелка манёвра, тепловая карта риска, поле потока."""
from __future__ import annotations

import math

import cv2
import numpy as np

from .types import ObstacleResult

SIDE_COLORS = {
    "none": (90, 200, 90),
    "left": (0, 200, 255),
    "right": (0, 200, 255),
    "stop": (0, 0, 255),
}


def heatmap_overlay(img: np.ndarray, risk: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """risk: дивергенция / порог по ячейкам. 1.0 = граница срабатывания."""
    h, w = img.shape[:2]
    norm = np.clip(risk / 2.0, 0.0, 1.0)              # 0..2 порога -> 0..1
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.resize(heat, (w, h), interpolation=cv2.INTER_NEAREST)
    return cv2.addWeighted(img, 1.0 - alpha, heat, alpha, 0)


def draw_flow(img: np.ndarray, flow: np.ndarray, step: int = 16, scale: float = 3.0) -> np.ndarray:
    h, w = img.shape[:2]
    fh, fw = flow.shape[:2]
    sx, sy = w / fw, h / fh
    for y in range(step // 2, fh, step):
        for x in range(step // 2, fw, step):
            u, v = flow[y, x]
            p0 = (int(x * sx), int(y * sy))
            p1 = (int((x + u * scale) * sx), int((y + v * scale) * sy))
            cv2.arrowedLine(img, p0, p1, (255, 255, 0), 1, tipLength=0.3)
    return img


def draw_side_arrow(img: np.ndarray, side: str) -> None:
    h, w = img.shape[:2]
    cx, cy = w // 2, h - 40
    color = SIDE_COLORS.get(side, (255, 255, 255))
    if side == "left":
        cv2.arrowedLine(img, (cx + 60, cy), (cx - 60, cy), color, 4, tipLength=0.4)
    elif side == "right":
        cv2.arrowedLine(img, (cx - 60, cy), (cx + 60, cy), color, 4, tipLength=0.4)
    elif side == "stop":
        cv2.circle(img, (cx, cy), 22, color, 4)
        cv2.line(img, (cx - 14, cy - 14), (cx + 14, cy + 14), color, 4)


def draw_result(img: np.ndarray, res: ObstacleResult, show_heat: bool = True,
                show_flow: bool = False, extra: list[str] | None = None) -> np.ndarray:
    out = img
    if show_heat and "risk" in res.debug:
        out = heatmap_overlay(out, res.debug["risk"])
    if show_flow and "flow" in res.debug:
        out = draw_flow(out, res.debug["flow"])
    color = SIDE_COLORS.get(res.side, (255, 255, 255))
    if res.bbox is not None:
        x1, y1, x2, y2 = res.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    draw_side_arrow(out, res.side)

    lines = [res.method]
    if res.proximity is not None and math.isfinite(res.proximity):
        unit = {"ttc_s": "TTC %.2f s", "dist_m": "dist %.2f m", "size_frac": "size %.2f"}
        lines.append(unit.get(res.proximity_kind, "%.2f") % res.proximity)
    if "ttc_center" in res.debug and math.isfinite(res.debug["ttc_center"]):
        lines.append("center TTC %.1f s" % res.debug["ttc_center"])
    if res.timing_ms:
        parts = [f"{k} {v:.1f}" for k, v in res.timing_ms.items() if k != "total"]
        lines.append(f"{res.timing_ms.get('total', 0):.1f} ms  (" + ", ".join(parts) + ")")
    if extra:
        lines.extend(extra)
    y = 22
    for line in lines:
        cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22
    return out
