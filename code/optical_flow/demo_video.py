"""Демо-видео работы выбранного детектора на синтетических сценах.

    python optical_flow/demo_video.py [--scenes post_v2 box_left passby wall_yaw25] [--config best]

Берёт настройку из results/tuning/best.json ("best" или имя метода из "per_method"),
прогоняет детектор по кадрам сцен из кэша и рисует: тепловую карту риска, рамку
препятствия, стрелку манёвра, оценку TTC и истинное TTC, рамку правильного ответа.
Результат: results/demo/<сцена>.mp4 и общий results/demo/demo.gif (для слайдов).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.sources import make_writer                   # noqa: E402
from common.types import Frame                           # noqa: E402
from common.viz import draw_result                       # noqa: E402
from optical_flow.detector import FlowObstacleDetector   # noqa: E402
from optical_flow.scenes import load_cached, scenario_by_name  # noqa: E402
from optical_flow.tune import CACHE, OUT, POST_GRID, make_cfg   # noqa: E402

TITLES = {
    "post_v2": "столб на пути, 2 м/с",
    "box_left": "ящик слева: обход справа",
    "passby": "столб сбоку: тревоги нет",
    "wall_yaw25": "стена при рыскании 25 °/с",
    "post_v6": "столб на пути, 6 м/с",
    "post_noisy": "шум и смаз",
}


def load_config(tag: str) -> tuple[dict, dict]:
    info = json.loads((OUT / "best.json").read_text(encoding="utf-8"))
    cfg = info["best"] if tag == "best" else info["per_method"][tag]
    combo = {k: cfg[k] for k in ("backend", "work_width", "frame_step")}
    post = {k: cfg[k] for k in POST_GRID}
    return combo, post


def render_scene(name: str, combo: dict, post: dict, thr: float = 4.0) -> list[np.ndarray]:
    frames, ts, labels = load_cached(scenario_by_name(name), CACHE)
    det = FlowObstacleDetector(make_cfg(combo, post, thr))
    out = []
    for k in range(len(frames)):
        res = det.process(Frame(k, float(ts[k]), frames[k]))
        bgr = cv2.cvtColor(np.ascontiguousarray(frames[k]), cv2.COLOR_GRAY2BGR)
        lab = labels[k]
        true_ttc = f"true TTC {lab['ttc_s']:.1f} s" if lab["ttc_s"] is not None else "true TTC -"
        vis = draw_result(bgr, res, show_heat=True, extra=[true_ttc])
        if lab["present"]:   # правильный ответ: препятствие опасно
            cv2.rectangle(vis, (2, 2), (vis.shape[1] - 3, vis.shape[0] - 3), (0, 0, 255), 3)
        out.append(vis)
    return out


def put_title(img: np.ndarray, text: str) -> np.ndarray:
    """Подпись кириллицей через Pillow (cv2.putText кириллицу не рисует)."""
    from PIL import Image, ImageDraw, ImageFont
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    w = d.textlength(text, font=font)
    x, y = (img.shape[1] - w) / 2, img.shape[0] - 34
    d.rectangle((x - 8, y - 4, x + w + 8, y + 28), fill=(0, 0, 0))
    d.text((x, y), text, font=font, fill=(255, 255, 255))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="+", default=["post_v2", "box_left", "passby", "wall_yaw25"])
    ap.add_argument("--config", default="best")
    ap.add_argument("--gif-width", type=int, default=400)
    ap.add_argument("--gif-step", type=int, default=3)
    args = ap.parse_args()

    combo, post = load_config(args.config)
    out_dir = ROOT / "results" / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    gif_frames = []
    for name in args.scenes:
        frames = render_scene(name, combo, post)
        fps = scenario_by_name(name).fps
        wr = make_writer(out_dir / f"{name}.mp4", (frames[0].shape[1], frames[0].shape[0]), fps)
        for f in frames:
            wr.write(put_title(f, TITLES.get(name, name)))
        wr.release()
        # для GIF: каждый 2-й кадр, уменьшенный размер
        for f in frames[::args.gif_step]:
            g = put_title(f, TITLES.get(name, name))
            h = int(g.shape[0] * args.gif_width / g.shape[1])
            gif_frames.append(cv2.cvtColor(cv2.resize(g, (args.gif_width, h), interpolation=cv2.INTER_AREA),
                                           cv2.COLOR_BGR2RGB))
        print(f"{name}: {len(frames)} frames -> {out_dir / (name + '.mp4')}")
    from PIL import Image
    imgs = [Image.fromarray(f).convert("P", palette=Image.ADAPTIVE, colors=64) for f in gif_frames]
    imgs[0].save(out_dir / "demo.gif", save_all=True, append_images=imgs[1:], duration=66, loop=0, optimize=True)
    print(f"gif: {len(imgs)} frames -> {out_dir / 'demo.gif'} ({(out_dir / 'demo.gif').stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
