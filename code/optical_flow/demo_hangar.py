"""Демо-видео детектора на полётах в ангаре UZH-FPV с выбранной настройкой.

    python optical_flow/demo_hangar.py [--preset hangar_gyro] [--flights indoor_forward_6 indoor_forward_10]

Кадры показаны только со стандартной отрисовкой детектора (common/viz.draw_result): тепловая карта
дивергенции относительно порога, рамка найденного препятствия с TTC, стрелка манёвра, время на кадр.
Никаких дополнительных надписей и правок. Результат: results/demo/hangar_<preset>.mp4
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.sources import VideoSource, make_writer           # noqa: E402
from common.types import Frame                                 # noqa: E402
from common.uzh import INDOOR_CAM0 as C, camera_orientation, gravity_in_camera  # noqa: E402
from common.viz import draw_result                             # noqa: E402
from optical_flow.detector import FlowObstacleDetector         # noqa: E402
from optical_flow.tune_real import make_cfg                    # noqa: E402

FONT = None


def text(img, s, xy, size=22, color=(255, 255, 255), bg=(0, 0, 0)):
    """Подпись кириллицей через Pillow."""
    global FONT
    from PIL import Image, ImageDraw, ImageFont
    if FONT is None or FONT.size != size:
        try:
            FONT = ImageFont.truetype("arial.ttf", size)
        except OSError:
            FONT = ImageFont.load_default()
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    w = d.textlength(s, font=FONT)
    x, y = xy
    if bg is not None:
        d.rectangle((x - 6, y - 3, x + w + 6, y + size + 5), fill=bg[::-1])
    d.text((x, y), s, font=FONT, fill=color[::-1])
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def render(flight: str, cfg, wr, title: str, args_hide_floor: bool = False) -> dict:
    video = ROOT / "data" / "uzh_fpv" / f"{flight}.mp4"
    lab = json.loads(video.with_name(flight + ".events.json").read_text(encoding="utf-8"))
    src = VideoSource(video)
    ts = np.array(src.timestamps)
    G = gravity_in_camera(video.with_suffix(""), ts)
    R = camera_orientation(video.with_suffix(""), ts)
    flying = ~np.isnan(G[:, 0])
    t0, t1 = ts[flying][0], ts[flying][-1]
    det = FlowObstacleDetector(cfg)
    n_alarm = n = 0
    for k, fr in enumerate(src):
        t = ts[k]
        if not (t0 <= t <= t1):
            continue
        fm = Frame(k, t, fr.bgr, gravity=G[k], R_wc=R[k])
        res = det.process(fm)
        if args_hide_floor:
            ground = det._ground_cells(fm)                # маска пола самого детектора (по горизонту)
            if ground is not None and "risk" in res.debug:
                res.debug["risk"] = np.where(ground, 0.0, res.debug["risk"])
        vis = draw_result(fr.bgr.copy(), res)        # стандартная отрисовка детектора
        n_alarm += int(res.present)
        wr.write(vis)
        n += 1
    src.release()
    return {"frames": n, "alarm_frames": n_alarm}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="hangar_gyro")
    ap.add_argument("--flights", nargs="+", default=["indoor_forward_6", "indoor_forward_10"])
    ap.add_argument("--hide-floor", action="store_true", help="не показывать на карте пол (ячейки, которые детектор отсекает)")
    args = ap.parse_args()
    presets = json.loads((ROOT / "optical_flow" / "presets.json").read_text(encoding="utf-8"))
    p = presets[args.preset]["config"]
    cfg = replace(make_cfg(p), cam_f=C["f"], cam_cx=C["cx"], cam_cy=C["cy"])   # кадры полного размера 640x480
    out = ROOT / "results" / "demo" / f"hangar_{args.preset}{'_nofloor' if args.hide_floor else ''}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    wr = make_writer(out, (640, 480), 28.0)
    for f in args.flights:
        print(f, render(f, cfg, wr, f"ангар, полёт {f.split('_')[-1]}", args.hide_floor), flush=True)
    wr.release()
    print(out, f"{out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
