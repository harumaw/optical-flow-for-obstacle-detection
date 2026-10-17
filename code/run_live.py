"""Живая демонстрация на вебкамере.

    python run_live.py                              # flow:farneback, камера 0
    python run_live.py --methods flow:farneback flow:dis flow:lk --tile
    python run_live.py --record data/raw/clip01.mp4  # параллельно пишет сырое видео + метки времени

Клавиши: q выход, 1..9 выбор метода, h тепловая карта, f стрелки потока,
         r сброс состояния, p пауза.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from common.sources import WebcamSource, make_writer, timestamps_path
from common.util import unique_names
from common.viz import draw_result
from methods import make_detector


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--methods", nargs="+", default=["flow:farneback"])
    ap.add_argument("--tile", action="store_true", help="считать все методы на каждом кадре и показывать рядом")
    ap.add_argument("--record", type=str, default=None, help="писать сырое видео в файл (mp4)")
    ap.add_argument("--no-heat", action="store_true")
    args = ap.parse_args()

    dets = [make_detector(s) for s in args.methods]
    unique_names(dets)
    active = 0
    show_heat = not args.no_heat
    show_flow = False
    paused = False

    src = WebcamSource(args.cam, args.width, args.height)
    print(f"camera {args.cam}: {src.width}x{src.height} @ {src.fps}")

    writer = None
    ts_file = None
    ts_writer = None
    if args.record:
        rec_path = Path(args.record)
        writer = make_writer(rec_path, (src.width, src.height), src.fps or 30.0)
        ts_file = open(timestamps_path(rec_path), "w", newline="", encoding="utf-8")
        ts_writer = csv.writer(ts_file)
        ts_writer.writerow(["frame", "t"])
        print(f"recording to {rec_path}")

    fps_ema = None
    try:
        for frame in src:
            if writer is not None:
                writer.write(frame.bgr)
                ts_writer.writerow([frame.index, f"{frame.t:.6f}"])
            if paused:
                key = cv2.waitKeyEx(30)
                if key in (ord("p"), ord(" ")):
                    paused = False
                elif key == ord("q"):
                    break
                continue

            if args.tile:
                panels = []
                for d in dets:
                    res = d.process(frame)
                    panels.append(draw_result(frame.bgr.copy(), res, show_heat, show_flow))
                vis = np.hstack(panels)
            else:
                res = dets[active].process(frame)
                inst = 1000.0 / max(res.timing_ms.get("total", 1e-3), 1e-3)
                fps_ema = inst if fps_ema is None else 0.9 * fps_ema + 0.1 * inst
                extra = [f"proc fps {fps_ema:.1f}   [{active + 1}/{len(dets)}] keys: 1-9 h f r p q"]
                vis = draw_result(frame.bgr.copy(), res, show_heat, show_flow, extra)

            cv2.imshow("obstacle", vis)
            key = cv2.waitKeyEx(1)
            if key == -1:
                continue
            if key == ord("q"):
                break
            if key == ord("h"):
                show_heat = not show_heat
            elif key == ord("f"):
                show_flow = not show_flow
            elif key == ord("r"):
                for d in dets:
                    d.reset()
            elif key in (ord("p"), ord(" ")):
                paused = True
            elif ord("1") <= key <= ord("9"):
                idx = key - ord("1")
                if idx < len(dets) and idx != active:
                    active = idx
                    dets[active].reset()
    finally:
        src.release()
        if writer is not None:
            writer.release()
            ts_file.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
