"""Запись сырого видео с вебкамеры вместе с реальными метками времени кадров.

    python record.py --out data/raw/wall_approach_01.mp4 [--cam 0] [--width 640 --height 480] [--seconds 30]

Рядом создаётся <имя>.timestamps.csv, из него источник берёт точные dt.
Клавиша q останавливает запись раньше времени.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from common.sources import WebcamSource, make_writer, timestamps_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--seconds", type=float, default=0, help="0 = до нажатия q")
    args = ap.parse_args()

    out = Path(args.out)
    src = WebcamSource(args.cam, args.width, args.height)
    writer = make_writer(out, (src.width, src.height), src.fps or 30.0)
    print(f"camera {src.width}x{src.height} @ {src.fps}; writing {out}; press q to stop")
    n = 0
    with open(timestamps_path(out), "w", newline="", encoding="utf-8") as f:
        ts = csv.writer(f)
        ts.writerow(["frame", "t"])
        try:
            for frame in src:
                writer.write(frame.bgr)
                ts.writerow([frame.index, f"{frame.t:.6f}"])
                n += 1
                vis = frame.bgr.copy()
                cv2.putText(vis, f"REC {frame.t:6.1f}s  frame {n}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow("record", vis)
                if cv2.waitKey(1) == ord("q"):
                    break
                if args.seconds and frame.t >= args.seconds:
                    break
        finally:
            src.release()
            writer.release()
            cv2.destroyAllWindows()
    print(f"saved {n} frames, {frame.t:.1f} s -> {out}")


if __name__ == "__main__":
    main()
