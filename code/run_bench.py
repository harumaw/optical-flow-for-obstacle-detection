"""Офлайн-сравнение методов на одном источнике (видео, папка кадров, TUM).

    python run_bench.py --source data/raw/clip01.mp4 --methods flow:farneback flow:dis flow:lk
    python run_bench.py --source data/tum/rgbd_dataset_freiburg2_pioneer_slam --labels data/tum/.../labels.csv

Все методы считаются на каждом кадре одного прохода, время меряется отдельно
для каждого. Результат: results/<имя>/<метод>.csv (по кадрам), summary.json и summary.md.
Разметка берётся из --labels или из <источник>.labels.csv, если файл есть.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2

from common.labels import default_labels_path, read_labels
from common.metrics import summarize_method
from common.sources import open_source
from common.util import safe_name, unique_names
from common.viz import draw_result
from methods import make_detector


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def summary_markdown(summary: dict) -> str:
    methods = list(summary["methods"])
    has_labels = any("f1" in summary["methods"][m] for m in methods)
    head = ["method", "frames", "mean ms", "median ms", "p95 ms", "fps"]
    if has_labels:
        head += ["precision", "recall", "F1", "side acc", "delay (frames)", "TTC rel err"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for m in methods:
        s = summary["methods"][m]
        t = s.get("timing_total", {})
        row = [m, s.get("frames", ""), t.get("mean_ms", ""), t.get("median_ms", ""), t.get("p95_ms", ""), t.get("fps", "")]
        if has_labels:
            row += [s.get("precision", ""), s.get("recall", ""), s.get("f1", ""),
                    s.get("side_accuracy", ""), s.get("mean_delay_frames", ""), s.get("ttc_median_rel_err", "")]
        lines.append("| " + " | ".join("" if v is None else str(v) for v in row) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--methods", nargs="+", default=["flow:farneback", "flow:dis", "flow:lk"])
    ap.add_argument("--labels", type=str, default=None)
    ap.add_argument("--out", type=str, default=None, help="папка результатов (по умолчанию results/<источник>)")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--fps", type=float, default=None, help="переопределить fps источника")
    ap.add_argument("--show", action="store_true", help="показывать первый метод во время прогона")
    args = ap.parse_args()

    src = open_source(args.source, fps=args.fps)
    out_dir = Path(args.out) if args.out else Path("results") / safe_name(src.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = None
    labels_path = Path(args.labels) if args.labels else default_labels_path(args.source)
    if labels_path.exists():
        labels = read_labels(labels_path)
        print(f"labels: {labels_path} ({len(labels)} frames)")
    else:
        print("labels: none (only timing will be reported)")

    dets = [make_detector(s) for s in args.methods]
    unique_names(dets)
    rows: dict[str, list[dict]] = {d.name: [] for d in dets}

    n = 0
    with src:
        for frame in src:
            for i, d in enumerate(dets):
                res = d.process(frame)
                rows[d.name].append(res.to_row())
                if args.show and i == 0:
                    cv2.imshow("bench", draw_result(frame.bgr.copy(), res))
                    if cv2.waitKey(1) == ord("q"):
                        args.max_frames = n
            n += 1
            if n % 100 == 0:
                print(f"  {n} frames", end="\r")
            if args.max_frames and n >= args.max_frames:
                break
    if args.show:
        cv2.destroyAllWindows()
    print(f"processed {n} frames from {src.name}")

    summary = {"source": args.source, "frames": n, "methods": {}}
    for d in dets:
        write_rows(out_dir / f"{safe_name(d.name)}.csv", rows[d.name])
        summary["methods"][d.name] = summarize_method(rows[d.name], labels)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    md = summary_markdown(summary)
    (out_dir / "summary.md").write_text(md + "\n", encoding="utf-8")
    print(md)
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
