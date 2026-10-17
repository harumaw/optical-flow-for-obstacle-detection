"""Автоматическая разметка последовательности TUM RGB-D по карте глубины и траектории камеры.

    python tum_labels.py data/tum/rgbd_dataset_freiburg2_pioneer_slam [--ttc 4.0] [--near 1.0]

Создаёт в папке последовательности два файла, по одному на критерий:

    labels_ttc.csv   препятствие есть, если время до столкновения меньше --ttc секунд.
                     Это тот же вопрос, на который отвечает оптический поток.
    labels_dist.csv  препятствие есть, если ближе --near метров что-то занимает
                     заметную часть центра кадра. Это вопрос для оценки глубины.

Для каждого кадра:
    dist_m  расстояние до ближайшего в центральной зоне (10-й перцентиль глубины Kinect);
    v_fwd   скорость камеры вдоль оптической оси из groundtruth.txt (система захвата движения);
    ttc_s   dist_m / v_fwd, если камера движется вперёд быстрее 0.05 м/с, иначе пусто.
Если groundtruth.txt нет, скорость сближения берётся из производной dist_m по времени.
Пол исключается: высота каждого пикселя под камерой считается по глубине и
параметрам Kinect, пиксели на уровне пола обнуляются (см. common/tum.py).
Кадры, где в центральной зоне мало валидной глубины, помечаются unknown и в метриках не участвуют.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common.decision import decide
from common.labels import write_labels
from common.tum import (TumSequence, auto_label, estimate_floor_height, forward_speed,
                        nearest_region, read_groundtruth, remove_floor)

MIN_SPEED = 0.05   # м/с: медленнее считаем, что камера не сближается


def depth_derivative_speed(times: np.ndarray, dists: np.ndarray, smooth: int = 5) -> np.ndarray:
    """Запасной вариант без траектории: скорость сближения = -d(dist)/dt."""
    valid = ~np.isnan(dists)
    out = np.full(len(dists), np.nan)
    if valid.sum() <= smooth:
        return out
    filled = np.interp(np.arange(len(dists)), np.flatnonzero(valid), dists[valid])
    d_s = np.convolve(filled, np.ones(smooth) / smooth, mode="same")
    out[1:-1] = -(d_s[2:] - d_s[:-2]) / (times[2:] - times[:-2])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--ttc", type=float, default=4.0, help="порог времени до столкновения, с")
    ap.add_argument("--near", type=float, default=1.0, help="порог расстояния, м")
    ap.add_argument("--center-frac", type=float, default=0.6)
    ap.add_argument("--min-frac", type=float, default=0.05)
    ap.add_argument("--keep-floor", action="store_true", help="не исключать пол из разметки")
    args = ap.parse_args()

    folder = Path(args.folder)
    seq = TumSequence(folder)
    floor_y = None
    if not args.keep_floor:
        floor_y = estimate_floor_height(seq)
        print(f"floor is {floor_y:.2f} m below the camera; floor pixels are excluded")
    times, dists, dist_rows, near_boxes, widths = [], [], [], [], []
    for i, (t, _, depth_path) in enumerate(seq):
        depth = seq.read_depth(depth_path)
        if floor_y is not None:
            depth = remove_floor(depth, floor_y)
        widths.append(depth.shape[1])
        lab = auto_label(depth, args.near, args.center_frac, args.min_frac)
        dist, box = nearest_region(depth, args.center_frac)
        times.append(t)
        dists.append(np.nan if dist is None else dist)
        near_boxes.append(box)
        dist_rows.append({"frame": i, "present": lab["present"], "side": lab["side"],
                          "dist_m": lab["dist_m"], "unknown": lab["unknown"]})
        if i % 100 == 0:
            print(f"  depth {i}/{len(seq)}", end="\r")
    times = np.array(times)
    dists = np.array(dists)

    gt = read_groundtruth(folder)
    if gt is not None:
        v = forward_speed(gt, times)
        src = "groundtruth trajectory"
    else:
        v = depth_derivative_speed(times, dists)
        src = "depth derivative (no groundtruth.txt)"

    ttc_rows = []
    for i in range(len(times)):
        unknown = bool(np.isnan(dists[i]) or np.isnan(v[i]))
        ttc = float(dists[i] / v[i]) if not unknown and v[i] > MIN_SPEED else None
        present = ttc is not None and ttc < args.ttc
        side = decide(near_boxes[i], widths[i]) if present and near_boxes[i] else "none"
        ttc_rows.append({"frame": i, "present": present, "side": side,
                         "dist_m": None if np.isnan(dists[i]) else float(dists[i]),
                         "ttc_s": ttc, "unknown": unknown})
        if ttc is not None:
            dist_rows[i]["ttc_s"] = ttc

    write_labels(folder / "labels_ttc.csv", ttc_rows)
    write_labels(folder / "labels_dist.csv", dist_rows)

    n = len(ttc_rows)
    n_unk = sum(r["unknown"] for r in ttc_rows)
    moving = int(np.sum(v > MIN_SPEED))
    print(f"\n{n} frames, speed from {src}")
    print(f"  moving forward (> {MIN_SPEED} m/s): {moving} frames; unknown: {n_unk}")
    vv = v[~np.isnan(v)]
    if vv.size:
        print(f"  forward speed: median {np.median(vv):.2f}  max {vv.max():.2f} m/s")
    dd = dists[~np.isnan(dists)]
    if dd.size:
        print(f"  nearest distance: median {np.median(dd):.2f}  min {dd.min():.2f} m")
    print(f"  labels_ttc.csv : {sum(r['present'] for r in ttc_rows)} frames with TTC < {args.ttc} s")
    print(f"  labels_dist.csv: {sum(r['present'] for r in dist_rows)} frames with obstacle < {args.near} m")


if __name__ == "__main__":
    main()
