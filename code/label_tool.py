"""Ручная разметка видео: на каждом кадре отмечается рекомендуемый манёвр.

    python label_tool.py data/raw/wall_approach_01.mp4 [--out data/raw/wall_approach_01.labels.csv]

Клавиши:
    n       нет препятствия (none)
    l / r   препятствие есть, обходить влево / вправо
    s       препятствие перекрывает кадр (stop)
    d, ->   следующий кадр (текущая метка переносится на него)
    a, <-   предыдущий кадр
    j / k   +30 / -30 кадров
    space   воспроизведение с текущей меткой / пауза
    q       сохранить и выйти
Разметка сохраняется в формате common.labels (frame, present, side).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from common.labels import default_labels_path, read_labels, write_labels
from common.viz import SIDE_COLORS

KEY_LEFT, KEY_RIGHT = 2424832, 2555904


class VideoIndex:
    """Доступ к кадрам по номеру с кэшем последних прочитанных."""

    def __init__(self, path: Path, cache: int = 200) -> None:
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open {path}")
        self.count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._next = 0
        self._cache: dict[int, object] = {}
        self._cache_n = cache

    def get(self, i: int):
        if i in self._cache:
            return self._cache[i]
        if i != self._next:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            self._next = i
        ok, img = self.cap.read()
        if not ok:
            return None
        self._next = i + 1
        self._cache[i] = img
        if len(self._cache) > self._cache_n:
            self._cache.pop(next(iter(self._cache)))
        return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    video = Path(args.video)
    out = Path(args.out) if args.out else default_labels_path(video)
    vid = VideoIndex(video)
    sides = ["none"] * vid.count
    if out.exists():
        for fi, rec in read_labels(out).items():
            if 0 <= fi < vid.count:
                sides[fi] = rec["side"] if rec["present"] else "none"
        print(f"loaded existing labels from {out}")

    i = 0
    state = sides[0]
    playing = False
    while True:
        img = vid.get(i)
        if img is None:
            i = max(0, i - 1)
            playing = False
            continue
        sides[i] = state
        vis = img.copy()
        color = SIDE_COLORS[state]
        cv2.rectangle(vis, (0, 0), (vis.shape[1] - 1, vis.shape[0] - 1), color, 6)
        txt = f"frame {i}/{vid.count - 1}   label: {state.upper()}   {'PLAY' if playing else 'PAUSE'}"
        cv2.putText(vis, txt, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, txt, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        cv2.imshow("label", vis)
        key = cv2.waitKeyEx(30 if playing else 0)

        if key == ord("q"):
            break
        elif key == ord("n"):
            state = "none"
        elif key == ord("l"):
            state = "left"
        elif key == ord("r"):
            state = "right"
        elif key == ord("s"):
            state = "stop"
        elif key == ord(" "):
            playing = not playing
        elif key in (ord("d"), KEY_RIGHT) or (playing and key == -1):
            if i < vid.count - 1:
                i += 1
            else:
                playing = False
        elif key in (ord("a"), KEY_LEFT):
            i = max(0, i - 1)
            state = sides[i]
        elif key == ord("j"):
            j = min(vid.count - 1, i + 30)
            for f in range(i, j):          # пропущенные кадры получают текущую метку
                sides[f] = state
            i = j
        elif key == ord("k"):
            i = max(0, i - 30)
            state = sides[i]

    rows = [{"frame": fi, "present": s != "none", "side": s} for fi, s in enumerate(sides)]
    write_labels(out, rows)
    n_pos = sum(1 for s in sides if s != "none")
    print(f"saved {len(rows)} labels ({n_pos} with obstacle) -> {out}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
