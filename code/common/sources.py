"""Источники кадров: вебкамера, видеофайл, папка с изображениями, датасет TUM RGB-D.

Все источники итерируются по Frame. Для видеофайла рядом может лежать
<имя>.timestamps.csv с реальными временными метками записи (пишет record.py),
тогда dt между кадрами берётся из него, а не из номинального fps.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
from time import perf_counter
from typing import Iterator

import cv2
import numpy as np

from .types import Frame

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


class FrameSource:
    fps: float | None = None
    name: str = "source"
    replayable: bool = True     # можно ли пройти повторно (вебкамера - нет)

    def __iter__(self) -> Iterator[Frame]:
        raise NotImplementedError

    def release(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


class WebcamSource(FrameSource):
    replayable = False

    def __init__(self, index: int = 0, width: int | None = None, height: int | None = None,
                 fps: float | None = None) -> None:
        self.name = f"cam{index}"
        self.cap = None
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if os.name == "nt" else [cv2.CAP_ANY]
        for b in backends:
            cap = cv2.VideoCapture(index, b)
            if cap.isOpened():
                self.cap = cap
                break
            cap.release()
        if self.cap is None:
            raise RuntimeError(f"Cannot open webcam {index}")
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or None
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __iter__(self) -> Iterator[Frame]:
        i = 0
        t0 = perf_counter()
        while True:
            ok, bgr = self.cap.read()
            if not ok:
                break
            yield Frame(i, perf_counter() - t0, bgr)
            i += 1

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def timestamps_path(video: Path) -> Path:
    return video.with_name(video.stem + ".timestamps.csv")


def read_timestamps(path: Path) -> list[float] | None:
    if not path.exists():
        return None
    ts: list[float] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ts.append(float(row["t"]))
    return ts


class VideoSource(FrameSource):
    def __init__(self, path: str | Path, fps_override: float | None = None) -> None:
        self.path = Path(path)
        self.name = self.path.stem
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video {self.path}")
        nominal = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps_override or (nominal if nominal and nominal > 0 else 30.0)
        self.timestamps = read_timestamps(timestamps_path(self.path))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self) -> Iterator[Frame]:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        i = 0
        while True:
            ok, bgr = self.cap.read()
            if not ok:
                break
            if self.timestamps is not None and i < len(self.timestamps):
                t = self.timestamps[i]
            else:
                t = i / self.fps
            yield Frame(i, t, bgr)
            i += 1

    def release(self) -> None:
        self.cap.release()


class ImageDirSource(FrameSource):
    def __init__(self, folder: str | Path, fps: float = 30.0) -> None:
        self.folder = Path(folder)
        self.name = self.folder.name
        self.fps = fps
        self.files = sorted(p for p in self.folder.iterdir() if p.suffix.lower() in IMAGE_EXT)
        if not self.files:
            raise RuntimeError(f"No images in {self.folder}")

    def __iter__(self) -> Iterator[Frame]:
        for i, p in enumerate(self.files):
            bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            yield Frame(i, i / self.fps, bgr)


class TumSource(FrameSource):
    """Последовательность TUM RGB-D: кадр + глубина в метрах."""

    def __init__(self, folder: str | Path) -> None:
        from .tum import TumSequence
        self.seq = TumSequence(folder)
        self.name = Path(folder).name
        self.fps = 30.0

    def __iter__(self) -> Iterator[Frame]:
        for i, (t, rgb_path, depth_path) in enumerate(self.seq):
            bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            depth = self.seq.read_depth(depth_path)
            yield Frame(i, t, bgr, depth)


def open_source(spec: str, width: int | None = None, height: int | None = None,
                fps: float | None = None) -> FrameSource:
    """spec: номер камеры ("0"), путь к видео, к папке с изображениями или к папке TUM."""
    if spec.isdigit():
        return WebcamSource(int(spec), width, height, fps)
    p = Path(spec)
    if p.is_file() and p.suffix.lower() in VIDEO_EXT:
        return VideoSource(p, fps)
    if p.is_dir():
        if (p / "rgb.txt").exists():
            return TumSource(p)
        return ImageDirSource(p, fps or 30.0)
    raise ValueError(f"Cannot interpret source {spec!r}")


def make_writer(path: str | Path, size: tuple[int, int], fps: float = 30.0) -> cv2.VideoWriter:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*("mp4v" if path.suffix.lower() == ".mp4" else "MJPG"))
    w = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not w.isOpened():
        raise RuntimeError(f"Cannot open video writer {path}")
    return w
