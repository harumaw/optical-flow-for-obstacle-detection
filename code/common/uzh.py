"""Датасет UZH-FPV: вектор силы тяжести в осях камеры по траектории и калибровке.

groundtruth.txt: "timestamp tx ty tz qx qy qz qw" — положение и ориентация IMU (body) в мировой системе,
ось z мира вверх. Калибровка Kalibr: T_cam_imu переводит точки из осей IMU в оси камеры.
Тяжесть в мире g_w = (0, 0, -1); в камере g_c = R_cam_imu · R_world_imuᵀ · g_w.
На настоящем дроне то же самое даёт крен и тангаж автопилота (MAVLink ATTITUDE).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .tum import quat_to_rot

# indoor_forward_calib_snapdragon, cam0 (левая камера), модель equidistant (рыбий глаз)
INDOOR_CAM0 = {
    "R_cam_imu": np.array([
        [-0.028228787368606456, -0.999601488301944, 1.2175294828553618e-05],
        [0.014401251861751119, -0.00041887083271471837, -0.9998962088597202],
        [0.999497743623523, -0.028225682131089447, 0.014407337010089172],
    ]),
    "f": 278.58, "cx": 319.75, "cy": 241.97, "width": 640, "model": "equidistant",
    "timeshift_cam_imu": -0.016684572091862235,   # t_imu = t_cam + timeshift (Kalibr)
}


def first_image_time(seq_root: str | Path) -> float:
    for line in open(Path(seq_root) / "left_images.txt", encoding="utf-8"):
        if not line.startswith("#"):
            return float(line.split()[1])
    raise RuntimeError("empty left_images.txt")


def gravity_in_camera(seq_root: str | Path, rel_times: np.ndarray, calib: dict = INDOOR_CAM0) -> np.ndarray:
    """(N, 3) единичные векторы тяжести в осях камеры; NaN вне участка с траекторией."""
    root = Path(seq_root)
    t0 = first_image_time(root)
    g = np.array([[float(x) for x in l.split()] for l in open(root / "groundtruth.txt", encoding="utf-8")
                  if not l.startswith("#")])
    gt_t, q = g[:, 0] - t0, g[:, 4:8]
    out = np.full((len(rel_times), 3), np.nan)
    gw = np.array([0.0, 0.0, -1.0])
    R_ci = calib["R_cam_imu"]
    for i, t in enumerate(rel_times):
        if t < gt_t[0] or t > gt_t[-1]:
            continue
        j = int(np.clip(np.searchsorted(gt_t, t), 0, len(gt_t) - 1))
        R_wi = quat_to_rot(q[j])
        v = R_ci @ (R_wi.T @ gw)
        out[i] = v / np.linalg.norm(v)
    return out


def camera_orientation(seq_root: str | Path, rel_times: np.ndarray, calib: dict = INDOOR_CAM0) -> np.ndarray:
    """(N, 3, 3) ориентации камеры в мире R_wc (камера -> мир); NaN вне участка с траекторией.
    Кватернионы траектории интерполируются сферически между соседними отсчётами."""
    root = Path(seq_root)
    t0 = first_image_time(root)
    g = np.array([[float(x) for x in l.split()] for l in open(root / "groundtruth.txt", encoding="utf-8")
                  if not l.startswith("#")])
    gt_t, q = g[:, 0] - t0, g[:, 4:8]
    R_ic = calib["R_cam_imu"].T
    out = np.full((len(rel_times), 3, 3), np.nan)
    for i, t in enumerate(np.asarray(rel_times) + calib.get("timeshift_cam_imu", 0.0)):
        if t < gt_t[0] or t > gt_t[-1]:
            continue
        j = int(np.clip(np.searchsorted(gt_t, t), 1, len(gt_t) - 1))
        a = (t - gt_t[j - 1]) / max(gt_t[j] - gt_t[j - 1], 1e-9)
        q0, q1 = q[j - 1], q[j]
        if np.dot(q0, q1) < 0:
            q1 = -q1
        qi = (1 - a) * q0 + a * q1                     # нормированная линейная интерполяция (шаг 2 мс)
        out[i] = quat_to_rot(qi / np.linalg.norm(qi)) @ R_ic
    return out
