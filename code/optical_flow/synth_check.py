"""Проверка детектора по оптическому потоку на синтетических последовательностях
с точно известным временем до столкновения.

    python optical_flow/synth_check.py [--out data/synth] [--backends farneback dis lk]

Последовательности (640x480, 30 кадров/с):
    approach_wall  сближение с фронтальной стеной: 8 м -> 1 м при 1 м/с, TTC известно
    object_left    статичный фон, объект слева сближается 6 м -> 1.5 м (ожидаем side=right)
    lateral        боковой сдвиг камеры 1.5 px/кадр (препятствия нет)
    rotation       поворот вокруг оптической оси 0.4 град/кадр (препятствия нет)
    static         неподвижная камера, только шум сенсора (препятствия нет)

Для каждой пишутся <имя>.mp4 и <имя>.labels.csv, чтобы run_bench.py мог
использовать их как обычный источник. Затем все бэкенды прогоняются
и печатается таблица: точность наличия, стороны, ошибка TTC, время на кадр.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.decision import decide                          # noqa: E402
from common.labels import write_labels                      # noqa: E402
from common.metrics import summarize_method                 # noqa: E402
from common.sources import make_writer                      # noqa: E402
from common.types import Frame                              # noqa: E402
from optical_flow.detector import FlowConfig, FlowObstacleDetector  # noqa: E402

W, H, FPS = 640, 480, 30.0
DT = 1.0 / FPS


def make_texture(h: int, w: int, rng: np.random.Generator, blobs: int = 60) -> np.ndarray:
    noise = rng.random((h, w), dtype=np.float32)
    tex = cv2.GaussianBlur(noise, (0, 0), 1.5)
    tex = (tex - tex.min()) / (tex.max() - tex.min() + 1e-6)
    img = (60 + tex * 140).astype(np.uint8)
    for _ in range(blobs):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = int(rng.integers(6, 40))
        g = int(rng.integers(0, 255))
        if rng.random() < 0.5:
            cv2.circle(img, (x, y), r, g, -1)
        else:
            cv2.rectangle(img, (x - r, y - r), (x + r, y + r), g, -1)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def zoom_crop(base: np.ndarray, scale: float, dx: float = 0.0, dy: float = 0.0, angle: float = 0.0) -> np.ndarray:
    """Центральный вырез WxH из base, увеличенный в scale раз, со сдвигом и поворотом."""
    bh, bw = base.shape[:2]
    cx, cy = bw / 2.0 + dx, bh / 2.0 + dy
    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    M[0, 2] += W / 2.0 - cx
    M[1, 2] += H / 2.0 - cy
    return cv2.warpAffine(base, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def add_noise(img: np.ndarray, rng: np.random.Generator, sigma: float = 2.0) -> np.ndarray:
    n = rng.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)


def seq_approach_wall(rng, z0=8.0, z1=1.0, v=1.0, ttc_thr=4.0):
    base = make_texture(2 * H, 2 * W, rng)
    z = z0
    k = 0
    while z > z1:
        s = z0 / z
        yield add_noise(zoom_crop(base, s), rng), {"frame": k, "present": z / v < ttc_thr,
                                                   "side": "stop" if z / v < ttc_thr else "none",
                                                   "dist_m": z, "ttc_s": z / v}
        z -= v * DT
        k += 1


def seq_object_left(rng, z0=6.0, z1=1.5, v=1.0, ttc_thr=4.0):
    base = make_texture(2 * H, 2 * W, rng, blobs=20)
    obj = make_texture(180, 140, rng, blobs=25)
    ocx, ocy = int(W * 0.28), H // 2
    z = z0
    k = 0
    while z > z1:
        s = z0 / z
        frame = zoom_crop(base, 1.0)
        oh, ow = int(obj.shape[0] * s), int(obj.shape[1] * s)
        patch = cv2.resize(obj, (ow, oh), interpolation=cv2.INTER_LINEAR)
        x0, y0 = ocx - ow // 2, ocy - oh // 2
        xs0, ys0 = max(0, x0), max(0, y0)
        xs1, ys1 = min(W, x0 + ow), min(H, y0 + oh)
        frame[ys0:ys1, xs0:xs1] = patch[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]
        ttc = z / v
        present = ttc < ttc_thr
        side = decide((xs0, ys0, xs1, ys1), W) if present else "none"
        yield add_noise(frame, rng), {"frame": k, "present": present, "side": side,
                                      "dist_m": z, "ttc_s": ttc}
        z -= v * DT
        k += 1


def seq_lateral(rng, n=150, px_per_frame=1.5):
    base = make_texture(2 * H, 2 * W, rng)
    for k in range(n):
        yield add_noise(zoom_crop(base, 1.0, dx=(k - n / 2) * px_per_frame), rng), \
            {"frame": k, "present": False, "side": "none"}


def seq_rotation(rng, n=150, deg_per_frame=0.4):
    base = make_texture(2 * H, 2 * W, rng)
    for k in range(n):
        yield add_noise(zoom_crop(base, 1.0, angle=k * deg_per_frame), rng), \
            {"frame": k, "present": False, "side": "none"}


def seq_static(rng, n=120):
    base = make_texture(2 * H, 2 * W, rng)
    frame = zoom_crop(base, 1.0)
    for k in range(n):
        yield add_noise(frame, rng), {"frame": k, "present": False, "side": "none"}


SEQUENCES = {
    "approach_wall": seq_approach_wall,
    "object_left": seq_object_left,
    "lateral": seq_lateral,
    "rotation": seq_rotation,
    "static": seq_static,
}


def generate(out: Path, names: list[str], seed: int = 0) -> dict[str, tuple[list[np.ndarray], list[dict]]]:
    out.mkdir(parents=True, exist_ok=True)
    data = {}
    for name in names:
        rng = np.random.default_rng(seed)
        frames, labels = [], []
        for img, lab in SEQUENCES[name](rng):
            frames.append(img)
            labels.append(lab)
        writer = make_writer(out / f"{name}.mp4", (W, H), FPS)
        for img in frames:
            writer.write(img)
        writer.release()
        write_labels(out / f"{name}.labels.csv", labels)
        data[name] = (frames, labels)
        print(f"  {name}: {len(frames)} frames -> {out / (name + '.mp4')}")
    return data


def evaluate(data, backends: list[str], ttc_thr: float, opts: dict | None = None) -> dict:
    results = {}
    for name, (frames, labels) in data.items():
        gt = {l["frame"]: {"present": l["present"], "side": l["side"], "ttc_s": l.get("ttc_s")} for l in labels}
        results[name] = {}
        for b in backends:
            bname, _, arg = b.partition(":")         # dis:medium, lk:12
            bkw = {}
            if arg and bname == "dis":
                bkw["preset"] = arg
            elif arg and bname == "lk":
                bkw["cell"] = int(arg)
            det = FlowObstacleDetector(FlowConfig(backend=bname, backend_kwargs=bkw,
                                                  ttc_threshold_s=ttc_thr, **(opts or {})))
            if bname == "lk" and arg:
                det.name += f"-{arg}"
            rows = [det.process(Frame(i, i * DT, img)).to_row() for i, img in enumerate(frames)]
            results[name][det.name] = (summarize_method(rows, gt), rows)
    return results


def print_table(results: dict) -> None:
    head = (f"{'sequence':<14}{'method':<16}{'ms':>7}{'acc':>7}{'prec':>7}{'rec':>7}"
            f"{'side':>7}{'sideLR':>8}{'delay':>7}{'ttcErr':>8}")
    print(head)
    print("-" * len(head))
    fmt = lambda v: "-" if v is None else f"{v:.2f}"
    for name, per in results.items():
        for m, (s, _) in per.items():
            t = s.get("timing_total", {}).get("mean_ms", float("nan"))
            print(f"{name:<14}{m:<16}{t:>7.1f}{s.get('accuracy', 0):>7.2f}{s.get('precision', 0):>7.2f}"
                  f"{s.get('recall', 0):>7.2f}{fmt(s.get('side_accuracy')):>7}{fmt(s.get('side_accuracy_lr')):>8}"
                  f"{str(s.get('mean_delay_frames', '-')):>7}{str(s.get('ttc_median_rel_err', '-')):>8}")


def plot_ttc(results: dict, data: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name in ("approach_wall", "object_left"):
        if name not in results:
            continue
        fig, ax = plt.subplots(figsize=(8, 3.6))
        labels = data[name][1]
        ax.plot([l["frame"] for l in labels], [l["ttc_s"] for l in labels], "k--", label="истинное TTC")
        for m, (_, rows) in results[name].items():
            pts = [(r["frame"], float(r["proximity"])) for r in rows if r["proximity"] != ""]
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], ".", ms=3, label=m)
        ax.set_xlabel("кадр")
        ax.set_ylabel("TTC, с")
        ax.set_ylim(0, 8)
        ax.set_title(f"Синтетика: {name}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"ttc_{name}.png", dpi=150)
    print(f"  plots -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=str, default="data/synth")
    ap.add_argument("--backends", nargs="+", default=["farneback", "dis", "lk"])
    ap.add_argument("--sequences", nargs="+", default=list(SEQUENCES))
    ap.add_argument("--ttc", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--opt", nargs="*", default=[], metavar="KEY=VAL",
                    help="параметры FlowConfig, например min_texture=2.0 min_persist=3")
    args = ap.parse_args()

    opts = {}
    for kv in args.opt:
        k, _, v = kv.partition("=")
        opts[k] = float(v) if "." in v else int(v)

    out = Path(args.out)
    print("generating sequences")
    data = generate(out, args.sequences, args.seed)
    print(f"evaluating {opts or 'defaults'}")
    results = evaluate(data, args.backends, args.ttc, opts)
    print_table(results)
    if any(n in results for n in ("approach_wall", "object_left")):
        figs = Path("results/synth")
        figs.mkdir(parents=True, exist_ok=True)
        plot_ttc(results, data, figs)


if __name__ == "__main__":
    main()
