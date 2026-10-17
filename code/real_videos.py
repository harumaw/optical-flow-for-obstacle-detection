"""Проверка детектора на реальных видео полётов (YouTube FPV, UZH-FPV) с ручной разметкой эпизодов.

    python real_videos.py sheets VIDEO [--step 0.5] [--start 0] [--end 0]   # листы кадров для разметки
    python real_videos.py eval   VIDEO [--labels VIDEO.events.json]        # прогон детектора и оценки
    python real_videos.py demo   VIDEO [--start S --end E]                 # видео с наложением детектора
    python real_videos.py report                                           # сводка по всем размеченным видео

Разметка: <видео>.events.json
{
  "fps_note": "...",
  "ignore": [[t0, t1], ...],              # куски, которые не оцениваются (титры, переходы, неясно)
  "events": [                              # опасные эпизоды
    {"start": 12.0,                        # препятствие на линии полёта начинает заметно приближаться
     "end": 14.5,                          # удар, либо момент, когда препятствие ушло с пути (облёт)
     "type": "crash" | "near_miss",
     "side": "left" | "right" | "stop" | "",   # правильный манёвр (обход слева/справа), "" если неясно
     "note": "дерево по центру"}
  ]
}
  "safe": [[t0, t1], ...]                 # необязательно: явно безопасные куски (путь открыт)
Если "safe" задан, ложные тревоги считаются только на этих кусках; иначе безопасным считается
всё вне событий и ignore. Для плотного леса нужен явный "safe": деревья рядом почти всегда.

Метрики:
  событие обнаружено   была тревога внутри [start, end]
  упреждение           end − время первой тревоги в событии (для crash это время до удара), с
  доля тревоги         доля кадров события с тревогой
  ложные тревоги       доля безопасных кадров с тревогой; отдельно — их число в минуту полёта
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from common.sources import VideoSource, make_writer
from common.types import Frame
from common.viz import draw_result
from optical_flow.detector import FlowConfig, FlowObstacleDetector

RESULTS = Path("results") / "real"


def events_path(video: Path) -> Path:
    return video.with_name(video.stem + ".events.json")


# ------------------------------------------------------------------ листы кадров
def cmd_sheets(args) -> None:
    video = Path(args.video)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    from common.sources import read_timestamps, timestamps_path
    ts = read_timestamps(timestamps_path(video))
    ts = np.array(ts) if ts else None
    dur = ts[-1] if ts is not None else n / fps
    t_end = args.end if args.end > 0 else dur
    times = np.arange(args.start, t_end, args.step)
    out_dir = RESULTS / video.stem / "sheets"
    out_dir.mkdir(parents=True, exist_ok=True)
    cols, rows = 4, 4
    tw, th = 400, 225
    per = cols * rows
    for s in range(0, len(times), per):
        tiles = []
        for t in times[s:s + per]:
            if ts is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(np.searchsorted(ts, t)))
            else:
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, f = cap.read()
            if not ok:
                f = np.zeros((th, tw, 3), np.uint8)
            f = cv2.resize(f, (tw, th), interpolation=cv2.INTER_AREA)
            # крест в центре кадра: линия полёта
            cv2.drawMarker(f, (tw // 2, th // 2), (0, 255, 255), cv2.MARKER_CROSS, 18, 1)
            cv2.putText(f, f"{t:6.1f}s", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(f, f"{t:6.1f}s", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            tiles.append(f)
        while len(tiles) < per:
            tiles.append(np.zeros((th, tw, 3), np.uint8))
        sheet = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])
        p = out_dir / f"sheet_{s // per:03d}.jpg"
        cv2.imwrite(str(p), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(p)
    print(f"{video.name}: {dur:.1f} s, {fps:.1f} fps, {n} frames, {math.ceil(len(times) / per)} sheets")


# ------------------------------------------------------------------ прогон детектора
def run_detector(video: Path, cfg: FlowConfig | None = None, t0: float = 0.0, t1: float = 0.0,
                 collect_frames: bool = False, overrides: dict | None = None):
    src = VideoSource(video)
    det = FlowObstacleDetector(cfg or FlowConfig())
    rows, frames = [], []
    start_frame = int(t0 * src.fps)
    src.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    k = start_frame
    while True:
        ok, bgr = src.cap.read()
        if not ok:
            break
        t = src.timestamps[k] if src.timestamps is not None and k < len(src.timestamps) else k / src.fps
        if t1 and t > t1:
            break
        res = det.process(Frame(k, t, bgr))
        rows.append({"frame": k, "t": t, "alarm": res.present, "ttc": res.proximity, "side": res.side,
                     "ms": res.timing_ms.get("total", 0.0)})
        if collect_frames:
            frames.append(draw_result(bgr.copy(), res))
        k += 1
    src.release()
    return rows, frames, src.fps


# Конфигурации детектора: "synthetic" — лучшая по синтетике (по умолчанию в FlowConfig);
# "real" — поправка для реальных FPV-кадров: земля занимает до половины кадра, препятствие ищется ближе к центру.
CONFIGS = {
    "synthetic": {},
    "real": {"ground_frac": 0.5, "corridor_frac": 0.4, "min_cells": 8},
}


def make_cfg(args) -> FlowConfig:
    return FlowConfig(ttc_threshold_s=args.ttc, **CONFIGS[args.config])


def in_any(t: float, spans) -> bool:
    return any(a <= t <= b for a, b in spans)


def evaluate(rows: list[dict], labels: dict) -> dict:
    ignore = [tuple(x) for x in labels.get("ignore", [])]
    events = labels.get("events", [])
    ev_spans = [(e["start"], e["end"]) for e in events]
    safe_spans = [tuple(x) for x in labels.get("safe", [])]
    safe = [r for r in rows if not in_any(r["t"], ignore) and not in_any(r["t"], ev_spans)
            and (not safe_spans or in_any(r["t"], safe_spans))]
    fa = sum(r["alarm"] for r in safe)
    fps = (len(rows) - 1) / (rows[-1]["t"] - rows[0]["t"]) if len(rows) > 1 else 30.0
    safe_minutes = len(safe) / fps / 60
    ev_out = []
    for e in events:
        er = [r for r in rows if e["start"] <= r["t"] <= e["end"]]
        alarms = [r for r in er if r["alarm"]]
        first = alarms[0]["t"] if alarms else None
        ev_out.append({
            **e,
            "detected": first is not None,
            "lead_s": round(e["end"] - first, 2) if first is not None else None,
            "alarm_share": round(len(alarms) / len(er), 3) if er else None,
            "side_ok": (sum(r["side"] == e["side"] for r in alarms) / len(alarms))
            if alarms and e.get("side") else None,
        })
    det = [e for e in ev_out if e["detected"]]
    crashes = [e for e in ev_out if e.get("type") == "crash"]
    return {
        "frames": len(rows),
        "duration_s": round(rows[-1]["t"] - rows[0]["t"], 1) if rows else 0,
        "safe_frames": len(safe),
        "fa_frames": fa,
        "fa_rate": round(fa / len(safe), 4) if safe else None,
        "fa_per_min": round(fa / safe_minutes, 1) if safe_minutes else None,
        "events": len(events),
        "detected": len(det),
        "crashes": len(crashes),
        "crashes_detected": sum(e["detected"] for e in crashes),
        "mean_lead_s": round(float(np.mean([e["lead_s"] for e in det])), 2) if det else None,
        "ms_per_frame": round(float(np.mean([r["ms"] for r in rows[1:]])), 2) if len(rows) > 1 else None,
        "per_event": ev_out,
    }


def cmd_eval(args) -> dict:
    video = Path(args.video)
    lp = Path(args.labels) if args.labels else events_path(video)
    labels = json.loads(lp.read_text(encoding="utf-8"))
    rows, _, fps = run_detector(video, make_cfg(args))
    m = evaluate(rows, labels)
    m["ttc_threshold_s"] = args.ttc
    m["config"] = args.config
    out = RESULTS / video.stem / f"{args.config}_ttc{args.ttc:g}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    import pandas as pd
    pd.DataFrame(rows).to_csv(out / "frames.csv", index=False)
    print(f"{video.name}: {m['detected']}/{m['events']} events detected "
          f"(crashes {m['crashes_detected']}/{m['crashes']}), mean lead {m['mean_lead_s']} s, "
          f"false alarms {m['fa_frames']}/{m['safe_frames']} frames ({m['fa_per_min']}/min), "
          f"{m['ms_per_frame']} ms/frame")
    for e in m["per_event"]:
        print(f"   {e['start']:6.1f}-{e['end']:6.1f} {e['type']:<9} detected={e['detected']!s:<5} "
              f"lead={e['lead_s']} share={e['alarm_share']}  {e.get('note', '')}")
    plot_timeline(rows, labels, out / "timeline.png", video.stem)
    return m


def plot_timeline(rows, labels, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.array([r["t"] for r in rows])
    a = np.array([r["alarm"] for r in rows], float)
    ttc = np.array([r["ttc"] if r["ttc"] is not None else np.nan for r in rows], float)
    fig, ax = plt.subplots(2, 1, figsize=(12, 4), sharex=True, gridspec_kw={"height_ratios": [1, 2]})
    for e in labels.get("events", []):
        c = "#c0392b" if e.get("type") == "crash" else "#e67e22"
        for axx in ax:
            axx.axvspan(e["start"], e["end"], color=c, alpha=0.18, lw=0)
    for s, e_ in labels.get("ignore", []):
        for axx in ax:
            axx.axvspan(s, e_, color="gray", alpha=0.25, lw=0)
    for s, e_ in labels.get("safe", []):
        ax[0].axvspan(s, e_, ymin=0, ymax=0.15, color="#27ae60", alpha=0.6, lw=0)
    ax[0].fill_between(t, 0, a, step="mid", color="#1f4e79")
    ax[0].set_yticks([])
    ax[0].set_ylabel("тревога")
    ax[1].plot(t, ttc, ".", ms=2, color="#1f4e79")
    ax[1].set_ylim(0, 4.5)
    ax[1].set_ylabel("TTC, с")
    ax[1].set_xlabel("время ролика, с (красное: удар, оранжевое: близкий пролёт, серое: не оценивается, зелёное: путь открыт)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def cmd_demo(args) -> None:
    video = Path(args.video)
    rows, frames, fps = run_detector(video, make_cfg(args), t0=args.start, t1=args.end,
                                     collect_frames=True)
    out = RESULTS / video.stem
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"demo_{args.start:g}-{args.end:g}_{args.config}_ttc{args.ttc:g}.mp4"
    wr = make_writer(p, (frames[0].shape[1], frames[0].shape[0]), fps)
    for f in frames:
        wr.write(f)
    wr.release()
    print(p)


def cmd_report(args) -> None:
    rows = []
    for ev in sorted(RESULTS.glob("*/*_ttc*/eval.json")):
        m = json.loads(ev.read_text(encoding="utf-8"))
        rows.append({"video": ev.parent.parent.name, **{k: v for k, v in m.items() if k != "per_event"}})
    import pandas as pd
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(RESULTS / "summary.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sheets"); s.add_argument("video"); s.add_argument("--step", type=float, default=0.5)
    s.add_argument("--start", type=float, default=0.0); s.add_argument("--end", type=float, default=0.0)
    e = sub.add_parser("eval"); e.add_argument("video"); e.add_argument("--labels", default=None)
    e.add_argument("--ttc", type=float, default=4.0, help="порог TTC, с")
    e.add_argument("--config", choices=list(CONFIGS), default="synthetic")
    d = sub.add_parser("demo"); d.add_argument("video"); d.add_argument("--start", type=float, default=0.0)
    d.add_argument("--end", type=float, default=0.0)
    d.add_argument("--ttc", type=float, default=4.0)
    d.add_argument("--config", choices=list(CONFIGS), default="synthetic")
    sub.add_parser("report")
    args = ap.parse_args()
    {"sheets": cmd_sheets, "eval": cmd_eval, "demo": cmd_demo, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
