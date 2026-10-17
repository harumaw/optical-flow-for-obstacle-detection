"""Чтение и запись разметки: CSV с колонками frame, present, side, dist_m, ttc_s."""
from __future__ import annotations

import csv
from pathlib import Path

LABEL_FIELDS = ["frame", "present", "side", "dist_m", "ttc_s", "unknown"]


def read_labels(path: str | Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fi = int(row["frame"])
            rec = {
                "present": row.get("present", "0") in ("1", "True", "true"),
                "side": row.get("side", "none") or "none",
                "dist_m": float(row["dist_m"]) if row.get("dist_m") else None,
                "ttc_s": float(row["ttc_s"]) if row.get("ttc_s") else None,
                "unknown": row.get("unknown", "0") in ("1", "True", "true"),
            }
            out[fi] = rec
    return out


def write_labels(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LABEL_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({
                "frame": r["frame"],
                "present": int(bool(r.get("present", False))),
                "side": r.get("side", "none"),
                "dist_m": "" if r.get("dist_m") is None else round(float(r["dist_m"]), 3),
                "ttc_s": "" if r.get("ttc_s") is None else round(float(r["ttc_s"]), 3),
                "unknown": int(bool(r.get("unknown", False))),
            })


def default_labels_path(source_path: str | Path) -> Path:
    p = Path(source_path)
    if p.is_dir():
        return p / "labels.csv"
    return p.with_name(p.stem + ".labels.csv")
