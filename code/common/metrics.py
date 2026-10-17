"""Метрики качества и времени для сравнения методов."""
from __future__ import annotations

import math

import numpy as np


def timing_stats(ms: list[float]) -> dict:
    if not ms:
        return {}
    a = np.asarray(ms, dtype=np.float64)
    mean = float(a.mean())
    return {
        "n": int(a.size),
        "mean_ms": round(mean, 3),
        "median_ms": round(float(np.median(a)), 3),
        "p95_ms": round(float(np.percentile(a, 95)), 3),
        "max_ms": round(float(a.max()), 3),
        "fps": round(1000.0 / mean, 2) if mean > 0 else None,
    }


def presence_metrics(pred: dict[int, bool], gt: dict[int, bool]) -> dict:
    """Бинарная классификация "препятствие есть" по кадрам, общим для pred и gt."""
    tp = fp = fn = tn = 0
    for fi, g in gt.items():
        if fi not in pred:
            continue
        p = pred[fi]
        if p and g:
            tp += 1
        elif p and not g:
            fp += 1
        elif not p and g:
            fn += 1
        else:
            tn += 1
    n = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "frames": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(f1, 4), "accuracy": round((tp + tn) / n, 4) if n else 0.0,
    }


def side_accuracy(pred_side: dict[int, str], gt_side: dict[int, str], gt_present: dict[int, bool]) -> dict:
    """Доля правильных сторон среди кадров, где препятствие есть по разметке и метод его увидел.
    side_accuracy_lr считается только по кадрам с разметкой left/right: там сравнивается
    направление обхода, а не граница между "обойти" и "stop"."""
    ok = total = 0
    ok_lr = total_lr = 0
    for fi, g in gt_side.items():
        if not gt_present.get(fi) or fi not in pred_side or pred_side[fi] == "none":
            continue
        total += 1
        ok += int(pred_side[fi] == g)
        if g in ("left", "right"):
            total_lr += 1
            ok_lr += int(pred_side[fi] == g)
    return {
        "side_frames": total,
        "side_accuracy": round(ok / total, 4) if total else None,
        "side_accuracy_lr": round(ok_lr / total_lr, 4) if total_lr else None,
    }


def detection_delay(pred: dict[int, bool], gt: dict[int, bool]) -> dict:
    """Задержка обнаружения: сколько кадров от начала каждого отрезка с препятствием
    до первого срабатывания метода. Отрезки без срабатывания считаются пропущенными."""
    frames = sorted(gt)
    delays: list[int] = []
    missed = 0
    i = 0
    while i < len(frames):
        if not gt[frames[i]]:
            i += 1
            continue
        j = i
        while j < len(frames) and gt[frames[j]]:
            j += 1
        seg = frames[i:j]
        hit = next((k for k, fi in enumerate(seg) if pred.get(fi)), None)
        if hit is None:
            missed += 1
        else:
            delays.append(hit)
        i = j
    return {
        "segments": len(delays) + missed,
        "segments_missed": missed,
        "mean_delay_frames": round(float(np.mean(delays)), 2) if delays else None,
    }


def ttc_error(pred_ttc: dict[int, float], gt_ttc: dict[int, float]) -> dict:
    """Ошибка времени до столкновения на кадрах, где оба значения конечны."""
    rel = []
    absd = []
    for fi, g in gt_ttc.items():
        p = pred_ttc.get(fi)
        if p is None or g is None or not (math.isfinite(p) and math.isfinite(g)) or g <= 0:
            continue
        absd.append(abs(p - g))
        rel.append(abs(p - g) / g)
    if not rel:
        return {"ttc_frames": 0}
    return {
        "ttc_frames": len(rel),
        "ttc_mae_s": round(float(np.mean(absd)), 3),
        "ttc_median_rel_err": round(float(np.median(rel)), 3),
    }


def summarize_method(rows: list[dict], labels: dict[int, dict] | None) -> dict:
    """rows: список ObstacleResult.to_row() одного метода."""
    out: dict = {"frames": len(rows)}
    for stage in ("total", "pre", "flow", "infer", "post"):
        key = f"ms_{stage}"
        vals = [r[key] for r in rows[1:] if key in r]     # первый кадр без потока не считаем
        if vals:
            out[f"timing_{stage}"] = timing_stats(vals)
    if labels:
        pred_present = {r["frame"]: bool(r["present"]) for r in rows}
        pred_side = {r["frame"]: r["side"] for r in rows}
        pred_ttc = {r["frame"]: float(r["proximity"]) for r in rows
                    if r.get("proximity_kind") == "ttc_s" and r.get("proximity") not in ("", None)}
        known = {fi: rec for fi, rec in labels.items() if not rec.get("unknown")}
        gt_present = {fi: rec["present"] for fi, rec in known.items()}
        gt_side = {fi: rec["side"] for fi, rec in known.items()}
        gt_ttc = {fi: rec["ttc_s"] for fi, rec in known.items() if rec.get("ttc_s") is not None}
        out.update(presence_metrics(pred_present, gt_present))
        out.update(side_accuracy(pred_side, gt_side, gt_present))
        out.update(detection_delay(pred_present, gt_present))
        if gt_ttc and pred_ttc:
            out.update(ttc_error(pred_ttc, gt_ttc))
    return out
