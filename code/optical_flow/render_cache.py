"""Рендер всех сцен в кэш (data/scenes_cache) параллельно. Запускается один раз.

    python optical_flow/render_cache.py [--workers 6]
"""
from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optical_flow.scenes import default_suite, load_cached  # noqa: E402

CACHE = Path(__file__).resolve().parents[1] / "data" / "scenes_cache"


def _one(name: str) -> str:
    from optical_flow.scenes import scenario_by_name
    t0 = time.perf_counter()
    frames, ts, labels = load_cached(scenario_by_name(name), CACHE)
    return f"{name}: {len(frames)} frames, {sum(l['present'] for l in labels)} with obstacle, {time.perf_counter() - t0:.0f} s"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    names = [s.name for s in default_suite()]
    with Pool(args.workers) as pool:
        for line in pool.imap_unordered(_one, names):
            print(line, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
