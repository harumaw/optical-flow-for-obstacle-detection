"""UZH-FPV: последовательность left-кадров -> mp4 + метки времени. python data/uzh_fpv/to_video.py indoor_forward_10"""
import csv, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.sources import make_writer
seq = sys.argv[1]; root = Path(__file__).parent / seq
rows = [l.split() for l in open(root / "left_images.txt") if not l.startswith("#")]
ts = np.array([float(r[1]) for r in rows]); files = [root / r[2] for r in rows]
img0 = cv2.imread(str(files[0])); out = root.with_suffix(".mp4")
wr = make_writer(out, (img0.shape[1], img0.shape[0]), 30.0)
with open(root.parent / f"{seq}.timestamps.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["frame", "t"])
    for i, p in enumerate(files):
        wr.write(cv2.imread(str(p))); w.writerow([i, f"{ts[i]-ts[0]:.6f}"])
wr.release()
g = np.array([[float(x) for x in l.split()] for l in open(root / "groundtruth.txt") if not l.startswith("#")])
print(f"{seq}: {len(files)} frames {ts[-1]-ts[0]:.1f}s, trajectory {g[0,0]-ts[0]:.1f}-{g[-1,0]-ts[0]:.1f}s")
