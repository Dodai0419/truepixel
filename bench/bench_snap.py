"""Time pixel-snap's per-cell loop against truepixel's batched cells on the same input, and check they agree.

    python bench/bench_snap.py

loop_cells() is the cell loop of pixel-snap 0.5.2 snap() (MIT, klavdiy-core), copied unchanged.
Only RGB inputs are used, where truepixel's edge-cell rule for transparency does not apply.
"""
import os, statistics, sys, time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import truepixel as tp  # noqa: E402


def loop_cells(arr, grid, mix_threshold=48.0, mix_fraction=0.2):
    has_alpha = arr.shape[2] == 4
    h, w = arr.shape[:2]
    xs, ys = tp._bounds(w, grid.scale_x, grid.offset_x), tp._bounds(h, grid.scale_y, grid.offset_y)
    gh, gw = len(ys) - 1, len(xs) - 1
    out = np.zeros((gh, gw, 4), dtype=np.uint8)
    mixed = np.zeros((gh, gw), dtype=bool)
    if grid.smooth:
        def window(a, b):
            return a + (b - a - 1) // 2, a + (b - a) // 2 + 1
    else:
        def window(a, b):
            return a + (b - a) // 4, b - (b - a) // 4
    for j in range(gh):
        wy0, wy1 = window(ys[j], ys[j + 1])
        for i in range(gw):
            wx0, wx1 = window(xs[i], xs[i + 1])
            cell = arr[wy0:wy1, wx0:wx1].reshape(-1, arr.shape[2]).astype(np.int32)
            med = np.median(cell, axis=0)
            out[j, i, :3] = np.rint(med[:3])
            out[j, i, 3] = (255 if med[3] >= 128 else 0) if has_alpha else 255
            dist = np.sqrt(((cell[:, :3] - med[:3]) ** 2).sum(1))
            mixed[j, i] = (dist > mix_threshold).mean() > mix_fraction
    return out, mixed


def batched_cells(arr, grid):
    res = tp.snap(arr, grid=grid)   # no colours/palette: only the cell step and bookkeeping
    img = res.image if res.image.shape[2] == 4 else np.dstack([res.image, np.full(res.image.shape[:2], 255, np.uint8)])
    return img, res.mixed


def median_ms(fn, runs):
    ts = []
    for _ in range(runs):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1000


if __name__ == "__main__":
    ex = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples")
    messy = np.asarray(Image.open(os.path.join(ex, "messy.png")).convert("RGB"))
    trex = np.asarray(Image.open(os.path.join(ex, "trex.jpg")).convert("RGB"))
    cases = [("messy.png, detected grid", messy, tp.detect_grid(messy)),
             ("trex.jpg 880x672, --scale 6", trex, tp.Grid(6, 6, 0, 0)),
             ("trex.jpg 880x672, --scale 4", trex, tp.Grid(4, 4, 0, 0))]
    for name, arr, grid in cases:
        a, b = loop_cells(arr, grid), batched_cells(arr, grid)
        same = np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
        t_loop, t_new = median_ms(lambda: loop_cells(arr, grid), 5), median_ms(lambda: batched_cells(arr, grid), 5)
        print(f"{name:32s} cells {a[1].size:6d}  identical {same}  pixel-snap loop {t_loop:7.1f} ms  "
              f"truepixel {t_new:6.1f} ms  ({t_loop / t_new:.1f}x)")
