#!/usr/bin/env python3
"""truepixel: turn AI "pixel art" into a real, game-ready sprite.

    python truepixel.py image.png [--scale N] [--colors 24 | --palette p.hex] [--out-dir DIR]
    python truepixel.py --demo

Removes a plain light background, finds the true pixel grid (fractional sizes included), rebuilds one real
pixel per cell, snaps colours, and bottom-centres the sprite on a square transparent canvas.
Writes <name>.sprite.png (true size) and <name>.preview.png (before | after, zoomed). numpy + Pillow only.

Grid detection and cell snapping are ported from pixel-snap (MIT); background removal and canvas placement
are rewritten from pxl-cleanup (MIT) without scipy and without the unfake package.
"""
import argparse, os, sys
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class Grid:
    scale_x: float
    scale_y: float
    offset_x: float
    offset_y: float
    smooth: bool = False   # smooth upscale: the true colour is only at the cell centre
    warning: str = ""


@dataclass
class Result:
    image: np.ndarray      # one pixel per cell, uint8 RGB or RGBA
    mixed: np.ndarray      # bool per cell: the source broke its own grid there
    grid: Grid
    xs: list
    ys: list
    colors: np.ndarray


# ---------- stage 1: background (pxl-cleanup) ----------

def _spread(mask, reach, axis):
    """Grow reach to every whole run of mask pixels it touches along one axis."""
    m, r = (mask, reach) if axis == 1 else (mask.T, reach.T)
    start = m & ~np.pad(m, ((0, 0), (1, 0)))[:, :-1]
    ids = np.cumsum(start.ravel()).reshape(m.shape) * m
    hit = np.zeros(ids.max() + 1, bool)
    hit[ids[r & m]] = True
    hit[0] = False
    out = hit[ids]
    return out if axis == 1 else out.T


def border_connected(mask):
    """Pixels of mask 4-connected to the image border (scipy ndimage.label + edge labels, without scipy)."""
    reach = np.zeros_like(mask)
    for sl in (np.s_[0, :], np.s_[-1, :], np.s_[:, 0], np.s_[:, -1]):
        reach[sl] = mask[sl]
    n = -1
    while n != reach.sum():
        n = reach.sum()
        reach = _spread(mask, _spread(mask, reach, 1), 0)
    return reach


def _dilate(m):
    out = m.copy()
    out[1:] |= m[:-1]; out[:-1] |= m[1:]; out[:, 1:] |= m[:, :-1]; out[:, :-1] |= m[:, 1:]
    return out


def remove_background(img, ground_band=0.80):
    """Remove a near-white background and a brown ground patch that touch the border, then crop.
    The sprite's dark outline stops the fill, so light areas inside it (teeth, eyes, a pale belly) stay."""
    rgb = np.array(img.convert("RGB")).astype(np.int16)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx, mn = rgb.max(-1), rgb.min(-1)
    h, w = mn.shape
    white = (mn > 215) & (mx - mn < 30)
    rows = np.arange(h)[:, None].repeat(w, 1)
    ground = (r > g) & (g > b) & (r - b > 35) & (mx > 70) & (rows > int(h * ground_band))
    background = border_connected(white | ground)
    background |= _dilate(_dilate(background)) & (mn > 150)   # the 1-2 px light JPEG halo
    alpha = np.where(background, 0, 255).astype(np.uint8)
    if not alpha.any():
        raise ValueError("background removal left nothing; is the image all background? Try --keep-background")
    out = np.dstack([rgb.astype(np.uint8), alpha])
    ys, xs = np.where(alpha > 0)
    return Image.fromarray(out[ys.min():ys.max() + 1, xs.min():xs.max() + 1], "RGBA")


# ---------- stage 2: grid (pixel-snap) ----------

def _edge_profile(rgb, axis, order=1):
    """Mean colour jump (order=1) or bend (order=2, smooth upscales) between neighbouring columns/rows."""
    d = np.abs(np.diff(rgb.astype(np.int32), n=order, axis=axis)).sum(axis=2)
    return d.mean(axis=0 if axis == 1 else 1)


def _coherence(edge, pos, periods):
    out = np.empty(len(periods), dtype=np.complex128)
    for k in range(0, len(periods), 512):
        p = periods[k:k + 512, None]
        out[k:k + 512] = (edge[None, :] * np.exp(2j * np.pi * pos[None, :] / p)).sum(1)
    return np.abs(out) / edge.sum(), np.angle(out)


def _best_period(edge, max_period=None, around=None):
    """(period, offset, coherence): cell borders add up in phase at the true period and its divisors,
    so the answer is the LARGEST strong peak of the edge spectrum."""
    n = len(edge) + 1
    if edge.max() <= 0 or n < 8:
        return 1.0, 0.0, 0.0
    edge = edge - np.median(edge)
    edge[edge < 0] = 0
    if edge.sum() <= 0:
        return 1.0, 0.0, 0.0
    pos = np.arange(1, n, dtype=np.float64)
    hi = min(max_period or n / 4, n / 4)
    lo = 2.0
    if around:
        lo, hi = max(2.0, around[0]), min(hi, around[1])
    if hi <= lo:
        return 1.0, 0.0, 0.0
    periods = np.exp(np.arange(np.log(lo), np.log(hi), 0.001))
    coh, _ = _coherence(edge, pos, periods)
    best = coh.max()
    if best < max(0.3, 5.5 / np.sqrt(n)):   # pixel-snap: noise reaches ~4.9/sqrt(n)
        return 1.0, 0.0, 0.0
    peaks = [i for i in range(1, len(coh) - 1)
             if coh[i] >= coh[i - 1] and coh[i] >= coh[i + 1] and coh[i] >= 0.8 * best]
    i = max(peaks) if peaks else int(coh.argmax())
    fine = np.linspace(periods[max(i - 2, 0)], periods[min(i + 2, len(periods) - 1)], 81)
    fcoh, fang = _coherence(edge, pos, fine)
    j = int(fcoh.argmax())
    p = float(fine[j])
    return p, float((fang[j] / (2 * np.pi) * p) % p), float(fcoh[j])


def _blur(rgb, sigma=1.0):
    """Separable Gaussian: keeps the low-frequency bends of a smooth upscale, drops JPEG block seams."""
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    out = rgb.astype(np.float64)
    for axis in (0, 1):
        pad = [(r, r) if a == axis else (0, 0) for a in range(out.ndim)]
        p = np.pad(out, pad, mode="reflect")
        out = sum(k[i] * np.take(p, np.arange(i, i + out.shape[axis]), axis=axis) for i in range(len(k)))
    return out


def _axis_grid(rgb, axis, max_scale, soft=None, around=None):
    p1, o1, c1 = _best_period(_edge_profile(rgb, axis), max_scale, around)
    p2, o2, c2 = _best_period(_edge_profile(rgb if soft is None else soft, axis, order=2), max_scale, around)
    if p2 > 1 and c2 > c1:   # bends sit on cell centres; the border is half a cell back
        return p2, (o2 + 0.5 - p2 / 2) % p2, True, c2
    return p1, o1, False, c1


def cell_fit(rgb, px, py, ox, oy):
    """Share of the colour variance the cells' own means explain (1 = every cell flat)."""
    a = np.asarray(rgb, dtype=np.float64)[..., :3]
    h, w = a.shape[:2]
    xs, ys = _bounds(w, px, ox), _bounds(h, py, oy)
    lx = np.searchsorted(xs, np.arange(w), side="right") - 1
    ly = np.searchsorted(ys, np.arange(h), side="right") - 1
    flat = (ly[:, None] * (len(xs) + 1) + lx[None, :]).ravel()
    n = np.bincount(flat).astype(np.float64)
    total = ((a - a.reshape(-1, 3).mean(0)) ** 2).sum()
    if total <= 0:
        return 1.0
    within = 0.0
    for c in range(3):
        v = a[..., c].ravel()
        s, s2 = np.bincount(flat, weights=v), np.bincount(flat, weights=v * v)
        within += (s2 - np.where(n > 0, s * s / np.maximum(n, 1), 0)).sum()
    return float(1 - within / total)


GRID_FIT = 0.75     # pixel-snap, measured: upscaled sprites >= 0.836, photos <= 0.73
SAME_PIXEL = 1.15   # axes within 15 % are one square pixel found twice


def detect_grid(img, max_scale=None, hint=None, square=True):
    rgb = np.asarray(img)[..., :3]
    soft = _blur(rgb)
    near = (hint / SAME_PIXEL, hint * SAME_PIXEL) if hint else None
    px, ox, sx, cx = _axis_grid(rgb, 1, max_scale, soft, near)
    py, oy, sy, cy = _axis_grid(rgb, 0, max_scale, soft, near)
    found = (px, py, ox, oy)
    warning = ""
    if square and (px > 1) != (py > 1):
        p = max(px, py)
        axis = 0 if px > 1 else 1
        q, o, s, _ = _axis_grid(rgb, axis, max_scale, soft, (p * 0.98, p * 1.02))
        o = o if q > 1 else 0.0
        if axis == 0:
            py, oy, sy = p, o, s
        else:
            px, ox, sx = p, o, s
        warning = (f"the {'y' if axis == 0 else 'x'} axis showed no grid; used the other axis's pixel size "
                   f"{p:.3g} for both (square pixels)")
    elif px > 1 and py > 1 and max(px, py) / min(px, py) > 1.03:
        r = max(px, py) / min(px, py)
        if square and r <= SAME_PIXEL:
            p, keep = (px, "x") if cx >= cy else (py, "y")
            q, o, s, _ = _axis_grid(rgb, 0 if keep == "x" else 1, max_scale, soft, (p * 0.98, p * 1.02))
            was = (px, py)
            if keep == "x":
                py, oy = p, (o if q > 1 else oy)
            else:
                px, ox = p, (o if q > 1 else ox)
            warning = (f"axes disagreed: x {was[0]:.3g}, y {was[1]:.3g}; used square {p:.3g} from the stronger "
                       f"{keep} axis. Check it, or pass --scale")
        else:
            warning = (f"pixels are not square: x {px:.3g}, y {py:.3g} (aspect 1:{r:.2g}); kept as found. "
                       f"Pass --scale to force one size")
    if px > 1 or py > 1:
        fx, fy, fox, foy = found if found[0] > 1 and found[1] > 1 else (px, py, ox, oy)
        fit = cell_fit(rgb, fx, fy, fox, foy)
        if fit < GRID_FIT:
            p = max(px, py)
            hint = (f"if this IS pixel art (1-px dithering through a smooth upscale fits poorly), pass --scale "
                    f"{round(p)}" if fit >= 0.6 and abs(p - round(p)) < 0.1 else "Pass --scale if you know the pixel size")
            return Grid(1.0, 1.0, 0.0, 0.0, False,
                        f"no pixel grid: the best grid found ({px:.3g} x {py:.3g} px) explains only {fit:.0%} "
                        f"of the colour, so this looks like a photo or a painting; the output is the input, 1:1. {hint}")
    if px <= 1 and py <= 1:
        warning = "no grid found: this does not look like an upscaled pixel image; the output is the input, 1:1"
    return Grid(px, py, ox, oy, sx or sy, warning)


def _bounds(n, scale, offset):
    if scale <= 1:
        return list(range(n + 1))
    k0 = int(np.floor(-offset / scale))
    starts = (offset + scale * np.arange(k0, k0 + int(n / scale) + 3))
    starts = [int(round(x)) for x in starts if 1 <= round(x) <= n - 1]
    return sorted(set([0] + starts + [n]))


# ---------- stage 3: colours (pixel-snap) ----------

def load_palette(path):
    """A .hex file (one colour per line) or a GIMP .gpl palette."""
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    cols = []
    if lines and lines[0].strip() == "GIMP Palette":
        for ln in lines[1:]:
            ln = ln.strip()
            if ln and not ln.startswith("#") and ":" not in ln:
                cols.append([int(v) for v in ln.split()[:3]])
    else:
        for ln in lines:
            ln = ln.split(";")[0].strip().lstrip("#")
            if ln:
                if len(ln) != 6:
                    raise ValueError(f"bad hex colour: {ln!r}")
                cols.append([int(ln[i:i + 2], 16) for i in (0, 2, 4)])
    if not cols:
        raise ValueError(f"no colours in {path}")
    return np.array(cols, dtype=np.uint8)


def _nearest(colors, palette):
    return ((colors.astype(np.int32)[:, None, :] - palette.astype(np.int32)[None]) ** 2).sum(2).argmin(1)


def _kmeans(colors, k, iters=30):
    """Deterministic k-means with farthest-point init."""
    pts = colors.astype(np.float64)
    uniq = np.unique(colors, axis=0)
    if len(uniq) <= k:
        return uniq.astype(np.uint8)
    centers = [np.median(pts, axis=0)]
    for _ in range(k - 1):
        d = np.min([((pts - c) ** 2).sum(1) for c in centers], axis=0)
        centers.append(pts[d.argmax()])
    centers = np.array(centers)
    for _ in range(iters):
        lab = ((pts[:, None, :] - centers[None]) ** 2).sum(2).argmin(1)
        new = np.array([pts[lab == i].mean(0) if (lab == i).any() else centers[i] for i in range(k)])
        if np.allclose(new, centers):
            break
        centers = new
    return np.clip(np.rint(centers), 0, 255).astype(np.uint8)


def snap(img, colors=None, palette=None, grid=None, mix_threshold=48.0, mix_fraction=0.2):
    """One colour per cell, read from the cell centre (borders hold the blur), then snap colours."""
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=2)
    has_alpha = arr.shape[2] == 4
    grid = grid or detect_grid(arr)
    h, w = arr.shape[:2]
    xs, ys = _bounds(w, grid.scale_x, grid.offset_x), _bounds(h, grid.scale_y, grid.offset_y)
    gh, gw = len(ys) - 1, len(xs) - 1
    out = np.zeros((gh, gw, 4), dtype=np.uint8)
    mixed = np.zeros((gh, gw), dtype=bool)
    if grid.smooth:
        window = lambda a, b: (a + (b - a - 1) // 2, a + (b - a) // 2 + 1)
    else:
        window = lambda a, b: (a + (b - a) // 4, b - (b - a) // 4)
    wy = [window(ys[j], ys[j + 1]) for j in range(gh)]
    wx = [window(xs[i], xs[i + 1]) for i in range(gw)]
    # cells come in a few window sizes: one batched median per size instead of one call per cell
    for hh in {b - a for a, b in wy}:
        J = [j for j in range(gh) if wy[j][1] - wy[j][0] == hh]
        rows = np.array([wy[j][0] for j in J])[:, None] + np.arange(hh)
        for ww in {b - a for a, b in wx}:
            I = [i for i in range(gw) if wx[i][1] - wx[i][0] == ww]
            cols = np.array([wx[i][0] for i in I])[:, None] + np.arange(ww)
            cells = arr[rows[:, :, None, None], cols[None, None, :, :]].astype(np.int32)   # (J, hh, I, ww, C)
            cells = cells.transpose(0, 2, 1, 3, 4).reshape(len(J), len(I), hh * ww, -1)
            med = np.median(cells, axis=2)
            dist = np.sqrt(((cells[..., :3] - med[:, :, None, :3]) ** 2).sum(-1))
            mix = (dist > mix_threshold).mean(-1) > mix_fraction
            if has_alpha:   # an edge cell: the removed background must not tint it
                for a, b in zip(*np.nonzero((med[..., 3] >= 128) & (cells[..., 3] < 128).any(-1))):
                    cell = cells[a, b][cells[a, b, :, 3] >= 128]
                    med[a, b] = np.median(cell, axis=0)
                    mix[a, b] = (np.sqrt(((cell[:, :3] - med[a, b, :3]) ** 2).sum(1)) > mix_threshold).mean() > mix_fraction
            jj, ii = np.ix_(J, I)
            out[jj, ii, :3] = np.rint(med[..., :3])
            out[jj, ii, 3] = np.where(med[..., 3] >= 128, 255, 0) if has_alpha else 255
            mixed[jj, ii] = mix
    opaque = out[..., 3] > 0
    cols = out[..., :3][opaque]
    if opaque.any() and palette is not None:
        palette = np.asarray(palette, dtype=np.uint8)
        out[..., :3][opaque] = palette[_nearest(cols, palette)]
    elif opaque.any() and colors:
        centers = _kmeans(cols, colors)
        out[..., :3][opaque] = centers[_nearest(cols, centers)]
    out[..., :3][~opaque] = 0
    used = np.unique(out[..., :3][opaque], axis=0) if opaque.any() else np.zeros((0, 3), np.uint8)
    return Result(out if has_alpha else out[..., :3], mixed, grid, xs, ys, used)


# ---------- stage 4: canvas (pxl-cleanup) ----------

def place_on_canvas(img, size=128, margin=4):
    """Bottom-centre the sprite so every sprite's feet land `margin` px above the bottom edge."""
    if img.width > size or img.height > size - margin:
        raise ValueError(f"sprite is {img.width}x{img.height}, too big for a {size}px canvas with a {margin}px "
                         "margin. Use a larger --scale or --canvas")
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(img, ((size - img.width) // 2, size - margin - img.height))
    return canvas


def clean(img, scale=None, colors=24, palette=None, canvas=128, margin=4, ground_band=0.80, keep_background=False):
    """The whole pipeline. Returns (RGBA image, info dict)."""
    src = img.convert("RGBA")
    if not keep_background and np.asarray(src)[..., 3].min() == 255:
        src = remove_background(src, ground_band)
    arr = np.asarray(src)
    grid = Grid(scale, scale, 0.0, 0.0) if scale else detect_grid(arr)
    res = snap(arr, colors=None if palette is not None else colors, palette=palette, grid=grid)
    sprite = Image.fromarray(res.image if res.image.shape[2] == 4 else np.dstack([res.image, np.full(res.image.shape[:2], 255, np.uint8)]), "RGBA")
    if sprite.getbbox():
        sprite = sprite.crop(sprite.getbbox())
    info = {"input": img.size, "area": src.size, "pixel": (round(grid.scale_x, 3), round(grid.scale_y, 3)), "sprite": sprite.size,
            "colors": len(res.colors), "mixed": int(res.mixed.sum()), "warning": grid.warning}
    return (place_on_canvas(sprite, canvas, margin) if canvas else sprite), info


def preview(src, sprite, height=384):
    """Before | after: the source scaled to `height`, the sprite zoomed with nearest neighbour on a checkerboard."""
    a = src.convert("RGB")
    a = a.resize((max(1, round(a.width * height / a.height)), height), Image.LANCZOS)
    z = max(1, height // sprite.height)
    b = sprite.resize((sprite.width * z, sprite.height * z), Image.NEAREST)
    yy, xx = np.mgrid[0:b.height, 0:b.width]
    board = np.where(((yy // 16 + xx // 16) % 2)[..., None] == 0, 236, 250).astype(np.uint8).repeat(3, 2)
    bg = Image.fromarray(board, "RGB")
    bg.paste(b, (0, 0), b)
    out = Image.new("RGB", (a.width + bg.width + 24, max(a.height, bg.height)), (255, 255, 255))
    out.paste(a, (0, 0))
    out.paste(bg, (a.width + 24, 0))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="truepixel", description="Turn AI pixel art into a real, game-ready sprite.")
    ap.add_argument("inputs", nargs="*")
    ap.add_argument("--out-dir", help="default: next to each input")
    ap.add_argument("--scale", type=float, help="the fake pixel size in the source (default: detect it)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--colors", type=int, default=24, help="palette size (default 24, 0 = keep every cell's colour)")
    g.add_argument("--palette", help="snap to a palette file (.hex or GIMP .gpl)")
    ap.add_argument("--canvas", type=int, default=128, help="square canvas size, 0 = no canvas (default 128)")
    ap.add_argument("--margin", type=int, default=4, help="empty pixels below the feet (default 4)")
    ap.add_argument("--ground-band", type=float, default=0.80, help="brown pixels below this share of the height are ground")
    ap.add_argument("--keep-background", action="store_true", help="do not remove the background")
    ap.add_argument("--demo", action="store_true", help="run on the bundled examples into ./dist")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(errors="replace")
    here = os.path.dirname(os.path.abspath(__file__))
    jobs = [(p, {}) for p in a.inputs]
    if a.demo:
        ex = os.path.join(here, "examples")
        jobs = [(os.path.join(ex, "trex.jpg"), {"scale": 6}),
                (os.path.join(ex, "messy.png"), {"palette": load_palette(os.path.join(ex, "palette.hex")),
                                                 "keep_background": True, "canvas": 0})]
        a.out_dir = a.out_dir or "dist"
    if not jobs:
        ap.print_help(); return 2
    palette = load_palette(a.palette) if a.palette else None
    failed = 0
    for path, over in jobs:
        try:
            src = Image.open(path); src.load()
        except (OSError, ValueError) as e:
            print(f"truepixel: cannot open {path}: {getattr(e, 'strerror', None) or e}", file=sys.stderr)
            failed += 1; continue
        opts = dict(scale=a.scale, colors=a.colors or None, palette=palette, canvas=a.canvas, margin=a.margin,
                    ground_band=a.ground_band, keep_background=a.keep_background)
        opts.update(over)
        try:
            out, info = clean(src, **opts)
        except ValueError as e:
            print(f"truepixel: {path}: {e}", file=sys.stderr)
            failed += 1; continue
        stem = os.path.join(a.out_dir or os.path.dirname(os.path.abspath(path)), os.path.splitext(os.path.basename(path))[0])
        os.makedirs(os.path.dirname(stem), exist_ok=True)
        out.save(stem + ".sprite.png")
        preview(src, out.crop(out.getbbox()) if out.getbbox() else out).save(stem + ".preview.png")
        crop = f" (sprite area {info['area'][0]}x{info['area'][1]})" if info["area"] != info["input"] else ""
        print(f"{os.path.basename(path)}: {info['input'][0]}x{info['input'][1]}{crop} -> {info['sprite'][0]}x{info['sprite'][1]} "
              f"sprite (pixel {info['pixel'][0]:g}), {info['colors']} colours, {info['mixed']} mixed cells")
        if info["warning"]:
            print(f"  WARNING {info['warning']}")
        print(f"  wrote {stem}.sprite.png and {stem}.preview.png")
    return 2 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
