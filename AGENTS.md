# truepixel (for Codex and other agents)

Use this when the user has an AI-generated or upscaled "pixel art" image and wants a real sprite: one pixel per cell, a small palette, transparent background, bottom-centred on a square canvas.

1. Run from the repo root (needs `pip install numpy pillow`):
   ```bash
   python truepixel.py <image> --out-dir <dir>
   ```
   It writes `<dir>/<name>.sprite.png` (the real sprite) and `<dir>/<name>.preview.png` (before | after, zoomed).
2. Read the first output line: `880x672 (sprite area 680x558) -> 114x93 sprite (pixel 6), 24 colours, ...`.
   - If it prints `WARNING no pixel grid`, the grid was not found. Measure one fake pixel's width (or try 4, 6, 8) and run again with `--scale N`.
   - A sprite that looks too big or too small means the pixel size is wrong: raise `--scale` for a smaller sprite.
3. Open the preview image and check: outline closed, eyes and small accents kept, no white halo, nothing cut off.
4. Options: `--colors 15`, `--palette p.hex` (.hex or GIMP .gpl), `--canvas 0` (tight crop), `--canvas 64 --margin 2`, `--keep-background`, `--ground-band 1.0` (keep brown ground).
5. Report the sprite size, colour count and the preview path. Do not call it done while step 3 fails.
