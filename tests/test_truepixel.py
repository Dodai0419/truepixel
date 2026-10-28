"""Tests for truepixel. The grid tests are ported from pixel-snap's own suite; the golden tests compare
against output images that pixel-snap and pxl-cleanup committed to their repos (tests/upstream/).

Run:  python -m unittest -q
"""
import os, subprocess, sys, tempfile, unittest

import numpy as np
from PIL import Image, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EX = os.path.join(ROOT, "examples")
UP = os.path.join(ROOT, "tests", "upstream")
sys.path.insert(0, ROOT)
import truepixel as tp  # noqa: E402

PALETTE = np.array([[20, 12, 28], [68, 36, 52], [48, 52, 109], [78, 74, 78],
                    [133, 76, 48], [52, 101, 36], [208, 70, 72], [222, 238, 214]], dtype=np.uint8)


def make_sprite(seed=1, size=16):
    return PALETTE[np.random.default_rng(seed).integers(0, len(PALETTE), size=(size, size))]


def degrade(sprite, scale=7, offset=(3, 5), noise=6, blur=0.8, seed=2):
    big = np.repeat(np.repeat(sprite, scale, axis=0), scale, axis=1)
    oy, ox = offset
    h, w = big.shape[:2]
    canvas = np.zeros((h + oy + 4, w + ox + 4, 3), dtype=np.uint8)
    canvas[:] = PALETTE[0]
    canvas[oy:oy + h, ox:ox + w] = big
    arr = np.asarray(Image.fromarray(canvas).filter(ImageFilter.GaussianBlur(blur))).astype(np.int16)
    arr += np.random.default_rng(seed).integers(-noise, noise + 1, size=arr.shape)
    return np.clip(arr, 0, 255).astype(np.uint8)


def circ(a, b, p):
    d = abs(a - b) % p
    return min(d, p - d)


def rgba(path):
    return np.asarray(Image.open(path).convert("RGBA"))


class Grid(unittest.TestCase):
    def test_finds_scale_and_offset(self):
        for scale in (4, 7, 8, 12):
            with self.subTest(scale=scale):
                g = tp.detect_grid(degrade(make_sprite(), scale=scale, offset=(3, 2)))
                self.assertAlmostEqual(g.scale_x, scale, delta=0.05)
                self.assertAlmostEqual(g.scale_y, scale, delta=0.05)
                self.assertLess(circ(g.offset_x, 2, scale), 0.5)
                self.assertLess(circ(g.offset_y, 3, scale), 0.5)

    def test_palette_returns_exact_sprite(self):
        sprite = make_sprite()
        res = tp.snap(degrade(sprite, scale=7, offset=(3, 5)), palette=PALETTE)
        np.testing.assert_array_equal(res.image[1:17, 1:17, :3], sprite)

    def test_n_colors_keeps_the_partition(self):
        sprite = make_sprite(seed=5)
        res = tp.snap(degrade(sprite, scale=7, offset=(0, 0)), colors=len(PALETTE))
        got = res.image[:16, :16, :3]
        src = np.unique(sprite.reshape(-1, 3), axis=0, return_inverse=True)[1].ravel()
        out = np.unique(got.reshape(-1, 3), axis=0, return_inverse=True)[1].ravel()
        pairs = set(zip(src.tolist(), out.tolist()))
        self.assertEqual(len(pairs), len(set(src.tolist())))
        self.assertEqual(len(pairs), len(set(out.tolist())))

    def test_mixed_cells_are_flagged(self):
        img = degrade(make_sprite(), scale=7, offset=(0, 0), blur=0, noise=0)
        img[7:14, 10:14] = (255, 255, 0)
        res = tp.snap(img, palette=PALETTE)
        self.assertTrue(res.mixed[1, 1])
        self.assertEqual(int(res.mixed.sum()), 1)

    def test_photo_is_not_a_grid(self):
        noise = np.random.default_rng(3).integers(0, 255, size=(96, 96, 3)).astype(np.uint8)
        self.assertEqual(tp.detect_grid(noise).scale_x, 1.0)


class UpstreamGolden(unittest.TestCase):
    def test_pixel_snap_example_is_pixel_identical(self):
        # pixel-snap README: python pixel_snap.py examples/messy.png -o examples/clean.png --palette examples/palette.hex --upscale 8
        src = np.asarray(Image.open(os.path.join(EX, "messy.png")).convert("RGB"))
        res = tp.snap(src, palette=tp.load_palette(os.path.join(EX, "palette.hex")))
        got = np.asarray(Image.fromarray(res.image).resize((128, 128), Image.NEAREST))
        want = np.asarray(Image.open(os.path.join(UP, "pixel-snap_messy_clean.png")).convert("RGB"))
        np.testing.assert_array_equal(got, want)

    def test_pxl_cleanup_background_is_pixel_identical(self):
        got = np.asarray(tp.remove_background(Image.open(os.path.join(EX, "trex.jpg"))))
        want = rgba(os.path.join(UP, "pxl-cleanup_trex_1_nobg.png"))
        self.assertEqual(got.shape, want.shape)
        np.testing.assert_array_equal(got[..., 3], want[..., 3])
        np.testing.assert_array_equal(got, want)

    def test_enclosed_light_areas_are_kept(self):
        a = np.full((40, 40, 3), 255, np.uint8)
        a[5:35, 5:35] = 20           # dark outline ...
        a[10:30, 10:30] = 250        # ... around a near-white belly
        out = np.asarray(tp.remove_background(Image.fromarray(a)))
        self.assertEqual(out.shape[:2], (30, 30))
        self.assertTrue((out[8:22, 8:22, 3] == 255).all())


class Pipeline(unittest.TestCase):
    def test_trex_becomes_a_game_ready_sprite(self):
        out, info = tp.clean(Image.open(os.path.join(EX, "trex.jpg")), scale=6, colors=24)
        a = np.asarray(out)
        self.assertEqual(a.shape, (128, 128, 4))
        self.assertTrue(set(np.unique(a[..., 3]).tolist()) <= {0, 255})
        self.assertLessEqual(len(np.unique(a[a[..., 3] > 0][:, :3], axis=0)), 24)
        rows = np.where(a[..., 3].any(1))[0]
        self.assertEqual(rows.max(), 128 - 4 - 1, "feet sit 4 px above the bottom")
        self.assertEqual(info["sprite"], (114, 93))   # 680/6 and 558/6 leave partial edge cells; unfake floors to 113x92

    def test_sprite_matches_pxl_cleanup_shape(self):
        # pxl-cleanup used the 'unfake' package for this step; compare silhouettes, not exact pixels
        out, _ = tp.clean(Image.open(os.path.join(EX, "trex.jpg")), scale=6, colors=24)
        mine = np.asarray(out)[..., 3] > 0
        theirs = rgba(os.path.join(UP, "pxl-cleanup_trex_4_final.png"))[..., 3] > 0
        iou = (mine & theirs).sum() / (mine | theirs).sum()
        self.assertGreater(iou, 0.85)

    def test_too_big_for_canvas(self):
        with self.assertRaises(ValueError):
            tp.place_on_canvas(Image.new("RGBA", (200, 10)), 128, 4)


class Cli(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, os.path.join(ROOT, "truepixel.py"), *args], capture_output=True, text=True)

    def test_cli_writes_sprite_and_preview(self):
        with tempfile.TemporaryDirectory() as td:
            r = self.run_cli(os.path.join(EX, "trex.jpg"), "--out-dir", td, "--scale", "6")
            self.assertEqual(r.returncode, 0, r.stderr)
            for name in ("trex.sprite.png", "trex.preview.png"):
                self.assertTrue(os.path.exists(os.path.join(td, name)), name)

    def test_cli_messy_matches_pixel_snap(self):
        with tempfile.TemporaryDirectory() as td:
            r = self.run_cli(os.path.join(EX, "messy.png"), "--palette", os.path.join(EX, "palette.hex"),
                             "--keep-background", "--canvas", "0", "--out-dir", td)
            self.assertEqual(r.returncode, 0, r.stderr)
            with Image.open(os.path.join(td, "messy.sprite.png")) as im:
                got = np.asarray(im.convert("RGB").resize((128, 128), Image.NEAREST))
        want = np.asarray(Image.open(os.path.join(UP, "pixel-snap_messy_clean.png")).convert("RGB"))
        np.testing.assert_array_equal(got, want)

    def test_missing_file_is_one_line_and_exit_2(self):
        r = self.run_cli("no-such-file.png")
        self.assertEqual(r.returncode, 2)
        self.assertIn("cannot open", r.stderr)
        self.assertNotIn("Traceback", r.stderr)


if __name__ == "__main__":
    unittest.main()
