import importlib.util
from pathlib import Path
import tempfile
import unittest

from PIL import Image, ImageChops

SPEC = importlib.util.spec_from_file_location("crop_center", Path(__file__).resolve().parents[1]/"experiments/crop_center.py")
crop = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(crop)


class CropTests(unittest.TestCase):
    def test_projection_is_unchanged_after_coordinate_shift(self):
        original = [13625.600752573875, 3504., 2336., .2]
        box = (2102, 1401, 4906, 3271)
        f, cx, cy, k = crop.crop_camera(original, box)
        self.assertEqual([cx, cy], [1402., 935.])
        for x, y in [(-.04, .02), (.03, -.01), (0., 0.)]:
            radial = 1+k*(x*x+y*y)
            raw_u, raw_v = original[0]*x*radial+original[1], original[0]*y*radial+original[2]
            self.assertAlmostEqual(f*x*radial+cx, raw_u-box[0])
            self.assertAlmostEqual(f*y*radial+cy, raw_v-box[1])

    def test_png_crop_preserves_decoded_pixels_and_orientation(self):
        root = Path(__file__).resolve().parents[1]/"results"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root, prefix="crop_test_") as tmp:
            p = Path(tmp)
            self.assertTrue(p.resolve().is_relative_to(root.resolve()))
            im = Image.new("RGB", (24, 16))
            im.putdata([(x*9, y*14, (x+y)*5) for y in range(16) for x in range(24)])
            exif = Image.Exif()
            exif[274] = 8
            exif[272] = "test camera"
            exif[34665] = {37386: 70., 33437: 8., 40962: 24, 40963: 16}
            source, target = p/"original.JPG", p/"crop.PNG"
            im.save(source, quality=95, exif=exif)
            box = (5, 3, 19, 13)
            crop.save_pixel_crop(source, target, box)
            with Image.open(source) as original, Image.open(target) as result:
                self.assertIsNone(ImageChops.difference(original.crop(box).convert("RGB"), result.convert("RGB")).getbbox())
                self.assertEqual(result.size, (14, 10))
                self.assertEqual(result.getexif()[274], 8)
                self.assertEqual(result.getexif().get_ifd(34665)[40962], 14)
                self.assertEqual(result.getexif().get_ifd(34665)[40963], 10)
            with self.assertRaises(FileExistsError):
                crop.save_pixel_crop(source, target, box)


if __name__ == "__main__":
    unittest.main()
