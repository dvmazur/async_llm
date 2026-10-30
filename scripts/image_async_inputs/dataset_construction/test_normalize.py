import tempfile
import unittest
from pathlib import Path

from PIL import Image

from normalize_pair import normalize
from pipeline import digest


class NormalizeTest(unittest.TestCase):
    def test_size_rgb_alpha_metadata_and_raw_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            before, after = root / "before.jpg", root / "after.png"
            Image.new("RGB", (200, 100), "blue").save(before, dpi=(300, 300))
            Image.new("RGBA", (100, 50), (255, 0, 0, 0)).save(after, dpi=(150, 150))
            hashes = digest(before), digest(after)
            result = normalize(before, after, root / "normalized")
            for stage in ("before", "after"):
                with Image.open(root / "normalized" / f"{stage}.png") as image:
                    self.assertEqual(image.size, (100, 50))
                    self.assertEqual(image.mode, "RGB")
                    self.assertEqual(image.info, {})
                    if stage == "after":
                        self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))
            self.assertEqual(hashes, (digest(before), digest(after)))
            self.assertEqual(result["processor_parity"]["status"], "pending_model_selection")
            with self.assertRaisesRegex(ValueError, "output exists"):
                normalize(before, after, root / "normalized")

    def test_large_aspect_change_rejected_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (200, 100)).save(root / "before.png")
            Image.new("RGB", (100, 100)).save(root / "after.png")
            with self.assertRaisesRegex(ValueError, "Aspect ratio mismatch"):
                normalize(root / "before.png", root / "after.png", root / "out")
            self.assertFalse((root / "out").exists())
