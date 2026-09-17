import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from PIL import Image

from package_dataset import digest, model_context, package, validate_package


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "source"
        self.root.mkdir()
        images = {}
        for state, color in (("before", "red"), ("after", "blue")):
            data = io.BytesIO()
            Image.new("RGB", (8, 6), color).save(data, format="PNG")
            (self.root / f"{state}.png").write_bytes(data.getvalue())
            images[state] = {"path": f"{state}.png", "sha256": digest(data.getvalue())}
        self.inp = {"id": "sample", "image_before": "before.png", "image_after": "after.png",
                    "text_shard_1": "Question?", "text_shard_2": "Recheck the corrected picture."}
        ann = {"id": "sample", "status": "accepted", "group_id": "source-hash",
               "source": {"dataset": "fixture", "source_id": "1", "source_split": "test",
                          "category": "test", "difficulty_group": "reasoning",
                          "question": "Question?", "answer": "secret-after", "original": {}},
               "review": {"decision": "accept", "answer_before": "secret-before",
                          "answer_after": "secret-after"},
               "proposal": {"solution_before": "secret-solution"},
               "before_sha256": "raw-before", "after_sha256": "raw-after",
               "normalization": {"images": images, "target_size": [8, 6],
                                 "processor_parity": {"status": "pending_model_selection"}}}
        (self.root / "inputs.jsonl").write_text(json.dumps(self.inp) + "\n")
        (self.root / "annotations.jsonl").write_text(json.dumps(ann) + "\n")
        self.manifest()
        self.output = Path(self.temp.name) / "dataset.parquet"

    def manifest(self):
        value = {"count": 1, **{f"{name}_sha256": digest(
            (self.root / f"{name}.jsonl").read_bytes()) for name in ("inputs", "annotations")}}
        (self.root / "manifest.json").write_text(json.dumps(value))

    def test_standalone_roundtrip_and_context(self):
        package(self.root, self.output)
        shutil.rmtree(self.root)  # The temporary fixture is no longer available.
        row = validate_package(self.output)[0]
        for stage, pixel in ((1, (255, 0, 0)), (2, (0, 0, 255))):
            context = model_context(row, stage)
            self.assertEqual(set(context), {"image", "text"})
            self.assertEqual(context["image"].getpixel((0, 0)), pixel)
            self.assertNotIn("secret", context["text"])
        self.assertEqual(model_context(row, 2)["text"],
                         self.inp["text_shard_1"] + "\n\n" + self.inp["text_shard_2"])
        with self.assertRaises(ValueError):
            model_context(row, 3)

    def test_refuses_overwrite(self):
        package(self.root, self.output)
        original = self.output.read_bytes()
        with self.assertRaises(FileExistsError):
            package(self.root, self.output)
        self.assertEqual(original, self.output.read_bytes())

    def test_detects_manifest_tampering(self):
        with (self.root / "inputs.jsonl").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            package(self.root, self.output)
        self.assertFalse(self.output.exists())

    def test_detects_image_tampering(self):
        (self.root / "before.png").write_bytes(b"invalid")
        with self.assertRaisesRegex(ValueError, "checksum"):
            package(self.root, self.output)

    def test_detects_duplicate_ids(self):
        (self.root / "inputs.jsonl").write_text((json.dumps(self.inp) + "\n") * 2)
        self.manifest()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            package(self.root, self.output)


if __name__ == "__main__":
    unittest.main()
