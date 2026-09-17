"""CPU-only integration checks using explicit synthetic fixtures, not benchmark data."""

import json
import tempfile
import unittest
from pathlib import Path

import pipeline


class PipelineTest(unittest.TestCase):
    def test_review_gate_export_and_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "records").mkdir()
            pilot = Path(__file__).parent / "pilot_v1"
            manifest = root / "source.jsonl"
            source = {"dataset": "synthetic_test_fixture", "source_id": "geometry",
                      "source_split": "fixture", "category": "geometry",
                      "question": "Find the square perimeter.", "answer": "68",
                      "image": str(pilot / "01_geometry_after.png"),
                      "source_url": "local:test-fixture", "license": "test-only"}
            manifest.write_text(json.dumps(source) + "\n")
            pipeline.import_sources(root, manifest)
            pipeline.import_sources(root, manifest)
            self.assertEqual(len(pipeline.records(root)), 1)
            sample_id = pipeline.records(root)[0]["id"]
            with self.assertRaisesRegex(ValueError, "No accepted"):
                pipeline.export(root, root / "export")
            proposal = {"edit_type": "length_label", "changed_fact_before": "6",
                        "changed_fact_after": "15", "expected_answer_before": "40",
                        "solution_before": "sqrt(64+36)*4=40",
                        "solution_after": "sqrt(64+225)*4=68",
                        "editing_method": "existing programmatic fixture",
                        "reasoning_change": "Recompute diagonal then perimeter"}
            pipeline.transition(root, sample_id, "propose", proposal)
            with self.assertRaisesRegex(ValueError, "identical"):
                pipeline.transition(root, sample_id, "attach", {
                    "image": source["image"], "approval_note": "test fixture"})
            pipeline.transition(root, sample_id, "attach", {
                "image": str(pilot / "01_geometry_before.png"),
                "approval_note": "Previously generated synthetic fixture only"})
            review = {"reviewer": "test", "notes": "Fixture checks only",
                      "answer_before": "40", "answer_after": "68", "decision": "accept"}
            with self.assertRaisesRegex(ValueError, "normalization first"):
                pipeline.transition(root, sample_id, "review", review)
            pipeline.normalize_record(root, sample_id)
            with self.assertRaisesRegex(ValueError, "isolated_edit"):
                pipeline.transition(root, sample_id, "review", review)
            review.update(dict.fromkeys(("isolated_edit", "readable", "both_solvable",
                                         "visual_required", "answers_distinct",
                                         "source_terms_checked", "same_dimensions"), True))
            pipeline.transition(root, sample_id, "review", review)
            pipeline.report(root)
            pipeline.export(root, root / "export")
            exported = json.loads((root / "export/inputs.jsonl").read_text())
            self.assertEqual(set(exported), {"id", "image_before", "image_after",
                                             "text_shard_1", "text_shard_2"})
            self.assertIn("/normalized/", exported["image_before"])
            with self.assertRaisesRegex(ValueError, "already exists"):
                pipeline.export(root, root / "export")
            record = pipeline.records(root)[0]
            (root / record["before_image"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Asset changed"):
                pipeline.export(root, root / "export2")

    def test_invalid_ids(self):
        with self.assertRaises(ValueError):
            pipeline.record_path(Path("/tmp"), "../../outside")


if __name__ == "__main__":
    unittest.main()
