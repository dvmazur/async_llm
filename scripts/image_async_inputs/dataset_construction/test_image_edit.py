import base64
import unittest

from image_edit_test import extract_image, metadata_only, parse_solve


class ImageResponseTest(unittest.TestCase):
    def test_fenced_json(self):
        value, repaired = parse_solve('```json\n{"answer":"45"}\n```')
        self.assertEqual(value["answer"], "45")
        self.assertTrue(repaired)

    def test_latex_escape_repair_preserves_answer(self):
        value, repaired = parse_solve(r'{"answer":"45","solution":"x^\circ"}')
        self.assertTrue(repaired)
        self.assertEqual(value["answer"], "45")
        self.assertEqual(value["solution"], r"x^\circ")

    def test_extract_and_strip_image_without_mutating_response(self):
        payload = b"image fixture bytes"
        response = {"choices": [{"message": {"images": [{"image_url": {
            "url": "data:image/png;base64," + base64.b64encode(payload).decode()}}]}}],
            "usage": {"cost": 0.1}}
        self.assertEqual(extract_image(response), payload)
        stripped = metadata_only(response)
        self.assertEqual(stripped["usage"]["cost"], 0.1)
        self.assertEqual(stripped["choices"][0]["message"]["images"], [{"saved_separately": True}])
        self.assertEqual(extract_image(response), payload)

    def test_external_urls_are_not_downloaded(self):
        with self.assertRaises(ValueError):
            extract_image({"choices": [{"message": {"images": [{"image_url": {
                "url": "https://example.org/image.png"}}]}}]})
