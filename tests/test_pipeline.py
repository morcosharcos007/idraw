import importlib.util
import unittest
from pathlib import Path

import cv2
import numpy as np

APP = Path(__file__).resolve().parents[1] / "app.py"
source = APP.read_text(encoding="utf-8")
source = source[:source.index("def render_index")]
source = source.replace("from flask import Flask, Response, abort, render_template, request, url_for", "")
source = source.replace('app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), "templates"))\napp.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024\n', "")
source = source.replace("STATE_DIR.mkdir(parents=True, exist_ok=True)", "")
module_path = APP.parent / "_pipeline_test_runtime.py"
module_path.write_text(source, encoding="utf-8")
spec = importlib.util.spec_from_file_location("pipeline", module_path)
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)
module_path.unlink(missing_ok=True)


class PipelineTests(unittest.TestCase):
    def test_svg_from_simple_strokes(self):
        image = np.full((500, 900, 3), 255, np.uint8)
        cv2.ellipse(image, (260, 260), (120, 75), -12, 180, 520, (20, 20, 20), 7)
        cv2.line(image, (130, 275), (460, 120), (20, 20, 20), 7)
        cv2.ellipse(image, (650, 260), (70, 45), 0, 0, 330, (20, 20, 20), 6)
        processed = pipeline.process(image, 160, 3)
        svg, stats = pipeline.handwriting_to_svg(processed, quality=3)
        pipeline.validate_svg(svg)
        self.assertGreater(stats["paths"], 0)
        self.assertGreater(stats["points"], 10)
        self.assertEqual(svg.count("<path "), stats["paths"])
        self.assertIn('fill="none"', svg)
        self.assertIn('stroke-linecap="round"', svg)

    def test_blank_image_is_rejected(self):
        image = np.full((300, 500, 3), 255, np.uint8)
        processed = pipeline.process(image, 160, 3)
        with self.assertRaises(ValueError):
            pipeline.handwriting_to_svg(processed, quality=3)


if __name__ == "__main__":
    unittest.main()
