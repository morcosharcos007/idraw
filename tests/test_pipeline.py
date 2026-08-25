import unittest

import cv2
import numpy as np

import wsgi


pipeline = wsgi.app_module


class PipelineTests(unittest.TestCase):
    def test_svg_from_simple_strokes_uses_production_reconstruction(self):
        image = np.full((500, 900, 3), 255, np.uint8)
        cv2.ellipse(
            image, (260, 260), (120, 75), -12, 180, 520, (20, 20, 20), 7
        )
        cv2.line(image, (130, 275), (460, 120), (20, 20, 20), 7)
        cv2.ellipse(
            image, (650, 260), (70, 45), 0, 0, 330, (20, 20, 20), 6
        )

        processed = pipeline.process(image, 160, 3)
        svg, stats = pipeline.handwriting_to_svg(processed, quality=3)
        pipeline.validate_svg(svg)

        self.assertGreater(stats["paths"], 0)
        self.assertGreater(stats["points"], 10)
        self.assertEqual(svg.count("<path "), stats["paths"])
        self.assertIn('fill="none"', svg)
        self.assertIn('stroke-linecap="round"', svg)
        self.assertEqual(stats["order"], "geometric-probable")

    def test_blank_image_is_rejected_by_production_reconstruction(self):
        image = np.full((300, 500, 3), 255, np.uint8)
        processed = pipeline.process(image, 160, 3)

        with self.assertRaises(ValueError):
            pipeline.handwriting_to_svg(processed, quality=3)


if __name__ == "__main__":
    unittest.main()
