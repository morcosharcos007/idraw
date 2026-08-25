import unittest

import cv2
import numpy as np

from stroke_reconstruction import handwriting_to_svg
from plotter_pipeline import DEFAULT_PROFILE, analyze_svg


class Stage2ReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.mask = np.full((500, 900), 255, np.uint8)
        cv2.putText(
            self.mask,
            "Mama",
            (80, 280),
            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            5,
            0,
            5,
            cv2.LINE_AA,
        )

    def test_balanced_reconstruction_is_plotter_svg(self):
        svg, stats = handwriting_to_svg(self.mask, quality=3, ordering="balanced")
        self.assertGreaterEqual(stats["paths"], 1)
        self.assertTrue(stats["bezier"])
        self.assertIn('data-stroke-order="1"', svg)
        self.assertIn("C", svg)

    def test_ordering_modes_are_accepted(self):
        for mode in ("natural", "balanced", "efficient"):
            svg, stats = handwriting_to_svg(self.mask, quality=3, ordering=mode)
            self.assertEqual(stats["order"], mode)
            self.assertIn(f"order={mode}", svg)


class Stage2PlotterTests(unittest.TestCase):
    def test_plan_reports_pen_lifts_and_efficiency(self):
        svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="210mm" height="297mm" viewBox="0 0 210 297">
        <path d="M10,10 L30,10"/><path d="M30.3,10 L50,10"/><path d="M180,280 L190,280"/>
        </svg>'''
        _, plan = analyze_svg(svg, DEFAULT_PROFILE)
        self.assertEqual(plan.strokes, 3)
        self.assertGreaterEqual(plan.pen_lifts, 1)
        self.assertGreater(plan.efficiency_percent, 0)
        self.assertTrue(plan.fits_bed)

    def test_parser_supports_cubic(self):
        svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="210mm" height="297mm" viewBox="0 0 210 297">
        <path d="M0,0 C10,0 10,10 20,10"/>
        </svg>'''
        machine_svg, plan = analyze_svg(svg, DEFAULT_PROFILE)
        self.assertIn('data-stroke-order="1"', machine_svg)
        self.assertEqual(plan.strokes, 1)


if __name__ == "__main__":
    unittest.main()
