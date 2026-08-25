import unittest

from plotter_pipeline import DEFAULT_PROFILE, analyze_svg, path_to_points


class PlotterPipelineTests(unittest.TestCase):
    def test_path_parser_handles_cubic(self):
        points = path_to_points("M0,0 C10,0 10,10 20,10")
        self.assertGreaterEqual(len(points), 3)
        self.assertAlmostEqual(points[-1][0], 20.0, places=4)
        self.assertAlmostEqual(points[-1][1], 10.0, places=4)

    def test_machine_plan(self):
        svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="210mm" height="297mm" viewBox="0 0 210 297">
        <path d="M10,10 L30,10"/><path d="M31,10 L50,10"/><path d="M180,280 L190,280"/>
        </svg>'''
        machine_svg, plan = analyze_svg(svg, DEFAULT_PROFILE)
        self.assertTrue(plan.fits_bed)
        self.assertEqual(plan.strokes, 3)
        self.assertIn('data-stroke-order="1"', machine_svg)
        self.assertIn('idraw-plot-plan', machine_svg)


if __name__ == "__main__":
    unittest.main()
