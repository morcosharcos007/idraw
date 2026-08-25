import xml.etree.ElementTree as ET

import cv2
import numpy as np

from stroke_reconstruction import handwriting_to_svg


def synthetic_handwriting():
    image = np.full((500, 900), 255, dtype=np.uint8)
    pts1 = np.array(
        [[80, 320], [140, 240], [210, 300], [290, 180], [390, 260]],
        np.int32,
    )
    pts2 = np.array(
        [[430, 270], [500, 220], [570, 290], [650, 210], [760, 270]],
        np.int32,
    )
    cv2.polylines(image, [pts1], False, 0, 8, cv2.LINE_AA)
    cv2.polylines(image, [pts2], False, 0, 8, cv2.LINE_AA)
    cv2.circle(image, (515, 150), 5, 0, -1, cv2.LINE_AA)
    return image


def test_quality_changes_geometry():
    image = synthetic_handwriting()
    results = [handwriting_to_svg(image, quality=q) for q in (1, 3, 5)]
    stats = [item[1] for item in results]

    for svg, stat in results:
        ET.fromstring(svg)
        assert "<path " in svg
        assert stat["order"] == "geometric-probable"

    assert len({stat["points"] for stat in stats}) == 3


def test_svg_keeps_original_canvas():
    image = synthetic_handwriting()
    svg, _ = handwriting_to_svg(image, quality=3)
    root = ET.fromstring(svg)
    assert root.attrib["viewBox"] == "0 0 900 500"


def test_reconstruction_does_not_force_left_to_right_orientation():
    # A descending stroke is deliberately supplied right-to-left. The
    # ordering code must be allowed to preserve that orientation instead of
    # reversing every stroke by x/y position.
    path = [[300, 100], [250, 140], [200, 190], [150, 250]]
    from stroke_reconstruction import _order_probable_strokes

    ordered = _order_probable_strokes([np.asarray(path, dtype=np.float32)])
    result = np.asarray(ordered[0], dtype=np.float32)

    # With one stroke there is no historical evidence that justifies a
    # left-to-right reversal. The first point therefore remains available.
    assert np.allclose(result[0], np.asarray(path[0], dtype=np.float32))
