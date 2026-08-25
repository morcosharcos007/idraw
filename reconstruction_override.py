"""Backward-compatible import shim.

There is intentionally only one reconstruction implementation. Production,
tests and any legacy import should all resolve to stroke_reconstruction.py.
"""

from stroke_reconstruction import (  # noqa: F401
    QUALITY,
    handwriting_to_svg,
)
