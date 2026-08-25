"""Production entry point.

The original Flask app remains the owner of routes and state handling.  The
reconstruction layer is injected here so the deployed service can be upgraded
without duplicating the whole application or creating a second route stack.
"""
import app as app_module
from stroke_reconstruction import handwriting_to_svg

app_module.handwriting_to_svg = handwriting_to_svg
app = app_module.app
