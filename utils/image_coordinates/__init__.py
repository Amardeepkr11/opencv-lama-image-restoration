"""Clickable image that returns the pixel position of a click (or a click-and-drag box).

Local copy of streamlit-image-coordinates 0.2.2 (MIT) with a fixed frontend; see
frontend/main.js for what was changed.
"""
import base64
from io import BytesIO
from pathlib import Path
from typing import Optional

import streamlit.components.v1 as components
from PIL import Image

_component_func = components.declare_component(
    "inpaint_image_coordinates",
    path=str((Path(__file__).parent / "frontend").absolute()),
)


def image_coordinates(image: Image.Image, width: Optional[int] = None, click_and_drag: bool = False,
                      key: Optional[str] = None) -> Optional[dict]:
    """
    Show a PIL image and return the last click on it.

    Returns None before the first click, otherwise a dict with "x", "y" (or "x1", "y1",
    "x2", "y2" when click_and_drag=True), the "width" and "height" the image was
    actually shown at in the browser, and "unix_time" of the click.
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG", compress_level=1)
    src = "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")
    return _component_func(src=src, width=width, height=None, use_column_width=None,
                           click_and_drag=click_and_drag, key=key, default=None)
