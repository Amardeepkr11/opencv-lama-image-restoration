"""Freehand brush painting over an image (plain HTML/JS Streamlit component).

Strokes come back in 0..1 image coordinates, so they do not depend on the size
the browser actually showed the image at.
"""
import base64
from io import BytesIO
from pathlib import Path
from typing import Optional

import streamlit.components.v1 as components
from PIL import Image

_component_func = components.declare_component(
    "inpaint_brush_canvas",
    path=str((Path(__file__).parent / "frontend").absolute()),
)


def brush_canvas(image: Image.Image, width: int, brush_radius: float, erase: bool = False,
                 ack_id: int = 0, key: Optional[str] = None) -> Optional[dict]:
    """
    Show a PIL image the user can paint on by dragging the mouse.

    brush_radius is normalised to the image width (radius_px / image_width).
    ack_id is the id of the newest stroke Python has already stored; the browser
    stops showing and resending strokes with an id up to and including it.

    Returns None before the first stroke, otherwise {"strokes": [...]} holding every
    stroke not yet acknowledged, each {"id": int, "erase": bool, "r": float,
    "points": [[x, y], ...]} with x and y between 0 and 1.
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG", compress_level=1)
    src = "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")
    return _component_func(src=src, width=int(width), radius=float(brush_radius), erase=bool(erase),
                           ack_id=int(ack_id), key=key, default=None)
