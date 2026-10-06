"""
AI co-drawing logic. THIS IS A STUB -- replace `generate_drawing` with your model.

The frontend sends the human's drawing every turn; whatever image this
function returns replaces the canvas in the browser (so it should contain
both the human's marks and the AI's additions).
"""

from __future__ import annotations

import random
from typing import Any

from PIL import Image, ImageDraw


def generate_drawing(
    image: Image.Image,
    actions: list[dict[str, Any]],
    turn: int,
    total_turns: int,
) -> tuple[Image.Image, str | None]:
    """
    Args:
        image: The current canvas (RGBA, e.g. 800x600) including all previous
            human and AI contributions. Use `numpy.asarray(image)` for a pixel array.
        actions: Vector record of what the human drew THIS turn. Each item is
            {"tool": "brush"|"eraser"|"line"|"rectangle"|"ellipse"|"fill",
             "color": "#rrggbb", "size": int, "opacity": float, "filled": bool,
             "points": [{"x": float, "y": float}, ...]}
        turn: 1-based turn number.
        total_turns: Total number of turns in this session (NUM_TURNS in the frontend).

    Returns:
        (new_image, message) -- the full drawing to show the user, and an optional
        short message displayed above the canvas (or None).
    """
    # TODO: replace this placeholder with the real AI model.
    # Placeholder: echo the drawing back with a few random circles so you can
    # see the round trip working end to end.
    result = image.convert("RGB").copy()
    draw = ImageDraw.Draw(result)
    w, h = result.size
    for _ in range(3):
        r = random.randint(15, 60)
        x, y = random.randint(r, w - r), random.randint(r, h - r)
        color = tuple(random.randint(0, 255) for _ in range(3))
        draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=4)

    message = (
        f"[stub] Turn {turn}/{total_turns}: received {len(actions)} human "
        f"action(s) and added 3 circles."
    )
    return result, message
