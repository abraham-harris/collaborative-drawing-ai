"""
AI co-drawing logic: the entry point the server calls every turn.

The frontend sends the human's drawing; the image returned here replaces the
canvas in the browser. The selected drawing model (see drawing_models/) adds
strokes, and `render` composites them so the human's existing ink is never
altered -- the AI can only add. The AI always draws something on its turn:
if the chosen model comes up empty, a fallback model takes over.
"""

from __future__ import annotations

import logging
import random
from typing import Any

from PIL import Image

from drawing_models import (DEFAULT_MODE, FALLBACKS, MODELS, RANDOM, SUBJECTS, DrawingContext,
                            render, stroke_to_action, strokes_from_actions)

log = logging.getLogger(__name__)


def generate_drawing(
    image: Image.Image,
    actions: list[dict[str, Any]],
    turn: int,
    total_turns: int,
    mode: str = DEFAULT_MODE,
    subject: str = RANDOM,
) -> tuple[Image.Image, str | None, list[dict[str, Any]], str]:
    """
    Args:
        image: The current canvas (RGBA, e.g. 800x600) including all previous
            human and AI contributions.
        actions: Vector record of what the human drew THIS turn. Each item is
            {"tool": "brush"|"eraser"|"line"|"rectangle"|"ellipse"|"fill",
             "color": "#rrggbb", "size": int, "opacity": float, "filled": bool,
             "points": [{"x": float, "y": float}, ...]}
        turn: 1-based turn number.
        total_turns: Total number of turns in this session (NUM_TURNS in the frontend).
        mode: Which drawing model to use -- a key of drawing_models.MODELS
            ("complete" or "continue").
        subject: What the AI should draw -- one of drawing_models.SUBJECTS, or
            "random" to pick one.

    Returns:
        (new_image, message, ai_actions, subject) -- the full drawing to show the
        user, a short message displayed above the canvas, the AI's strokes in the
        same DrawAction format the frontend sends, and the subject actually drawn.

    Raises:
        RuntimeError: if no model could draw anything (e.g. no network access to
            download SketchRNN weights).
    """
    if mode not in MODELS:
        raise ValueError(f"Unknown mode {mode!r}; expected one of {sorted(MODELS)}")
    rng = random.Random()
    if subject == RANDOM:
        subject = rng.choice(SUBJECTS)
    elif subject not in SUBJECTS:
        raise ValueError(f"Unknown subject {subject!r}")

    ctx = DrawingContext.build(image, strokes_from_actions(actions), subject, rng=rng)
    errors = []
    for attempt in [mode, *FALLBACKS[mode]]:
        try:
            ai_strokes, message = MODELS[attempt](ctx)
        except Exception as exc:  # noqa: BLE001 -- try the next model instead
            log.exception("drawing model %r failed", attempt)
            errors.append(f"{attempt}: {exc}")
            continue
        if ai_strokes:
            if attempt != mode:
                message = f"({mode.capitalize()} couldn't add anything this time, so I used {attempt.capitalize()}.) {message}"
            result = render(image, ai_strokes)
            return result, message, [stroke_to_action(s) for s in ai_strokes], subject
        errors.append(f"{attempt}: drew nothing")

    raise RuntimeError("The AI couldn't draw this turn (" + "; ".join(errors) + ")")
