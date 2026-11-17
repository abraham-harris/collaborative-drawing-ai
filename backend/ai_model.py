"""Generate two limited-ink alternatives from the same canvas."""

from __future__ import annotations

import logging
import random
from typing import Any

from PIL import Image

from drawing_models import (DEFAULT_MODE, MODELS, SUBJECTS, DrawingContext, Stroke,
                            render, stroke_to_action, strokes_from_actions)
from drawing_models.strokes import dist

log = logging.getLogger(__name__)


INK_LIMIT = 300.0
MAX_SUBJECT_ATTEMPTS = 6


def trim_strokes(strokes: list[Stroke], limit: float = INK_LIMIT) -> list[Stroke]:
    """Keep strokes in order, interpolating the final point at the budget edge."""
    kept = []
    remaining = limit
    for stroke in strokes:
        if remaining <= 0:
            break
        if not stroke.points:
            continue
        points = [stroke.points[0]]
        for point in stroke.points[1:]:
            length = dist(points[-1], point)
            if length > remaining:
                fraction = remaining / length
                last = points[-1]
                points.append((last[0] + (point[0] - last[0]) * fraction,
                               last[1] + (point[1] - last[1]) * fraction))
                remaining = 0
                break
            points.append(point)
            remaining -= length
        if len(points) > 1:
            kept.append(Stroke(points=points, color=stroke.color, size=stroke.size,
                               opacity=stroke.opacity, closed=False, kind=stroke.kind,
                               meta=stroke.meta.copy()))
    return kept


def generate_drawing(
    image: Image.Image,
    actions: list[dict[str, Any]],
    context_actions: list[dict[str, Any]] | None = None,
    mode: str = DEFAULT_MODE,
    rng: random.Random | None = None,
) -> list[tuple[Image.Image, str, list[dict[str, Any]], str]]:
    """Return two candidates with distinct random subjects, both in the chosen mode."""
    if mode not in MODELS:
        raise ValueError(f"Unknown mode {mode!r}; expected one of {sorted(MODELS)}")
    rng = rng or random.Random()
    candidates = []
    errors = []
    strokes = strokes_from_actions(context_actions if context_actions is not None else actions)
    for subject in rng.sample(SUBJECTS, min(MAX_SUBJECT_ATTEMPTS, len(SUBJECTS))):
        ctx = DrawingContext.build(image, strokes, subject,
                                   rng=random.Random(rng.getrandbits(64)))
        try:
            ai_strokes, _ = MODELS[mode](ctx)
        except Exception as exc:  # noqa: BLE001 -- try another subject
            log.exception("drawing model %r failed for %r", mode, subject)
            errors.append(f"{subject}: {exc}")
            continue
        limited = trim_strokes(ai_strokes)
        if limited:
            message = "The AI added to your drawing."
            candidates.append((render(image, limited), message,
                               [stroke_to_action(s) for s in limited], subject))
            if len(candidates) == 2:
                return candidates
        else:
            errors.append(f"{subject}: drew nothing")
    log.warning("Could not make two AI choices: %s", "; ".join(errors))
    raise RuntimeError("The AI couldn't make two choices this turn")
