"""COMPLETE: treat the human's strokes as the start of the subject and finish it.

This is SketchRNN's "predictive" mode. The model, trained on Quick Draw
doodles of the subject, first reads the human's strokes as if it had drawn
them itself, then keeps drawing until it decides the doodle is done. If you
draw a cat's head and the subject is "cat", it should add the ears, whiskers,
and so on, in place on your drawing.
"""

from __future__ import annotations

import numpy as np

from . import sketch_rnn
from .canvas import DrawingContext, place_object
from .strokes import Point, Stroke, simplify

CANDIDATES = 6


def start_in_open_space(model: sketch_rnn.SketchRNN, ctx: DrawingContext,
                        nrng: np.random.Generator) -> list[list[Point]] | None:
    """Blank canvas: draw a whole subject in the open."""
    sketch = next((s for s in (model.draw(None, ctx.temperature, nrng) for _ in range(CANDIDATES))
                   if s.strokes), None)
    return place_object(ctx, sketch.strokes) if sketch else None


def finish(model: sketch_rnn.SketchRNN, lines: list[list[Point]], k: float, ctx: DrawingContext,
           nrng: np.random.Generator) -> list[list[Point]] | None:
    """Have the model read `lines` (canvas px; `k` px per model unit) and draw the rest.
    Returns the new strokes in canvas px, or None if the model adds nothing."""
    origin = lines[0][0]
    cursor = model.prime([[(x / k, y / k) for x, y in line] for line in lines])
    for temperature in (ctx.temperature, ctx.temperature + 0.15, ctx.temperature + 0.3):
        best = None
        for _ in range(CANDIDATES):
            sk = model.generate(cursor, temperature, nrng)
            if not sk.strokes:
                continue
            new = [[(origin[0] + x * k, origin[1] + y * k) for x, y in s] for s in sk.strokes]
            pts = [p for s in new for p in s]
            off_canvas = sum(not ctx.in_bounds(p) for p in pts) / len(pts)
            score = off_canvas * 3 + (0 if sk.finished else 0.5)
            if best is None or score < best[0]:
                best = (score, new)
        if best is not None:
            return [[(min(max(x, 0.0), ctx.width), min(max(y, 0.0), ctx.height)) for x, y in s]
                    for s in best[1]]
    return None


def generate(ctx: DrawingContext) -> tuple[list[Stroke], str]:
    model = sketch_rnn.load(ctx.subject)
    nrng = np.random.default_rng(ctx.rng.getrandbits(63))
    name = ctx.subject.replace("_", " ")
    style = ctx.ai_style()

    if not ctx.strokes:
        placed = start_in_open_space(model, ctx, nrng)
        if not placed:
            return [], ""
        return [Stroke(points=p, **style) for p in placed], f"Nothing to finish yet, so I started a {name}."

    # Scale: canvas px per model unit, so the human's drawing reads as most of a full doodle.
    x0, y0, x1, y1 = ctx.user_box()
    k = sketch_rnn.canvas_scale(model, max(x1 - x0, y1 - y0), min(ctx.width, ctx.height), nrng)
    lines = finish(model, [simplify(s.points, 1.5) for s in ctx.strokes], k, ctx, nrng)
    if not lines:
        return [], ""
    n = len(lines)
    return ([Stroke(points=p, **style) for p in lines],
            f"Finished your {name} with {n} more stroke{'s' if n != 1 else ''}.")
