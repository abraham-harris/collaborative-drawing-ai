"""COMPLETE: treat the human's strokes as the start of the subject and finish it.

This is SketchRNN's "predictive" mode. The model, trained on Quick Draw
doodles of the subject, first reads the human's strokes as if it had drawn
them itself, then keeps drawing until it decides the doodle is done. If you
draw a cat's head and the subject is "cat", it should add the ears, whiskers,
and so on, in place on your drawing.
"""

from __future__ import annotations

import statistics

import numpy as np

from . import sketch_rnn
from .canvas import DrawingContext, place_object
from .strokes import Stroke, bbox, simplify

CANDIDATES = 6
# Assume the human has drawn about this fraction of the finished drawing's size.
ASSUMED_PROGRESS = 0.7
_typical_size: dict[str, float] = {}


def typical_size(model: sketch_rnn.SketchRNN, rng: np.random.Generator) -> float:
    """Median size (data units) of the drawings this model makes on its own."""
    if model.name not in _typical_size:
        sizes = []
        for _ in range(3):
            sk = model.draw(None, 0.25, rng)
            if sk.strokes:
                x0, y0, x1, y1 = bbox([p for s in sk.strokes for p in s])
                sizes.append(max(x1 - x0, y1 - y0))
        _typical_size[model.name] = statistics.median(sizes) if sizes else 900.0
    return _typical_size[model.name]


def generate(ctx: DrawingContext) -> tuple[list[Stroke], str]:
    model = sketch_rnn.load(ctx.subject)
    nrng = np.random.default_rng(ctx.rng.getrandbits(63))
    name = ctx.subject.replace("_", " ")
    style = ctx.ai_style()

    if not ctx.strokes:
        sketch = next((s for s in (model.draw(None, ctx.temperature, nrng) for _ in range(CANDIDATES))
                       if s.strokes), None)
        placed = place_object(ctx, sketch.strokes) if sketch else None
        if not placed:
            return [], ""
        return [Stroke(points=p, **style) for p in placed], f"Nothing to finish yet, so I started a {name}."

    # Scale: canvas px per model unit, so the human's drawing reads as most of a full doodle.
    x0, y0, x1, y1 = ctx.user_box()
    user_dim = max(x1 - x0, y1 - y0, 40.0)
    full = typical_size(model, nrng)
    k = min(user_dim / (ASSUMED_PROGRESS * full), 0.95 * min(ctx.width, ctx.height) / full)

    origin = ctx.strokes[0].points[0]
    prime = [[(x / k, y / k) for x, y in simplify(s.points, 1.5)] for s in ctx.strokes]

    for temperature in (ctx.temperature, ctx.temperature + 0.15, ctx.temperature + 0.3):
        best = None
        for _ in range(CANDIDATES):
            sk = model.draw(prime, temperature, nrng)
            if not sk.strokes:
                continue
            lines = [[(origin[0] + x * k, origin[1] + y * k) for x, y in s] for s in sk.strokes]
            pts = [p for s in lines for p in s]
            off_canvas = sum(not ctx.in_bounds(p) for p in pts) / len(pts)
            score = off_canvas * 3 + (0 if sk.finished else 0.5)
            if best is None or score < best[0]:
                best = (score, lines)
        if best is not None:
            lines = [[(min(max(x, 0.0), ctx.width), min(max(y, 0.0), ctx.height)) for x, y in s]
                     for s in best[1]]
            n = len(lines)
            return ([Stroke(points=p, **style) for p in lines],
                    f"Finished your {name} with {n} more stroke{'s' if n != 1 else ''}.")
    return [], ""
