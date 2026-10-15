"""CONTINUE: the AI picks up the pen where the human stopped and draws the subject.

SketchRNN (trained on Quick Draw) generates a complete drawing of the subject.
The drawing is positioned so its first stroke starts exactly at the end of
the human's last line, so the human's line flows into the AI's drawing. It's
scaled to the size of the human's drawing, mirrored or shrunk if needed to stay
on the canvas and out of existing ink.
"""

from __future__ import annotations

import numpy as np

from . import sketch_rnn
from .canvas import DrawingContext, place_object
from .strokes import Stroke, bbox

CANDIDATES = 6
EDGE_MARGIN = 4.0


def _anchored(sketch: sketch_rnn.Sketch, start, k: float, mirror: bool):
    ox, oy = sketch.strokes[0][0]
    return [
        [(start[0] + ((ox - x) if mirror else (x - ox)) * k, start[1] + (y - oy) * k) for x, y in s]
        for s in sketch.strokes
    ]


def generate(ctx: DrawingContext) -> tuple[list[Stroke], str]:
    model = sketch_rnn.load(ctx.subject)
    nrng = np.random.default_rng(ctx.rng.getrandbits(63))
    sketches = [s for s in (model.draw(None, ctx.temperature, nrng) for _ in range(CANDIDATES)) if s.strokes]
    if not sketches:
        return [], ""
    sketches.sort(key=lambda s: not s.finished)  # prefer drawings the model chose to finish
    name = ctx.subject.replace("_", " ")
    style = ctx.ai_style()

    open_strokes = [s for s in ctx.strokes if not s.closed] or ctx.strokes
    if open_strokes:
        start = open_strokes[-1].points[-1]
        size = ctx.target_size()
        best = None
        for sk in sketches:
            x0, y0, x1, y1 = bbox([p for s in sk.strokes for p in s])
            dim = max(x1 - x0, y1 - y0, 1e-6)
            for factor in (1.0, 0.8, 0.65, 0.5):
                for mirror in (False, True):
                    lines = _anchored(sk, start, size * factor / dim, mirror)
                    if not all(ctx.in_bounds(p, EDGE_MARGIN) for s in lines for p in s):
                        continue
                    ink = sum(ctx.overlap(s) for s in lines) / len(lines)
                    score = ink * 3 + (1 - factor) * 0.6 + (0 if sk.finished else 0.3)
                    if best is None or score < best[0]:
                        best = (score, lines)
        if best is not None:
            strokes = [Stroke(points=p, **style) for p in best[1]]
            return strokes, f"Picked up where your line ended and kept drawing: a {name}."

    # Nothing to continue from (or no room at the line's end): draw it in open space instead.
    placed = place_object(ctx, sketches[0].strokes)
    if placed is None:
        return [], ""
    reason = "There was no room at the end of your line, so I" if open_strokes else "I"
    return [Stroke(points=p, **style) for p in placed], f"{reason} drew a {name} nearby."
