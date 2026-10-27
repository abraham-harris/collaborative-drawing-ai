"""CONTINUE: the AI extends the human's strokes, then finishes the drawing from there.

1. Extend: for every human stroke, SketchRNN (trained on Quick Draw doodles of
   the subject) first reads the entire drawing: the other human strokes, any
   extensions it has already made, and finally that stroke. The pen is held
   down at the stroke's end, so the model's next move continues that line,
   and its shape depends on everything already on the page.
2. Finish: with the human's strokes and its extensions in memory, the model
   keeps drawing until the subject is complete, so the result reads as a
   whole object built out of the human's lines.

Unlike Complete, which only adds separate new strokes, Continue's first marks
always grow out of the ends of the human's lines.
"""

from __future__ import annotations

import math

import numpy as np

from . import sketch_rnn
from .canvas import DrawingContext
from .complete_model import finish, start_in_open_space
from .strokes import Point, Stroke, path_length, simplify

CANDIDATES = 5          # alternative extensions sampled per stroke
MAX_STEPS = 60          # model steps per extension
MAX_EXTENDED = 8        # only extend the most recent strokes (keeps a turn fast)
MIN_LENGTH = 12.0       # px; shorter extensions are invisible next to the stroke
# The model tends to lift the pen early; make each extension at least this
# fraction of the stroke it continues (capped at MAX_FORCED px).
FORCED_FRACTION = 0.6
MAX_FORCED = 250.0


def _extend(model: sketch_rnn.SketchRNN, lines: list[list[Point]], k: float, ctx: DrawingContext,
            target_length: float, nrng: np.random.Generator) -> list[Point] | None:
    """Continue the LAST of `lines` (canvas px), with the model having read all of them."""
    origin = lines[0][0]
    cursor = model.prime([[(x / k, y / k) for x, y in line] for line in lines], hold_last=True)
    target = max(target_length, 30.0)
    for temperature in (ctx.temperature, ctx.temperature + 0.2):
        best = None
        for _ in range(CANDIDATES):
            sk = model.generate(cursor, temperature, nrng, max_steps=MAX_STEPS, stop_at_lift=True,
                                min_length=min(FORCED_FRACTION * target_length, MAX_FORCED) / k)
            if not sk.strokes:
                continue
            line = [(origin[0] + x * k, origin[1] + y * k) for x, y in sk.strokes[0]]
            length = path_length(line)
            if length < MIN_LENGTH:
                continue
            off_canvas = sum(not ctx.in_bounds(p) for p in line) / len(line)
            # Prefer on-canvas extensions about as long as the stroke they continue.
            score = off_canvas * 3 + abs(math.log(length / target)) * 0.5 + (0 if sk.finished else 0.3)
            if best is None or score < best[0]:
                best = (score, line)
        if best is not None:
            return [(min(max(x, 0.0), ctx.width), min(max(y, 0.0), ctx.height)) for x, y in best[1]]
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
        return [Stroke(points=p, **style) for p in placed], f"Nothing to continue yet, so I started a {name}."

    x0, y0, x1, y1 = ctx.user_box()
    k = sketch_rnn.canvas_scale(model, max(x1 - x0, y1 - y0), min(ctx.width, ctx.height), nrng)
    human = [simplify(s.points, 1.5) for s in ctx.strokes]

    # 1. Extend the human's strokes, each with the whole drawing in the model's memory.
    extensions: list[list[Point]] = []
    for i in range(max(0, len(human) - MAX_EXTENDED), len(human)):
        others = [h for j, h in enumerate(human) if j != i] + extensions
        line = _extend(model, others + [human[i]], k, ctx, ctx.strokes[i].length, nrng)
        if line:
            extensions.append(line)

    # 2. Finish the drawing with the human's strokes and the extensions as its start.
    rest = finish(model, human + extensions, k, ctx, nrng) or []

    if not extensions and not rest:
        return [], ""
    parts = []
    if extensions:
        n, total = len(extensions), len(human)
        if total == 1:
            parts.append("extended your stroke")
        elif n == total:
            parts.append(f"extended each of your {total} strokes")
        else:
            parts.append(f"extended {n} of your {total} strokes")
    if rest:
        parts.append(f"added {len(rest)} more to make a {name}")
    else:
        parts.append(f"steering toward a {name}")
    msg = "I " + " and ".join(parts) + "."
    return [Stroke(points=p, **style) for p in extensions + rest], msg
