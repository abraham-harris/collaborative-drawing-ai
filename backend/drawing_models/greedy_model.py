"""GREEDY: try many limited-ink additions, keep the one AuDrA rates most creative.

Each turn the AI may use only as much ink (px of line) as the human drew this
turn, split between strokes however the drawing it generates splits it.

1. Generate `ctx.iterations` candidates with SketchRNN in one batch per strategy
   (split across several random subjects, which aren't shown to the player):
   - complete: read the whole drawing, then add new strokes
   - continue: read the drawing with the pen held down at the end of one of the
     human's strokes, extend that line, then keep drawing
   Temperatures are spread from cautious to adventurous, so candidates range
   from typical to surprising. Each one stops when it runs out of ink and is
   trimmed to exactly the budget.
2. Render every candidate onto the canvas and score them all with AuDrA in one batch.
   (Candidates are rendered without anti-aliasing: 9x faster, and AuDrA's scores
   move by at most ~0.005. The returned drawing is rendered at full quality.)
3. Keep the highest-scoring candidate: the best next step, with no lookahead.
"""

from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import sketch_rnn
from .canvas import DrawingContext, render
from .strokes import Point, Stroke, path_length, simplify, trim_to_length

TEMPERATURES = (0.2, 0.9)   # range spread across the candidates
MIN_INK = 3.0               # px; ignore candidates that drew (almost) nothing
MAX_CONTINUE_GROUPS = 2     # human strokes to try extending per turn


def _strategies(n: int, human: list[list[Point]], rng) -> list[tuple[str, int | None, int]]:
    """Split n candidates: about half "complete", the rest extending up to two human strokes
    (the most recent one, plus one other picked at random)."""
    targets = [len(human) - 1]
    if len(human) > 1:
        targets.append(rng.randrange(len(human) - 1))
    targets = targets[:MAX_CONTINUE_GROUPS]
    n_complete = math.ceil(n / 2) if n > 1 else 1
    groups = [("complete", None, n_complete)]
    rest = n - n_complete
    for g, i in enumerate(targets):
        share = rest // len(targets) + (1 if g < rest % len(targets) else 0)
        if share:
            groups.append(("continue", i, share))
    return groups


def _generate_candidates(subject: str, n: int, temps: list[float], ctx: DrawingContext,
                         human: list[list[Point]], budget: float, nrng: np.random.Generator):
    """n candidates for one subject, using temperatures from `temps`."""
    model = sketch_rnn.load(subject)
    x0, y0, x1, y1 = ctx.user_box()
    k = sketch_rnn.canvas_scale(model, max(x1 - x0, y1 - y0), min(ctx.width, ctx.height), nrng)
    candidates = []
    for strategy, target, count in _strategies(n, human, ctx.rng):
        if target is None:
            lines = human
            cursor = model.prime([[(x / k, y / k) for x, y in h] for h in lines])
        else:
            lines = [h for j, h in enumerate(human) if j != target] + [human[target]]
            cursor = model.prime([[(x / k, y / k) for x, y in h] for h in lines], hold_last=True)
        origin = lines[0][0]
        group_temps, temps = temps[:count], temps[count:]
        sketches = model.generate_batch(cursor, group_temps, nrng, stop_length=budget / k)
        for temperature, sk in zip(group_temps, sketches):
            new = [[(min(max(origin[0] + x * k, 0.0), ctx.width), min(max(origin[1] + y * k, 0.0), ctx.height))
                    for x, y in s] for s in sk.strokes]
            new = trim_to_length(new, budget)
            ink = sum(path_length(s) for s in new)
            if ink >= MIN_INK:
                candidates.append({"subject": subject, "strategy": strategy, "extends": target,
                                   "temperature": float(temperature), "lines": new, "ink": ink})
    return candidates


def generate(ctx: DrawingContext) -> tuple[list[Stroke], str]:
    budget = ctx.ink_budget or 0.0
    report = ctx.report.setdefault("greedy", {"candidates": [], "iterations": ctx.iterations})
    if budget < MIN_INK or not ctx.strokes:
        return [], "You didn't draw anything this turn, so I have no ink to spend."

    nrng = np.random.default_rng(ctx.rng.getrandbits(63))
    human = [simplify(s.points, 1.5) for s in ctx.strokes]
    style = ctx.ai_style()

    t = time.perf_counter()
    temps = list(np.linspace(*TEMPERATURES, ctx.iterations))
    ctx.rng.shuffle(temps)
    subjects = ctx.subjects or [ctx.subject]
    if len(subjects) > 1:  # download/load several subjects' weights at once
        with ThreadPoolExecutor(len(subjects)) as pool:
            list(pool.map(sketch_rnn.load, subjects))
    candidates = []
    for g, subject in enumerate(subjects):
        n = ctx.iterations // len(subjects) + (1 if g < ctx.iterations % len(subjects) else 0)
        if n:
            group_temps, temps = temps[:n], temps[n:]
            candidates += _generate_candidates(subject, n, group_temps, ctx, human, budget, nrng)
    timings = ctx.report.setdefault("timings", {})
    timings["generate"] = time.perf_counter() - t
    if not candidates:
        return [], ""

    # Score every candidate as part of the whole drawing, in one batch.
    t = time.perf_counter()
    scores: list[float | None] = [None] * len(candidates)
    if ctx.scorer is not None:
        images = [render(ctx.image, [Stroke(points=p, **style) for p in c["lines"]], ctx.mask, supersample=1)
                  for c in candidates]
        timings["renderCandidates"] = time.perf_counter() - t
        t = time.perf_counter()
        try:
            scores = [float(s) for s in ctx.scorer(images)]
        except Exception as exc:  # noqa: BLE001 -- still take a turn, just unranked
            report["scoringError"] = str(exc) or type(exc).__name__
        timings["scoreCandidates"] = time.perf_counter() - t
    else:
        report["scoringError"] = "creativity scorer unavailable"

    scored = [s for s in scores if s is not None]
    best = max(range(len(candidates)), key=lambda i: scores[i]) if scored else 0
    chosen = candidates[best]
    report.update({
        # Subjects stay hidden from the player, as in a duel.
        "candidates": [{"strategy": c["strategy"], "extends": c["extends"],
                        "temperature": c["temperature"], "ink": c["ink"], "score": s}
                       for c, s in zip(candidates, scores)],
        "averageScore": sum(scored) / len(scored) if scored else None,
        "chosenIndex": best,
        "chosenScore": scores[best],
    })

    ctx.report["chosenSubject"] = chosen["subject"]
    how = "new strokes" if chosen["strategy"] == "complete" else "extending one of your lines"
    verdict = f"rated most creative (AuDrA {scores[best]:.2f})" if scored else "unscored, so I kept the first"
    msg = (f"I tried {len(candidates)} ways to spend {budget:.0f}px of ink and kept the one "
           f"{verdict}: {how}.")
    return [Stroke(points=p, **style) for p in chosen["lines"]], msg
