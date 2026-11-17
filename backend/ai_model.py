"""Generate the AI's turn from the canvas.

Subjects come from recognizing the drawing (drawing_models/recognizer.py): the
AI picks among its top few interpretations, avoiding the most obvious one more
often than not. On a blank canvas, or with recognition off, they're random.

- Complete / Continue (the dueling_artists design): two limited-ink alternatives
  with distinct subjects; the human chooses one to keep.
- Greedy: the AI tries many alternatives limited to the ink the human drew this
  turn and returns the one AuDrA rates most creative, so there's nothing to choose.

`score_options` adds AuDrA scores for the console report.
"""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Any

from PIL import Image

from drawing_models import (DEFAULT_MODE, MODELS, SUBJECTS, DrawingContext, Stroke, recognizer,
                            render, stroke_to_action, strokes_from_actions)
from drawing_models.canvas import Scorer
from drawing_models.strokes import dist, ink_length

log = logging.getLogger(__name__)


INK_LIMIT = 300.0
MAX_SUBJECT_ATTEMPTS = 6
# Greedy: how many candidates it tries, and how many random subjects it spreads them across.
DEFAULT_ITERATIONS = 10
GREEDY_SUBJECTS = 3
MIN_INK = 3.0  # px; below this the human effectively drew nothing
# The models re-read the drawing every turn; keep only the most recent strokes so
# long sessions (rounds are unlimited) stay fast.
MAX_CONTEXT_STROKES = 40


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


def _audra_scorer() -> Scorer | None:
    """AuDrA's batch scorer, or None if it's turned off (AUDRA_SCORE_TURNS=0) or not installed."""
    if os.environ.get("AUDRA_SCORE_TURNS", "1") == "0":
        return None
    try:
        import audra_service
    except ImportError:
        return None
    return audra_service.score_drawings


def pick_subjects(strokes: list[Stroke], n: int, rng: random.Random, recognize: bool,
                  report: dict[str, Any]) -> list[str]:
    """n distinct subjects to try, best first: interpretations of the drawing (see
    recognizer.choose_subjects), then random ones to fill in."""
    picked: list[str] = []
    if recognize and strokes and recognizer.available():
        t = time.perf_counter()
        guesses = recognizer.recognize([s.points for s in strokes])
        picked = recognizer.choose_subjects(guesses, n, rng)
        report["recognition"] = {
            "guesses": [{"subject": s, "probability": p} for s, p in guesses[:recognizer.TOP_K]],
        }
        report["timings"]["recognize"] = time.perf_counter() - t
        rest = [s for s in SUBJECTS if s not in picked]
        return picked + rng.sample(rest, n - len(picked))
    return rng.sample(SUBJECTS, n)


def generate_drawing(
    image: Image.Image,
    actions: list[dict[str, Any]],
    context_actions: list[dict[str, Any]] | None = None,
    mode: str = DEFAULT_MODE,
    rng: random.Random | None = None,
    *,
    iterations: int = DEFAULT_ITERATIONS,
    scorer: Scorer | None | str = "audra",
    report: dict[str, Any] | None = None,
    recognize: bool = True,
) -> list[tuple[Image.Image, str, list[dict[str, Any]], str]]:
    """Return two candidates with distinct subjects, both in the chosen mode.

    Greedy returns a single candidate instead (see `_greedy`). `iterations` and
    `scorer` only affect Greedy. With `recognize`, subjects are interpretations of
    the drawing; otherwise random. Pass a `report` dict to collect ink and timings.
    """
    if mode not in MODELS:
        raise ValueError(f"Unknown mode {mode!r}; expected one of {sorted(MODELS)}")
    rng = rng or random.Random()
    report = {} if report is None else report
    timings = report.setdefault("timings", {})
    report["humanInk"] = ink_length(strokes_from_actions(actions))
    strokes = strokes_from_actions(context_actions if context_actions is not None else actions)
    strokes = strokes[-MAX_CONTEXT_STROKES:]
    if mode == "greedy":
        subjects = pick_subjects(strokes, GREEDY_SUBJECTS, rng, recognize, report)
        return _greedy(image, strokes, subjects, rng, iterations, scorer, report)

    report["inkBudget"] = INK_LIMIT
    start = time.perf_counter()
    candidates = []
    errors = []
    for subject in pick_subjects(strokes, min(MAX_SUBJECT_ATTEMPTS, len(SUBJECTS)), rng, recognize, report):
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
            candidates.append((render(image, limited, ctx.mask), message,
                               [stroke_to_action(s) for s in limited], subject))
            if len(candidates) == 2:
                timings["model"] = time.perf_counter() - start
                return candidates
        else:
            errors.append(f"{subject}: drew nothing")
    log.warning("Could not make two AI choices: %s", "; ".join(errors))
    raise RuntimeError("The AI couldn't make two choices this turn")


def _greedy(image: Image.Image, strokes: list[Stroke], subjects: list[str], rng: random.Random,
            iterations: int, scorer: Scorer | None | str, report: dict[str, Any]):
    """One candidate: the most creative of `iterations` tries across `subjects`, using
    only as much ink as the human drew this turn."""
    budget = report["humanInk"]
    report["inkBudget"] = budget
    if budget < MIN_INK:
        return [(image.convert("RGB"), "You didn't draw anything this turn, so I have no ink to spend.", [], "")]
    if scorer == "audra":
        scorer = _audra_scorer()

    ctx = DrawingContext.build(image, strokes, subjects[0], rng=random.Random(rng.getrandbits(64)),
                               ink_budget=budget, iterations=max(1, iterations), subjects=subjects,
                               scorer=scorer, report=report)
    t = time.perf_counter()
    try:
        ai_strokes, message = MODELS["greedy"](ctx)
    except Exception as exc:  # noqa: BLE001
        log.exception("greedy model failed")
        raise RuntimeError("The AI couldn't draw this turn") from exc
    finally:
        report["timings"]["model"] = time.perf_counter() - t
    ai_strokes = trim_strokes(ai_strokes, budget)
    if not ai_strokes:
        raise RuntimeError("The AI couldn't draw this turn")

    t = time.perf_counter()
    result = render(image, ai_strokes, ctx.mask)
    report["timings"]["render"] = time.perf_counter() - t
    subject = report.pop("chosenSubject", subjects[0])  # returned in the tuple, not the report
    return [(result, message, [stroke_to_action(s) for s in ai_strokes], subject)]


def score_options(image: Image.Image, options: list[tuple[Image.Image, list[dict[str, Any]]]],
                  report: dict[str, Any], scorer: Scorer | None | str = "audra") -> None:
    """Add AuDrA scores to `report` for the console, in one batch: the drawing before
    this turn, then each option (the whole drawing with it, and its AI strokes alone).
    Also records each option's ink. Never raises; failures go in report["scoringError"]."""
    report["options"] = [{"ink": ink_length(strokes_from_actions(ai)), "strokes": len(ai)}
                         for _, ai in options]
    if scorer == "audra":
        scorer = _audra_scorer()
    if scorer is None:
        report["scoringError"] = "creativity scorer unavailable"
        return
    blank = Image.new("RGB", image.size, "white")
    images = ([image.convert("RGB")] + [result.convert("RGB") for result, _ in options]
              + [render(blank, strokes_from_actions(ai)) for _, ai in options])
    t = time.perf_counter()
    try:
        scores = [float(s) for s in scorer(images)]
    except Exception:  # noqa: BLE001 -- a turn shouldn't fail because scoring did
        log.exception("AuDrA scoring failed")
        report["scoringError"] = "creativity scoring failed"
        return
    finally:
        report.setdefault("timings", {})["scoreDrawing"] = time.perf_counter() - t
    n = len(options)
    report["before"] = scores[0]
    for i, option in enumerate(report["options"]):
        option["after"], option["aiOnly"] = scores[1 + i], scores[1 + n + i]
