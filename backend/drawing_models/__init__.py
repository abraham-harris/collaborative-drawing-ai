"""Drawing models the AI can use to add to the human's drawing.

Both draw a recognizable object of the chosen subject using SketchRNN, a neural
network trained on the Quick, Draw! dataset.

Each model is a function `generate(ctx: DrawingContext) -> (new_strokes, message)`.
Models only return NEW strokes. `canvas.render` draws them so that existing ink
is never changed, which means a model cannot erase the human's work.

To add a model, write a module with a `generate` function and register it in MODELS
(and in AI_MODES in frontend/src/config.ts).
"""

from __future__ import annotations

from typing import Callable

from . import complete_model, continue_model
from .canvas import DrawingContext, render
from .strokes import Stroke, stroke_to_action, strokes_from_actions
from .subjects import RANDOM, SUBJECTS

ModelFn = Callable[[DrawingContext], tuple[list[Stroke], str]]

MODELS: dict[str, ModelFn] = {
    "complete": complete_model.generate,
    "continue": continue_model.generate,
}
DEFAULT_MODE = "complete"

# If a model draws nothing (or fails), try these next so the AI always takes its turn.
FALLBACKS: dict[str, list[str]] = {
    "complete": ["continue"],
    "continue": ["complete"],
}

__all__ = [
    "MODELS", "DEFAULT_MODE", "FALLBACKS", "SUBJECTS", "RANDOM", "DrawingContext", "Stroke",
    "render", "stroke_to_action", "strokes_from_actions",
]
