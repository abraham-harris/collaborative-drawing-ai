"""Recognize what's on the canvas, so the AI can draw something that fits it.

A small CNN (trained by training/train_recognizer.py on Quick, Draw!'s 28x28
bitmaps) scores the drawing against the 95 subjects the AI can draw. It runs in
numpy, so recognition needs no PyTorch and takes a few milliseconds.

`choose_subjects` then deliberately avoids always taking the top guess: it
samples among the top few interpretations, with the most likely one down-weighted,
so the AI sometimes "sees" the drawing as something else.
"""

from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .strokes import Point

WEIGHTS = Path(__file__).with_name("recognizer.npz")

# Rasterize like Quick Draw's bitmaps: drawing centered, longest side 26 px of a
# 28x28 frame, lines ~1.6 px wide, drawn at 10x and averaged down.
SIZE, FIT, SCALE, LINE_WIDTH = 28, 26, 10, 16

# Subject choice: consider the top guesses that are at least this likely...
TOP_K = 5
MIN_PROBABILITY = 0.02
# ...and make the single most likely guess this much less likely to be picked.
OBVIOUS_WEIGHT = 0.35


def rasterize(strokes: list[list[Point]]) -> np.ndarray:
    """Polylines (any coordinates) -> 28x28 float image, ink 1.0 on 0.0, like Quick Draw."""
    points = [p for s in strokes for p in s]
    big = Image.new("L", (SIZE * SCALE, SIZE * SCALE), 0)
    if not points:
        return np.zeros((SIZE, SIZE), np.float32)
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    k = FIT * SCALE / max(x1 - x0, y1 - y0, 1e-6)
    ox = (SIZE * SCALE - (x1 - x0) * k) / 2
    oy = (SIZE * SCALE - (y1 - y0) * k) / 2
    draw = ImageDraw.Draw(big)
    r = LINE_WIDTH / 2
    for s in strokes:
        pts = [(ox + (x - x0) * k, oy + (y - y0) * k) for x, y in s]
        if len(pts) > 1:
            draw.line(pts, fill=255, width=LINE_WIDTH, joint="curve")
        for x, y in (pts[0], pts[-1]):
            draw.ellipse((x - r, y - r, x + r, y + r), fill=255)
    return np.asarray(big.resize((SIZE, SIZE), Image.Resampling.BOX), np.float32) / 255


def _conv3x3_relu(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """3x3 convolution with padding 1, then ReLU. x: (C, H, W), w: (O, C, 3, 3)."""
    c, h, wd = x.shape
    p = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    cols = np.stack([p[:, i:i + h, j:j + wd] for i in range(3) for j in range(3)], axis=1)  # (C, 9, H, W)
    out = w.reshape(w.shape[0], -1) @ cols.reshape(c * 9, h * wd) + b[:, None]
    return np.maximum(out, 0).reshape(-1, h, wd)


def _maxpool2(x: np.ndarray) -> np.ndarray:
    c, h, w = x.shape
    h, w = h // 2 * 2, w // 2 * 2
    return x[:, :h, :w].reshape(c, h // 2, 2, w // 2, 2).max(axis=(2, 4))


class Recognizer:
    def __init__(self, path: Path = WEIGHTS):
        with np.load(path) as z:
            self.classes = [str(c) for c in z["classes"]]
            self.convs = [(z[f"conv{i}_w"], z[f"conv{i}_b"]) for i in range(3)]
            self.fcs = [(z[f"fc{i}_w"], z[f"fc{i}_b"]) for i in range(2)]

    def predict(self, image: np.ndarray) -> list[tuple[str, float]]:
        """28x28 image -> [(subject, probability)], most likely first."""
        x = image[None].astype(np.float32)
        for w, b in self.convs:
            x = _maxpool2(_conv3x3_relu(x, w, b))
        x = x.reshape(-1)
        (w1, b1), (w2, b2) = self.fcs
        logits = w2 @ np.maximum(w1 @ x + b1, 0) + b2
        p = np.exp(logits - logits.max())
        p /= p.sum()
        order = np.argsort(-p)
        return [(self.classes[i], float(p[i])) for i in order]


@lru_cache(maxsize=1)
def _default() -> Recognizer:
    return Recognizer()


def available() -> bool:
    return WEIGHTS.exists()


def recognize(strokes: list[list[Point]]) -> list[tuple[str, float]]:
    """What the drawing looks like, as [(subject, probability)], most likely first."""
    return _default().predict(rasterize(strokes))


def choose_subjects(guesses: list[tuple[str, float]], n: int, rng: random.Random) -> list[str]:
    """Pick n different subjects among the top guesses, by probability, with the most
    obvious one down-weighted. Returns fewer than n if there aren't enough plausible guesses."""
    pool = [(s, p) for s, p in guesses[:TOP_K] if p >= MIN_PROBABILITY] or guesses[:1]
    weights = [p * (OBVIOUS_WEIGHT if i == 0 else 1.0) for i, (_, p) in enumerate(pool)]
    chosen: list[str] = []
    while pool and len(chosen) < n:
        i = rng.choices(range(len(pool)), weights=weights)[0]
        chosen.append(pool.pop(i)[0])
        weights.pop(i)
    return chosen
