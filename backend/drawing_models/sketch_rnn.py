"""SketchRNN (Ha & Eck, 2018, https://arxiv.org/abs/1704.03477) decoder in numpy.

Google trained these models on the Quick, Draw! dataset, one per category, and
hosts the weights publicly. This is a port of the magenta-js inference code
(magenta-js/sketch/src/sketch_rnn/model.ts): an LSTM reads pen movements one
at a time and outputs a probability distribution over the next movement.

Coordinates in this module are "data units": the space the Quick Draw
drawings were recorded in, where a whole drawing spans roughly 100-250 units.
Callers scale the results to canvas pixels.

Speed: several alternative drawings are sampled as one batch (one matrix
multiply per step for all of them), and decoded weights are cached as .npz
so loading a subject the second time skips the 11 MB JSON parse.
"""

from __future__ import annotations

import base64
import json
import os
import tempfile
import threading
import urllib.request
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# "large_models" use a 1024-unit LSTM (~11 MB per category); "models" use 512 units (~3 MB).
MODEL_URL = "https://storage.googleapis.com/quickdraw-models/sketchRNN/{size}/{name}.gen.json"
MODEL_SIZE = os.environ.get("SKETCH_RNN_SIZE", "large_models")
CACHE_DIR = os.path.join(tempfile.gettempdir(), "sketch_rnn_cache")
MAX_MODELS_IN_MEMORY = 4
N_MIXTURE = 20

Polyline = list[tuple[float, float]]


def _decode(b64: str, shape: list[int]) -> np.ndarray:
    """Weights are int16 arrays, base64 encoded, scaled to [-10, 10]."""
    raw = np.frombuffer(base64.b64decode(b64), dtype="<i2").astype(np.float32)
    return (raw * (10.0 / 32767.0)).reshape(shape)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _softmax_rows(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _sample_rows(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One categorical sample per row of probabilities."""
    idx = (np.cumsum(p, axis=1) < rng.random((p.shape[0], 1))).sum(axis=1)
    return np.minimum(idx, p.shape[1] - 1)


@dataclass
class Sketch:
    """A generated drawing: polylines in data units, starting at the pen origin."""

    strokes: list[Polyline]
    finished: bool  # the model chose to stop (vs. hitting the step limit)


class SketchRNN:
    def __init__(self, info: dict, out_w: np.ndarray, out_b: np.ndarray,
                 kernel: np.ndarray, bias: np.ndarray):
        self.info = info
        self.name: str = info["name"]
        self.scale_factor: float = float(info["scale_factor"])
        self.max_seq_len: int = int(info["max_seq_len"])
        self.out_b, self.bias = out_b, bias
        # Stored transposed: this OpenBLAS build multiplies a small batch by a wide
        # matrix ~10x faster as (W^T @ X^T)^T than as X @ W.
        self.out_w_t = np.ascontiguousarray(out_w.T)
        self.kernel_t = np.ascontiguousarray(kernel.T)
        self.units: int = out_w.shape[0]

    @classmethod
    def from_spec(cls, spec: list) -> "SketchRNN":
        """Build from Google's .gen.json: [info, weight shapes, base64 weights]."""
        info, dims, weights = spec
        # TF's BasicLSTMCell multiplies [input, hidden] by one stacked kernel.
        kernel = np.concatenate([_decode(weights[2], dims[2]), _decode(weights[3], dims[3])], 0)
        return cls(info, _decode(weights[0], dims[0]), _decode(weights[1], dims[1]),
                   kernel, _decode(weights[4], dims[4]))

    # --- the network -------------------------------------------------------------

    def zero_state(self) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(self.units, np.float32), np.zeros(self.units, np.float32)

    def _lstm(self, x: np.ndarray, c: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """One LSTM step for a batch: x (n, 5) in model units, c and h (n, units)."""
        z = (self.kernel_t @ np.concatenate([x, h], axis=1).T).T + self.bias
        i, j, f, o = np.split(z, 4, axis=1)
        c = c * _sigmoid(f + 1.0) + _sigmoid(i) * np.tanh(j)  # forget bias = 1.0
        return c, np.tanh(c) * _sigmoid(o)

    def update(self, stroke: list[float], state: tuple[np.ndarray, np.ndarray]):
        """Feed one [dx, dy, pen_down, pen_up, end] step (data units) into the LSTM."""
        x = np.array([[stroke[0] / self.scale_factor, stroke[1] / self.scale_factor,
                       stroke[2], stroke[3], stroke[4]]], np.float32)
        c, h = self._lstm(x, state[0][None], state[1][None])
        return c[0], h[0]

    def _sample_batch(self, h: np.ndarray, temps: np.ndarray, rng: np.random.Generator):
        """Sample the next move for each row of h from the mixture density output.
        Returns (dx, dy) in model units and the pen index (0 down, 1 up, 2 end)."""
        z = (self.out_w_t @ h.T).T + self.out_b
        pen_logits, rest = z[:, :3], z[:, 3:]
        pi, mu_x, mu_y, s_x, s_y, corr = np.split(rest, 6, axis=1)
        discrete = (0.5 + temps * 0.5)[:, None]
        pen = _softmax_rows(pen_logits / discrete)
        k = _sample_rows(_softmax_rows(pi / discrete), rng)
        rows = np.arange(h.shape[0])
        sqrt_t = np.sqrt(temps)
        sx, sy = np.exp(s_x[rows, k]) * sqrt_t, np.exp(s_y[rows, k]) * sqrt_t
        rho = np.tanh(corr[rows, k])
        z1, z2 = rng.standard_normal((2, h.shape[0]))
        dx = np.sqrt(1 - rho ** 2) * sx * z1 + rho * sx * z2 + mu_x[rows, k]
        dy = sy * z2 + mu_y[rows, k]
        return dx, dy, _sample_rows(pen, rng)

    # --- drawing -----------------------------------------------------------------

    def prime(self, lines: list[Polyline] | None, hold_last: bool = False) -> "Cursor":
        """Have the model read existing polylines (data units, absolute) as if it
        had drawn them itself. The first point of the first line is the origin.

        With `hold_last`, the pen stays down at the end of the last line, so the
        next thing the model draws extends that line instead of starting a new one."""
        state = self.update([0, 0, 1, 0, 0], self.zero_state())
        if not lines:
            return Cursor(state, 0.0, 0.0, pen_down=True)
        ox, oy = lines[0][0]
        px, py = ox, oy
        for i, line in enumerate(lines):
            for j, (lx, ly) in enumerate(line):
                if i == 0 and j == 0:
                    continue  # the origin itself isn't a step
                lift = j == len(line) - 1 and not (hold_last and i == len(lines) - 1)
                state = self.update([lx - px, ly - py, 0 if lift else 1, 1 if lift else 0, 0], state)
                px, py = lx, ly
        return Cursor(state, px - ox, py - oy, pen_down=hold_last)

    def generate_batch(self, cursor: "Cursor", temperatures: Sequence[float], rng: np.random.Generator,
                       max_steps: int | None = None, stop_at_lift: bool = False,
                       min_length: float = 0.0, stop_length: float | None = None) -> list[Sketch]:
        """Sample one drawing per temperature, all continuing from `cursor`, as one batch.

        Returned strokes are relative to the origin (see `prime`).
        - stop_at_lift: stop a drawing the first time its pen lifts.
        - min_length (data units): the pen can't lift until it has drawn that far.
        - stop_length (data units): stop a drawing once it has drawn that much ink.
        """
        n = len(temperatures)
        ids = np.arange(n)  # which sketch each active row belongs to
        temps = np.asarray(temperatures, np.float64)
        c = np.repeat(cursor.state[0][None], n, 0)
        h = np.repeat(cursor.state[1][None], n, 0)
        x, y = np.full(n, cursor.x), np.full(n, cursor.y)
        pen_down = np.full(n, cursor.pen_down)
        drawn = np.zeros(n)
        strokes: list[list[Polyline]] = [[] for _ in range(n)]
        current: list[Polyline | None] = [None] * n
        finished = [False] * n

        def close(i: int) -> None:
            if current[i] is not None and len(current[i]) > 1:
                strokes[i].append(current[i])
            current[i] = None

        for _ in range(max_steps or self.max_seq_len):
            if not ids.size:
                break
            dx, dy, pen = self._sample_batch(h, temps, rng)
            pen[pen_down & (drawn < min_length)] = 0  # keep the pen down until min_length
            px_dx, px_dy = dx * self.scale_factor, dy * self.scale_factor
            keep = np.ones(ids.size, bool)
            for r, i in enumerate(ids):
                if pen[r] == 2:  # the model chose to end the drawing
                    close(i)
                    finished[i], keep[r] = True, False
                    continue
                nx, ny = x[r] + px_dx[r], y[r] + px_dy[r]
                if pen_down[r]:
                    if current[i] is None:
                        current[i] = [(float(x[r]), float(y[r]))]
                    current[i].append((float(nx), float(ny)))
                    drawn[r] += float(np.hypot(px_dx[r], px_dy[r]))
                if pen[r] == 1 and current[i] is not None:  # pen lifts after this point
                    close(i)
                    if stop_at_lift:
                        finished[i], keep[r] = True, False
                if stop_length is not None and drawn[r] >= stop_length:
                    close(i)
                    finished[i], keep[r] = True, False
                x[r], y[r] = nx, ny
            pen_down = pen == 0
            step = np.zeros((ids.size, 5), np.float32)
            step[:, 0], step[:, 1] = dx, dy
            step[np.arange(ids.size), 2 + pen] = 1.0
            # Only advance the drawings that are still going.
            ids, temps, x, y, pen_down, drawn = (a[keep] for a in (ids, temps, x, y, pen_down, drawn))
            if ids.size:
                c, h = self._lstm(step[keep], c[keep], h[keep])
        for i in ids:
            close(i)
        return [Sketch(strokes[i], finished[i]) for i in range(n)]

    def generate(self, cursor: "Cursor", temperature: float, rng: np.random.Generator,
                 max_steps: int | None = None, stop_at_lift: bool = False,
                 min_length: float = 0.0) -> Sketch:
        """Sample a single drawing (see `generate_batch`)."""
        return self.generate_batch(cursor, [temperature], rng, max_steps, stop_at_lift, min_length)[0]

    def draw(self, prime: list[Polyline] | None, temperature: float, rng: np.random.Generator,
             max_steps: int | None = None) -> Sketch:
        """Generate a sketch, optionally after reading `prime` (see `prime`)."""
        return self.generate(self.prime(prime), temperature, rng, max_steps)


@dataclass(frozen=True)
class Cursor:
    """The model's memory and pen position after reading part of a drawing."""

    state: tuple[np.ndarray, np.ndarray]
    x: float
    y: float
    pen_down: bool


_typical_size: dict[str, float] = {}


def typical_size(model: SketchRNN, rng: np.random.Generator) -> float:
    """Median size (data units) of the drawings this model makes on its own."""
    if model.name not in _typical_size:
        sizes = []
        for sk in model.generate_batch(model.prime(None), [0.25] * 3, rng):
            if sk.strokes:
                xs = [p[0] for s in sk.strokes for p in s]
                ys = [p[1] for s in sk.strokes for p in s]
                sizes.append(max(max(xs) - min(xs), max(ys) - min(ys)))
        _typical_size[model.name] = float(np.median(sizes)) if sizes else 900.0
    return _typical_size[model.name]


def canvas_scale(model: SketchRNN, user_size: float, canvas_size: float, rng: np.random.Generator,
                 progress: float = 0.7) -> float:
    """Canvas px per data unit, chosen so a human drawing `user_size` px across reads
    as `progress` of a full drawing, while a full drawing still fits on the canvas."""
    full = typical_size(model, rng)
    return min(max(user_size, 40.0) / (progress * full), 0.95 * canvas_size / full)


# --- loading -------------------------------------------------------------------

_models: OrderedDict[str, SketchRNN] = OrderedDict()
_lock = threading.Lock()


def _download(name: str, path: str) -> None:
    url = MODEL_URL.format(size=MODEL_SIZE, name=name)
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = resp.read()
    tmp = f"{path}.{os.getpid()}.part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def load(name: str) -> SketchRNN:
    """Load a pretrained model by Quick Draw category (underscored, e.g. "fire_hydrant").
    Downloads once to a temp-dir cache, keeps a few in memory."""
    with _lock:
        if name in _models:
            _models.move_to_end(name)
            return _models[name]

    os.makedirs(CACHE_DIR, exist_ok=True)
    base = os.path.join(CACHE_DIR, f"{MODEL_SIZE}-{name}")
    try:
        with np.load(base + ".npz") as z:
            model = SketchRNN(json.loads(str(z["info"])), z["out_w_t"].T, z["out_b"], z["kernel_t"].T, z["bias"])
    except (OSError, KeyError, ValueError):
        if not os.path.exists(base + ".gen.json"):
            _download(name, base + ".gen.json")
        with open(base + ".gen.json", encoding="utf-8") as f:
            model = SketchRNN.from_spec(json.load(f))
        # Save the decoded weights: np.load is ~20x faster than parsing the JSON again.
        tmp = f"{base}.{os.getpid()}.part.npz"
        np.savez(tmp, info=json.dumps(model.info), out_w_t=model.out_w_t, out_b=model.out_b,
                 kernel_t=model.kernel_t, bias=model.bias)
        os.replace(tmp, base + ".npz")

    with _lock:
        _models[name] = model
        while len(_models) > MAX_MODELS_IN_MEMORY:
            _models.popitem(last=False)
    return model
