"""SketchRNN (Ha & Eck, 2018, https://arxiv.org/abs/1704.03477) decoder in numpy.

Google trained these models on the Quick, Draw! dataset, one per category, and
hosts the weights publicly. This is a port of the magenta-js inference code
(magenta-js/sketch/src/sketch_rnn/model.ts): an LSTM reads pen movements one
at a time and outputs a probability distribution over the next movement.

Coordinates in this module are "data units": the space the Quick Draw
drawings were recorded in, where a whole drawing spans roughly 100-250 units.
Callers scale the results to canvas pixels.
"""

from __future__ import annotations

import base64
import json
import os
import tempfile
import threading
import urllib.request
from collections import OrderedDict
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


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


@dataclass
class Sketch:
    """A generated drawing: polylines in data units, starting at the pen origin."""

    strokes: list[Polyline]
    finished: bool  # the model chose to stop (vs. hitting the step limit)


class SketchRNN:
    def __init__(self, spec: list):
        info, dims, weights = spec
        self.name: str = info["name"]
        self.scale_factor: float = float(info["scale_factor"])
        self.max_seq_len: int = int(info["max_seq_len"])
        self.out_w = _decode(weights[0], dims[0])
        self.out_b = _decode(weights[1], dims[1])
        # TF's BasicLSTMCell multiplies [input, hidden] by one stacked kernel.
        self.kernel = np.concatenate([_decode(weights[2], dims[2]), _decode(weights[3], dims[3])], 0)
        self.bias = _decode(weights[4], dims[4])
        self.units: int = dims[0][0]

    # --- the network -------------------------------------------------------------

    def zero_state(self) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(self.units, np.float32), np.zeros(self.units, np.float32)

    def update(self, stroke: list[float], state: tuple[np.ndarray, np.ndarray]):
        """Feed one [dx, dy, pen_down, pen_up, end] step (data units) into the LSTM."""
        c, h = state
        x = np.array([stroke[0] / self.scale_factor, stroke[1] / self.scale_factor,
                      stroke[2], stroke[3], stroke[4]], np.float32)
        z = np.concatenate([x, h]) @ self.kernel + self.bias
        i, j, f, o = np.split(z, 4)
        c = c * _sigmoid(f + 1.0) + _sigmoid(i) * np.tanh(j)  # forget bias = 1.0
        h = np.tanh(c) * _sigmoid(o)
        return c, h

    def sample(self, h: np.ndarray, temperature: float, rng: np.random.Generator,
               keep_pen_down: bool = False) -> list[float]:
        """Sample the next [dx, dy, pen_down, pen_up, end] from the mixture density output.
        With `keep_pen_down`, lifting the pen or ending the drawing is not allowed."""
        z = h @ self.out_w + self.out_b
        pen_logits, rest = z[:3], z[3:]
        pi, mu_x, mu_y, s_x, s_y, corr = np.split(rest, 6)
        discrete_temp = 0.5 + temperature * 0.5
        pen = _softmax(pen_logits / discrete_temp)
        pi = _softmax(pi / discrete_temp)
        s_x = np.exp(s_x) * np.sqrt(temperature)
        s_y = np.exp(s_y) * np.sqrt(temperature)
        corr = np.tanh(corr)

        k = rng.choice(N_MIXTURE, p=pi)
        z1, z2 = rng.standard_normal(2)
        dx = np.sqrt(1 - corr[k] ** 2) * s_x[k] * z1 + corr[k] * s_x[k] * z2 + mu_x[k]
        dy = s_y[k] * z2 + mu_y[k]
        p = [0.0, 0.0, 0.0]
        p[0 if keep_pen_down else rng.choice(3, p=pen)] = 1.0
        return [float(dx) * self.scale_factor, float(dy) * self.scale_factor, *p]

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

    def generate(self, cursor: "Cursor", temperature: float, rng: np.random.Generator,
                 max_steps: int | None = None, stop_at_lift: bool = False,
                 min_length: float = 0.0) -> Sketch:
        """Keep drawing from `cursor` (which is not modified, so it can be reused to
        sample several alternatives). Returned strokes are relative to the origin.
        With `stop_at_lift`, stop the first time the model lifts the pen.
        With `min_length` (data units), the pen can't lift until it has drawn that far."""
        state, x, y, pen_down = cursor.state, cursor.x, cursor.y, cursor.pen_down
        drawn = 0.0
        strokes: list[Polyline] = []
        current: Polyline | None = None
        for _ in range(max_steps or self.max_seq_len):
            step = self.sample(state[1], temperature, rng, keep_pen_down=pen_down and drawn < min_length)
            if step[4] == 1.0:
                if current and len(current) > 1:
                    strokes.append(current)
                return Sketch(strokes, finished=True)
            nx, ny = x + step[0], y + step[1]
            if pen_down:
                if current is None:
                    current = [(x, y)]
                current.append((nx, ny))
                drawn += float(np.hypot(step[0], step[1]))
            if step[3] == 1.0 and current is not None:  # pen lifts after this point
                if len(current) > 1:
                    strokes.append(current)
                current = None
                if stop_at_lift:
                    return Sketch(strokes, finished=True)
            pen_down = step[2] == 1.0
            x, y = nx, ny
            state = self.update(step, state)
        if current and len(current) > 1:
            strokes.append(current)
        return Sketch(strokes, finished=False)

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
        for _ in range(3):
            sk = model.draw(None, 0.25, rng)
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


def load(name: str) -> SketchRNN:
    """Load a pretrained model by Quick Draw category (underscored, e.g. "fire_hydrant").
    Downloads once to a temp-dir cache, keeps a few in memory."""
    with _lock:
        if name in _models:
            _models.move_to_end(name)
            return _models[name]

    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{MODEL_SIZE}-{name}.gen.json")
    if not os.path.exists(path):
        url = MODEL_URL.format(size=MODEL_SIZE, name=name)
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = resp.read()
        tmp = f"{path}.{os.getpid()}.part"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    with open(path, encoding="utf-8") as f:
        model = SketchRNN(json.load(f))

    with _lock:
        _models[name] = model
        while len(_models) > MAX_MODELS_IN_MEMORY:
            _models.popitem(last=False)
    return model
