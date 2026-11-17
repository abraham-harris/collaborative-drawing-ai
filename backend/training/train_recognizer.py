"""Train the doodle recognizer used by drawing_models/recognizer.py.

A small CNN learns to tell apart the 95 subjects the AI can draw, from Quick,
Draw!'s 28x28 bitmaps (Google, CC BY 4.0). Each category file is ~100 MB, so
this downloads only slices of rows with HTTP Range requests (~330 MB total,
cached in the temp directory). Training takes a few minutes on a laptop CPU.

The weights are exported to drawing_models/recognizer.npz, which the app runs
with numpy alone (no PyTorch at runtime).

Run from backend/ (needs torch, which is in the repository's root requirements.txt):
    python training/train_recognizer.py
"""

from __future__ import annotations

import json
import os
import random
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import nn

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from drawing_models import recognizer  # noqa: E402
from drawing_models.subjects import SUBJECTS  # noqa: E402

BITMAP_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{name}.npy"
STROKES_URL = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/{name}.ndjson"
CACHE = Path(tempfile.gettempdir()) / "quickdraw_recognizer_data"
HEADER_BYTES = 80                 # every numpy_bitmap .npy has an 80-byte header
CHUNKS, CHUNK_ROWS = 4, 1000      # training rows per class: 4 random slices of 1000
VAL_ROWS = 500                    # held-out rows per class, from another slice
EPOCHS = 8
SEED = 0


def _url(template: str, name: str) -> str:
    return template.format(name=urllib.parse.quote(name.replace("_", " ")))


def _get(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def fetch_bitmaps(name: str) -> tuple[np.ndarray, np.ndarray]:
    """(train, val) uint8 arrays of shape (n, 28, 28) for one category, cached."""
    path = CACHE / f"{name}.npz"
    if path.exists():
        with np.load(path) as z:
            return z["train"], z["val"]
    url = _url(BITMAP_URL, name)
    header = _get(url, 0, HEADER_BYTES - 1).decode("latin1")
    total = int(header.split("'shape': (")[1].split(",")[0])
    rng = random.Random(f"{SEED}-{name}")
    # Non-overlapping slices spread through the file: CHUNKS for training, 1 for validation.
    slots = rng.sample(range(total // CHUNK_ROWS), CHUNKS + 1)
    parts = []
    for slot in slots:
        rows = CHUNK_ROWS if len(parts) < CHUNKS else VAL_ROWS
        start = HEADER_BYTES + slot * CHUNK_ROWS * 784
        parts.append(np.frombuffer(_get(url, start, start + rows * 784 - 1), np.uint8).reshape(-1, 28, 28))
    train, val = np.concatenate(parts[:CHUNKS]), parts[CHUNKS]
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, train=train, val=val)
    return train, val


def fetch_stroke_drawings(name: str, n: int = 40) -> list[list[list[tuple[float, float]]]]:
    """A few recognized drawings in vector form, to test our own rasterizer end to end."""
    path = CACHE / f"{name}.strokes.json"
    if path.exists():
        return json.loads(path.read_text())
    url = _url(STROKES_URL, name)
    start = 2_000_000  # a slice away from the start of the file
    body = _get(url, start, start + 200_000)
    drawings = []
    for line in body.split(b"\n")[1:-1]:
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if record.get("recognized"):
            drawings.append([list(zip(xs, ys)) for xs, ys, *_ in record["drawing"]])
        if len(drawings) == n:
            break
    path.write_text(json.dumps(drawings))
    return drawings


class Net(nn.Module):
    """Same layers as recognizer.Recognizer (kept in sync by export())."""

    def __init__(self, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),     # 14x14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 7x7
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 3x3
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 3 * 3, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def augment(x: torch.Tensor) -> torch.Tensor:
    """Small random shifts, so slightly off-center user drawings still match."""
    dx, dy = random.randint(-2, 2), random.randint(-2, 2)
    return torch.roll(x, shifts=(dy, dx), dims=(2, 3))


def export(net: Net, path: Path) -> None:
    convs = [m for m in net.features if isinstance(m, nn.Conv2d)]
    fcs = [m for m in net.classifier if isinstance(m, nn.Linear)]
    arrays = {}
    for i, m in enumerate(convs):
        arrays[f"conv{i}_w"], arrays[f"conv{i}_b"] = m.weight.detach().numpy(), m.bias.detach().numpy()
    for i, m in enumerate(fcs):
        arrays[f"fc{i}_w"], arrays[f"fc{i}_b"] = m.weight.detach().numpy(), m.bias.detach().numpy()
    np.savez_compressed(path, classes=np.array(SUBJECTS), **{k: v.astype(np.float32) for k, v in arrays.items()})


def main() -> None:
    torch.manual_seed(SEED)
    random.seed(SEED)
    t = time.time()
    with ThreadPoolExecutor(16) as pool:
        data = list(pool.map(fetch_bitmaps, SUBJECTS))
    print(f"data: {len(SUBJECTS)} classes in {time.time() - t:.0f}s")

    x_train = np.concatenate([d[0] for d in data]).astype(np.float32)[:, None] / 255
    y_train = np.concatenate([np.full(len(d[0]), i) for i, d in enumerate(data)])
    x_val = np.concatenate([d[1] for d in data]).astype(np.float32)[:, None] / 255
    y_val = np.concatenate([np.full(len(d[1]), i) for i, d in enumerate(data)])
    x_train, y_train = torch.from_numpy(x_train), torch.from_numpy(y_train)
    x_val, y_val = torch.from_numpy(x_val), torch.from_numpy(y_val)

    net = Net(len(SUBJECTS))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=EPOCHS * (len(x_train) // 256 + 1))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    for epoch in range(EPOCHS):
        net.train()
        perm = torch.randperm(len(x_train))
        t = time.time()
        for i in range(0, len(perm), 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            loss = loss_fn(net(augment(x_train[idx])), y_train[idx])
            loss.backward()
            opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            logits = torch.cat([net(x_val[i:i + 2048]) for i in range(0, len(x_val), 2048)])
        top1 = (logits.argmax(1) == y_val).float().mean().item()
        top5 = (logits.topk(5, 1).indices == y_val[:, None]).any(1).float().mean().item()
        print(f"epoch {epoch + 1}: val top-1 {top1:.3f} top-5 {top5:.3f} ({time.time() - t:.0f}s)")

    out = BACKEND / "drawing_models" / "recognizer.npz"
    export(net, out)
    print(f"exported {out} ({os.path.getsize(out) / 1e6:.1f} MB)")

    # End to end: vector drawings -> our rasterizer -> numpy model, as the app does it.
    model = recognizer.Recognizer(out)
    with ThreadPoolExecutor(16) as pool:
        strokes = list(pool.map(fetch_stroke_drawings, SUBJECTS))
    hits1 = hits5 = total = 0
    for label, drawings in zip(SUBJECTS, strokes):
        for d in drawings:
            ranked = [s for s, _ in model.predict(recognizer.rasterize(d))]
            hits1 += ranked[0] == label
            hits5 += label in ranked[:5]
            total += 1
    print(f"end to end on {total} vector drawings: top-1 {hits1 / total:.3f} top-5 {hits5 / total:.3f}")


if __name__ == "__main__":
    main()
