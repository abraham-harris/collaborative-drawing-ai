"""Load AuDrA's private checkpoint and score a single drawing."""

from __future__ import annotations

import hashlib
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from urllib.parse import unquote, urlsplit

from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
LOCAL_CHECKPOINT = ROOT / "AuDrA" / "AuDrA_trained.ckpt"
CHECKPOINT_SHA256 = "6de2b8dbf49ad3dfb964958179e263c4dca2902f82f47b16850434c6fb94776d"
MAX_CHECKPOINT_BYTES = 160 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stream_checkpoint(pathname: str):
    from vercel.blob import sync as blob

    return blob.stream_blob(pathname, access="private")


def _checkpoint_path() -> Path:
    if LOCAL_CHECKPOINT.is_file():
        return LOCAL_CHECKPOINT

    blob_url = os.environ.get("AUDRA_CHECKPOINT_BLOB_URL")
    if not blob_url or not (os.environ.get("BLOB_STORE_ID") or os.environ.get("BLOB_READ_WRITE_TOKEN")):
        raise RuntimeError("AuDrA checkpoint storage is not configured")

    # Use a pathname so the connected store determines the destination, even if
    # the saved URL contains an older store hostname.
    pathname = unquote(urlsplit(blob_url).path.lstrip("/"))
    if not pathname:
        raise RuntimeError("AuDrA checkpoint URL has no pathname")

    cached = Path(tempfile.gettempdir()) / f"audra-{CHECKPOINT_SHA256[:16]}.ckpt"
    if cached.is_file() and _sha256(cached) == CHECKPOINT_SHA256:
        return cached

    digest = hashlib.sha256()
    size = 0
    with tempfile.NamedTemporaryFile(dir=cached.parent, prefix="audra-", suffix=".tmp", delete=False) as file:
        temp_path = Path(file.name)
        try:
            with _stream_checkpoint(pathname) as download:
                for chunk in download:
                    size += len(chunk)
                    if size > MAX_CHECKPOINT_BYTES:
                        raise RuntimeError("AuDrA checkpoint is larger than expected")
                    digest.update(chunk)
                    file.write(chunk)
            if digest.hexdigest() != CHECKPOINT_SHA256:
                raise RuntimeError("AuDrA checkpoint checksum did not match")
            os.replace(temp_path, cached)
        finally:
            temp_path.unlink(missing_ok=True)
    return cached


# More threads than this made batch scoring slower and less predictable on a
# 12-thread laptop with other apps open (worst case 1.97 s -> 0.83 s at 6).
MAX_TORCH_THREADS = 6


@lru_cache(maxsize=1)
def _model():
    import torch
    from AuDrA.score import load_model

    torch.set_num_threads(max(1, min(torch.get_num_threads(), MAX_TORCH_THREADS)))
    return load_model(_checkpoint_path())


def warm() -> None:
    """Import PyTorch and load the checkpoint now (~4 s), so the first scored turn is fast."""
    _model()


def score_drawings(images: list[Image.Image]) -> list[float]:
    """Score same-size drawings in one batch; used to rank the AI's candidates."""
    from flask import has_request_context, request

    if has_request_context():
        from vercel.headers import set_headers

        set_headers(request.headers)
    from AuDrA.score import score_batch

    return score_batch(images, _model())


def score_drawing(image: Image.Image) -> float:
    from flask import has_request_context, request
    from vercel.headers import set_headers

    if has_request_context():
        set_headers(request.headers)
    from AuDrA.score import score_image

    return score_image(image, _model())
