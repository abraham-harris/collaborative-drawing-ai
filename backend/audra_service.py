"""Load AuDrA's private checkpoint and score a single drawing."""

from __future__ import annotations

import hashlib
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from urllib.request import Request, urlopen

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


def _checkpoint_path() -> Path:
    if LOCAL_CHECKPOINT.is_file():
        return LOCAL_CHECKPOINT

    blob_url = os.environ.get("AUDRA_CHECKPOINT_BLOB_URL")
    token = os.environ.get("BLOB_READ_WRITE_TOKEN")
    if not blob_url or not token:
        raise RuntimeError("AuDrA checkpoint storage is not configured")

    cached = Path(tempfile.gettempdir()) / f"audra-{CHECKPOINT_SHA256[:16]}.ckpt"
    if cached.is_file() and _sha256(cached) == CHECKPOINT_SHA256:
        return cached

    request = Request(blob_url, headers={"Authorization": f"Bearer {token}"})
    digest = hashlib.sha256()
    size = 0
    with tempfile.NamedTemporaryFile(dir=cached.parent, prefix="audra-", suffix=".tmp", delete=False) as file:
        temp_path = Path(file.name)
        try:
            with urlopen(request, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
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


@lru_cache(maxsize=1)
def _model():
    from AuDrA.score import load_model

    return load_model(_checkpoint_path())


def score_drawing(image: Image.Image) -> float:
    from AuDrA.score import score_image

    return score_image(image, _model())
