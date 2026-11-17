"""
HTTP bridge between the React frontend and the AI model.

Run:  python server.py   (listens on http://127.0.0.1:5000)
The Vite dev server proxies /api/* here, so no CORS setup is needed.
"""

from __future__ import annotations

import base64
import binascii
import io
import math
import os
import sys
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, request
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_model import DEFAULT_ITERATIONS, generate_drawing, score_options
from drawing_models import DEFAULT_MODE, MODELS, RANDOM, SUBJECTS

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB
MAX_SCORE_REQUEST_BYTES = 4 * 1024 * 1024
MAX_SCORE_PIXELS = 2_000_000
MAX_ITERATIONS = 30


def decode_data_url(data_url: str) -> Image.Image:
    """'data:image/png;base64,....' -> PIL image (RGBA)."""
    _, _, encoded = data_url.partition(",")
    return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGBA")


def encode_data_url(image: Image.Image) -> str:
    """PIL image -> 'data:image/png;base64,....'"""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


@app.get("/api/health")
def health():
    return jsonify({"ok": True})


@app.get("/api/subjects")
def subjects():
    return jsonify({"subjects": SUBJECTS, "random": RANDOM})


@app.post("/api/turn")
def turn():
    payload = request.get_json(silent=True)
    if not payload or "image" not in payload:
        return jsonify({"error": "Expected JSON body with an 'image' data URL"}), 400

    try:
        image = decode_data_url(payload["image"])
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Could not decode image: {exc}"}), 400

    mode = payload.get("mode", DEFAULT_MODE)
    if mode not in MODELS:
        return jsonify({"error": f"Unknown mode {mode!r}; expected one of {sorted(MODELS)}"}), 400

    try:
        iterations = max(1, min(MAX_ITERATIONS, int(payload.get("greedyIterations", DEFAULT_ITERATIONS))))
    except (TypeError, ValueError):
        return jsonify({"error": "greedyIterations must be a whole number"}), 400

    # Ink, AuDrA scores, and timings for the page's console.
    start = time.perf_counter()
    report = {"turn": payload.get("turn"), "totalTurns": payload.get("totalTurns"), "mode": mode}
    try:
        results = generate_drawing(
            image=image,
            actions=payload.get("actions", []),
            context_actions=payload.get("contextActions"),
            mode=mode,
            iterations=iterations,
            report=report,
            recognize=payload.get("subjectSource", "recognize") != "random",
        )
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 502
    score_options(image, [(result, ai_actions) for result, _message, ai_actions, _subject in results], report)
    if "recognition" in report:
        # Which interpretation each option drew. The page reveals this only after the
        # player chooses; with random subjects it's never sent (as in the original duel).
        report["recognition"]["optionSubjects"] = [subject for *_rest, subject in results]
    report.setdefault("timings", {})["total"] = time.perf_counter() - start

    return jsonify({"candidates": [
        {"image": encode_data_url(result), "mode": mode,
         "aiActions": ai_actions, "message": message}
        for result, message, ai_actions, _subject in results
    ], "report": report})


@app.post("/api/score")
def score():
    if request.content_length is not None and request.content_length > MAX_SCORE_REQUEST_BYTES:
        return jsonify({"error": "Image is too large to score"}), 413

    payload = request.get_json(silent=True)
    data_url = payload.get("image") if isinstance(payload, dict) else None
    prefix = "data:image/png;base64,"
    if not isinstance(data_url, str) or not data_url.startswith(prefix):
        return jsonify({"error": "Expected a PNG image data URL"}), 400
    if len(data_url) > MAX_SCORE_REQUEST_BYTES:
        return jsonify({"error": "Image is too large to score"}), 413

    try:
        image_bytes = base64.b64decode(data_url[len(prefix):], validate=True)
        with Image.open(io.BytesIO(image_bytes)) as source:
            if source.format != "PNG" or source.width * source.height > MAX_SCORE_PIXELS:
                raise ValueError("Expected a PNG of at most 2 million pixels")
            image = source.convert("RGB")
    except (binascii.Error, Image.DecompressionBombError, OSError, ValueError) as exc:
        return jsonify({"error": f"Could not decode image: {exc}"}), 400

    try:
        from audra_service import score_drawing

        value = score_drawing(image)
        if not math.isfinite(value):
            raise RuntimeError("AuDrA returned a non-finite score")
    except Exception:  # noqa: BLE001
        app.logger.exception("AuDrA scoring failed")
        return jsonify({"error": "Creativity scoring is temporarily unavailable"}), 503

    response = jsonify({"score": value})
    response.headers["Cache-Control"] = "no-store"
    return response


def _warm_audra() -> None:
    try:
        from audra_service import warm

        warm()
        app.logger.info("AuDrA creativity model loaded")
    except Exception:  # noqa: BLE001 -- scoring is optional; turns report it as unavailable
        app.logger.warning("AuDrA creativity model unavailable; turns will be unscored", exc_info=True)


if __name__ == "__main__":
    # Load PyTorch + AuDrA (~4-6 s) in the background now instead of during the first turn.
    # With debug=True the reloader runs this file twice; only warm the process that serves.
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        threading.Thread(target=_warm_audra, daemon=True).start()
    app.run(host="127.0.0.1", port=5000, debug=True)
