"""
HTTP bridge between the React frontend and the AI model.

Run:  python server.py   (listens on http://127.0.0.1:5000)
The Vite dev server proxies /api/* here, so no CORS setup is needed.
"""

from __future__ import annotations

import base64
import io

from flask import Flask, jsonify, request
from PIL import Image

from ai_model import generate_drawing

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB


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


@app.post("/api/turn")
def turn():
    payload = request.get_json(silent=True)
    if not payload or "image" not in payload:
        return jsonify({"error": "Expected JSON body with an 'image' data URL"}), 400

    try:
        image = decode_data_url(payload["image"])
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Could not decode image: {exc}"}), 400

    result, message = generate_drawing(
        image=image,
        actions=payload.get("actions", []),
        turn=int(payload.get("turn", 1)),
        total_turns=int(payload.get("totalTurns", 1)),
    )

    response = {"image": encode_data_url(result)}
    if message:
        response["message"] = message
    return jsonify(response)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
