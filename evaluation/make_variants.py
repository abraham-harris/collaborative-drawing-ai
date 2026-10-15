import base64
import io
import requests
from PIL import Image

BASE_URL = "http://127.0.0.1:5000"
CANVAS_SIZE = (800, 600)  # from CANVAS_WIDTH, CANVAS_HEIGHT in config.ts

# Fixed stroke so every one of the 10 runs gets the exact same human input.
FIXED_ACTIONS = [
    {"tool": "brush", "color": "#000000", "size": 4, "opacity": 1.0,
     "filled": False,
     "points": [{"x": 100, "y": 100}, {"x": 200, "y": 150}, {"x": 300, "y": 100}]}
]

def image_to_data_url(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

def data_url_to_image(data_url):
    _, _, encoded = data_url.partition(",")
    return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGBA")

def run_one_session():
    blank_canvas = Image.new("RGBA", CANVAS_SIZE, (255, 255, 255, 255))
    payload = {
        "image": image_to_data_url(blank_canvas),
        "actions": FIXED_ACTIONS,
        "turn": 1,
        "totalTurns": 1,
    }
    resp = requests.post(f"{BASE_URL}/api/turn", json=payload)
    resp.raise_for_status()
    return data_url_to_image(resp.json()["image"])

if __name__ == "__main__":
    for i in range(1, 11):
        final = run_one_session()
        final.convert("RGB").save(f"variant_{i:02d}.png")
        print(f"saved variant_{i:02d}.png")