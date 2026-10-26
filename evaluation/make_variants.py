import base64
import io
import json
import requests
from PIL import Image, ImageDraw, ImageColor

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


def _quadratic_curve_points(points, samples_per_segment=12):
    """Port of the frontend's drawFreehand (frontend/src/lib/draw.ts): a
    smooth path through `points` using quadraticCurveTo through each
    segment's midpoint, ending with a straight lineTo the last point.
    Returns a flattened list of (x, y) tuples approximating that path, for
    PIL (which has no native curve-stroke primitive) to draw as segments.
    """
    pts = [(p["x"], p["y"]) for p in points]
    if len(pts) == 1:
        return pts  # caller draws a single dot for this case

    path = [pts[0]]
    current = pts[0]
    for i in range(1, len(pts) - 1):
        control = pts[i]
        end = ((pts[i][0] + pts[i + 1][0]) / 2, (pts[i][1] + pts[i + 1][1]) / 2)
        for s in range(1, samples_per_segment + 1):
            t = s / samples_per_segment
            x = (1 - t) ** 2 * current[0] + 2 * (1 - t) * t * control[0] + t ** 2 * end[0]
            y = (1 - t) ** 2 * current[1] + 2 * (1 - t) * t * control[1] + t ** 2 * end[1]
            path.append((x, y))
        current = end
    path.append(pts[-1])  # final ctx.lineTo(last)
    return path


def render_actions(actions, canvas_size, background=(255, 255, 255, 255)):
    """Isolate ONLY the AI's own strokes (the aiActions the backend returns)
    by rendering them onto a blank canvas -- rather than embedding the
    whole final image, which also contains the fixed human seed stroke
    diluting the signal.

    This ports frontend/src/lib/draw.ts's drawAction/drawFreehand (checked
    against the group's repo on the `draw` branch): round line caps/joins,
    quadratic-curve smoothing through segment midpoints, globalAlpha for
    opacity. Only the 'brush' tool is implemented, which is deliberate, not
    a shortcut: backend/drawing_models/strokes.py's stroke_to_action()
    always emits aiActions with tool='brush', filled=False -- the AI never
    uses line/rectangle/ellipse/fill/eraser, so there's nothing else to
    port for this use case. Those are still skipped defensively (with a
    note) rather than silently mis-rendered, in case that ever changes.
    """
    canvas = Image.new("RGBA", canvas_size, background)
    skipped_tools = set()

    for action in actions or []:
        tool = action.get("tool", "brush")
        points = action.get("points") or []
        if not points:
            continue

        if tool != "brush":
            skipped_tools.add(tool)
            continue

        color = action.get("color", "#000000")
        size = int(action.get("size", 4))
        opacity = float(action.get("opacity", 1.0))
        alpha = max(0, min(255, round(opacity * 255)))
        rgba = (*ImageColor.getrgb(color), alpha)

        # Draw this stroke on its own transparent layer so opacity blends
        # correctly even where strokes overlap, then composite onto canvas
        # (matches ctx.globalAlpha applying per-stroke, not per-pixel-sample).
        layer = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)

        if len(points) == 1:
            r = size / 2
            x, y = points[0]["x"], points[0]["y"]
            draw.ellipse([x - r, y - r, x + r, y + r], fill=rgba)
        else:
            xy = _quadratic_curve_points(points)
            draw.line(xy, fill=rgba, width=size, joint="curve")
            r = size / 2
            for x, y in xy:  # round caps/joins (canvas lineCap/lineJoin = 'round')
                draw.ellipse([x - r, y - r, x + r, y + r], fill=rgba)

        canvas = Image.alpha_composite(canvas, layer)

    if skipped_tools:
        print(f"  [note] render_actions skipped unsupported tool type(s): {sorted(skipped_tools)} "
              f"-- only 'brush' is implemented (the AI model never emits the others)")

    return canvas


def run_one_session():
    """Calls the backend once with the fixed seed and returns
    (full_image, ai_only_image, subject, n_ai_actions)."""
    blank_canvas = Image.new("RGBA", CANVAS_SIZE, (255, 255, 255, 255))
    payload = {
        "image": image_to_data_url(blank_canvas),
        "actions": FIXED_ACTIONS,
        "turn": 1,
        "totalTurns": 1,
    }
    resp = requests.post(f"{BASE_URL}/api/turn", json=payload)
    resp.raise_for_status()
    data = resp.json()

    full_image = data_url_to_image(data["image"])
    subject = data.get("subject")
    ai_actions = data.get("aiActions") or []
    ai_only_image = render_actions(ai_actions, CANVAS_SIZE)

    return full_image, ai_only_image, subject, len(ai_actions)


if __name__ == "__main__":
    subjects = {}
    for i in range(1, 11):
        full_image, ai_only_image, subject, n_actions = run_one_session()

        full_filename = f"variant_{i:02d}.png"
        ai_only_filename = f"ai_only_{i:02d}.png"

        full_image.convert("RGB").save(full_filename)
        ai_only_image.convert("RGB").save(ai_only_filename)
        subjects[full_filename] = subject

        print(f"saved {full_filename} + {ai_only_filename}  (subject: {subject}, {n_actions} ai strokes)")

    with open("subjects.json", "w") as f:
        json.dump(subjects, f, indent=2)
    print("saved subjects.json")