"""Stroke representation and geometry helpers shared by all drawing models."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

Point = tuple[float, float]


@dataclass
class Stroke:
    """A polyline in canvas pixel coordinates plus its paint style."""

    points: list[Point]
    color: str = "#1e1e1e"
    size: float = 8.0
    opacity: float = 1.0
    closed: bool = False
    # The frontend tool this came from ("brush", "line", "rectangle", ...).
    kind: str = "brush"
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> float:
        return path_length(self.points)


# --- conversion to/from the frontend's DrawAction format ---------------------


def strokes_from_actions(actions: list[dict[str, Any]]) -> list[Stroke]:
    """Convert the frontend's DrawAction list into polylines.

    Eraser and fill actions are skipped: they aren't lines the models can build on.
    """
    strokes: list[Stroke] = []
    for action in actions:
        tool = action.get("tool")
        pts = [(float(p["x"]), float(p["y"])) for p in action.get("points", [])]
        if tool in ("eraser", "fill") or not pts:
            continue

        style = dict(
            color=action.get("color", "#1e1e1e"),
            size=float(action.get("size", 8)),
            opacity=float(action.get("opacity", 1)),
            kind=tool,
        )
        a, b = pts[0], pts[-1]
        if tool == "brush":
            stroke = Stroke(points=pts, **style)
        elif tool == "line":
            stroke = Stroke(points=[a, b], **style)
        elif tool == "rectangle":
            (x0, y0), (x1, y1) = a, b
            corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
            stroke = Stroke(points=corners, closed=True, **style)
        elif tool == "ellipse":
            cx, cy = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
            rx, ry = abs(b[0] - a[0]) / 2, abs(b[1] - a[1]) / 2
            ring = [
                (cx + rx * math.cos(t), cy + ry * math.sin(t))
                for t in (2 * math.pi * i / 64 for i in range(65))
            ]
            stroke = Stroke(points=ring, closed=True, **style)
        else:
            continue

        if stroke.length >= 2:  # ignore single-click dots
            strokes.append(stroke)
    return strokes


def stroke_to_action(stroke: Stroke) -> dict[str, Any]:
    """Convert an AI stroke back to the frontend's DrawAction shape (always a brush path)."""
    return {
        "tool": "brush",
        "color": stroke.color,
        "size": stroke.size,
        "opacity": stroke.opacity,
        "filled": False,
        "points": [{"x": round(x, 2), "y": round(y, 2)} for x, y in stroke.points],
    }


# --- geometry ------------------------------------------------------------------


def dist(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def path_length(points: list[Point]) -> float:
    return sum(dist(points[i], points[i + 1]) for i in range(len(points) - 1))


def resample(points: list[Point], spacing: float) -> list[Point]:
    """Return points evenly spaced `spacing` px apart along the path."""
    if len(points) < 2:
        return list(points)
    out = [points[0]]
    carry = 0.0  # distance travelled since the last emitted point
    for i in range(len(points) - 1):
        a, b = points[i], points[i + 1]
        seg = dist(a, b)
        if seg == 0:
            continue
        t = spacing - carry
        while t <= seg:
            out.append((a[0] + (b[0] - a[0]) * t / seg, a[1] + (b[1] - a[1]) * t / seg))
            t += spacing
        carry = seg - (t - spacing)
    if dist(out[-1], points[-1]) > spacing * 0.25:
        out.append(points[-1])
    return out


def bbox(points: list[Point]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def simplify(points: list[Point], epsilon: float) -> list[Point]:
    """Ramer-Douglas-Peucker line simplification (Quick Draw data was simplified this way)."""
    if len(points) < 3:
        return list(points)
    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        lo, hi = stack.pop()
        (ax, ay), (bx, by) = points[lo], points[hi]
        seg = math.hypot(bx - ax, by - ay)
        best_d, best_i = -1.0, lo
        for i in range(lo + 1, hi):
            px, py = points[i]
            if seg == 0:
                d = math.hypot(px - ax, py - ay)
            else:
                d = abs((bx - ax) * (ay - py) - (ax - px) * (by - ay)) / seg
            if d > best_d:
                best_d, best_i = d, i
        if best_d > epsilon:
            keep[best_i] = True
            stack.append((lo, best_i))
            stack.append((best_i, hi))
    return [p for p, k in zip(points, keep) if k]
