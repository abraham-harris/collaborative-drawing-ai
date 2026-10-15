"""Canvas-level helpers: where ink already is, where an object fits, and safe rendering."""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, field

from PIL import Image, ImageChops, ImageDraw

from .strokes import Point, Stroke, bbox, resample

BACKGROUND = (255, 255, 255)
# A pixel counts as existing ink if any channel differs from the background by more than this.
INK_THRESHOLD = 12
SUPERSAMPLE = 2  # draw AI strokes at 2x and downsample for anti-aliasing

# How the AI's strokes look. One distinct color makes it easy to see who drew what.
AI_COLOR = "#1971c2"
AI_DEFAULT_SIZE = 4.0

Polyline = list[Point]


def ink_mask(image: Image.Image) -> Image.Image:
    """'L' mask: 255 where the canvas already has ink, 0 where it is background."""
    diff = ImageChops.difference(image.convert("RGB"), Image.new("RGB", image.size, BACKGROUND))
    r, g, b = diff.split()
    strongest = ImageChops.lighter(ImageChops.lighter(r, g), b)
    return strongest.point(lambda v: 255 if v > INK_THRESHOLD else 0)


@dataclass
class DrawingContext:
    """Everything a model needs to know about the canvas this turn."""

    width: int
    height: int
    # The human's strokes from this turn (vectors).
    strokes: list[Stroke]
    # Quick Draw category the AI should draw, e.g. "cat".
    subject: str
    # Coarse grid of ALL ink on the canvas (any turn, human or AI).
    occupancy: list[list[bool]]
    cell: int
    # Bounding box of all ink on the canvas, or None if it is blank.
    ink_box: tuple[float, float, float, float] | None
    temperature: float = 0.25
    rng: random.Random = field(default_factory=random.Random)
    _table: list[list[int]] | None = field(default=None, repr=False)

    @classmethod
    def build(cls, image: Image.Image, strokes: list[Stroke], subject: str,
              temperature: float = 0.25, cell: int = 10, rng: random.Random | None = None) -> "DrawingContext":
        w, h = image.size
        mask = ink_mask(image)
        gw, gh = -(-w // cell), -(-h // cell)
        px = mask.resize((gw, gh), Image.Resampling.BOX).load()
        occupancy = [[px[x, y] > 0 for x in range(gw)] for y in range(gh)]
        return cls(w, h, strokes, subject, occupancy, cell, mask.getbbox(), temperature,
                   rng or random.Random())

    # --- queries -----------------------------------------------------------------

    def in_bounds(self, p: Point, margin: float = 0.0) -> bool:
        return margin <= p[0] <= self.width - margin and margin <= p[1] <= self.height - margin

    def occupied(self, p: Point) -> bool:
        gx, gy = int(p[0] // self.cell), int(p[1] // self.cell)
        if 0 <= gy < len(self.occupancy) and 0 <= gx < len(self.occupancy[0]):
            return self.occupancy[gy][gx]
        return True  # off-canvas counts as blocked

    def overlap(self, points: Polyline) -> float:
        """Fraction of a path that lands on existing ink (or off-canvas)."""
        samples = resample(points, self.cell / 2)
        if not samples:
            return 0.0
        return sum(self.occupied(p) for p in samples) / len(samples)

    def ink_in_box(self, x0: float, y0: float, x1: float, y1: float) -> float:
        """Fraction of grid cells inside the box that contain ink (O(1) via a summed-area table)."""
        if self._table is None:
            rows, cols = len(self.occupancy), len(self.occupancy[0])
            t = [[0] * (cols + 1) for _ in range(rows + 1)]
            for y in range(rows):
                run = 0
                for x in range(cols):
                    run += self.occupancy[y][x]
                    t[y + 1][x + 1] = t[y][x + 1] + run
            self._table = t
        rows, cols = len(self.occupancy), len(self.occupancy[0])
        gx0, gy0 = max(0, int(x0 // self.cell)), max(0, int(y0 // self.cell))
        gx1, gy1 = min(cols, int(-(-x1 // self.cell))), min(rows, int(-(-y1 // self.cell)))
        if gx1 <= gx0 or gy1 <= gy0:
            return 0.0
        t = self._table
        filled = t[gy1][gx1] - t[gy0][gx1] - t[gy1][gx0] + t[gy0][gx0]
        return filled / ((gx1 - gx0) * (gy1 - gy0))

    def user_box(self) -> tuple[float, float, float, float] | None:
        pts = [p for s in self.strokes for p in s.points]
        return bbox(pts) if pts else None

    def target_size(self) -> float:
        """How big (px) the AI's object should be: about the size of what the human drew."""
        box = self.user_box()
        limit = 0.6 * min(self.width, self.height)
        if box is None:
            return 0.45 * min(self.width, self.height)
        return max(120.0, min(limit, max(box[2] - box[0], box[3] - box[1])))

    def ai_style(self) -> dict:
        sizes = [s.size for s in self.strokes]
        size = statistics.median(sizes) if sizes else AI_DEFAULT_SIZE
        return dict(color=AI_COLOR, size=max(2.0, min(8.0, size)), opacity=1.0, kind="brush")

    # --- placement ---------------------------------------------------------------

    def find_spot(self, w: float, h: float, near: Point | None = None,
                  margin: float = 10.0) -> tuple[float, float, float] | None:
        """Best top-left corner for a w x h object: empty space first, then close to `near`.
        Returns (x, y, ink_fraction), or None if it can't fit on the canvas at all."""
        if w > self.width - 2 * margin or h > self.height - 2 * margin:
            return None
        if near is None:
            near = (self.width / 2, self.height / 2)
        diag = (self.width ** 2 + self.height ** 2) ** 0.5
        best = None
        step = 15
        y = margin
        while y + h <= self.height - margin:
            x = margin
            while x + w <= self.width - margin:
                # Pad the box so the object doesn't end up touching existing ink.
                ink = self.ink_in_box(x - 15, y - 15, x + w + 15, y + h + 15)
                cx, cy = x + w / 2, y + h / 2
                d = ((cx - near[0]) ** 2 + (cy - near[1]) ** 2) ** 0.5 / diag
                score = ink * 5 + d
                if best is None or score < best[0]:
                    best = (score, x, y, ink)
                x += step
            y += step
        return None if best is None else (best[1], best[2], best[3])


def fit_into(strokes: list[Polyline], x: float, y: float, size: float, mirror: bool = False) -> list[Polyline]:
    """Scale polylines so their larger side is `size` px and put their top-left at (x, y)."""
    x0, y0, x1, y1 = bbox([p for s in strokes for p in s])
    k = size / max(x1 - x0, y1 - y0, 1e-6)
    out = []
    for s in strokes:
        pts = []
        for px, py in s:
            dx = (x1 - px) if mirror else (px - x0)
            pts.append((x + dx * k, y + (py - y0) * k))
        out.append(pts)
    return out


def place_object(ctx: DrawingContext, strokes: list[Polyline], near: Point | None = None,
                 max_ink: float = 0.08) -> list[Polyline] | None:
    """Put a drawing (any units) into the emptiest nearby spot, shrinking it if needed."""
    x0, y0, x1, y1 = bbox([p for s in strokes for p in s])
    aspect_w = (x1 - x0) / max(x1 - x0, y1 - y0, 1e-6)
    aspect_h = (y1 - y0) / max(x1 - x0, y1 - y0, 1e-6)
    size = ctx.target_size()
    best = None
    for factor in (1.0, 0.8, 0.65, 0.5):
        s = size * factor
        spot = ctx.find_spot(s * aspect_w, s * aspect_h, near)
        if spot is None:
            continue
        x, y, ink = spot
        if best is None or ink < best[3]:
            best = (x, y, s, ink)
        if ink <= max_ink:
            break
    if best is None:
        return None
    x, y, s, _ = best
    return fit_into(strokes, x, y, s, mirror=ctx.rng.random() < 0.5)


def render(image: Image.Image, strokes: list[Stroke]) -> Image.Image:
    """Draw AI strokes onto the canvas WITHOUT changing any existing ink.

    Strokes are painted on a transparent layer and composited over the canvas,
    then every pixel that was already inked is pasted back unchanged. So the AI
    can only add; its lines appear to pass behind existing ones.
    """
    base = image.convert("RGBA")
    if not strokes:
        return base.convert("RGB")

    w, h = base.size
    layer = Image.new("RGBA", (w * SUPERSAMPLE, h * SUPERSAMPLE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for s in strokes:
        rgba = hex_to_rgb(s.color) + (round(255 * max(0.0, min(1.0, s.opacity))),)
        width = max(1, round(s.size * SUPERSAMPLE))
        pts = [(x * SUPERSAMPLE, y * SUPERSAMPLE) for x, y in s.points]
        if len(pts) > 1:
            draw.line(pts, fill=rgba, width=width, joint="curve")
        if not s.closed:  # round end caps
            r = width / 2
            for x, y in (pts[0], pts[-1]):
                draw.ellipse((x - r, y - r, x + r, y + r), fill=rgba)

    # Downsample in premultiplied alpha so edges don't pick up dark fringes.
    layer = layer.convert("RGBa").resize((w, h), Image.Resampling.LANCZOS).convert("RGBA")
    combined = Image.alpha_composite(base, layer)
    combined.paste(base, mask=ink_mask(base))
    return combined.convert("RGB")


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    c = color.lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    n = int(c[:6], 16)
    return (n >> 16) & 255, (n >> 8) & 255, n & 255
