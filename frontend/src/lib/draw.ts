import { BACKGROUND_COLOR } from '../config';
import type { DrawAction, Point } from '../types';

/** Renders a single stroke or shape onto a 2D context. (Fill is handled by floodFill.) */
export function drawAction(ctx: CanvasRenderingContext2D, action: DrawAction): void {
  const { tool, points, size, filled } = action;
  if (points.length === 0 || tool === 'fill') return;

  const color = tool === 'eraser' ? BACKGROUND_COLOR : action.color;
  ctx.save();
  ctx.globalAlpha = tool === 'eraser' ? 1 : action.opacity;
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = size;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  switch (tool) {
    case 'brush':
    case 'eraser':
      drawFreehand(ctx, points, size);
      break;
    case 'line': {
      const [a, b = a] = points;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
      break;
    }
    case 'rectangle': {
      const [a, b = a] = points;
      const x = Math.min(a.x, b.x);
      const y = Math.min(a.y, b.y);
      const w = Math.abs(b.x - a.x);
      const h = Math.abs(b.y - a.y);
      if (filled) ctx.fillRect(x, y, w, h);
      else ctx.strokeRect(x, y, w, h);
      break;
    }
    case 'ellipse': {
      const [a, b = a] = points;
      ctx.beginPath();
      ctx.ellipse(
        (a.x + b.x) / 2,
        (a.y + b.y) / 2,
        Math.abs(b.x - a.x) / 2,
        Math.abs(b.y - a.y) / 2,
        0,
        0,
        Math.PI * 2,
      );
      if (filled) ctx.fill();
      else ctx.stroke();
      break;
    }
  }
  ctx.restore();
}

/** Smooth freehand path using quadratic curves through segment midpoints. */
function drawFreehand(ctx: CanvasRenderingContext2D, points: Point[], size: number): void {
  if (points.length === 1) {
    ctx.beginPath();
    ctx.arc(points[0].x, points[0].y, size / 2, 0, Math.PI * 2);
    ctx.fill();
    return;
  }
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (let i = 1; i < points.length - 1; i++) {
    const midX = (points[i].x + points[i + 1].x) / 2;
    const midY = (points[i].y + points[i + 1].y) / 2;
    ctx.quadraticCurveTo(points[i].x, points[i].y, midX, midY);
  }
  const last = points[points.length - 1];
  ctx.lineTo(last.x, last.y);
  ctx.stroke();
}

function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace('#', '');
  const full = h.length === 3 ? h.split('').map((c) => c + c).join('') : h;
  const n = parseInt(full, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

/**
 * Paint-bucket fill: replaces the contiguous region around `seed` whose color is
 * within `tolerance` (0..255 per channel) of the seed color.
 */
export function floodFill(
  ctx: CanvasRenderingContext2D,
  seed: Point,
  color: string,
  opacity: number,
  tolerance = 32,
): void {
  const { width, height } = ctx.canvas;
  const sx = Math.floor(seed.x);
  const sy = Math.floor(seed.y);
  if (sx < 0 || sy < 0 || sx >= width || sy >= height) return;

  const imageData = ctx.getImageData(0, 0, width, height);
  const data = imageData.data;
  const start = (sy * width + sx) * 4;
  const target = [data[start], data[start + 1], data[start + 2], data[start + 3]];
  const [r, g, b] = hexToRgb(color);

  const matches = (i: number) =>
    Math.abs(data[i] - target[0]) <= tolerance &&
    Math.abs(data[i + 1] - target[1]) <= tolerance &&
    Math.abs(data[i + 2] - target[2]) <= tolerance &&
    Math.abs(data[i + 3] - target[3]) <= tolerance;

  const visited = new Uint8Array(width * height);
  const stack: number[] = [sx, sy];

  while (stack.length) {
    const y = stack.pop()!;
    let x = stack.pop()!;
    // Walk left to the start of this horizontal span.
    while (x >= 0 && !visited[y * width + x] && matches((y * width + x) * 4)) x--;
    x++;
    let spanUp = false;
    let spanDown = false;
    while (x < width && !visited[y * width + x] && matches((y * width + x) * 4)) {
      const p = y * width + x;
      const i = p * 4;
      visited[p] = 1;
      data[i] = Math.round(r * opacity + data[i] * (1 - opacity));
      data[i + 1] = Math.round(g * opacity + data[i + 1] * (1 - opacity));
      data[i + 2] = Math.round(b * opacity + data[i + 2] * (1 - opacity));
      data[i + 3] = 255;

      if (y > 0) {
        const up = (p - width) * 4;
        const ok = !visited[p - width] && matches(up);
        if (ok && !spanUp) stack.push(x, y - 1);
        spanUp = ok;
      }
      if (y < height - 1) {
        const down = (p + width) * 4;
        const ok = !visited[p + width] && matches(down);
        if (ok && !spanDown) stack.push(x, y + 1);
        spanDown = ok;
      }
      x++;
    }
  }
  ctx.putImageData(imageData, 0, 0);
}
