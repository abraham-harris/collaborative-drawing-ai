import type { DrawAction, Point } from '../types';

/** Render one pen stroke onto a canvas context. */
export function drawAction(ctx: CanvasRenderingContext2D, action: DrawAction): void {
  const { points, size } = action;
  if (!points.length) return;
  ctx.save();
  ctx.globalAlpha = action.opacity;
  ctx.strokeStyle = action.color;
  ctx.fillStyle = action.color;
  ctx.lineWidth = size;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  drawFreehand(ctx, points, size);
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
