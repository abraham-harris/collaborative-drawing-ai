import type { DrawAction, Point } from '../types';

export function pathLength(points: Point[]): number {
  return points.slice(1).reduce((length, point, i) =>
    length + Math.hypot(point.x - points[i].x, point.y - points[i].y), 0);
}

export function inkUsed(actions: DrawAction[]): number {
  return actions.reduce((total, action) => total + Math.max(action.size, pathLength(action.points)), 0);
}

/** Add one pen point, cutting the segment exactly where the path budget ends. */
export function appendWithinBudget(points: Point[], next: Point, allowance: number): void {
  const last = points[points.length - 1];
  const remaining = allowance - pathLength(points);
  if (!last || remaining <= 0) return;
  const distance = Math.hypot(next.x - last.x, next.y - last.y);
  if (!distance) return;
  if (distance > remaining) {
    points.push({ x: last.x + (next.x - last.x) * remaining / distance,
      y: last.y + (next.y - last.y) * remaining / distance });
  } else {
    points.push(next);
  }
}
