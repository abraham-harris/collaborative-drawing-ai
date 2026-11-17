import assert from 'node:assert/strict';
import test from 'node:test';
import { appendWithinBudget, inkUsed, pathLength } from '../src/lib/ink.ts';
import type { DrawAction, Point } from '../src/types';

test('pen path stops at the exact budget', () => {
  const points: Point[] = [{ x: 0, y: 0 }];
  appendWithinBudget(points, { x: 200, y: 0 }, 300);
  appendWithinBudget(points, { x: 400, y: 0 }, 300);
  appendWithinBudget(points, { x: 500, y: 0 }, 300);
  assert.equal(pathLength(points), 300);
  assert.deepEqual(points.at(-1), { x: 300, y: 0 });
});

test('removing an action refunds its ink and a tap costs pen width', () => {
  const action: DrawAction = {
    tool: 'brush', color: '#000000', size: 8, opacity: 1, filled: false,
    points: [{ x: 0, y: 0 }],
  };
  assert.equal(inkUsed([action]), 8);
  assert.equal(inkUsed([]), 0);
});
