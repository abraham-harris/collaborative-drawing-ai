import type { Tool } from '../types';

export const TOOLS: { id: Tool; label: string; icon: string; key: string }[] = [
  { id: 'brush', label: 'Brush', icon: '✎', key: 'B' },
  { id: 'eraser', label: 'Eraser', icon: '⌫', key: 'E' },
  { id: 'line', label: 'Line', icon: '╱', key: 'L' },
  { id: 'rectangle', label: 'Rectangle', icon: '▭', key: 'R' },
  { id: 'ellipse', label: 'Ellipse', icon: '◯', key: 'O' },
  { id: 'fill', label: 'Fill', icon: '◧', key: 'F' },
];

/** Lowercase keyboard key -> tool. */
export const TOOL_SHORTCUTS: Record<string, Tool> = Object.fromEntries(
  TOOLS.map((t) => [t.key.toLowerCase(), t.id]),
);
