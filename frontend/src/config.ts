/**
 * Number of human -> AI round trips in one session.
 * Each turn: the human draws, clicks Submit, and the AI returns a new drawing.
 */
export const NUM_TURNS = 1;

/** Internal canvas resolution (the canvas is scaled with CSS to fit the screen). */
export const CANVAS_WIDTH = 800;
export const CANVAS_HEIGHT = 600;

/** Canvas background; the eraser paints with this color. */
export const BACKGROUND_COLOR = '#ffffff';

/** Maximum number of undo steps kept in memory. */
export const MAX_HISTORY = 40;

/**
 * Base URL of the Python AI server. Empty string means "same origin", which in
 * development goes through the Vite proxy (see vite.config.ts) to the Flask
 * server. Override with VITE_API_URL in a .env file if needed.
 */
export const API_BASE_URL: string = import.meta.env.VITE_API_URL ?? '';

export const PALETTE = [
  '#1e1e1e', '#ffffff', '#7a7a7a', '#e03131', '#f08c00', '#fcc419',
  '#2f9e44', '#1971c2', '#6741d9', '#c2255c', '#8d5524', '#66d9e8',
];
