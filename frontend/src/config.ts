import type { AIMode, SubjectSource } from './types';

/** Path length available to each participant on each turn, in canvas pixels. */
export const INK_LIMIT = 300;

/** Default and maximum number of turns in a game (editable on the page; "Finish drawing" ends early). */
export const DEFAULT_TURNS = 5;
export const MAX_TURNS = 20;

/** Where the AI's subjects come from (editable on the page). */
export const SUBJECT_SOURCES: { id: SubjectSource; label: string }[] = [
  { id: 'recognize', label: 'Recognize drawing' },
  { id: 'random', label: 'Random' },
];
export const DEFAULT_SUBJECT_SOURCE: SubjectSource = 'recognize';

/** How many candidates Greedy generates and ranks by creativity (editable on the page). */
export const DEFAULT_GREEDY_ITERATIONS = 10;
export const MAX_GREEDY_ITERATIONS = 30;

/** The AI drawing models the user can choose between (see backend/drawing_models). */
export const AI_MODES: { id: AIMode; label: string; description: string }[] = [
  {
    id: 'complete',
    label: 'Complete',
    description: 'Adds short, separate strokes based on your drawing (SketchRNN).',
  },
  {
    id: 'continue',
    label: 'Continue',
    description: 'Extends the ends of your lines (SketchRNN).',
  },
  {
    id: 'greedy',
    label: 'Greedy',
    description:
      'Uses only as much ink as you drew. Tries several drawings and keeps the one AuDrA rates most creative.',
  },
];
export const DEFAULT_AI_MODE: AIMode = 'complete';

/** Internal canvas resolution (the canvas is scaled with CSS to fit the screen). */
export const CANVAS_WIDTH = 800;
export const CANVAS_HEIGHT = 600;

/** Canvas background. */
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
