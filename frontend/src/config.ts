import type { AIMode } from './types';

/**
 * Number of human -> AI round trips in one session.
 * Each turn: the human draws, clicks Submit, and the AI returns a new drawing.
 */
export const NUM_TURNS = 1;

/** The AI drawing models the user can choose between (see backend/drawing_models). */
export const AI_MODES: { id: AIMode; label: string; description: string }[] = [
  {
    id: 'complete',
    label: 'Complete',
    description: 'Treats your strokes as the start of the subject and finishes it (SketchRNN).',
  },
  {
    id: 'continue',
    label: 'Continue',
    description: 'Picks up the pen where your last line ended and draws the subject from there (SketchRNN).',
  },
];
export const DEFAULT_AI_MODE: AIMode = 'complete';

/** Sent as the subject to let the server pick one at random each turn. */
export const RANDOM_SUBJECT = 'random';
export const DEFAULT_SUBJECT = RANDOM_SUBJECT;

/**
 * What the AI can draw: Quick, Draw! categories with a pretrained SketchRNN
 * model. Must match SUBJECTS in backend/drawing_models/subjects.py.
 */
export const SUBJECTS: string[] = [
  'alarm_clock', 'ambulance', 'angel', 'ant', 'backpack', 'barn', 'basket', 'bear', 'bee',

  'bicycle', 'bird', 'book', 'brain', 'bridge', 'bulldozer', 'bus', 'butterfly', 'cactus',

  'calendar', 'castle', 'cat', 'chair', 'couch', 'crab', 'cruise_ship', 'diving_board', 'dog',

  'dolphin', 'duck', 'elephant', 'eye', 'face', 'fan', 'fire_hydrant', 'firetruck', 'flamingo',

  'flower', 'frog', 'garden', 'hand', 'hedgehog', 'helicopter', 'kangaroo', 'key', 'lantern',

  'lighthouse', 'lion', 'lobster', 'map', 'mermaid', 'monkey', 'mosquito', 'octopus', 'owl',

  'paintbrush', 'palm_tree', 'parrot', 'passport', 'peas', 'penguin', 'pig', 'pineapple',

  'pool', 'postcard', 'power_outlet', 'rabbit', 'radio', 'rain', 'rhinoceros', 'rifle',

  'roller_coaster', 'sandwich', 'scorpion', 'sea_turtle', 'sheep', 'skull', 'snail',

  'snowflake', 'speedboat', 'spider', 'squirrel', 'steak', 'stove', 'strawberry', 'swan',

  'swing_set', 'tiger', 'toothbrush', 'toothpaste', 'tractor', 'trombone', 'truck', 'whale',

  'windmill', 'yoga',
];

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
