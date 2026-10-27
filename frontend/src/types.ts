export type Tool = 'brush' | 'eraser' | 'line' | 'rectangle' | 'ellipse' | 'fill';

export interface Point {
  x: number;
  y: number;
}

export interface BrushSettings {
  tool: Tool;
  color: string;
  size: number;
  /** 0..1 */
  opacity: number;
  /** Fill rectangles/ellipses instead of outlining them. */
  filled: boolean;
}

/**
 * One thing the human did on the canvas, in canvas pixel coordinates.
 * - brush/eraser: `points` is the full stroke path
 * - line/rectangle/ellipse: `points` is [start, end]
 * - fill: `points` is [seed point]
 */
export interface DrawAction {
  tool: Tool;
  color: string;
  size: number;
  opacity: number;
  filled: boolean;
  points: Point[];
}

/** Which drawing model the AI uses (keys of MODELS in backend/drawing_models). */
export type AIMode = 'complete' | 'continue';

/** Body POSTed to the Python server when the human clicks Submit. */
export interface SubmitPayload {
  /** 1-based index of the current turn. */
  turn: number;
  totalTurns: number;
  width: number;
  height: number;
  /** The full current canvas as a PNG data URL ("data:image/png;base64,..."). */
  image: string;
  /** Vector record of what the human drew during this turn. */
  actions: DrawAction[];
  /** Which drawing model should respond. */
  mode: AIMode;
  /** What the AI should draw: one of SUBJECTS, or 'random'. */
  subject: string;
}

/** Response from the Python server. */
export interface AIResponse {
  /** The combined human + AI drawing as a PNG data URL. Replaces the canvas. */
  image: string;
  /** Optional text from the AI (e.g. what it added). */
  message?: string;
  /** The model that responded. */
  mode?: AIMode;
  /** The subject the AI drew (resolved if 'random' was requested). */
  subject?: string;
  /** The strokes the AI added, in the same format as the human's actions. */
  aiActions?: DrawAction[];
}

export type Phase = 'human' | 'ai' | 'done';
