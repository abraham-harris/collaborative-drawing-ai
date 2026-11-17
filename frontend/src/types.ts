export type Tool = 'brush';

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
  filled: boolean;
}

/**
 * One thing the human did on the canvas, in canvas pixel coordinates.
 * `points` is the full pen stroke path.
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
  width: number;
  height: number;
  /** The full current canvas as a PNG data URL ("data:image/png;base64,..."). */
  image: string;
  /** Vector record of what the human drew during this turn. */
  actions: DrawAction[];
  /** Accepted human and AI strokes from earlier rounds, plus this turn. */
  contextActions: DrawAction[];
  /** Which drawing mode both candidates should use. */
  mode: AIMode;
}

export interface AICandidate {
  /** The combined human + AI drawing as a PNG data URL. */
  image: string;
  message: string;
  mode: AIMode;
  aiActions: DrawAction[];
}

export interface AIResponse { candidates: [AICandidate, AICandidate] }

export type Phase = 'human' | 'ai' | 'choose' | 'done';
