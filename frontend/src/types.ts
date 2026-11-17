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
export type AIMode = 'complete' | 'continue' | 'greedy';

/** Body POSTed to the Python server when the human clicks Submit. */
export interface SubmitPayload {
  /** 1-based index of the current turn. */
  turn: number;
  /** How many turns this game has. */
  totalTurns: number;
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
  /** How many candidates Greedy generates. */
  greedyIterations: number;
  /** Where the AI's subjects come from: interpretations of the drawing, or random. */
  subjectSource: SubjectSource;
}

export type SubjectSource = 'recognize' | 'random';

export interface AICandidate {
  /** The combined human + AI drawing as a PNG data URL. */
  image: string;
  message: string;
  mode: AIMode;
  aiActions: DrawAction[];
}

/** Two candidates to choose between, or (Greedy) one the AI already chose. */
export interface AIResponse {
  candidates: [AICandidate, AICandidate] | [AICandidate];
  report?: TurnReport;
}

/** One of Greedy's candidates (subjects are hidden, as in a duel). */
export interface GreedyCandidate {
  strategy: 'complete' | 'continue';
  /** Index (in the drawing so far) of the stroke a "continue" candidate extended. */
  extends: number | null;
  temperature: number;
  /** px of line drawn. */
  ink: number;
  /** AuDrA score of the whole drawing with this candidate added (null if unscored). */
  score: number | null;
}

/** What happened during one AI turn (see backend/server.py and ai_model.py). Times are seconds. */
export interface TurnReport {
  turn: number;
  totalTurns: number;
  mode: AIMode;
  /** px of line the human drew this turn, and the most the AI could use. */
  humanInk?: number;
  inkBudget?: number;
  /** AuDrA score of the drawing before the AI's turn. */
  before?: number;
  /** Each returned drawing: its ink, and AuDrA scores for the whole drawing with it and its strokes alone. */
  options?: { ink: number; strokes: number; after?: number; aiOnly?: number }[];
  scoringError?: string;
  /** What the recognizer thinks the drawing is (only when subjects come from recognition). */
  recognition?: {
    guesses: { subject: string; probability: number }[];
    /** The interpretation each returned option drew. */
    optionSubjects?: string[];
  };
  greedy?: {
    iterations: number;
    candidates: GreedyCandidate[];
    averageScore?: number | null;
    chosenIndex?: number;
    chosenScore?: number | null;
    scoringError?: string;
  };
  timings: Partial<Record<'total' | 'recognize' | 'model' | 'generate' | 'renderCandidates' | 'scoreCandidates' | 'render' | 'scoreDrawing', number>>;
}

export type Phase = 'human' | 'ai' | 'choose' | 'done';
