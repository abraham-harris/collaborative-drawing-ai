import { AI_MODES, MAX_GREEDY_ITERATIONS, MAX_TURNS, SUBJECT_SOURCES } from '../config';
import type { AIMode, SubjectSource } from '../types';

interface Props {
  mode: AIMode;
  onChange: (mode: AIMode) => void;
  disabled: boolean;
  totalTurns: number;
  /** Lowest allowed turn count (the current turn: you can't end before it). */
  minTurns: number;
  onTotalTurnsChange: (turns: number) => void;
  iterations: number;
  onIterationsChange: (iterations: number) => void;
  /** Turn and candidate counts can still change while the AI's options are shown. */
  settingsDisabled: boolean;
  subjectSource: SubjectSource;
  onSubjectSourceChange: (source: SubjectSource) => void;
}

/** Parse a number input, clamped to [min, max]; null while the field is empty or invalid. */
function clampInput(value: string, min: number, max: number): number | null {
  const n = Math.round(Number(value));
  if (value.trim() === '' || !Number.isFinite(n)) return null;
  return Math.min(max, Math.max(min, n));
}

export default function ModeSelector({ mode, onChange, disabled, totalTurns, minTurns, onTotalTurnsChange,
  iterations, onIterationsChange, settingsDisabled, subjectSource, onSubjectSourceChange }: Props) {
  const current = AI_MODES.find((m) => m.id === mode);
  return (
    <div className="mode-bar">
      <span className="mode-title" id="mode-label">AI partner</span>
      <div className="mode-options" role="radiogroup" aria-labelledby="mode-label">
        {AI_MODES.map((m) => (
          <button key={m.id} type="button" role="radio" aria-checked={mode === m.id}
            className={`mode-btn${mode === m.id ? ' active' : ''}`}
            onClick={() => onChange(m.id)} disabled={disabled} title={m.description}>
            {m.label}
          </button>
        ))}
      </div>
      <label className="subject-select"
        title="Recognize: the AI guesses what you drew and picks among its likelier interpretations, not always the most obvious">
        <span className="mode-title">Subjects</span>
        <select value={subjectSource} disabled={disabled}
          onChange={(e) => onSubjectSourceChange(e.target.value as SubjectSource)}>
          {SUBJECT_SOURCES.map((s) => <option key={s.id} value={s.id}>{s.label}</option>)}
        </select>
      </label>
      <label className="number-setting" title="How many turns you and the AI take; Finish drawing ends early">
        <span className="mode-title">Turns</span>
        <input type="number" min={minTurns} max={MAX_TURNS} value={totalTurns} disabled={settingsDisabled}
          onChange={(e) => {
            const n = clampInput(e.target.value, minTurns, MAX_TURNS);
            if (n !== null) onTotalTurnsChange(n);
          }} />
      </label>
      {mode === 'greedy' && (
        <label className="number-setting" title="How many drawings Greedy tries before keeping the most creative">
          <span className="mode-title">Candidates</span>
          <input type="number" min={1} max={MAX_GREEDY_ITERATIONS} value={iterations} disabled={disabled}
            onChange={(e) => {
              const n = clampInput(e.target.value, 1, MAX_GREEDY_ITERATIONS);
              if (n !== null) onIterationsChange(n);
            }} />
        </label>
      )}
      {current && <p className="mode-description">{current.description}</p>}
    </div>
  );
}
