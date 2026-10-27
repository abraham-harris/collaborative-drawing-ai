import { AI_MODES, RANDOM_SUBJECT, SUBJECTS } from '../config';
import type { AIMode } from '../types';

interface Props {
  mode: AIMode;
  onChange: (mode: AIMode) => void;
  subject: string;
  onSubjectChange: (subject: string) => void;
  disabled: boolean;
}

const label = (subject: string) => subject.replace(/_/g, ' ');

export default function ModeSelector({ mode, onChange, subject, onSubjectChange, disabled }: Props) {
  const current = AI_MODES.find((m) => m.id === mode);
  return (
    <div className="mode-bar">
      <span className="mode-title" id="mode-label">AI partner</span>
      <div className="mode-options" role="radiogroup" aria-labelledby="mode-label">
        {AI_MODES.map((m) => (
          <button
            key={m.id}
            type="button"
            role="radio"
            aria-checked={mode === m.id}
            className={`mode-btn${mode === m.id ? ' active' : ''}`}
            onClick={() => onChange(m.id)}
            disabled={disabled}
            title={m.description}
          >
            {m.label}
          </button>
        ))}
      </div>
      <label className="subject-select">
        <span className="mode-title">Subject</span>
        <select value={subject} onChange={(e) => onSubjectChange(e.target.value)} disabled={disabled}>
          <option value={RANDOM_SUBJECT}>Surprise me</option>
          {SUBJECTS.map((s) => (
            <option key={s} value={s}>
              {label(s)}
            </option>
          ))}
        </select>
      </label>
      {current && <p className="mode-description">{current.description}</p>}
    </div>
  );
}
