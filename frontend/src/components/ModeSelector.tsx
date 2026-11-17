import { AI_MODES } from '../config';
import type { AIMode } from '../types';

interface Props {
  mode: AIMode;
  onChange: (mode: AIMode) => void;
  disabled: boolean;
}

export default function ModeSelector({ mode, onChange, disabled }: Props) {
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
      {current && <p className="mode-description">{current.description}</p>}
    </div>
  );
}
