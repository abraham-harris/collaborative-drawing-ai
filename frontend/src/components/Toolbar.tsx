import { PALETTE } from '../config';
import { TOOLS } from '../lib/tools';
import type { BrushSettings } from '../types';

interface Props {
  settings: BrushSettings;
  onChange: (patch: Partial<BrushSettings>) => void;
  disabled: boolean;
  canUndo: boolean;
  canRedo: boolean;
  onUndo: () => void;
  onRedo: () => void;
  onClear: () => void;
  onDownload: () => void;
}

export default function Toolbar({
  settings,
  onChange,
  disabled,
  canUndo,
  canRedo,
  onUndo,
  onRedo,
  onClear,
  onDownload,
}: Props) {
  const isShape = settings.tool === 'rectangle' || settings.tool === 'ellipse';
  const usesColor = settings.tool !== 'eraser';

  return (
    <aside className="toolbar" aria-label="Drawing tools">
      <section>
        <h2>Tools</h2>
        <div className="tool-grid">
          {TOOLS.map((t) => (
            <button
              key={t.id}
              type="button"
              className={`tool-btn${settings.tool === t.id ? ' active' : ''}`}
              onClick={() => onChange({ tool: t.id })}
              disabled={disabled}
              title={`${t.label} (${t.key})`}
              aria-pressed={settings.tool === t.id}
            >
              <span className="tool-icon" aria-hidden>{t.icon}</span>
              <span className="tool-label">{t.label}</span>
            </button>
          ))}
        </div>
        {isShape && (
          <label className="check">
            <input
              type="checkbox"
              checked={settings.filled}
              onChange={(e) => onChange({ filled: e.target.checked })}
              disabled={disabled}
            />
            Filled shape
          </label>
        )}
      </section>

      <section className={usesColor ? '' : 'muted'}>
        <h2>Color</h2>
        <div className="swatches">
          {PALETTE.map((c) => (
            <button
              key={c}
              type="button"
              className={`swatch${settings.color === c ? ' active' : ''}`}
              style={{ background: c }}
              onClick={() => onChange({ color: c })}
              disabled={disabled}
              title={c}
              aria-label={`Color ${c}`}
            />
          ))}
        </div>
        <label className="color-custom">
          <input
            type="color"
            value={settings.color}
            onChange={(e) => onChange({ color: e.target.value })}
            disabled={disabled}
          />
          <span>Custom <code>{settings.color}</code></span>
        </label>
      </section>

      <section>
        <h2>Brush</h2>
        <label className="slider">
          <span>Size <output>{settings.size}px</output></span>
          <input
            type="range"
            min={1}
            max={80}
            value={settings.size}
            onChange={(e) => onChange({ size: Number(e.target.value) })}
            disabled={disabled}
          />
        </label>
        <label className={`slider${usesColor ? '' : ' muted'}`}>
          <span>Opacity <output>{Math.round(settings.opacity * 100)}%</output></span>
          <input
            type="range"
            min={5}
            max={100}
            value={Math.round(settings.opacity * 100)}
            onChange={(e) => onChange({ opacity: Number(e.target.value) / 100 })}
            disabled={disabled}
          />
        </label>
        <div className="brush-preview" aria-hidden>
          <span
            style={{
              width: Math.min(settings.size, 60),
              height: Math.min(settings.size, 60),
              background: settings.tool === 'eraser' ? '#fff' : settings.color,
              opacity: settings.tool === 'eraser' ? 1 : settings.opacity,
            }}
          />
        </div>
      </section>

      <section>
        <h2>Edit</h2>
        <div className="edit-row">
          <button type="button" onClick={onUndo} disabled={disabled || !canUndo} title="Undo (Ctrl+Z)">
            ↶ Undo
          </button>
          <button type="button" onClick={onRedo} disabled={disabled || !canRedo} title="Redo (Ctrl+Y)">
            ↷ Redo
          </button>
        </div>
        <div className="edit-row">
          <button type="button" onClick={onClear} disabled={disabled}>Clear</button>
          <button type="button" onClick={onDownload}>Save PNG</button>
        </div>
      </section>
    </aside>
  );
}
