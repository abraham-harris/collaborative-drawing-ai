import { PALETTE } from '../config';
import type { BrushSettings } from '../types';

interface Props {
  settings: BrushSettings;
  onChange: (patch: Partial<BrushSettings>) => void;
  disabled: boolean;
  canUndo: boolean;
  canRedo: boolean;
  onUndo: () => void;
  onRedo: () => void;
  onDownload: () => void;
}

export default function Toolbar({ settings, onChange, disabled, canUndo, canRedo,
  onUndo, onRedo, onDownload }: Props) {
  return (
    <aside className="toolbar" aria-label="Drawing tools">
      <section>
        <h2>Pen color</h2>
        <div className="swatches">
          {PALETTE.map((color) => (
            <button key={color} type="button" className={`swatch${settings.color === color ? ' active' : ''}`}
              style={{ background: color }} onClick={() => onChange({ color })}
              disabled={disabled} title={color} aria-label={`Color ${color}`} />
          ))}
        </div>
        <label className="color-custom">
          <input type="color" value={settings.color}
            onChange={(e) => onChange({ color: e.target.value })} disabled={disabled} />
          <span>Custom <code>{settings.color}</code></span>
        </label>
      </section>
      <section>
        <h2>Pen</h2>
        <label className="slider">
          <span>Size <output>{settings.size}px</output></span>
          <input type="range" min={1} max={80} value={settings.size}
            onChange={(e) => onChange({ size: Number(e.target.value) })} disabled={disabled} />
        </label>
        <label className="slider">
          <span>Opacity <output>{Math.round(settings.opacity * 100)}%</output></span>
          <input type="range" min={5} max={100} value={Math.round(settings.opacity * 100)}
            onChange={(e) => onChange({ opacity: Number(e.target.value) / 100 })} disabled={disabled} />
        </label>
        <div className="brush-preview" aria-hidden>
          <span style={{ width: Math.min(settings.size, 60), height: Math.min(settings.size, 60),
            background: settings.color, opacity: settings.opacity }} />
        </div>
      </section>
      <section>
        <h2>Edit</h2>
        <div className="edit-row">
          <button type="button" onClick={onUndo} disabled={disabled || !canUndo} title="Undo (Ctrl+Z)">↶ Undo</button>
          <button type="button" onClick={onRedo} disabled={disabled || !canRedo} title="Redo (Ctrl+Y)">↷ Redo</button>
        </div>
        <button type="button" onClick={onDownload}>Save PNG</button>
      </section>
    </aside>
  );
}
