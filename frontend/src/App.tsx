import { useCallback, useEffect, useRef, useState } from 'react';
import { submitDrawing } from './api';
import DrawingCanvas, { type DrawingCanvasHandle } from './components/DrawingCanvas';
import Toolbar from './components/Toolbar';
import { TOOL_SHORTCUTS } from './lib/tools';
import { CANVAS_HEIGHT, CANVAS_WIDTH, NUM_TURNS } from './config';
import type { BrushSettings, Phase } from './types';
import './App.css';

export default function App() {
  const canvasRef = useRef<DrawingCanvasHandle>(null);
  const [settings, setSettings] = useState<BrushSettings>({
    tool: 'brush',
    color: '#1e1e1e',
    size: 8,
    opacity: 1,
    filled: false,
  });
  const [history, setHistory] = useState({ canUndo: false, canRedo: false });
  const [turn, setTurn] = useState(1);
  const [phase, setPhase] = useState<Phase>('human');
  const [error, setError] = useState<string | null>(null);
  const [aiMessage, setAiMessage] = useState<string | null>(null);

  const drawingDisabled = phase !== 'human';
  const updateSettings = (patch: Partial<BrushSettings>) => setSettings((s) => ({ ...s, ...patch }));

  const handleSubmit = useCallback(async () => {
    const canvas = canvasRef.current;
    if (!canvas || phase !== 'human') return;
    setPhase('ai');
    setError(null);
    try {
      const response = await submitDrawing({
        turn,
        totalTurns: NUM_TURNS,
        width: CANVAS_WIDTH,
        height: CANVAS_HEIGHT,
        image: canvas.exportPNG(),
        actions: canvas.getActions(),
      });
      await canvas.loadImage(response.image);
      setAiMessage(response.message ?? null);
      if (turn >= NUM_TURNS) {
        setPhase('done');
      } else {
        setTurn((t) => t + 1);
        setPhase('human');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setPhase('human');
    }
  }, [phase, turn]);

  const handleRestart = () => {
    canvasRef.current?.reset();
    setTurn(1);
    setPhase('human');
    setError(null);
    setAiMessage(null);
  };

  const handleDownload = () => {
    const url = canvasRef.current?.exportPNG();
    if (!url) return;
    const a = document.createElement('a');
    a.href = url;
    a.download = `co-drawing-turn-${turn}.png`;
    a.click();
  };

  // Keyboard shortcuts: Ctrl+Z / Ctrl+Y (or Ctrl+Shift+Z), and single-letter tool keys.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (phase !== 'human') return;
      const target = e.target as HTMLElement;
      if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA') return;
      const mod = e.ctrlKey || e.metaKey;
      const key = e.key.toLowerCase();
      if (mod && key === 'z' && !e.shiftKey) {
        e.preventDefault();
        canvasRef.current?.undo();
      } else if (mod && (key === 'y' || (key === 'z' && e.shiftKey))) {
        e.preventDefault();
        canvasRef.current?.redo();
      } else if (!mod && !e.altKey && TOOL_SHORTCUTS[key]) {
        setSettings((s) => ({ ...s, tool: TOOL_SHORTCUTS[key] }));
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [phase]);

  const status =
    phase === 'human'
      ? `Your turn — draw, then submit`
      : phase === 'ai'
        ? 'The AI is drawing…'
        : 'Finished! Save your drawing or start over.';

  return (
    <div className="app">
      <header className="app-header">
        <h1>Co-Drawing with AI</h1>
        <div className="turn-info">
          <span className="turn-badge">
            Turn {turn} / {NUM_TURNS}
          </span>
          <span className={`status status-${phase}`} role="status">
            {phase === 'ai' && <span className="spinner" aria-hidden />}
            {status}
          </span>
        </div>
        <div className="header-actions">
          {phase === 'done' ? (
            <button type="button" className="primary" onClick={handleRestart}>
              Start over
            </button>
          ) : (
            <button type="button" className="primary" onClick={handleSubmit} disabled={phase !== 'human'}>
              {phase === 'ai' ? 'Waiting for AI…' : 'Submit to AI'}
            </button>
          )}
        </div>
      </header>

      {error && (
        <div className="banner banner-error" role="alert">
          <strong>Couldn't reach the AI:</strong> {error}
          <button type="button" className="link" onClick={() => setError(null)}>
            Dismiss
          </button>
        </div>
      )}
      {aiMessage && !error && <div className="banner banner-ai">🤖 {aiMessage}</div>}

      <main className="workspace">
        <Toolbar
          settings={settings}
          onChange={updateSettings}
          disabled={drawingDisabled}
          canUndo={history.canUndo}
          canRedo={history.canRedo}
          onUndo={() => canvasRef.current?.undo()}
          onRedo={() => canvasRef.current?.redo()}
          onClear={() => canvasRef.current?.clear()}
          onDownload={handleDownload}
        />
        <DrawingCanvas
          ref={canvasRef}
          settings={settings}
          disabled={drawingDisabled}
          onHistoryChange={setHistory}
        />
      </main>
    </div>
  );
}
