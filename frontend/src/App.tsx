import { useCallback, useEffect, useRef, useState } from 'react';
import { scoreDrawing, submitDrawing } from './api';
import DrawingCanvas, { type DrawingCanvasHandle } from './components/DrawingCanvas';
import ModeSelector from './components/ModeSelector';
import Toolbar from './components/Toolbar';
import { TOOL_SHORTCUTS } from './lib/tools';
import { CANVAS_HEIGHT, CANVAS_WIDTH, DEFAULT_AI_MODE, DEFAULT_SUBJECT, NUM_TURNS } from './config';
import type { AIMode, BrushSettings, Phase } from './types';
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
  const [aiMode, setAiMode] = useState<AIMode>(DEFAULT_AI_MODE);
  const [subject, setSubject] = useState<string>(DEFAULT_SUBJECT);
  const [creativityScore, setCreativityScore] = useState<number | null>(null);
  const [scoreError, setScoreError] = useState<string | null>(null);
  const [isScoring, setIsScoring] = useState(false);
  const scoreAbortRef = useRef<AbortController | null>(null);

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
        mode: aiMode,
        subject,
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
  }, [phase, turn, aiMode, subject]);

  const handleRestart = () => {
    scoreAbortRef.current?.abort();
    scoreAbortRef.current = null;
    canvasRef.current?.reset();
    setTurn(1);
    setPhase('human');
    setError(null);
    setAiMessage(null);
    setCreativityScore(null);
    setScoreError(null);
    setIsScoring(false);
  };

  const handleRate = async () => {
    const canvas = canvasRef.current;
    if (!canvas || phase !== 'done' || isScoring) return;
    const controller = new AbortController();
    scoreAbortRef.current = controller;
    setIsScoring(true);
    setScoreError(null);
    try {
      const value = await scoreDrawing(canvas.exportPNG(), controller.signal);
      if (!controller.signal.aborted) setCreativityScore(value);
    } catch (err) {
      if (!controller.signal.aborted) setScoreError(err instanceof Error ? err.message : String(err));
    } finally {
      if (!controller.signal.aborted) {
        setIsScoring(false);
        scoreAbortRef.current = null;
      }
    }
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
            <>
              <button type="button" onClick={handleRate} disabled={isScoring}>
                {isScoring ? 'Rating creativity…' : 'Rate creativity'}
              </button>
              <button type="button" className="primary" onClick={handleRestart}>
                Start over
              </button>
            </>
          ) : (
            <button type="button" className="primary" onClick={handleSubmit} disabled={phase !== 'human'}>
              {phase === 'ai' ? 'Waiting for AI…' : 'Submit to AI'}
            </button>
          )}
        </div>
      </header>

      <ModeSelector
        mode={aiMode}
        onChange={setAiMode}
        subject={subject}
        onSubjectChange={setSubject}
        disabled={phase === 'ai'}
      />

      {error && (
        <div className="banner banner-error" role="alert">
          <strong>Couldn't reach the AI:</strong> {error}
          <button type="button" className="link" onClick={() => setError(null)}>
            Dismiss
          </button>
        </div>
      )}
      {aiMessage && !error && <div className="banner banner-ai">🤖 {aiMessage}</div>}
      {scoreError && (
        <div className="banner banner-error" role="alert">
          <strong>Couldn't rate creativity:</strong> {scoreError}
        </div>
      )}
      {creativityScore !== null && (
        <div className="banner banner-score" role="status">
          AuDrA creativity estimate: <strong>{creativityScore.toFixed(2)}</strong>
          <span>For drawings similar to AuDrA's training examples.</span>
        </div>
      )}

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

      <footer className="credits">
        AI drawings by{' '}
        <a href="https://arxiv.org/abs/1704.03477" target="_blank" rel="noreferrer">SketchRNN</a>, trained on the{' '}
        <a href="https://quickdraw.withgoogle.com/data/" target="_blank" rel="noreferrer">Quick, Draw! dataset</a>{' '}
        (Google, CC BY 4.0).
        {' '}Creativity estimates by{' '}
        <a href="https://doi.org/10.31234/osf.io/t63dm" target="_blank" rel="noreferrer">AuDrA</a>{' '}
        (Patterson et al., CC BY-NC-SA 4.0).
      </footer>
    </div>
  );
}
