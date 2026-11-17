import { useCallback, useEffect, useRef, useState } from 'react';
import { scoreDrawing, submitDrawing } from './api';
import DrawingCanvas, { type DrawingCanvasHandle } from './components/DrawingCanvas';
import ModeSelector from './components/ModeSelector';
import TurnConsole, { type ConsoleEntry } from './components/TurnConsole';
import Toolbar from './components/Toolbar';
import {
  CANVAS_HEIGHT, CANVAS_WIDTH, DEFAULT_AI_MODE, DEFAULT_GREEDY_ITERATIONS, DEFAULT_TURNS, INK_LIMIT,
} from './config';
import type { AICandidate, AIMode, BrushSettings, DrawAction, Phase } from './types';
import './App.css';

export default function App() {
  const canvasRef = useRef<DrawingCanvasHandle>(null);
  const acceptedActionsRef = useRef<DrawAction[]>([]);
  const pendingActionsRef = useRef<DrawAction[]>([]);
  const [settings, setSettings] = useState<BrushSettings>({
    tool: 'brush', color: '#1e1e1e', size: 8, opacity: 1, filled: false,
  });
  const [history, setHistory] = useState({ canUndo: false, canRedo: false });
  const [remaining, setRemaining] = useState(INK_LIMIT);
  const [turn, setTurn] = useState(1);
  const [phase, setPhase] = useState<Phase>('human');
  const [candidates, setCandidates] = useState<AICandidate[] | null>(null);
  const [totalTurns, setTotalTurns] = useState(DEFAULT_TURNS);
  const [greedyIterations, setGreedyIterations] = useState(DEFAULT_GREEDY_ITERATIONS);
  const [consoleEntries, setConsoleEntries] = useState<ConsoleEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [aiMessage, setAiMessage] = useState<string | null>(null);
  const [aiMode, setAiMode] = useState<AIMode>(DEFAULT_AI_MODE);
  const [creativityScore, setCreativityScore] = useState<number | null>(null);
  const [scoreError, setScoreError] = useState<string | null>(null);
  const [isScoring, setIsScoring] = useState(false);
  const scoreAbortRef = useRef<AbortController | null>(null);

  const updateSettings = (patch: Partial<BrushSettings>) => setSettings((s) => ({ ...s, ...patch }));

  const handleSubmit = useCallback(async () => {
    const canvas = canvasRef.current;
    if (!canvas || phase !== 'human') return;
    const image = canvas.exportPNG();
    const actions = canvas.getActions();
    pendingActionsRef.current = actions;
    setPhase('ai');
    setError(null);
    setAiMessage(null);
    try {
      const response = await submitDrawing({
        turn, totalTurns, width: CANVAS_WIDTH, height: CANVAS_HEIGHT, image, actions,
        contextActions: [...acceptedActionsRef.current, ...actions], mode: aiMode, greedyIterations,
      });
      const { report } = response;
      if (report) setConsoleEntries((entries) => [...entries, { report, kept: null }]);
      if (response.candidates.length === 1) {
        // Greedy already chose its drawing: keep it without asking.
        await acceptCandidate(response.candidates[0], 0);
      } else {
        setCandidates(response.candidates);
        setPhase('choose');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setPhase('human');
    }
  // acceptCandidate only reads refs and state setters.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase, turn, totalTurns, aiMode, greedyIterations]);

  /** Put a candidate on the canvas, remember its strokes, and move to the next turn (or finish). */
  const acceptCandidate = async (candidate: AICandidate, index: number) => {
    await canvasRef.current?.loadImage(candidate.image);
    acceptedActionsRef.current = [
      ...acceptedActionsRef.current, ...pendingActionsRef.current, ...candidate.aiActions,
    ];
    pendingActionsRef.current = [];
    setAiMessage(candidate.message);
    setCandidates(null);
    setConsoleEntries((entries) => entries.map((e, i) => (i === entries.length - 1 ? { ...e, kept: index } : e)));
    if (turn >= totalTurns) {
      setPhase('done');
    } else {
      setTurn((value) => value + 1);
      setPhase('human');
    }
  };

  const handleChoose = async (candidate: AICandidate, index: number) => {
    if (phase !== 'choose') return;
    setPhase('ai');
    try {
      await acceptCandidate(candidate, index);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setPhase('choose');
    }
  };

  // Raising the turn count after the last turn picks the game back up.
  const handleTotalTurnsChange = (value: number) => {
    setTotalTurns(value);
    if (phase === 'done' && value > turn) {
      setTurn((t) => t + 1);
      setPhase('human');
      setCreativityScore(null);
    }
  };

  const handleFinish = () => {
    if (phase !== 'human') return;
    canvasRef.current?.exportPNG();
    setAiMessage(null);
    setPhase('done');
  };

  const handleRestart = () => {
    scoreAbortRef.current?.abort();
    scoreAbortRef.current = null;
    canvasRef.current?.reset();
    acceptedActionsRef.current = [];
    pendingActionsRef.current = [];
    setConsoleEntries([]);
    setTurn(1);
    setPhase('human');
    setCandidates(null);
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

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (phase !== 'human') return;
      const target = event.target as HTMLElement;
      if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA') return;
      const mod = event.ctrlKey || event.metaKey;
      const key = event.key.toLowerCase();
      if (mod && key === 'z' && !event.shiftKey) {
        event.preventDefault();
        canvasRef.current?.undo();
      } else if (mod && (key === 'y' || (key === 'z' && event.shiftKey))) {
        event.preventDefault();
        canvasRef.current?.redo();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [phase]);

  const status = phase === 'human'
    ? remaining <= 0 ? 'Ink used up — submit or finish' : 'Your turn — draw, then submit'
    : phase === 'ai' ? 'The AI is drawing…'
      : phase === 'choose' ? 'Choose one AI drawing to keep'
        : 'Finished! Save your drawing or start over.';

  return (
    <div className="app">
      <header className="app-header">
        <h1>Co-Drawing with AI</h1>
        <div className="turn-info">
          <span className="turn-badge">Turn {turn} / {totalTurns}</span>
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
              <button type="button" className="primary" onClick={handleRestart}>Start over</button>
            </>
          ) : phase === 'human' ? (
            <>
              <button type="button" onClick={handleFinish}>Finish drawing</button>
              <button type="button" className="primary" onClick={handleSubmit}>Submit to AI</button>
            </>
          ) : null}
        </div>
      </header>

      <ModeSelector mode={aiMode} onChange={setAiMode} disabled={phase !== 'human'}
        totalTurns={totalTurns} minTurns={turn} onTotalTurnsChange={handleTotalTurnsChange}
        iterations={greedyIterations} onIterationsChange={setGreedyIterations}
        settingsDisabled={phase === 'ai'} />

      {error && <div className="banner banner-error" role="alert">
        <strong>Couldn't continue:</strong> {error}
        <button type="button" className="link" onClick={() => setError(null)}>Dismiss</button>
      </div>}
      {aiMessage && !error && <div className="banner banner-ai">🤖 {aiMessage}</div>}
      {scoreError && <div className="banner banner-error" role="alert">
        <strong>Couldn't rate creativity:</strong> {scoreError}
      </div>}
      {creativityScore !== null && <div className="banner banner-score" role="status">
        AuDrA creativity estimate: <strong>{creativityScore.toFixed(2)}</strong>
        <span>For drawings similar to AuDrA's training examples.</span>
      </div>}

      {phase === 'choose' && candidates && (
        <section className="candidate-section" aria-label="AI drawing choices">
          <h2>Choose one drawing to keep</h2>
          <div className="candidate-grid">
            {candidates.map((candidate, index) => (
              <button key={index} type="button" className="candidate"
                onClick={() => void handleChoose(candidate, index)}>
                <img src={candidate.image} alt={`AI drawing option ${index + 1}`} />
                <span>Keep option {index + 1}</span>
              </button>
            ))}
          </div>
        </section>
      )}

      <main className="workspace">
        <Toolbar settings={settings} onChange={updateSettings} disabled={phase !== 'human'}
          canUndo={history.canUndo} canRedo={history.canRedo}
          onUndo={() => canvasRef.current?.undo()} onRedo={() => canvasRef.current?.redo()}
          onDownload={handleDownload} />
        <div className="drawing-column">
          {phase === 'human' && <div className="ink-meter" role="status">
            Ink left: <strong>{Math.max(0, Math.floor(remaining))} / {INK_LIMIT}px</strong>
            <progress max={INK_LIMIT} value={remaining} aria-label="Ink remaining" />
          </div>}
          <DrawingCanvas ref={canvasRef} settings={settings} disabled={phase !== 'human'}
            onHistoryChange={setHistory} onInkChange={setRemaining} />
        </div>
      </main>

      <TurnConsole entries={consoleEntries} onClear={() => setConsoleEntries([])} />

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
