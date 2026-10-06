import { useCallback, useEffect, useImperativeHandle, useRef, type PointerEvent, type Ref } from 'react';
import { BACKGROUND_COLOR, CANVAS_HEIGHT, CANVAS_WIDTH, MAX_HISTORY } from '../config';
import { drawAction, floodFill } from '../lib/draw';
import type { BrushSettings, DrawAction, Point } from '../types';

export interface DrawingCanvasHandle {
  /** Current canvas as a PNG data URL. */
  exportPNG(): string;
  /** Everything the human drew since the last reset / AI image. */
  getActions(): DrawAction[];
  /** Replace the canvas with an image (e.g. the AI's drawing) and reset history. */
  loadImage(dataUrl: string): Promise<void>;
  /** Clear to background as an undoable step. */
  clear(): void;
  /** Clear to background and wipe history (new session). */
  reset(): void;
  undo(): void;
  redo(): void;
}

interface HistoryEntry {
  image: ImageData;
  actions: DrawAction[];
}

interface Props {
  settings: BrushSettings;
  disabled: boolean;
  onHistoryChange: (state: { canUndo: boolean; canRedo: boolean }) => void;
  ref?: Ref<DrawingCanvasHandle>;
}

export default function DrawingCanvas({ settings, disabled, onHistoryChange, ref }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const previewRef = useRef<HTMLCanvasElement>(null);
  const historyRef = useRef<HistoryEntry[]>([]);
  const indexRef = useRef(-1);
  const currentRef = useRef<DrawAction | null>(null);

  const ctx = () => canvasRef.current!.getContext('2d', { willReadFrequently: true })!;
  const previewCtx = () => previewRef.current!.getContext('2d')!;

  const emitHistory = useCallback(() => {
    onHistoryChange({
      canUndo: indexRef.current > 0,
      canRedo: indexRef.current < historyRef.current.length - 1,
    });
  }, [onHistoryChange]);

  /** Snapshot the canvas as a new history state. */
  const commit = useCallback(
    (actions: DrawAction[]) => {
      const history = historyRef.current.slice(0, indexRef.current + 1);
      history.push({ image: ctx().getImageData(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT), actions });
      while (history.length > MAX_HISTORY) history.shift();
      historyRef.current = history;
      indexRef.current = history.length - 1;
      emitHistory();
    },
    [emitHistory],
  );

  const resetHistory = useCallback(() => {
    historyRef.current = [];
    indexRef.current = -1;
    commit([]);
  }, [commit]);

  const paintBackground = () => {
    const c = ctx();
    c.save();
    c.globalAlpha = 1;
    c.fillStyle = BACKGROUND_COLOR;
    c.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    c.restore();
  };

  const restore = (index: number) => {
    indexRef.current = index;
    ctx().putImageData(historyRef.current[index].image, 0, 0);
    emitHistory();
  };

  const currentActions = () => historyRef.current[indexRef.current]?.actions ?? [];

  useEffect(() => {
    paintBackground();
    resetHistory();
    // Only on mount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useImperativeHandle(
    ref,
    () => ({
      exportPNG: () => canvasRef.current!.toDataURL('image/png'),
      getActions: () => currentActions(),
      loadImage: (dataUrl) =>
        new Promise<void>((resolve, reject) => {
          const img = new Image();
          img.onload = () => {
            paintBackground();
            ctx().drawImage(img, 0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
            resetHistory();
            resolve();
          };
          img.onerror = () => reject(new Error('Could not decode the image returned by the AI'));
          img.src = dataUrl;
        }),
      clear: () => {
        paintBackground();
        commit([]);
      },
      reset: () => {
        paintBackground();
        resetHistory();
      },
      undo: () => {
        if (indexRef.current > 0) restore(indexRef.current - 1);
      },
      redo: () => {
        if (indexRef.current < historyRef.current.length - 1) restore(indexRef.current + 1);
      },
    }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [commit, resetHistory, emitHistory],
  );

  /** Convert a pointer position to canvas pixel coordinates (canvas is CSS-scaled). */
  const toCanvasPoint = (clientX: number, clientY: number): Point => {
    const rect = canvasRef.current!.getBoundingClientRect();
    return {
      x: ((clientX - rect.left) / rect.width) * CANVAS_WIDTH,
      y: ((clientY - rect.top) / rect.height) * CANVAS_HEIGHT,
    };
  };

  const renderPreview = () => {
    const p = previewCtx();
    p.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    if (currentRef.current) drawAction(p, currentRef.current);
  };

  const handlePointerDown = (e: PointerEvent<HTMLCanvasElement>) => {
    if (disabled || e.button !== 0) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    const point = toCanvasPoint(e.clientX, e.clientY);
    const action: DrawAction = {
      tool: settings.tool,
      color: settings.color,
      size: settings.size,
      opacity: settings.opacity,
      filled: settings.filled,
      points: [point],
    };

    if (settings.tool === 'fill') {
      floodFill(ctx(), point, settings.color, settings.opacity);
      commit([...currentActions(), action]);
      return;
    }
    currentRef.current = action;
    renderPreview();
  };

  const handlePointerMove = (e: PointerEvent<HTMLCanvasElement>) => {
    const current = currentRef.current;
    if (!current) return;
    if (current.tool === 'brush' || current.tool === 'eraser') {
      // Coalesced events give smoother strokes on fast movement.
      const events = e.nativeEvent.getCoalescedEvents?.() ?? [e.nativeEvent];
      for (const ev of events.length ? events : [e.nativeEvent]) {
        current.points.push(toCanvasPoint(ev.clientX, ev.clientY));
      }
    } else {
      current.points = [current.points[0], toCanvasPoint(e.clientX, e.clientY)];
    }
    renderPreview();
  };

  const finishStroke = () => {
    const current = currentRef.current;
    if (!current) return;
    currentRef.current = null;
    previewCtx().clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    drawAction(ctx(), current);
    commit([...currentActions(), current]);
  };

  // If drawing is disabled mid-stroke (e.g. submit via keyboard), drop the preview.
  useEffect(() => {
    if (disabled) finishStroke();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [disabled]);

  const cursor = disabled ? 'not-allowed' : settings.tool === 'fill' ? 'cell' : 'crosshair';

  return (
    <div className={`canvas-wrap${disabled ? ' is-disabled' : ''}`}>
      <canvas ref={canvasRef} width={CANVAS_WIDTH} height={CANVAS_HEIGHT} className="canvas" />
      <canvas
        ref={previewRef}
        width={CANVAS_WIDTH}
        height={CANVAS_HEIGHT}
        className="canvas canvas-preview"
        style={{ cursor }}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={finishStroke}
        onPointerCancel={finishStroke}
      />
    </div>
  );
}
