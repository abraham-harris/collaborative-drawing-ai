import { useEffect, useRef } from 'react';
import { AI_MODES } from '../config';
import type { TurnReport } from '../types';

export interface ConsoleEntry {
  report: TurnReport;
  /** Which option was kept (always 0 for Greedy); null while the player is still choosing. */
  kept: number | null;
}

interface Props {
  entries: ConsoleEntry[];
  onClear: () => void;
}

const score = (n: number | null | undefined) => (n == null ? '—' : n.toFixed(3));
const secs = (n: number | undefined) => (n == null ? '—' : n.toFixed(2));
const px = (n: number | undefined) => (n == null ? '—' : `${Math.round(n)} px`);
const signed = (n: number) => `${n >= 0 ? '+' : ''}${n.toFixed(3)}`;
const modeLabel = (id: string) => AI_MODES.find((m) => m.id === id)?.label ?? id;
const subjectLabel = (s: string) => s.replace(/_/g, ' ');

function lines({ report: r, kept }: ConsoleEntry): { key: string; label: string; text: string }[] {
  const out = [];
  const t = r.timings;
  const options = r.options ?? [];
  const isDuel = options.length > 1;

  const inkUsed = options.map((o) => px(o.ink)).join(' / ');
  out.push({
    key: 'ink',
    label: 'ink',
    text: `you ${px(r.humanInk)} → AI ${inkUsed || '—'}${isDuel ? ' (option 1 / 2)' : ''}`
      + (r.inkBudget != null ? ` · limit ${px(r.inkBudget)}` : ''),
  });

  const rec = r.recognition;
  if (rec) {
    out.push({
      key: 'sees',
      label: 'sees',
      text: rec.guesses.map((g) => `${subjectLabel(g.subject)} ${Math.round(g.probability * 100)}%`).join(' · '),
    });
    const drew = rec.optionSubjects ?? [];
    if (drew.length) {
      out.push({
        key: 'drew',
        label: 'drew',
        // In a duel, keep the choice blind: reveal the interpretations after it's made.
        text: isDuel && kept == null
          ? 'two interpretations — shown after you choose'
          : drew.map((s, i) => (isDuel ? `option ${i + 1}${i === kept ? ' (kept)' : ''}: ` : '') + subjectLabel(s)
            + (s === rec.guesses[0]?.subject ? ' (top guess)' : '')).join(' · '),
      });
    }
  }

  const g = r.greedy;
  if (g && g.candidates.length) {
    const scored = g.candidates.some((c) => c.score != null);
    const chosen = g.chosenIndex != null ? g.candidates[g.chosenIndex] : undefined;
    out.push({
      key: 'greedy',
      label: 'greedy',
      text: `${g.candidates.length} generated`
        + (scored ? ` · average ${score(g.averageScore)} · chosen #${(g.chosenIndex ?? 0) + 1} ${score(g.chosenScore)}` : ' · unscored')
        + (chosen ? ` (${chosen.strategy}, temp ${chosen.temperature.toFixed(2)})` : '')
        + (g.scoringError ? ` · ${g.scoringError}` : ''),
    });
    if (scored) {
      out.push({
        key: 'candidates',
        label: 'scores',
        text: g.candidates.map((c, i) => (i === g.chosenIndex ? `[${score(c.score)}]` : score(c.score))).join(' '),
      });
    }
  }

  // AuDrA for the whole drawing. In a duel, wait until the player has chosen so the
  // scores can't influence the choice.
  if (r.before == null) {
    out.push({ key: 'audra', label: 'AuDrA', text: `unavailable${r.scoringError ? `: ${r.scoringError}` : ''}` });
  } else if (isDuel && kept == null) {
    out.push({ key: 'audra', label: 'AuDrA', text: 'scored — shown after you choose' });
  } else if (isDuel) {
    out.push({
      key: 'audra',
      label: 'AuDrA',
      text: options.map((o, i) => `option ${i + 1}${i === kept ? ' (kept)' : ''} ${score(o.after)}`
        + ` (AI alone ${score(o.aiOnly)})`).join(' · '),
    });
    const k = options[kept ?? 0];
    if (k?.after != null) {
      out.push({ key: 'drawing', label: 'drawing', text: `${score(r.before)} → ${score(k.after)} (${signed(k.after - r.before)})` });
    }
  } else if (options[0]?.after != null) {
    const o = options[0];
    out.push({
      key: 'audra',
      label: 'AuDrA',
      text: `drawing ${score(r.before)} → ${score(o.after)} (${signed(o.after! - r.before)}) · AI strokes alone ${score(o.aiOnly)}`,
    });
  }

  const parts = [`${secs(t.total)} s total`];
  if (t.recognize != null) parts.push(`recognize ${secs(t.recognize)}`);
  parts.push(`model ${secs(t.model)}`);
  if (t.generate != null) {
    parts[parts.length - 1] += ` (generate ${secs(t.generate)}, render ${secs(t.renderCandidates)}, score ${secs(t.scoreCandidates)})`;
  }
  if (t.render != null) parts.push(`render ${secs(t.render)}`);
  parts.push(`score ${secs(t.scoreDrawing)}`);
  out.push({ key: 'time', label: 'time', text: parts.join(' · ') });
  return out;
}

/** A console-style log under the canvas: one block per AI turn. */
export default function TurnConsole({ entries, onClear }: Props) {
  const bodyRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    bodyRef.current?.scrollTo({ top: bodyRef.current.scrollHeight });
  }, [entries]);

  return (
    <section className="console" aria-label="AI turn console">
      <div className="console-head">
        <span>Console</span>
        <button type="button" className="link" onClick={onClear} disabled={!entries.length}>
          Clear
        </button>
      </div>
      <div className="console-body" ref={bodyRef} role="log">
        {!entries.length && <div className="console-empty">AI turn reports will appear here.</div>}
        {entries.map((entry, i) => (
          <div className="console-entry" key={i}>
            <div className="console-title">
              ── Turn {entry.report.turn}/{entry.report.totalTurns} · {modeLabel(entry.report.mode)}
            </div>
            {lines(entry).map((l) => (
              <div className="console-line" key={l.key}>
                <span className="console-label">{l.label}</span>
                <span>{l.text}</span>
              </div>
            ))}
          </div>
        ))}
      </div>
    </section>
  );
}
