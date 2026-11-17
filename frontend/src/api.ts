import { API_BASE_URL } from './config';
import type { AIResponse, SubmitPayload } from './types';

/** Sends the current drawing to the Python AI server and returns its drawing. */
export async function submitDrawing(payload: SubmitPayload): Promise<AIResponse> {
  const res = await fetch(`${API_BASE_URL}/api/turn`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => '');
    throw new Error(`AI server responded ${res.status}${detail ? `: ${detail}` : ''}`);
  }
  const data = (await res.json()) as AIResponse;
  // Two candidates to choose between, or one that Greedy already chose.
  if (!Array.isArray(data.candidates) || data.candidates.length < 1 || data.candidates.length > 2 ||
      data.candidates.some((candidate) => typeof candidate.image !== 'string' ||
        !candidate.image.startsWith('data:image/png;base64,') ||
        !Array.isArray(candidate.aiActions))) {
    throw new Error('AI server response is missing valid candidates');
  }
  return data;
}

/** Sends the finished canvas to AuDrA and returns its raw estimate. */
export async function scoreDrawing(image: string, signal?: AbortSignal): Promise<number> {
  const res = await fetch(`${API_BASE_URL}/api/score`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image }),
    signal,
  });
  const data = (await res.json().catch(() => null)) as { score?: unknown; error?: unknown } | null;
  if (!res.ok) {
    const detail = typeof data?.error === 'string' ? `: ${data.error}` : '';
    throw new Error(`Scoring server responded ${res.status}${detail}`);
  }
  if (typeof data?.score !== 'number' || !Number.isFinite(data.score)) {
    throw new Error('Scoring server response is missing a valid score');
  }
  return data.score;
}
