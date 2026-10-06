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
  if (typeof data.image !== 'string' || !data.image.startsWith('data:image')) {
    throw new Error('AI server response is missing a valid "image" data URL');
  }
  return data;
}
