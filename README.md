# collaborative-drawing-ai

A human (or another AI) and an AI model take turns on one canvas. The human draws and clicks **Submit to AI**. The drawing goes to a Python server, and the image the AI sends back replaces the canvas.

```
frontend/   React + TypeScript (Vite) drawing app
backend/    Flask server + AI drawing models (backend/drawing_models/)
```

## Running

**Hosted app:** [collaborative-drawing-ai.vercel.app](https://collaborative-drawing-ai.vercel.app/)

To run locally, start the backend and frontend in separate terminals:

**Backend** (terminal 1):
```bash
cd backend
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python server.py                # http://127.0.0.1:5000
```

**Frontend** (terminal 2):
```bash
cd frontend
npm install
npm run dev                     # http://localhost:5173 (proxies /api to :5000)
```

## Where things live

| What | File |
|---|---|
| Number of turns (`NUM_TURNS`, default 1), canvas size | `frontend/src/config.ts` |
| AI mode list shown on the page (`AI_MODES`, `DEFAULT_AI_MODE`) | `frontend/src/config.ts` |
| AI entry point: picks the model, renders its strokes | `backend/ai_model.py` → `generate_drawing()` |
| The drawing models | `backend/drawing_models/` |
| HTTP endpoint `POST /api/turn` | `backend/server.py` |
| Request/response types | `frontend/src/types.ts` |

## API contract

`POST /api/turn` request:
```json
{
  "turn": 1, "totalTurns": 1, "width": 800, "height": 600,
  "image": "data:image/png;base64,...",
  "actions": [{ "tool": "brush", "color": "#1e1e1e", "size": 8, "opacity": 1,
                "filled": false, "points": [{ "x": 10, "y": 20 }, ...] }],
  "mode": "complete", "subject": "cat"
}
```
`image` holds the whole canvas. `actions` lists only what the human drew this turn, as vectors.

`mode` is `"complete"` (the default) or `"continue"`. `subject` is a category from `GET /api/subjects`, or `"random"` (the default).

Response: `{ "image": "data:image/png;base64,...", "message": "optional text", "mode": "complete", "subject": "cat", "aiActions": [ ...same shape as actions ] }`

## AI drawing models

Choose a mode with the **AI partner** switch and what to draw with **Subject**: one of 95 categories, or "Surprise me". Every mode draws a recognizable object of the subject, and the AI always draws on its turn. If the chosen mode comes up empty or fails, a fallback mode takes over (see `FALLBACKS` in `drawing_models/__init__.py`), and the message says so.

| Mode | File | What it does |
|---|---|---|
| **Complete** | `complete_model.py` | SketchRNN reads the human's strokes as the start of the subject, then finishes the drawing in place. Draw a cat's head, and it adds the ears and whiskers. |
| **Continue** | `continue_model.py` | SketchRNN extends the human's strokes: for each one, it reads the whole drawing and keeps the pen down at that stroke's end, so its line grows out of the human's. Then it keeps drawing until the subject is complete. Its first marks always grow out of the human's lines; Complete only adds separate new strokes. |

Where the drawings come from:

- **SketchRNN** (`sketch_rnn.py`): [Ha & Eck, 2018](https://arxiv.org/abs/1704.03477). It's a neural network that draws one pen movement at a time. Google trained one per category on the [Quick, Draw! dataset](https://quickdraw.withgoogle.com/data/) (50M human doodles, CC BY 4.0). This is a numpy port of the magenta-js inference code. It loads Google's pretrained weights the first time a category is used and caches them in the temp directory (set `SKETCH_RNN_SIZE=models` to use the smaller 3 MB models).

The 95 subjects are the Quick Draw categories that have a pretrained SketchRNN model. They're listed in `drawing_models/subjects.py` and `SUBJECTS` in `frontend/src/config.ts`.

The AI only *adds* strokes, and it draws them in its own color (`AI_COLOR` in `canvas.py`). `canvas.render` pastes every already-inked pixel back on top, so the AI can never erase or paint over existing work.

To add a mode, write a `generate(ctx) -> (strokes, message)` function. Register it in `MODELS` and `FALLBACKS` in `drawing_models/__init__.py` and in `AI_MODES` in `frontend/src/config.ts`.
