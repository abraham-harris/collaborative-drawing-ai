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

## Rate drawings with AuDrA

The hosted app can score a completed human/AI drawing with **Rate creativity**.
It sends the current PNG to `POST /api/score`, scores it in memory, and displays
AuDrA's estimate. The app does not save drawings or scores on the server.

The same model also has a local batch scorer. Download the [AuDrA checkpoint from OSF](https://osf.io/kqn9v/files/eahmk)
and save it as `AuDrA/AuDrA_trained.ckpt` (it is not stored in Git). Then put
the drawings you want to rate in `AuDrA/user_images/`. The
scorer rates every supported image in that folder, including any examples you
leave there.

Use Python 3.12. From the repository root, create a separate environment and
run the scorer with the commands for your system:

**Linux / WSL (x86-64)**

```bash
python3.12 -m venv AuDrA/.venv
AuDrA/.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu -r AuDrA/requirements-inference.txt
AuDrA/.venv/bin/python AuDrA/score.py
```

**Windows PowerShell (64-bit Python)**

```powershell
py -3.12 -m venv AuDrA\.venv
AuDrA\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu -r AuDrA\requirements-inference.txt
AuDrA\.venv\Scripts\python.exe AuDrA\score.py
```

**Apple Silicon macOS 14+**

```bash
python3.12 -m venv AuDrA/.venv
AuDrA/.venv/bin/python -m pip install -r AuDrA/requirements-inference.txt
AuDrA/.venv/bin/python AuDrA/score.py
```

Scores are written to [AuDrA_predictions.csv](AuDrA/AuDrA_predictions.csv)
with `filenames` and `predictions` columns. The CSV is created locally after
the first run and is not committed to Git; running the command again replaces
it. Predictions are intended as normalized creativity ratings for drawings like
AuDrA's training examples, usually around 0–1, though the model does not
enforce that range. See [AuDrA's local setup notes](AuDrA/README.md) for input
options and platform limits.

### Vercel preview setup

The checkpoint stays out of Git. Create a **private** Vercel Blob store linked
to the project and upload the local file with `vercel blob put
AuDrA/AuDrA_trained.ckpt --access private`. Copy the returned private blob URL
into the Preview environment variable `AUDRA_CHECKPOINT_BLOB_URL`. Make sure
`BLOB_READ_WRITE_TOKEN` for that store is also available to Preview functions.
The server verifies the checkpoint's SHA-256 before loading it. For older Vercel
projects, set `VERCEL_SUPPORT_LARGE_FUNCTIONS=1` in the Preview environment to
opt into Large Functions. A branch push produces a preview URL for live testing
without changing the production site. The first score on a cold function will
take longer because it downloads and loads the checkpoint.

## Where things live

| What | File |
|---|---|
| Number of turns (`NUM_TURNS`, default 1), canvas size | `frontend/src/config.ts` |
| AI mode list shown on the page (`AI_MODES`, `DEFAULT_AI_MODE`) | `frontend/src/config.ts` |
| AI entry point: picks the model, renders its strokes | `backend/ai_model.py` → `generate_drawing()` |
| The drawing models | `backend/drawing_models/` |
| HTTP endpoint `POST /api/turn` | `backend/server.py` |
| HTTP endpoint `POST /api/score` | `backend/server.py` and `backend/audra_service.py` |
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

`POST /api/score` request: `{ "image": "data:image/png;base64,..." }`

Response: `{ "score": 0.42 }`. The number is AuDrA's raw estimate; it is not
clamped to 0–1 and should not be interpreted as a general creativity score.

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
