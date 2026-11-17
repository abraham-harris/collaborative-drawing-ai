# collaborative-drawing-ai

A human and SketchRNN take turns adding small amounts of ink to one canvas. Each AI turn offers two drawings with different randomly selected subjects; the human chooses one to keep and decides when the drawing is finished.

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
into the Preview environment variable `AUDRA_CHECKPOINT_BLOB_URL`. The linked
store must provide `BLOB_STORE_ID` to Preview functions; the Python Blob client
uses the function's OIDC token to read from that store. A read-write token remains
an option for local development outside Vercel. The server verifies the
checkpoint's SHA-256 before loading it. For older Vercel
projects, set `VERCEL_SUPPORT_LARGE_FUNCTIONS=1` in the Preview environment to
opt into Large Functions. A branch push produces a preview URL for live testing
without changing the production site. The first score on a cold function will
take longer because it downloads and loads the checkpoint.

## Where things live

| What | File |
|---|---|
| Ink limit, canvas size, AI modes | `frontend/src/config.ts` |
| AI entry point: picks subjects, limits and renders strokes | `backend/ai_model.py` → `generate_drawing()` |
| The drawing models | `backend/drawing_models/` |
| HTTP endpoint `POST /api/turn` | `backend/server.py` |
| HTTP endpoint `POST /api/score` | `backend/server.py` and `backend/audra_service.py` |
| Request/response types | `frontend/src/types.ts` |

## API contract

`POST /api/turn` request:
```json
{
  "turn": 1, "width": 800, "height": 600,
  "image": "data:image/png;base64,...",
  "actions": [{ "tool": "brush", "color": "#1e1e1e", "size": 8, "opacity": 1,
                "filled": false, "points": [{ "x": 10, "y": 20 }, ...] }],
  "contextActions": [ ...accepted strokes and this turn's actions ],
  "mode": "complete"
}
```
`image` holds the whole canvas. `actions` lists the human's pen strokes this turn; `contextActions` includes accepted strokes from earlier rounds for model context.

`mode` is `"complete"` (the default) or `"continue"`. The server picks two different subjects.

Response: `{ "candidates": [{ "image": "data:image/png;base64,...", "mode": "complete", "aiActions": [ ... ], "message": "..." }, { ... }] }`

`POST /api/score` request: `{ "image": "data:image/png;base64,..." }`

Response: `{ "score": 0.42 }`. The number is AuDrA's raw estimate; it is not
clamped to 0–1 and should not be interpreted as a general creativity score.

## AI drawing models

The **AI partner** switch selects Complete or Continue for both AI choices. Subjects are drawn randomly from 95 Quick, Draw! categories.

| Mode | File | What it does |
|---|---|---|
| **Complete** | `complete_model.py` | SketchRNN treats the human's strokes as the start of a subject and adds separate strokes. |
| **Continue** | `continue_model.py` | SketchRNN extends the ends of the human's lines. |

Where the drawings come from:

- **SketchRNN** (`sketch_rnn.py`): [Ha & Eck, 2018](https://arxiv.org/abs/1704.03477). It's a neural network that draws one pen movement at a time. Google trained one per category on the [Quick, Draw! dataset](https://quickdraw.withgoogle.com/data/) (50M human doodles, CC BY 4.0). This is a numpy port of the magenta-js inference code. It loads Google's pretrained weights the first time a category is used and caches them in the temp directory (set `SKETCH_RNN_SIZE=models` to use the smaller 3 MB models).

The 95 subjects are the Quick Draw categories that have a pretrained SketchRNN model. They're listed in `drawing_models/subjects.py`.

The AI only *adds* strokes, and it draws them in its own color (`AI_COLOR` in `canvas.py`). `canvas.render` pastes every already-inked pixel back on top, so the AI can never erase or paint over existing work.

To add a mode, write a `generate(ctx) -> (strokes, message)` function and register it in `MODELS` in `drawing_models/__init__.py` and `AI_MODES` in `frontend/src/config.ts`.
