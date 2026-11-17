# collaborative-drawing-ai

A human and SketchRNN take turns adding small amounts of ink to one canvas. Each AI turn offers two drawings with different randomly selected subjects; the human chooses one to keep and decides when the drawing is finished. In **Greedy** mode the AI instead tries many drawings, using only as much ink as the human just drew, and keeps the one AuDrA rates most creative. A console under the canvas reports every turn's ink, AuDrA scores, and timings.

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
| Ink limit, canvas size, AI modes, default turns and Greedy candidates | `frontend/src/config.ts` |
| Console under the canvas (per-turn report) | `frontend/src/components/TurnConsole.tsx` |
| Drawing recognition and how it picks subjects | `backend/drawing_models/recognizer.py`; trained by `backend/training/train_recognizer.py` |
| Tests (no network or checkpoint needed) | `backend/test_ai_model.py`, `backend/test_greedy.py`, `backend/test_recognizer.py`, `backend/test_score.py`, `frontend/tests/` |
| AI entry point: picks subjects, limits and renders strokes | `backend/ai_model.py` → `generate_drawing()` |
| The drawing models | `backend/drawing_models/` |
| HTTP endpoint `POST /api/turn` | `backend/server.py` |
| HTTP endpoint `POST /api/score` | `backend/server.py` and `backend/audra_service.py` |
| Request/response types | `frontend/src/types.ts` |

## API contract

`POST /api/turn` request:
```json
{
  "turn": 1, "totalTurns": 5, "width": 800, "height": 600,
  "image": "data:image/png;base64,...",
  "actions": [{ "tool": "brush", "color": "#1e1e1e", "size": 8, "opacity": 1,
                "filled": false, "points": [{ "x": 10, "y": 20 }, ...] }],
  "contextActions": [ ...accepted strokes and this turn's actions ],
  "mode": "complete", "greedyIterations": 10, "subjectSource": "recognize"
}
```
`image` holds the whole canvas. `actions` lists the human's pen strokes this turn; `contextActions` includes accepted strokes from earlier rounds for model context.

`mode` is `"complete"` (the default), `"continue"`, or `"greedy"`. `subjectSource` is `"recognize"` (the default) or `"random"`. For Complete and Continue the server picks two different subjects. `greedyIterations` (1–30, default 10) is how many candidates Greedy tries.

Response: `{ "candidates": [{ "image": "data:image/png;base64,...", "mode": "complete", "aiActions": [ ... ], "message": "..." }, { ... }], "report": { ... } }`. Greedy returns a single candidate, which the page keeps without asking.

`report` is shown in the page's console: `humanInk` and `inkBudget` (px of line), each returned option's `ink` with AuDrA scores for the whole drawing with it (`after`) and its AI strokes alone (`aiOnly`), the score `before` the turn, Greedy's candidates and choice (`greedy`), the recognizer's top guesses and the subject each option drew (`recognition`, only when subjects come from recognition), and step `timings` in seconds. With random subjects, subjects are never included. If AuDrA isn't set up, the scores are missing and `scoringError` says why. See `TurnReport` in `frontend/src/types.ts`.

`POST /api/score` request: `{ "image": "data:image/png;base64,..." }`

Response: `{ "score": 0.42 }`. The number is AuDrA's raw estimate; it is not
clamped to 0–1 and should not be interpreted as a general creativity score.

## AI drawing models

The **AI partner** switch selects Complete or Continue for both AI choices, or Greedy. **Turns** sets how long the game lasts; **Finish drawing** still ends it early, and raising Turns after the last turn picks the game back up. **Subjects** chooses where the AI's subjects (out of 95 Quick, Draw! categories) come from: *Recognize drawing* (the default; see below) or *Random*.

| Mode | File | What it does |
|---|---|---|
| **Complete** | `complete_model.py` | SketchRNN treats the human's strokes as the start of a subject and adds separate strokes. |
| **Continue** | `continue_model.py` | SketchRNN extends the ends of the human's lines. |
| **Greedy** | `greedy_model.py` (run by `ai_model._greedy`) | The AI may draw only as many px of line as the human drew this turn. It generates **Candidates** (default 10) alternatives in one batch, spread across 3 random subjects: half add new strokes, half extend one of the human's strokes, at temperatures from cautious (0.2) to adventurous (0.9). AuDrA scores each one as part of the whole drawing, and the highest-scoring one is kept. |

Where the drawings come from:

- **SketchRNN** (`sketch_rnn.py`): [Ha & Eck, 2018](https://arxiv.org/abs/1704.03477). It's a neural network that draws one pen movement at a time. Google trained one per category on the [Quick, Draw! dataset](https://quickdraw.withgoogle.com/data/) (50M human doodles, CC BY 4.0). This is a numpy port of the magenta-js inference code. It loads Google's pretrained weights the first time a category is used and caches them in the temp directory (set `SKETCH_RNN_SIZE=models` to use the smaller 3 MB models).

The 95 subjects are the Quick Draw categories that have a pretrained SketchRNN model. They're listed in `drawing_models/subjects.py`.

The AI only *adds* strokes, and it draws them in its own color (`AI_COLOR` in `canvas.py`). `canvas.render` pastes every already-inked pixel back on top, so the AI can never erase or paint over existing work.

To add a mode, write a `generate(ctx) -> (strokes, message)` function and register it in `MODELS` in `drawing_models/__init__.py` and `AI_MODES` in `frontend/src/config.ts`.

### Recognizing the drawing

With **Subjects: Recognize drawing**, the AI first guesses what's on the canvas, then draws a subject that fits it, but not always the most obvious one.

- **Recognizer** (`drawing_models/recognizer.py`): a small CNN that scores the drawing against the 95 subjects the AI can draw. It was trained on 28×28 [Quick, Draw!](https://quickdraw.withgoogle.com/data/) bitmaps by `training/train_recognizer.py`; the weights are `drawing_models/recognizer.npz` (about 2 MB). It runs in numpy, so it needs no PyTorch and takes a few milliseconds. The canvas's strokes (human and AI) are drawn at 28×28 the way Quick Draw's bitmaps were, then classified.
- **Choosing subjects** (`recognizer.choose_subjects`): the AI samples among the top 5 guesses that are at least 2% likely, in proportion to probability, but with the single most likely guess weighted down to 35% (`OBVIOUS_WEIGHT`). So a drawing that looks most like a cat might be continued as a tiger or a lion. A duel's two options are two different interpretations; Greedy spreads its candidates across three. If a subject fails to draw, the next interpretation is tried, then random ones.
- **Console:** the `sees` line lists the top guesses. The `drew` line shows which interpretation each option used; in a duel it's revealed only after the player chooses, like the scores.
- On a blank canvas, or with **Random**, subjects are random and never reported, as in the original duel.

To retrain (from `backend/`, with torch installed): `python training/train_recognizer.py`. It downloads slices of the Quick Draw bitmaps (~330 MB, cached in the temp directory) and prints validation and end-to-end accuracy.

### Console and creativity scores

Every AI turn adds a block to the console under the canvas: ink used, AuDrA scores, and how long each step took. In a duel, both options are scored, but the scores are only shown after the player chooses so they can't influence the choice. Scoring during turns needs PyTorch in the backend's environment (`pip install -r requirements.txt` from the repository root) and the AuDrA checkpoint described in "Rate drawings with AuDrA". Without them, turns still work and the console says scoring is unavailable; Greedy then keeps its first candidate. `python server.py` loads AuDrA in the background at startup (about 5 s) so the first turn isn't slowed down. Set `AUDRA_SCORE_TURNS=0` to skip scoring during turns, for example on a deployment where loading PyTorch would slow every cold start.

### Speed

An AI turn should take under 5 s. Measured on a 12-thread Windows laptop with other apps open, including AuDrA scoring:

| Mode | Turn time |
|---|---|
| Complete / Continue duel (2 options) | about 1–2 s |
| Greedy, 10 candidates | 1.5–2.5 s once its subjects are downloaded; up to ~4.6 s when all 3 random subjects are new |

What made it faster:
- SketchRNN samples all candidates as one batch instead of one at a time. The LSTM weights are stored transposed, because this numpy/OpenBLAS build multiplies a small batch by a wide matrix about 10x faster that way (10 drawings: 1.08 s → 0.20 s).
- Decoded SketchRNN weights are cached as `.npz` next to the downloaded JSON (load 0.8 s → 0.04 s), and Greedy downloads its subjects in parallel.
- Greedy renders candidates for scoring without anti-aliasing (0.55 s → 0.06 s for 10; scores change by ≤ 0.005), and all AuDrA scoring is batched.
- AuDrA is loaded at server start, and PyTorch is capped at 6 threads, which made scoring steadier when other apps were busy.
- The models re-read only the 40 most recent strokes, so long games stay fast.
- Tried and rejected: scoring candidates at lower resolution. At 160 px, AuDrA picked the same winner only 3 times out of 12.
