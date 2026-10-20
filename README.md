# collaborative-drawing-ai

A human (or another AI) and an AI model take turns on one canvas. The human draws and clicks **Submit to AI**. The drawing goes to a Python server, and the image the AI sends back replaces the canvas.

```
frontend/   React + TypeScript (Vite) drawing app
backend/    Flask server + AI model stub
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

AuDrA is a separate local CPU scorer; it does not change the drawing app's AI
turns. Download the [AuDrA checkpoint from OSF](https://osf.io/kqn9v/files/eahmk)
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

## Where things live

| What | File |
|---|---|
| Number of turns (`NUM_TURNS`, default 1), canvas size | `frontend/src/config.ts` |
| AI logic (**stub — replace this**) | `backend/ai_model.py` → `generate_drawing()` |
| HTTP endpoint `POST /api/turn` | `backend/server.py` |
| Request/response types | `frontend/src/types.ts` |

## API contract

`POST /api/turn` request:
```json
{
  "turn": 1, "totalTurns": 1, "width": 800, "height": 600,
  "image": "data:image/png;base64,...",
  "actions": [{ "tool": "brush", "color": "#1e1e1e", "size": 8, "opacity": 1,
                "filled": false, "points": [{ "x": 10, "y": 20 }, ...] }]
}
```
`image` holds the whole canvas. `actions` lists only what the human drew this turn, as vectors.

Response: `{ "image": "data:image/png;base64,...", "message": "optional text" }`
