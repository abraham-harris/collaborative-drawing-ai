# collaborative-drawing-ai

A human (or another AI) and an AI model take turns on one canvas. The human draws and clicks **Submit to AI**. The drawing goes to a Python server, and the image the AI sends back replaces the canvas.

```
frontend/   React + TypeScript (Vite) drawing app
backend/    Flask server + AI model stub
```

## Running

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
