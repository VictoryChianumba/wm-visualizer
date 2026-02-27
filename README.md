# WM Visualizer

Real-time interpretability visualiser for the [IRIS](https://github.com/eloialonso/iris) world model.  A FastAPI backend runs IRIS inference and streams attention weights and residual-stream activation norms to a Next.js frontend over WebSocket — every frame, live, as the agent plays.

---

## What it does

| Pane | Content |
|---|---|
| **Left** | Raw game frame at native resolution (`imageSmoothingEnabled = false`) |
| **Middle** | Per-head attention heatmap for the selected transformer layer.  Axes are labelled with token types (`o0`…`o15`, `act`) derived from the model config at runtime. |
| **Right (top)** | Residual-stream activation norm bar chart — one bar per layer, colour-coded by magnitude. |
| **Right (bottom)** | Live metrics (inference FPS, episode, return, queue depth, hook latency, drop rate) and an event stream (agent loaded, episode start/end, errors, WebSocket reconnects). |

The **layer slider** in the top bar sweeps the attention view across all transformer layers in real time.  The **agent dropdown** switches between available checkpoints without restarting the page.

---

## Architecture

```
┌──────────────────────────────┐       WebSocket /ws          ┌────────────────────────────┐
│       FastAPI backend        │ ──────────────────────────▶  │    Next.js frontend        │
│                              │                              │                            │
│  InferenceEngine             │  JSON frames @ infer rate    │  useVisualizerSocket()     │
│   └─ background thread       │                              │   ├─ GameFrame (canvas)    │
│       ├─ agent.act()  ──▶ Atari env                         │   ├─ AttentionHeatmap(visx)│
│       ├─ world_model() ──▶ hooks fire                       │   ├─ ActivationNorms (visx)│
│       └─ encode frame (PNG)  │                              │   ├─ ControlBar            │
│                              │  POST /control               │   └─ LogPane               │
│  REST: /agents  /config      │ ◀────────────────────────── │                            │
└──────────────────────────────┘                              └────────────────────────────┘
```

Hook extraction uses **PyTorch forward hooks only** — no files under `iris/src/` are ever modified.

### Hook targets

| Hook site | Captured tensor | Shape |
|---|---|---|
| `block.attn.attn_drop` | `inp[0]` — post-softmax attention before dropout | `(1, nh, T_q, T_k)` |
| `block` (residual output) | `out[0, -1].norm()` — L2 norm of last token | scalar |

With no KV cache (the strategy used here): `T_q = T_k = tokens_per_block = 17`, giving a consistent `4 × 17 × 17` attention tensor every step.

---

## Project structure

```
wm-visualizer/
├── backend/
│   ├── hooks.py          # IrisHookExtractor — forward hook registration / teardown
│   ├── inference.py      # InferenceEngine — background thread, bounded queue, frame encoding
│   ├── main.py           # FastAPI app — WebSocket /ws, REST /agents /config /control
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx
│   │   │   └── page.tsx
│   │   ├── components/
│   │   │   ├── GameFrame.tsx          # canvas, imageSmoothingEnabled=false
│   │   │   ├── AttentionHeatmap.tsx   # visx heatmap, token-labelled axes
│   │   │   ├── ActivationNorms.tsx    # visx bar chart
│   │   │   ├── ControlBar.tsx         # agent/layer controls, connection status
│   │   │   ├── LogPane.tsx            # metrics + event stream
│   │   │   └── MorphingGraph.tsx      # D3 placeholder (scaffold)
│   │   └── hooks/
│   │       └── useVisualizerSocket.ts # WebSocket lifecycle, auto-reconnect
│   ├── src/__tests__/
│   │   └── useVisualizerSocket.test.ts
│   ├── package.json
│   └── Dockerfile
├── tests/
│   └── test_backend.py
├── docker-compose.yml
└── README.md             ← you are here
```

---

## WebSocket message schema

### Frame message (every inference step)

```jsonc
{
  "type": "frame",
  "frame": "<base64 PNG>",          // raw game frame — before any preprocessing
  "attention": {
    "0": [[[…]]],                   // layer_idx → [num_heads][T_q][T_k]
    "1": [[[…]]],
    "9": [[[…]]]
  },
  "norms": [1.2, 3.4, …],          // L2 norm of last residual token, one per layer
  "metrics": {
    "infer_fps": 14.8,
    "step": 42,
    "episode": 3,
    "queue_depth": 1,
    "hook_latency_ms": 2.1,
    "drop_rate": 0.0,
    "return": 120.0
  },
  "token_layout": {
    "tokens_per_block": 17,
    "obs_per_block": 16,
    "labels": ["o0", "o1", …, "o15", "act"]  // derived from model config at runtime
  }
}
```

### Event messages (interleaved)

```jsonc
{ "type": "event", "event": "agent_loaded",  "data": { "agent": "Breakout", "layers": 10, "heads": 4 } }
{ "type": "event", "event": "episode_start", "data": { "episode": 1 } }
{ "type": "event", "event": "episode_end",   "data": { "episode": 1, "return": 42.0, "steps": 1700 } }
{ "type": "event", "event": "error",         "data": { "message": "…" } }
```

### Config message (sent once on connect)

```jsonc
{
  "type": "config",
  "num_layers": 10, "num_heads": 4, "embed_dim": 256,
  "tokens_per_block": 17, "max_blocks": 20,
  "agents": [
    { "id": "Breakout", "name": "Breakout",
      "path": "/abs/Breakout.pt", "env_id": "BreakoutNoFrameskip-v4" }
  ]
}
```

---

## Local development setup

### Prerequisites

- Python 3.10 with IRIS venv already set up (`iris/.venv310`)
- Node.js 20+
- IRIS checkpoints in `iris/checkpoints/` (e.g. `Breakout.pt`, `Alien.pt`)
- Atari ROMs installed: `ale-import-roms /path/to/roms/`

### 1. Backend

```bash
cd wm-visualizer/backend

# Option A: reuse iris/.venv310 (all deps already installed)
/path/to/iris/.venv310/bin/pip install fastapi uvicorn[standard] websockets

# Option B: fresh venv
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Set env vars and start
export IRIS_ROOT=/Users/temp/Desktop/projects/world_models/iris
export IRIS_SRC=$IRIS_ROOT/src
export CHECKPOINT_DIR=$IRIS_ROOT/checkpoints

uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Frontend

```bash
cd wm-visualizer/frontend
npm install
npm run dev        # → http://localhost:3000
```

Open `http://localhost:3000`.  REST calls are proxied to `http://localhost:8000` via the Next.js rewrite in `next.config.js`.

### 3. Running tests

**Backend:**

```bash
cd wm-visualizer
IRIS_ROOT=/path/to/iris \
  /path/to/iris/.venv310/bin/pytest tests/test_backend.py -v
```

**Frontend:**

```bash
cd wm-visualizer/frontend
npm test
```

---

## How to run locally (short form)

```bash
# Terminal 1 — backend
cd wm-visualizer/backend
IRIS_ROOT=../../iris uvicorn main:app --port 8000

# Terminal 2 — frontend
cd wm-visualizer/frontend
npm run dev
```

---

## Docker deployment

```bash
# from wm-visualizer/
docker-compose up --build
```

- Backend → `http://localhost:8000`
- Frontend → `http://localhost:3000`

`docker-compose.yml` mounts the IRIS repo as a **read-only** volume at `/iris`.  Nothing under `iris/src/` is ever written.

### Changing the IRIS mount path

Edit `docker-compose.yml` → `services.backend.volumes`:

```yaml
volumes:
  - /your/iris/path:/iris:ro
```

---

## How to add new agents

1. Drop a `.pt` checkpoint file into `iris/checkpoints/`.
2. The file stem becomes the agent name shown in the dropdown.
3. The Atari env ID is inferred automatically — `Breakout.pt` → `BreakoutNoFrameskip-v4`.
4. For non-standard names add an entry to `_KNOWN_ENV_IDS` in `backend/main.py`.
5. Refresh the browser — the new agent appears immediately (list fetched on each WS connect from `GET /agents`).

---

## Switching agents and games during a session

**Via the UI:** select a new agent in the dropdown.  The frontend clears the current visualisation instantly and shows a loading spinner; the backend simultaneously stops the inference thread, flushes the queue, detaches old hooks, loads the new checkpoint, attaches fresh hooks, and restarts — within a single `POST /control` call.

**Via REST:**

```bash
curl -X POST http://localhost:8000/control \
  -H 'Content-Type: application/json' \
  -d '{"command":"switch_agent","payload":{
        "checkpoint_path":"/iris/checkpoints/Alien.pt",
        "env_id":"AlienNoFrameskip-v4"}}'
```

**Other controls:**

| `command` | `payload` | Effect |
|---|---|---|
| `restart` | — | Reset current episode |
| `pause` | — | Halt inference (queue stops filling) |
| `resume` | — | Continue inference |
| `loop` | `{"enabled": true\|false}` | Toggle episode looping |
| `switch_agent` | `{"checkpoint_path":…, "env_id":…}` | Hot-swap agent |

---

## How to read the visualisations

### Attention heatmap (middle pane)

- **Grid:** one subplot per head.  4 heads → 2 × 2 grid.
- **Axes:** rows = query tokens (attending *from*), columns = key tokens (attending *to*).  Labels come from `token_layout.labels` in the WebSocket message — `o0`…`o15` are VQVAE image tokens, `act` is the action token.
- **Colour:** dark blue = low attention, bright indigo = high attention.
- **Layer slider:** sweep through 0 → num_layers − 1.  Early layers tend to attend locally; later layers show longer-range patterns and inter-token mixing.
- The diagonal of the causal mask is visible at the bottom-left corner for observation tokens.

### Activation norm chart (right-top pane)

- **X-axis:** transformer layer index (0 = first).
- **Y-axis:** L2 norm of the residual stream at the last token after each block.
- **Colour:** plasma scale — brighter = larger norm.
- Layers with tall bars are doing the most computation for the current token; near-zero layers have little effect.

---

## Log file location and structure

The backend logs to stdout.  To persist:

```bash
uvicorn main:app --port 8000 2>&1 | tee wm-visualizer.log
```

Format: `YYYY-MM-DD HH:MM:SS [LEVEL] module: message`

Hook latency warnings are emitted at `WARNING` level when extraction exceeds 10 ms.

---

## Known limitations

- **Single WebSocket client:** the engine is a singleton; a second browser tab shares the same inference loop.
- **No KV cache in WM forward:** hook extraction uses a fresh single-block pass (no KV cache) each step — consistent 17 × 17 attention, but does not reflect the model's true autoregressive context.
- **CPU-only default:** set `DEFAULT_DEVICE=mps` or `cuda` to use accelerators; MPS is untested with IRIS models.
- **Frame dropping:** when the browser renders slower than inference, frames are silently dropped; monitor `drop_rate` in the log pane.
- **Atari ROMs:** must be installed separately before first run.
- **D3 morphing graph:** scaffolded as a placeholder in `MorphingGraph.tsx`; implementation pending.
