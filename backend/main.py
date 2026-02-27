"""
FastAPI application and WebSocket server for the WM Visualizer.

Endpoints
---------
  WS  /ws                   — stream FrameData JSON; accepts query params
                              ?agent=<name>&env_id=<id>&device=<cpu|cuda>
  GET /agents               — list available checkpoints
  GET /config               — current model config
  POST /control             — episode control (loop, restart, pause, resume,
                              switch_agent)
"""

import asyncio
import logging
import os
from pathlib import Path
from queue import Empty
from typing import Any, Callable, Dict, List, Optional

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from inference import InferenceEngine

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config from environment variables
# ---------------------------------------------------------------------------

IRIS_ROOT = os.environ.get(
    "IRIS_ROOT",
    str(Path(__file__).parent.parent.parent / "iris"),
)
IRIS_SRC = os.environ.get("IRIS_SRC", str(Path(IRIS_ROOT) / "src"))
CHECKPOINT_DIR = os.environ.get(
    "CHECKPOINT_DIR",
    str(Path(IRIS_ROOT) / "checkpoints"),
)
DEFAULT_DEVICE = os.environ.get("DEFAULT_DEVICE", "cpu")

# Map checkpoint stem → Atari env ID (fallback: append NoFrameskip-v4)
_KNOWN_ENV_IDS: Dict[str, str] = {
    "Alien": "AlienNoFrameskip-v4",
    "Breakout": "BreakoutNoFrameskip-v4",
    "Pong": "PongNoFrameskip-v4",
    "SpaceInvaders": "SpaceInvadersNoFrameskip-v4",
    "Seaquest": "SeaquestNoFrameskip-v4",
    "Freeway": "FreewayNoFrameskip-v4",
    "MsPacman": "MsPacmanNoFrameskip-v4",
    "Qbert": "QbertNoFrameskip-v4",
}


def _infer_env_id(checkpoint_stem: str) -> str:
    return _KNOWN_ENV_IDS.get(checkpoint_stem, f"{checkpoint_stem}NoFrameskip-v4")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(title="WM Visualizer", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = InferenceEngine(iris_src=IRIS_SRC, iris_root=IRIS_ROOT)


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/agents")
async def list_agents() -> List[dict]:
    """
    Scan the checkpoint directory and return all available agents.

    Each entry::

        {"id": "Breakout", "name": "Breakout",
         "path": "/abs/path/to/Breakout.pt",
         "env_id": "BreakoutNoFrameskip-v4"}
    """
    ckpt_dir = Path(CHECKPOINT_DIR)
    agents = []
    if ckpt_dir.is_dir():
        for pt in sorted(ckpt_dir.glob("*.pt")):
            stem = pt.stem
            agents.append({
                "id": stem,
                "name": stem,
                "path": str(pt.resolve()),
                "env_id": _infer_env_id(stem),
            })
    return agents


@app.get("/config")
async def get_config() -> dict:
    """Return the current model config or an empty dict if no agent is loaded."""
    return engine.get_config()


class ControlCommand(BaseModel):
    command: str                          # loop | restart | pause | resume | switch_agent
    payload: Optional[Dict[str, Any]] = None


@app.post("/control")
async def control(cmd: ControlCommand) -> dict:
    """
    Send a control command to the inference engine.

    switch_agent payload::

        {"checkpoint_path": "/path/to/Agent.pt",
         "env_id": "BreakoutNoFrameskip-v4",
         "device": "cpu"}
    """
    loop = asyncio.get_event_loop()

    if cmd.command == "loop":
        engine.set_loop(bool((cmd.payload or {}).get("enabled", True)))

    elif cmd.command == "restart":
        engine.restart_episode()

    elif cmd.command == "pause":
        engine.pause()

    elif cmd.command == "resume":
        engine.resume()

    elif cmd.command == "switch_agent":
        p = cmd.payload or {}
        ckpt = Path(p["checkpoint_path"])
        env_id = p.get("env_id") or _infer_env_id(ckpt.stem)
        device = p.get("device", DEFAULT_DEVICE)
        # Run blocking operation in thread pool
        await loop.run_in_executor(
            None, engine.switch_agent, ckpt, env_id, device
        )

    else:
        return {"status": "error", "detail": f"Unknown command: {cmd.command!r}"}

    return {"status": "ok"}


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_handler(
    ws: WebSocket,
    agent: Optional[str] = Query(None),
    env_id: Optional[str] = Query(None),
    device: Optional[str] = Query(None),
) -> None:
    """
    Single WebSocket endpoint.

    On connect (if agent is provided and no inference is running) the engine is
    started with the requested agent.  Frames are streamed at inference rate.
    Events (agent_loaded, episode_start/end, error) are interleaved as they occur.
    """
    await ws.accept()
    loop = asyncio.get_event_loop()

    # Thread-safe event queue: inference thread → async WS sender
    from queue import Queue as SyncQueue
    event_q: SyncQueue = SyncQueue()

    def event_cb(event_name: str, data: dict) -> None:
        event_q.put_nowait({"type": "event", "event": event_name, "data": data})

    engine.register_event_callback(event_cb)

    # Start inference if not running and an agent was requested
    if agent and not engine.is_running:
        # Find checkpoint path
        ckpt_path = _resolve_checkpoint(agent)
        resolved_env = env_id or _infer_env_id(agent)
        resolved_dev = device or DEFAULT_DEVICE
        try:
            await loop.run_in_executor(
                None, engine.start, ckpt_path, resolved_env, resolved_dev
            )
        except Exception as exc:
            await ws.send_json({
                "type": "event",
                "event": "error",
                "data": {"message": str(exc)},
            })
            engine.unregister_event_callback(event_cb)
            return

    # Send initial config snapshot
    cfg = engine.get_config()
    agents_list = await list_agents()
    await ws.send_json({
        "type": "config",
        **cfg,
        "agents": agents_list,
    })

    try:
        while True:
            # Flush any queued events first (non-blocking)
            while not event_q.empty():
                try:
                    await ws.send_json(event_q.get_nowait())
                except Empty:
                    break

            # Fetch next frame (0.05 s timeout)
            frame = await loop.run_in_executor(None, engine.get_frame, 0.05)
            if frame is not None:
                await ws.send_json(frame)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as exc:
        logger.error("WebSocket error: %s", exc)
    finally:
        engine.unregister_event_callback(event_cb)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_checkpoint(agent_name: str) -> Path:
    """
    Resolve agent name to absolute checkpoint path.
    Accepts bare stem ("Breakout") or absolute path.
    """
    p = Path(agent_name)
    if p.is_absolute() and p.exists():
        return p
    # Look in checkpoint dir
    candidate = Path(CHECKPOINT_DIR) / f"{agent_name}.pt"
    if candidate.exists():
        return candidate
    # Maybe the name already includes .pt
    candidate2 = Path(CHECKPOINT_DIR) / agent_name
    if candidate2.exists():
        return candidate2
    raise FileNotFoundError(
        f"Checkpoint not found for agent {agent_name!r} in {CHECKPOINT_DIR}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
