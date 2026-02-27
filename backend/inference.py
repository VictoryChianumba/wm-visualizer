"""
IRIS inference engine.

Architecture
------------
InferenceEngine runs one background daemon thread that:
  1. Loads an IRIS Agent from a checkpoint.
  2. Steps the agent through a real Atari environment.
  3. Runs a single-block, no-KV-cache WorldModel forward pass for hook extraction
     (gives a consistent (1, nh, 17, 17) attention shape every step).
  4. Encodes the raw RGB frame as a base64 PNG.
  5. Pushes FrameData onto a bounded Queue.

If the queue is full the frame is dropped — inference is never blocked by a slow
consumer.  Agent switching stops the thread, flushes the queue, and restarts with
the new agent.
"""

import base64
import io
import logging
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from hooks import IrisHookExtractor

logger = logging.getLogger(__name__)

_QUEUE_MAXSIZE = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_token_labels(num_tokens: int, tokens_per_block: int) -> List[str]:
    """Token labels derived from model config — never hardcoded."""
    return [
        "act" if i % tokens_per_block == tokens_per_block - 1
        else f"o{i % tokens_per_block}"
        for i in range(num_tokens)
    ]


class _FpsCounter:
    """Rolling 1-second FPS counter (thread-safe for single writer)."""

    def __init__(self) -> None:
        self._ts: deque = deque()
        self.fps: float = 0.0

    def tick(self) -> float:
        now = time.perf_counter()
        self._ts.append(now)
        cutoff = now - 1.0
        while self._ts and self._ts[0] < cutoff:
            self._ts.popleft()
        self.fps = float(len(self._ts))
        return self.fps


def _encode_frame(obs_np: np.ndarray) -> str:
    """Encode (H, W, 3) uint8 array → base64 PNG string."""
    img = Image.fromarray(obs_np.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# FrameData
# ---------------------------------------------------------------------------

@dataclass
class FrameData:
    """One complete inference step, ready for JSON serialisation."""

    type: str = "frame"
    frame: str = ""                              # base64 PNG
    attention: Dict[str, List] = field(default_factory=dict)
    # str(layer_idx) → list[nh][T_q][T_k]
    norms: List[float] = field(default_factory=list)   # per layer
    metrics: Dict[str, Any] = field(default_factory=dict)
    token_layout: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "frame": self.frame,
            "attention": self.attention,
            "norms": self.norms,
            "metrics": self.metrics,
            "token_layout": self.token_layout,
        }


# ---------------------------------------------------------------------------
# Agent loader
# ---------------------------------------------------------------------------

def _load_agent(iris_root: Path, checkpoint_path: Path, env_id: str, device_str: str):
    """
    Load an IRIS Agent from a checkpoint.

    Uses Hydra's compose API (not @hydra.main) so it can be called at runtime.
    GlobalHydra state is cleared before and after to allow repeated calls.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate

    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(
            config_dir=str(iris_root / "config"),
        ):
            cfg = compose(
                config_name="trainer",
                overrides=[
                    f"env.train.id={env_id}",
                    f"env.test.id={env_id}",
                    f"initialization.path_to_checkpoint={checkpoint_path}",
                    f"common.device={device_str}",
                    "wandb.mode=disabled",
                ],
            )
    finally:
        GlobalHydra.instance().clear()

    # Import IRIS modules (iris/src must already be on sys.path)
    from agent import Agent
    from envs import SingleProcessEnv
    from models.actor_critic import ActorCritic
    from models.world_model import WorldModel

    device = torch.device(device_str)
    env_fn = partial(instantiate, config=cfg.env.test)
    env = SingleProcessEnv(env_fn)

    tokenizer = instantiate(cfg.tokenizer)
    world_model = WorldModel(
        obs_vocab_size=tokenizer.vocab_size,
        act_vocab_size=env.num_actions,
        config=instantiate(cfg.world_model),
    )
    actor_critic = ActorCritic(**cfg.actor_critic, act_vocab_size=env.num_actions)
    agent = Agent(tokenizer, world_model, actor_critic).to(device)

    ckpt = cfg.initialization.path_to_checkpoint
    if ckpt is None:
        ckpt = str(iris_root / "checkpoints" / "last.pt")
    agent.load(Path(ckpt), device)
    agent.eval()

    return agent, env, cfg


# ---------------------------------------------------------------------------
# InferenceEngine
# ---------------------------------------------------------------------------

class InferenceEngine:
    """
    Manages one IRIS inference loop in a background daemon thread.

    Usage::

        engine = InferenceEngine(iris_src="/iris/src", iris_root="/iris")
        engine.start(Path("Breakout.pt"), "BreakoutNoFrameskip-v4")
        frame_dict = engine.get_frame()          # None on timeout
        engine.switch_agent(Path("Alien.pt"), "AlienNoFrameskip-v4")
        engine.stop()

    Thread safety: all public methods are safe to call from any thread.
    """

    def __init__(self, iris_src: str, iris_root: str) -> None:
        self._iris_src = Path(iris_src)
        self._iris_root = Path(iris_root)

        # Add IRIS src/ to path once at construction time
        if str(self._iris_src) not in sys.path:
            sys.path.insert(0, str(self._iris_src))

        self._queue: Queue = Queue(maxsize=_QUEUE_MAXSIZE)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()          # not paused initially
        self._reset_requested = threading.Event()
        self._loop_episodes = True

        self._hooks = IrisHookExtractor()
        self._agent = None
        self._env = None
        self._cfg = None

        # Counters (written only from inference thread)
        self._step_count = 0
        self._episode_count = 0
        self._drop_count = 0
        self._total_frames = 0
        self._infer_fps = _FpsCounter()

        # Event callbacks: called from inference thread with (event_name, data)
        self._event_callbacks: List[Callable] = []
        self._cb_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    def start(
        self,
        checkpoint_path: Path,
        env_id: str,
        device_str: str = "cpu",
        loop: bool = True,
    ) -> None:
        """Load agent and start background inference thread."""
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Engine already running; call stop() or switch_agent() first")

        self._stop_event.clear()
        self._pause_event.set()
        self._reset_requested.clear()
        self._loop_episodes = loop
        self._step_count = 0
        self._episode_count = 0
        self._drop_count = 0
        self._total_frames = 0
        self._infer_fps = _FpsCounter()

        logger.info("Loading agent from %s (env=%s, device=%s)", checkpoint_path, env_id, device_str)
        try:
            self._agent, self._env, self._cfg = _load_agent(
                self._iris_root, checkpoint_path, env_id, device_str
            )
        except Exception as exc:
            logger.error("Agent load failed: %s", exc)
            self._emit_event("error", {"message": f"Agent load failed: {exc}"})
            raise

        self._hooks.attach(self._agent.world_model)

        num_layers = len(self._agent.world_model.transformer.blocks)
        num_heads = self._agent.world_model.transformer.config.num_heads
        tpb = self._agent.world_model.config.tokens_per_block

        self._emit_event("agent_loaded", {
            "agent": Path(checkpoint_path).stem,
            "env_id": env_id,
            "layers": num_layers,
            "heads": num_heads,
            "tokens_per_block": tpb,
        })

        self._thread = threading.Thread(
            target=self._run_safe, name="iris-inference", daemon=True
        )
        self._thread.start()
        logger.info("Inference started (%d layers, %d heads, tpb=%d)", num_layers, num_heads, tpb)

    def stop(self) -> None:
        """Signal the thread to stop, wait up to 10 s, then clean up."""
        self._stop_event.set()
        self._pause_event.set()  # unblock if paused
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            if self._thread.is_alive():
                logger.warning("Inference thread did not stop within 10 s")
            self._thread = None

        if self._hooks.num_layers > 0:
            self._hooks.detach()
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None
        self._agent = None

    def switch_agent(
        self,
        checkpoint_path: Path,
        env_id: str,
        device_str: str = "cpu",
        loop: bool = True,
    ) -> None:
        """Stop current agent, flush queue, start fresh with new agent."""
        self.stop()
        self._drain_queue()
        self.start(checkpoint_path, env_id, device_str, loop)

    # ------------------------------------------------------------------
    # Runtime controls
    # ------------------------------------------------------------------

    def pause(self) -> None:
        self._pause_event.clear()

    def resume(self) -> None:
        self._pause_event.set()

    def restart_episode(self) -> None:
        self._reset_requested.set()

    def set_loop(self, enabled: bool) -> None:
        self._loop_episodes = enabled

    def get_frame(self, timeout: float = 0.05) -> Optional[dict]:
        """Return the next FrameData dict or None if queue empty within timeout."""
        try:
            return self._queue.get(timeout=timeout).to_dict()
        except Empty:
            return None

    def register_event_callback(self, cb: Callable) -> None:
        with self._cb_lock:
            self._event_callbacks.append(cb)

    def unregister_event_callback(self, cb: Callable) -> None:
        with self._cb_lock:
            try:
                self._event_callbacks.remove(cb)
            except ValueError:
                pass

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def drop_rate(self) -> float:
        if self._total_frames == 0:
            return 0.0
        return self._drop_count / self._total_frames

    def get_config(self) -> dict:
        """Current model config (returns empty dict if no agent loaded)."""
        if self._agent is None:
            return {}
        wm = self._agent.world_model
        return {
            "num_layers": len(wm.transformer.blocks),
            "num_heads": wm.transformer.config.num_heads,
            "embed_dim": wm.transformer.config.embed_dim,
            "tokens_per_block": wm.config.tokens_per_block,
            "max_blocks": wm.config.max_blocks,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _drain_queue(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Empty:
                break

    def _emit_event(self, event_name: str, data: dict) -> None:
        with self._cb_lock:
            cbs = list(self._event_callbacks)
        for cb in cbs:
            try:
                cb(event_name, data)
            except Exception as exc:
                logger.debug("Event callback error: %s", exc)

    def _run_safe(self) -> None:
        try:
            self._run_inner()
        except Exception as exc:
            logger.error("Inference thread crashed: %s", exc, exc_info=True)
            self._emit_event("error", {"message": str(exc)})

    def _run_inner(self) -> None:
        agent = self._agent
        env = self._env
        hooks = self._hooks
        wm = agent.world_model
        tokenizer = agent.tokenizer
        device = agent.device

        tpb = wm.config.tokens_per_block
        num_layers = len(wm.transformer.blocks)
        token_layout = {
            "tokens_per_block": tpb,
            "obs_per_block": tpb - 1,
            "labels": get_token_labels(tpb, tpb),
        }

        # --- Episode init ---
        obs = env.reset()                          # (1, H, W, C) uint8
        obs_tensor = _to_tensor(obs, device)       # (1, C, H, W) float [0,1]
        agent.actor_critic.reset(n=1)
        self._episode_count += 1
        ep_return = 0.0
        self._emit_event("episode_start", {"episode": self._episode_count})

        while not self._stop_event.is_set():
            # Respect pause
            self._pause_event.wait()
            if self._stop_event.is_set():
                break

            # Manual episode reset
            if self._reset_requested.is_set():
                self._reset_requested.clear()
                obs = env.reset()
                obs_tensor = _to_tensor(obs, device)
                agent.actor_critic.reset(n=1)
                self._episode_count += 1
                ep_return = 0.0
                self._emit_event("episode_start", {"episode": self._episode_count})

            # --- Agent step ---
            with torch.no_grad():
                act = agent.act(obs_tensor, should_sample=True).cpu().numpy()  # (1,)

            next_obs, reward, done, _ = env.step(act)
            r = float(reward[0]) if hasattr(reward, "__len__") else float(reward)
            ep_return += r

            # --- World model forward (single block, no KV cache) for hooks ---
            t_wm = time.perf_counter()
            attn_data: Optional[Dict[int, torch.Tensor]] = None
            norms_data: Optional[Dict[int, float]] = None
            try:
                with torch.no_grad():
                    obs_tokens = tokenizer.encode(
                        obs_tensor, should_preprocess=True
                    ).tokens  # (1, K=16)
                    act_tensor = torch.tensor(
                        [[act[0]]], dtype=torch.long, device=device
                    )  # (1, 1)
                    tokens = torch.cat([obs_tokens, act_tensor], dim=1)  # (1, 17)
                    wm(tokens, past_keys_values=None)
                attn_data, norms_data = hooks.get_data()
            except Exception as exc:
                logger.debug("WM forward failed: %s", exc)
            hook_ms = (time.perf_counter() - t_wm) * 1000.0

            # --- Update obs ---
            obs = next_obs
            obs_tensor = _to_tensor(obs, device)
            self._step_count += 1
            self._total_frames += 1
            self._infer_fps.tick()

            # --- Encode raw frame ---
            raw = _get_raw_frame(env)
            frame_b64 = _encode_frame(raw)

            # --- Build attention payload ---
            attention_payload: Dict[str, list] = {}
            if attn_data is not None:
                for li, attn_t in attn_data.items():
                    # (1, nh, T_q, T_k) → nested Python list [nh][T_q][T_k]
                    attention_payload[str(li)] = attn_t.squeeze(0).cpu().tolist()

            norms_payload = [
                norms_data[i] if norms_data and i in norms_data else 0.0
                for i in range(num_layers)
            ]

            frame_data = FrameData(
                frame=frame_b64,
                attention=attention_payload,
                norms=norms_payload,
                metrics={
                    "infer_fps": round(self._infer_fps.fps, 1),
                    "step": self._step_count,
                    "episode": self._episode_count,
                    "queue_depth": self._queue.qsize(),
                    "hook_latency_ms": round(hook_ms, 2),
                    "drop_rate": round(self.drop_rate, 3),
                    "return": round(ep_return, 1),
                },
                token_layout=token_layout,
            )

            # Drop frame if consumer is behind — never block inference
            try:
                self._queue.put_nowait(frame_data)
            except Full:
                self._drop_count += 1

            # --- Episode end ---
            done_val = bool(done[0]) if hasattr(done, "__len__") else bool(done)
            if done_val:
                self._emit_event("episode_end", {
                    "episode": self._episode_count,
                    "return": round(ep_return, 1),
                    "steps": self._step_count,
                })
                ep_return = 0.0
                if self._loop_episodes:
                    obs = env.reset()
                    obs_tensor = _to_tensor(obs, device)
                    agent.actor_critic.reset(n=1)
                    self._episode_count += 1
                    self._emit_event("episode_start", {"episode": self._episode_count})
                else:
                    break


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _to_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    """(1, H, W, C) uint8  →  (1, C, H, W) float32 [0, 1]."""
    import torch
    from einops import rearrange
    return rearrange(torch.FloatTensor(obs).div(255), "n h w c -> n c h w").to(device)


def _get_raw_frame(env) -> np.ndarray:
    """
    Return the original human-readable RGB frame (H, W, 3) uint8.

    Tries env.env.unwrapped.original_obs (set by IRIS ALE wrappers) first,
    then falls back to ALE render, then to a placeholder.
    """
    try:
        raw = env.env.unwrapped.original_obs
        if raw is not None:
            return np.asarray(raw, dtype=np.uint8)
    except AttributeError:
        pass
    try:
        raw = env.env.unwrapped.render("rgb_array")
        if raw is not None:
            return np.asarray(raw, dtype=np.uint8)
    except Exception:
        pass
    return np.zeros((210, 160, 3), dtype=np.uint8)
