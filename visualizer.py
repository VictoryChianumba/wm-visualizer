#!/usr/bin/env python3
"""
World Model Interpretability Visualizer
========================================
Standalone tool that runs a pretrained Ha & Schmidhuber World Model
(VAE + MDN-RNN + Controller) and displays real-time internal-state
visualizations alongside gameplay.

This project is separate from the world-models training code.
Point it at any trained checkpoint directory via --agent_dir, and
at the world-models source tree via --world_models_dir (or the
WM_PROJECT_DIR environment variable).

Panes
-----
  Top-left     : Raw game frame (before any preprocessing)
  Top-right    : MDN mixture-weight heatmap  (rows = mixture components,
                 cols = recent time steps; brightness = weight)
  Bottom-left  : Latent activation norms  (|z_i| per VAE latent dimension)
  Bottom-right : Live log / running stats

Controls (top bar)
------------------
  ◀ / ▶ Agent   : cycle through discovered agents
  ◀ / ▶ Game    : cycle through available environments
  Loop / Restart / Pause : episode flow
  Mode slider   : switch heatmap between mixture weights (0),
                  mixture-mean L2 norms (1), and uncertainty σ (2)
  Path text box : specify agent-checkpoint directory; click Scan to refresh
  Quit          : clean shutdown
"""

# ── std ───────────────────────────────────────────────────────────────────────
import sys
import os
import time
import queue
import threading
import logging
import json
import argparse
import traceback
import platform
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

# ── Resolve world-models source path BEFORE any model imports ────────────────
# Priority: --world_models_dir flag > WM_PROJECT_DIR env var > adjacent sibling
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--world_models_dir", default=None)
_pre_args, _ = _pre.parse_known_args()

_WM_DIR: str = (
    _pre_args.world_models_dir
    or os.environ.get("WM_PROJECT_DIR", "")
    or str(Path(__file__).parent.parent / "world-models-carracing")
)
if _WM_DIR not in sys.path:
    sys.path.insert(0, _WM_DIR)

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn

# ── Colab / backend detection ─────────────────────────────────────────────────
IN_COLAB: bool = "google.colab" in sys.modules

import matplotlib  # noqa: E402
if IN_COLAB:
    matplotlib.use("Agg")
elif platform.system() == "Darwin":
    try:
        matplotlib.use("macosx")
    except Exception:
        matplotlib.use("TkAgg")
else:
    matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button, TextBox

# ── world-models imports (resolved via sys.path above) ───────────────────────
try:
    from models.vae import VAE
    from models.mdnrnn import MDNRNN
    from models.controller import Controller, squash_action
    from config import Config
    from utils import preprocess_frame, normalize_img_uint8
except ImportError as _e:
    sys.exit(
        f"Cannot import world-models source from '{_WM_DIR}'.\n"
        f"  → {_e}\n"
        "Pass --world_models_dir /path/to/world-models-carracing "
        "or set the WM_PROJECT_DIR environment variable."
    )

# ── constants ─────────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).parent
LOGS_DIR: Path = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

MAX_SESSION_LOGS: int = 10
QUEUE_MAXSIZE: int = 4
HISTORY_LEN: int = 60   # time steps shown in the heatmap


# ── data classes ──────────────────────────────────────────────────────────────

@dataclass
class HookData:
    """Internal model state extracted at one inference step."""
    z: np.ndarray              # (z_dim,)
    logvar: np.ndarray         # (z_dim,)
    h_state: np.ndarray        # (hidden,)
    mixture_weights: np.ndarray  # (n_mix,)  softmax'd
    mu_mix: np.ndarray         # (n_mix, z_dim)
    log_sigma_mix: np.ndarray  # (n_mix, z_dim)
    hook_latency_ms: float = 0.0


@dataclass
class FrameData:
    """One inference step bundled for the render queue."""
    raw_frame: np.ndarray   # (H, W, 3) uint8 RGB
    hook: HookData
    step: int
    episode: int
    reward: float
    done: bool
    is_sentinel: bool = False


# ── session logging ───────────────────────────────────────────────────────────

def _rotate_logs(logs_dir: Path, max_keep: int = MAX_SESSION_LOGS) -> None:
    files = sorted(logs_dir.glob("session_*.log"), key=lambda p: p.stat().st_mtime)
    for f in files[: max(0, len(files) - max_keep + 1)]:
        try:
            f.unlink()
        except OSError:
            pass


def _setup_session_logger() -> logging.Logger:
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"session_{ts}.log"
    _rotate_logs(LOGS_DIR)

    logger = logging.getLogger("wm_visualizer")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    class _JSONHandler(logging.FileHandler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                entry: Dict[str, Any] = {
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "level": record.levelname,
                    "msg": record.getMessage(),
                }
                if record.exc_info:
                    entry["exc"] = self.formatException(record.exc_info)
                self.stream.write(json.dumps(entry) + "\n")
                self.stream.flush()
            except Exception:
                pass

    fh = _JSONHandler(log_path)
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.propagate = False
    return logger


LOG: logging.Logger = _setup_session_logger()


# ── hook extractor ────────────────────────────────────────────────────────────

class HookExtractor:
    """
    Attaches read-only forward hooks to VAE and MDNRNN sub-modules.

    Hooks are placed on:
      vae.enc_fc   — captures concatenated [mu | logvar]  shape (B, 2*z_dim)
      rnn.rnn      — captures LSTM (output, (h_n, c_n))
      rnn.fc       — captures raw mixture params           shape (B, T, total)

    Thread-safe: all _data reads/writes are protected by a Lock.
    No files in the world-models source tree are modified.
    """

    def __init__(self) -> None:
        self._handles: list = []
        self._data: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def attach(self, vae: "VAE", rnn: "MDNRNN") -> None:
        """Register hooks. Removes any previously registered hooks first."""
        self.detach()

        def _vae_enc_fc(module: nn.Module, inp: Any, out: torch.Tensor) -> None:
            with self._lock:
                arr = out.detach().cpu().float().numpy()   # (B, 2*z_dim)
                zd = arr.shape[-1] // 2
                self._data["z"] = arr[0, :zd].copy()
                self._data["logvar"] = arr[0, zd:].copy()

        def _rnn_lstm(module: nn.Module, inp: Any, out: Any) -> None:
            # out = (output_tensor, (h_n, c_n))
            with self._lock:
                _, (h_n, _) = out
                arr = h_n.detach().cpu().float().numpy()  # (layers, B, hidden)
                self._data["h_state"] = arr[0, 0, :].copy()

        def _rnn_fc(module: nn.Module, inp: Any, out: torch.Tensor) -> None:
            # out shape: (B, T, n_mix + 2*n_mix*z_dim)
            with self._lock:
                arr = out.detach().cpu().float().numpy()
                p = arr[0, -1, :]           # last time-step, first batch item
                km = rnn.n_mix
                zd = rnn.z_dim
                pi_raw = p[:km]
                mu_flat = p[km: km + km * zd]
                ls_flat = p[km + km * zd:]
                # numerically-stable softmax
                pi_shift = pi_raw - pi_raw.max()
                pi_exp = np.exp(pi_shift)
                self._data["mixture_weights"] = (pi_exp / pi_exp.sum()).copy()
                self._data["mu_mix"] = mu_flat.reshape(km, zd).copy()
                self._data["log_sigma_mix"] = ls_flat.reshape(km, zd).copy()

        h1 = vae.enc_fc.register_forward_hook(_vae_enc_fc)
        h2 = rnn.rnn.register_forward_hook(_rnn_lstm)
        h3 = rnn.fc.register_forward_hook(_rnn_fc)
        self._handles = [h1, h2, h3]
        LOG.debug("HookExtractor: attached (vae.enc_fc, rnn.rnn, rnn.fc)")

    def detach(self) -> None:
        """Remove all registered hooks and clear cached data."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
        with self._lock:
            self._data.clear()
        LOG.debug("HookExtractor: all hooks removed")

    def get_data(self, n_mix: int, z_dim: int) -> Optional[HookData]:
        """Return a snapshot of hook data, or None if not yet populated."""
        required = ("z", "logvar", "h_state", "mixture_weights", "mu_mix", "log_sigma_mix")
        with self._lock:
            if not all(k in self._data for k in required):
                return None
            return HookData(
                z=self._data["z"].copy(),
                logvar=self._data["logvar"].copy(),
                h_state=self._data["h_state"].copy(),
                mixture_weights=self._data["mixture_weights"].copy(),
                mu_mix=self._data["mu_mix"].copy(),
                log_sigma_mix=self._data["log_sigma_mix"].copy(),
            )


# ── agent bundle ──────────────────────────────────────────────────────────────

@dataclass
class AgentBundle:
    """All models + config for one trained agent."""
    name: str
    path: Path
    vae: "VAE"
    rnn: "MDNRNN"
    controller: "Controller"
    cfg: "Config"
    device: torch.device

    @classmethod
    def load(cls, agent_dir: Path, device: torch.device) -> "AgentBundle":
        """
        Load VAE, MDNRNN, and Controller from agent_dir.

        Expected files:
          vae_final.pt        — {"state_dict": ..., "cfg": {...}}
          mdnrnn_final.pt     — {"state_dict": ..., "cfg": {...}}
          controller_best.npz — {best_theta, best_score, cfg}
        """
        vae_path  = agent_dir / "vae_final.pt"
        rnn_path  = agent_dir / "mdnrnn_final.pt"
        ctrl_path = agent_dir / "controller_best.npz"

        for p in (vae_path, rnn_path, ctrl_path):
            if not p.exists():
                raise FileNotFoundError(f"Missing checkpoint: {p}")

        # ── VAE ──────────────────────────────────────────────────────────────
        vae_ckpt = torch.load(vae_path, map_location=device, weights_only=False)
        vae_cfg: Dict[str, Any] = vae_ckpt.get("cfg", {})
        in_ch = 1 if vae_cfg.get("grayscale", True) else 3
        z_dim: int = int(vae_cfg.get("z_dim", 32))
        vae = VAE(in_ch=in_ch, z_dim=z_dim).to(device)
        vae.load_state_dict(vae_ckpt["state_dict"])
        vae.eval()

        # ── MDN-RNN ───────────────────────────────────────────────────────────
        rnn_ckpt = torch.load(rnn_path, map_location=device, weights_only=False)
        rnn_cfg: Dict[str, Any] = rnn_ckpt.get("cfg", {})
        n_mix: int      = int(rnn_cfg.get("mdn_mixtures", 10))
        rnn_hidden: int = int(rnn_cfg.get("rnn_hidden", 256))
        rnn = MDNRNN(z_dim=z_dim, action_dim=3, hidden=rnn_hidden, n_mix=n_mix).to(device)
        rnn.load_state_dict(rnn_ckpt["state_dict"])
        rnn.eval()

        # ── Controller ────────────────────────────────────────────────────────
        ctrl_ckpt = np.load(ctrl_path, allow_pickle=True)
        best_theta = ctrl_ckpt["best_theta"].astype(np.float32)
        raw_cfg = ctrl_ckpt["cfg"]
        ctrl_cfg_dict: Dict[str, Any] = (
            raw_cfg.item() if hasattr(raw_cfg, "item") else
            raw_cfg if isinstance(raw_cfg, dict) else {}
        )
        ctrl_hidden: int = int(ctrl_cfg_dict.get("controller_hidden", 0))
        in_dim: int = z_dim + rnn_hidden
        controller = Controller(in_dim=in_dim, hidden=ctrl_hidden, action_dim=3).to(device)
        offset = 0
        with torch.no_grad():
            for p in controller.parameters():
                n = p.numel()
                p.copy_(torch.from_numpy(best_theta[offset: offset + n]).view_as(p))
                offset += n
        controller.eval()

        # ── Merged config ─────────────────────────────────────────────────────
        cfg = Config()
        for src in (vae_cfg, rnn_cfg, ctrl_cfg_dict):
            for k, v in src.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)

        LOG.info(
            f"Agent loaded: '{agent_dir.name}'  "
            f"z={z_dim}  n_mix={n_mix}  hidden={rnn_hidden}"
        )
        return cls(
            name=agent_dir.name,
            path=agent_dir,
            vae=vae,
            rnn=rnn,
            controller=controller,
            cfg=cfg,
            device=device,
        )


def discover_agents(agent_path: Path) -> List[Path]:
    """
    Return directories under (and including) agent_path that contain all three
    required checkpoint files.
    """
    required = {"vae_final.pt", "mdnrnn_final.pt", "controller_best.npz"}
    candidates: List[Path] = []
    if agent_path.is_dir():
        candidates.append(agent_path)
        try:
            candidates.extend(p for p in agent_path.iterdir() if p.is_dir())
        except PermissionError:
            pass
    return [c for c in candidates if all((c / r).exists() for r in required)]


# ── inference thread ──────────────────────────────────────────────────────────

class InferenceThread(threading.Thread):
    """
    Producer thread: runs one rollout per episode, pushes FrameData to the
    bounded queue.  Drops frames (put_nowait) when the queue is full — never
    blocks the render thread.

    LSTM hidden state (h, c) is carried between steps; this is the analogue of
    KV-caching in transformer-based world models.  Hook shapes are verified to
    be consistent whether h is fresh (None) or carried.
    """

    def __init__(
        self,
        agent: AgentBundle,
        env_id: str,
        frame_queue: "queue.Queue[FrameData]",
        hook_extractor: HookExtractor,
        shutdown_event: threading.Event,
        pause_event: threading.Event,
        loop: bool = True,
    ) -> None:
        super().__init__(name="InferenceThread", daemon=True)
        self.agent = agent
        self.env_id = env_id
        self.frame_queue = frame_queue
        self.hooks = hook_extractor
        self.shutdown = shutdown_event
        self.pause = pause_event
        self.loop = loop
        # Public stats read by the render thread (benign races)
        self._episode: int = 0
        self._dropped: int = 0
        self._inference_fps: float = 0.0

    def run(self) -> None:
        LOG.info(f"InferenceThread start  agent={self.agent.name}  env={self.env_id}")
        try:
            self._rollout_loop()
        except Exception:
            LOG.error(f"InferenceThread unhandled:\n{traceback.format_exc()}")
        finally:
            self._push_sentinel()
            LOG.info("InferenceThread stopped")

    # ── private ───────────────────────────────────────────────────────────────

    def _push_sentinel(self) -> None:
        n_mix   = self.agent.rnn.n_mix
        z_dim   = self.agent.rnn.z_dim
        hidden  = self.agent.cfg.rnn_hidden
        sentinel = FrameData(
            raw_frame=np.zeros((96, 96, 3), dtype=np.uint8),
            hook=HookData(
                z=np.zeros(z_dim),
                logvar=np.zeros(z_dim),
                h_state=np.zeros(hidden),
                mixture_weights=np.ones(n_mix) / n_mix,
                mu_mix=np.zeros((n_mix, z_dim)),
                log_sigma_mix=np.zeros((n_mix, z_dim)),
            ),
            step=0, episode=self._episode, reward=0.0, done=True,
            is_sentinel=True,
        )
        try:
            self.frame_queue.put_nowait(sentinel)
        except queue.Full:
            pass

    def _rollout_loop(self) -> None:
        cfg    = self.agent.cfg
        device = self.agent.device

        while not self.shutdown.is_set():
            if self.pause.is_set():
                time.sleep(0.05)
                continue

            env = None
            try:
                import gymnasium as gym
                env = gym.make(self.env_id, render_mode="rgb_array")
                obs, _ = env.reset()
                self._episode += 1
                LOG.info(f"Episode {self._episode} started  env={self.env_id}")

                # LSTM hidden state: (1, 1, hidden)  — zeros = conventional init
                h = (
                    torch.zeros(1, 1, cfg.rnn_hidden, device=device),
                    torch.zeros(1, 1, cfg.rnn_hidden, device=device),
                )
                action     = np.zeros(3, dtype=np.float32)
                step       = 0
                ep_reward  = 0.0
                step_times: List[float] = []

                while not self.shutdown.is_set():
                    if self.pause.is_set():
                        time.sleep(0.05)
                        continue

                    t0 = time.perf_counter()

                    # Raw display frame (before any preprocessing)
                    raw_frame = env.render()   # (H, W, 3) uint8

                    # Preprocess exactly as train_controller_cmaes.eval_controller
                    frame_proc = preprocess_frame(raw_frame, cfg.frame_size, cfg.grayscale)
                    frame_norm = normalize_img_uint8(frame_proc)
                    x = np.transpose(frame_norm, (2, 0, 1))[None, ...]   # (1,C,H,W)
                    x_t = torch.from_numpy(x).float().to(device)

                    t_hook_start = time.perf_counter()
                    with torch.no_grad():
                        # VAE encode  →  hook fires on vae.enc_fc
                        mu, logvar = self.agent.vae.encode(x_t)
                        z = mu   # (1, z_dim)

                        # RNN step: (z_t, a_{t-1})  →  hook fires on rnn.rnn + rnn.fc
                        za = torch.cat(
                            [z, torch.from_numpy(action[None, :]).float().to(device)],
                            dim=-1,
                        ).unsqueeze(1)                               # (1, 1, z_dim+3)
                        _, h = self.agent.rnn(za, h)

                        # Controller: [z_t, h_t]  →  a_t
                        h_flat = h[0].squeeze(0).squeeze(0)          # (hidden,)
                        feat   = torch.cat([z.squeeze(0), h_flat], dim=-1).unsqueeze(0)
                        raw_action = self.agent.controller(feat).squeeze(0).cpu().numpy()
                    t_hook_end = time.perf_counter()

                    hook_latency_ms = (t_hook_end - t_hook_start) * 1000.0
                    action = squash_action(raw_action)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    ep_reward += float(reward)
                    step += 1

                    # Collect hook data (falls back to direct tensors if hooks not ready)
                    hd = self.hooks.get_data(
                        n_mix=self.agent.rnn.n_mix,
                        z_dim=self.agent.rnn.z_dim,
                    )
                    if hd is None:
                        hd = HookData(
                            z=mu.squeeze().cpu().numpy(),
                            logvar=logvar.squeeze().cpu().numpy(),
                            h_state=h[0].squeeze().cpu().numpy(),
                            mixture_weights=np.ones(self.agent.rnn.n_mix) / self.agent.rnn.n_mix,
                            mu_mix=np.zeros((self.agent.rnn.n_mix, self.agent.rnn.z_dim)),
                            log_sigma_mix=np.zeros((self.agent.rnn.n_mix, self.agent.rnn.z_dim)),
                        )
                    hd.hook_latency_ms = hook_latency_ms

                    fd = FrameData(
                        raw_frame=raw_frame,
                        hook=hd,
                        step=step,
                        episode=self._episode,
                        reward=ep_reward,
                        done=done,
                    )
                    # Non-blocking: drop frame rather than stall inference
                    try:
                        self.frame_queue.put_nowait(fd)
                    except queue.Full:
                        self._dropped += 1

                    step_times.append(time.perf_counter() - t0)
                    if len(step_times) > 30:
                        step_times.pop(0)
                    if step_times:
                        avg = sum(step_times) / len(step_times)
                        self._inference_fps = 1.0 / avg if avg > 0 else 0.0

                    if done:
                        LOG.info(
                            f"Episode {self._episode} done  "
                            f"steps={step}  reward={ep_reward:.1f}"
                        )
                        break

            except Exception:
                LOG.error(f"Rollout error:\n{traceback.format_exc()}")
            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass

            if not self.loop or self.shutdown.is_set():
                break


# ── visualizer app (all matplotlib on main thread) ────────────────────────────

class VisualizerApp:
    """
    Main application.  Builds the matplotlib figure and drives the render loop
    via FuncAnimation (desktop) or a manual IPython loop (Colab).

    ALL matplotlib calls — imshow, set_data, axes methods, widget callbacks —
    are guarded by an assertion that they run on the main thread.
    """

    VIS_MODES = ("Mix Weights", "Mix Mean Norms", "Uncertainty σ")

    def __init__(
        self,
        agent_dirs: List[Path],
        env_ids: List[str],
        device: torch.device,
        loop: bool = True,
    ) -> None:
        assert (
            threading.current_thread() is threading.main_thread()
        ), "VisualizerApp must be created on the main thread"

        self.agent_dirs = agent_dirs
        self.env_ids    = env_ids
        self.device     = device
        self.loop       = loop

        self._agent_idx = 0
        self._env_idx   = 0
        self._paused    = False
        self._vis_mode  = 0

        # Threading primitives
        self._frame_queue: "queue.Queue[FrameData]" = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self._shutdown_event = threading.Event()
        self._pause_event    = threading.Event()
        self._inference_thread: Optional[InferenceThread] = None

        self._hooks = HookExtractor()

        # Load initial agent
        self._agent: Optional[AgentBundle] = None
        self._load_agent(agent_dirs[0])
        assert self._agent is not None

        n_mix = self._agent.rnn.n_mix
        z_dim = self._agent.rnn.z_dim
        self._heatmap_history = np.zeros((n_mix, HISTORY_LEN), dtype=np.float32)
        self._z_dim_current   = z_dim

        # Stats
        self._render_times: List[float] = []
        self._render_fps: float = 0.0
        self._dropped_total: int = 0
        self._step_count: int   = 0
        self._episode_count: int = 0
        self._last_hook_lat: float = 0.0
        self._log_lines: List[str] = []

        self._build_figure()
        self._start_inference()

    # ═══════════════════════════════════════════════════════════ figure build

    def _build_figure(self) -> None:
        assert threading.current_thread() is threading.main_thread()

        plt.rcParams.update({
            "figure.facecolor": "#1a1a2e",
            "axes.facecolor":   "#0f1e35",
            "text.color":       "#dce4f0",
            "axes.labelcolor":  "#dce4f0",
            "xtick.color":      "#9ab0c8",
            "ytick.color":      "#9ab0c8",
            "axes.edgecolor":   "#2e4565",
            "axes.titlecolor":  "#7aa8d8",
            "axes.titlesize":   9,
            "axes.labelsize":   8,
        })

        self._fig = plt.figure(figsize=(16, 10), facecolor="#1a1a2e")
        try:
            self._fig.canvas.manager.set_window_title(
                "World Model Interpretability Visualizer"
            )
        except AttributeError:
            pass

        outer = gridspec.GridSpec(
            2, 1, figure=self._fig,
            height_ratios=[0.07, 0.93],
            hspace=0.06,
        )
        content = gridspec.GridSpecFromSubplotSpec(
            2, 2, subplot_spec=outer[1],
            hspace=0.38, wspace=0.28,
        )

        self._ax_frame = self._fig.add_subplot(content[0, 0])
        self._ax_heat  = self._fig.add_subplot(content[0, 1])
        self._ax_bar   = self._fig.add_subplot(content[1, 0])
        self._ax_log   = self._fig.add_subplot(content[1, 1])

        self._ax_frame.set_title("Raw Game Frame")
        self._ax_heat.set_title("MDN Mixture Heatmap")
        self._ax_bar.set_title("Latent Activation Norms  |z|")
        self._ax_log.set_title("Live Log")

        # Top-left: raw frame
        self._im_frame = self._ax_frame.imshow(
            np.zeros((96, 96, 3), dtype=np.uint8), aspect="auto"
        )
        self._ax_frame.axis("off")

        # Top-right: MDN heatmap
        n_mix = self._agent.rnn.n_mix
        self._im_heat = self._ax_heat.imshow(
            self._heatmap_history,
            aspect="auto", interpolation="nearest",
            vmin=0.0, vmax=1.0, cmap="inferno", origin="lower",
        )
        self._ax_heat.set_xlabel("Time step (recent →)")
        self._ax_heat.set_ylabel("Mixture component")
        self._ax_heat.set_yticks(range(n_mix))
        self._ax_heat.set_yticklabels([str(i) for i in range(n_mix)], fontsize=6)
        self._heat_cbar = self._fig.colorbar(
            self._im_heat, ax=self._ax_heat, fraction=0.046, pad=0.04
        )
        self._heat_cbar.ax.tick_params(labelsize=6, colors="#9ab0c8")

        # Bottom-left: latent norm bars
        z_dim = self._agent.rnn.z_dim
        self._bar_x = np.arange(z_dim)
        self._bars  = self._ax_bar.bar(
            self._bar_x, np.zeros(z_dim),
            color="#3d7ab5", edgecolor="none", width=0.8,
        )
        self._ax_bar.set_xlabel("Latent dimension  i")
        self._ax_bar.set_ylabel("|z_i|")
        self._ax_bar.set_xlim(-0.5, z_dim - 0.5)
        self._ax_bar.set_xticks(range(0, z_dim, max(1, z_dim // 8)))
        self._ax_bar.set_ylim(0, 3)
        from matplotlib.patches import Patch
        self._ax_bar.legend(
            handles=[
                Patch(color="#3d7ab5", label="positive z"),
                Patch(color="#c94040", label="negative z"),
            ],
            fontsize=6, loc="upper right", framealpha=0.4,
        )

        # Bottom-right: live log
        self._ax_log.axis("off")
        self._log_text = self._ax_log.text(
            0.02, 0.98, "",
            transform=self._ax_log.transAxes,
            va="top", ha="left", fontsize=7, family="monospace", color="#7ee88a",
        )

        self._build_controls()
        self._fig.canvas.mpl_connect("close_event", self._on_window_close)
        if not IN_COLAB:
            plt.ion()

    def _build_controls(self) -> None:
        assert threading.current_thread() is threading.main_thread()
        fig = self._fig

        # Agent navigation
        ax_pa = fig.add_axes([0.010, 0.928, 0.025, 0.040])
        ax_na = fig.add_axes([0.038, 0.928, 0.025, 0.040])
        ax_al = fig.add_axes([0.065, 0.920, 0.090, 0.055])
        self._btn_prev_agent = Button(ax_pa, "◀", color="#1f3248", hovercolor="#2e4a6a")
        self._btn_next_agent = Button(ax_na, "▶", color="#1f3248", hovercolor="#2e4a6a")
        ax_al.axis("off")
        self._lbl_agent = ax_al.text(
            0.5, 0.5, f"Agent: {self._agent.name}",
            ha="center", va="center", fontsize=7,
            color="#dce4f0", transform=ax_al.transAxes,
        )
        self._btn_prev_agent.on_clicked(self._on_prev_agent)
        self._btn_next_agent.on_clicked(self._on_next_agent)

        # Game navigation
        ax_pg = fig.add_axes([0.170, 0.928, 0.025, 0.040])
        ax_ng = fig.add_axes([0.198, 0.928, 0.025, 0.040])
        ax_gl = fig.add_axes([0.225, 0.920, 0.100, 0.055])
        self._btn_prev_game = Button(ax_pg, "◀", color="#1f3248", hovercolor="#2e4a6a")
        self._btn_next_game = Button(ax_ng, "▶", color="#1f3248", hovercolor="#2e4a6a")
        ax_gl.axis("off")
        self._lbl_game = ax_gl.text(
            0.5, 0.5, f"Game: {self.env_ids[self._env_idx]}",
            ha="center", va="center", fontsize=7,
            color="#dce4f0", transform=ax_gl.transAxes,
        )
        self._btn_prev_game.on_clicked(self._on_prev_game)
        self._btn_next_game.on_clicked(self._on_next_game)

        # Episode controls
        ax_loop    = fig.add_axes([0.340, 0.928, 0.055, 0.040])
        ax_restart = fig.add_axes([0.400, 0.928, 0.055, 0.040])
        ax_pause   = fig.add_axes([0.460, 0.928, 0.055, 0.040])
        self._btn_loop    = Button(ax_loop,    "Loop: ON", color="#1a3d1a", hovercolor="#245424")
        self._btn_restart = Button(ax_restart, "Restart",  color="#1f3248", hovercolor="#2e4a6a")
        self._btn_pause   = Button(ax_pause,   "Pause",    color="#1f3248", hovercolor="#2e4a6a")
        self._btn_loop.on_clicked(self._on_toggle_loop)
        self._btn_restart.on_clicked(self._on_restart)
        self._btn_pause.on_clicked(self._on_toggle_pause)
        for b in (self._btn_loop, self._btn_restart, self._btn_pause):
            b.label.set_fontsize(7)

        # Mode (layer) slider
        ax_mode_lbl = fig.add_axes([0.535, 0.960, 0.120, 0.020])
        ax_mode_lbl.axis("off")
        self._lbl_mode = ax_mode_lbl.text(
            0.5, 0.5, f"View: {self.VIS_MODES[0]}",
            ha="center", va="center", fontsize=7,
            color="#7aa8d8", transform=ax_mode_lbl.transAxes,
        )
        ax_slider = fig.add_axes([0.535, 0.930, 0.120, 0.022])
        self._slider_mode = Slider(
            ax_slider, "Mode",
            valmin=0, valmax=len(self.VIS_MODES) - 1,
            valinit=0, valstep=1, color="#3d5a80",
        )
        self._slider_mode.label.set_color("#dce4f0")
        self._slider_mode.label.set_fontsize(7)
        self._slider_mode.valtext.set_color("#dce4f0")
        self._slider_mode.valtext.set_fontsize(7)
        self._slider_mode.on_changed(self._on_mode_changed)

        # Agent-path text box + Scan
        ax_path = fig.add_axes([0.675, 0.930, 0.175, 0.040])
        self._tb_path = TextBox(
            ax_path, "Path: ",
            initial=str(self.agent_dirs[0].parent if self.agent_dirs else "."),
            color="#0e1e30", hovercolor="#162940",
        )
        self._tb_path.label.set_fontsize(7)
        self._tb_path.on_submit(self._on_path_submit)

        ax_scan = fig.add_axes([0.855, 0.928, 0.045, 0.040])
        self._btn_scan = Button(ax_scan, "Scan", color="#1f3248", hovercolor="#2e4a6a")
        self._btn_scan.label.set_fontsize(7)
        self._btn_scan.on_clicked(lambda e: self._on_path_submit(self._tb_path.text))

        ax_quit = fig.add_axes([0.905, 0.928, 0.045, 0.040])
        self._btn_quit = Button(ax_quit, "Quit", color="#3d1a1a", hovercolor="#6a2424")
        self._btn_quit.label.set_fontsize(7)
        self._btn_quit.on_clicked(self._on_quit)

    # ═══════════════════════════════════════════════════════ inference management

    def _load_agent(self, agent_dir: Path) -> None:
        self._agent = AgentBundle.load(agent_dir, self.device)
        self._hooks.attach(self._agent.vae, self._agent.rnn)

    def _start_inference(self) -> None:
        assert threading.current_thread() is threading.main_thread()
        self._shutdown_event.clear()
        self._pause_event.clear()
        env_id = self.env_ids[self._env_idx]
        self._inference_thread = InferenceThread(
            agent=self._agent,
            env_id=env_id,
            frame_queue=self._frame_queue,
            hook_extractor=self._hooks,
            shutdown_event=self._shutdown_event,
            pause_event=self._pause_event,
            loop=self.loop,
        )
        self._inference_thread.start()
        LOG.info(f"Inference started: agent={self._agent.name}  game={env_id}")

    def _stop_inference(self, timeout: float = 5.0) -> None:
        assert threading.current_thread() is threading.main_thread()
        if self._inference_thread and self._inference_thread.is_alive():
            LOG.info("Stopping InferenceThread …")
            self._shutdown_event.set()
            self._inference_thread.join(timeout=timeout)
            if self._inference_thread.is_alive():
                LOG.warning("InferenceThread did not stop in time")
        self._inference_thread = None
        while True:
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break
        self._hooks.detach()
        LOG.info("Inference stopped; queue flushed; hooks detached")

    def _switch_agent(self, new_dir: Path) -> None:
        assert threading.current_thread() is threading.main_thread()
        old_name = self._agent.name if self._agent else "—"
        self._stop_inference()
        try:
            self._load_agent(new_dir)
            self._heatmap_history = np.zeros(
                (self._agent.rnn.n_mix, HISTORY_LEN), dtype=np.float32
            )
            self._z_dim_current = self._agent.rnn.z_dim
            self._rebuild_heat_axes()
            self._rebuild_bar_axes()
            self._start_inference()
            msg = f"Agent: {old_name} → {self._agent.name}"
            LOG.info(msg)
            self._add_log(msg)
        except Exception as exc:
            msg = f"WARN: agent load failed — {exc}"
            LOG.error(f"{msg}\n{traceback.format_exc()}")
            self._add_log(msg)

    def _rebuild_heat_axes(self) -> None:
        assert threading.current_thread() is threading.main_thread()
        n_mix = self._agent.rnn.n_mix
        self._im_heat.set_data(self._heatmap_history)
        self._ax_heat.set_yticks(range(n_mix))
        self._ax_heat.set_yticklabels([str(i) for i in range(n_mix)], fontsize=6)

    def _rebuild_bar_axes(self) -> None:
        assert threading.current_thread() is threading.main_thread()
        z_dim = self._agent.rnn.z_dim
        self._ax_bar.cla()
        self._ax_bar.set_facecolor("#0f1e35")
        self._ax_bar.set_title("Latent Activation Norms  |z|", fontsize=9,
                               color="#7aa8d8")
        self._bar_x = np.arange(z_dim)
        self._bars  = self._ax_bar.bar(
            self._bar_x, np.zeros(z_dim),
            color="#3d7ab5", edgecolor="none", width=0.8,
        )
        self._ax_bar.set_xlabel("Latent dimension  i", fontsize=8)
        self._ax_bar.set_ylabel("|z_i|", fontsize=8)
        self._ax_bar.set_xlim(-0.5, z_dim - 0.5)
        self._ax_bar.set_xticks(range(0, z_dim, max(1, z_dim // 8)))
        self._ax_bar.set_ylim(0, 3)

    # ═══════════════════════════════════════════════════════════ render loop

    def _update_render(self, _frame: int) -> None:
        """Called by FuncAnimation on the main thread every ~50 ms."""
        assert threading.current_thread() is threading.main_thread()
        t0 = time.perf_counter()

        latest: Optional[FrameData] = None
        while True:
            try:
                fd = self._frame_queue.get_nowait()
            except queue.Empty:
                break
            if fd.is_sentinel:
                if not self.loop:
                    self._add_log("Episode ended.")
                break
            latest = fd

        if latest is not None:
            self._render_frame(latest)

        self._render_times.append(time.perf_counter() - t0)
        if len(self._render_times) > 30:
            self._render_times.pop(0)
        avg = sum(self._render_times) / len(self._render_times) if self._render_times else 1.0
        self._render_fps = 1.0 / avg if avg > 0 else 0.0
        self._update_log_pane()

    def _render_frame(self, fd: FrameData) -> None:
        assert threading.current_thread() is threading.main_thread()

        self._step_count     = fd.step
        self._episode_count  = fd.episode
        self._last_hook_lat  = fd.hook.hook_latency_ms
        if self._inference_thread:
            self._dropped_total = self._inference_thread._dropped

        # Raw frame
        self._im_frame.set_data(fd.raw_frame)

        # Heatmap column
        mode  = self._vis_mode
        n_mix = self._agent.rnn.n_mix

        if mode == 0:
            col = fd.hook.mixture_weights.copy()
        elif mode == 1:
            col = np.linalg.norm(fd.hook.mu_mix, axis=1)
            mx  = col.max() or 1.0
            col /= mx
        else:
            sigma = np.exp(np.clip(fd.hook.log_sigma_mix, -5, 5))
            col   = sigma.mean(axis=1)
            mx    = col.max() or 1.0
            col  /= mx

        if col.shape[0] != n_mix:
            col = np.ones(n_mix) / n_mix

        self._heatmap_history = np.roll(self._heatmap_history, -1, axis=1)
        self._heatmap_history[:, -1] = col
        self._im_heat.set_data(self._heatmap_history)

        # Latent norm bars
        z     = fd.hook.z
        z_abs = np.abs(z)
        for bar, h_val, v in zip(self._bars, z_abs, z):
            bar.set_height(h_val)
            bar.set_color("#3d7ab5" if v >= 0 else "#c94040")
        self._ax_bar.set_ylim(0, max(z_abs.max() * 1.25, 0.5))

    def _update_log_pane(self) -> None:
        assert threading.current_thread() is threading.main_thread()
        inf_fps = (
            self._inference_thread._inference_fps
            if self._inference_thread else 0.0
        )
        stats = [
            f"{'Agent':<8}: {self._agent.name if self._agent else '—'}",
            f"{'Game':<8}: {self.env_ids[self._env_idx]}",
            f"{'Ep/Step':<8}: {self._episode_count} / {self._step_count}",
            f"{'FPS inf':<8}: {inf_fps:6.1f}",
            f"{'FPS rnd':<8}: {self._render_fps:6.1f}",
            f"{'Queue':<8}: {self._frame_queue.qsize()}/{QUEUE_MAXSIZE}"
            f"  drops={self._dropped_total}",
            f"{'Hook ms':<8}: {self._last_hook_lat:.1f}",
            f"{'View':<8}: {self.VIS_MODES[self._vis_mode]}",
            "─" * 30,
        ] + self._log_lines[-10:]
        self._log_text.set_text("\n".join(stats))

    def _add_log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self._log_lines.append(f"[{ts}] {msg}")
        if len(self._log_lines) > 60:
            self._log_lines = self._log_lines[-60:]

    # ══════════════════════════════════════════════════════════ widget callbacks

    def _on_prev_agent(self, _: Any) -> None:
        assert threading.current_thread() is threading.main_thread()
        if len(self.agent_dirs) > 1:
            self._agent_idx = (self._agent_idx - 1) % len(self.agent_dirs)
            self._lbl_agent.set_text(f"Agent: {self.agent_dirs[self._agent_idx].name}")
            self._switch_agent(self.agent_dirs[self._agent_idx])

    def _on_next_agent(self, _: Any) -> None:
        assert threading.current_thread() is threading.main_thread()
        if len(self.agent_dirs) > 1:
            self._agent_idx = (self._agent_idx + 1) % len(self.agent_dirs)
            self._lbl_agent.set_text(f"Agent: {self.agent_dirs[self._agent_idx].name}")
            self._switch_agent(self.agent_dirs[self._agent_idx])

    def _on_prev_game(self, _: Any) -> None:
        assert threading.current_thread() is threading.main_thread()
        self._env_idx = (self._env_idx - 1) % len(self.env_ids)
        self._do_switch_game()

    def _on_next_game(self, _: Any) -> None:
        assert threading.current_thread() is threading.main_thread()
        self._env_idx = (self._env_idx + 1) % len(self.env_ids)
        self._do_switch_game()

    def _do_switch_game(self) -> None:
        assert threading.current_thread() is threading.main_thread()
        env_id = self.env_ids[self._env_idx]
        self._lbl_game.set_text(f"Game: {env_id}")
        self._add_log(f"Game → {env_id}")
        self._stop_inference()
        self._hooks.attach(self._agent.vae, self._agent.rnn)
        self._start_inference()

    def _on_toggle_loop(self, _: Any) -> None:
        assert threading.current_thread() is threading.main_thread()
        self.loop = not self.loop
        self._btn_loop.label.set_text(f"Loop: {'ON' if self.loop else 'OFF'}")
        if self._inference_thread:
            self._inference_thread.loop = self.loop

    def _on_restart(self, _: Any) -> None:
        assert threading.current_thread() is threading.main_thread()
        self._add_log("Restart")
        self._stop_inference()
        self._heatmap_history[:] = 0.0
        self._hooks.attach(self._agent.vae, self._agent.rnn)
        self._start_inference()

    def _on_toggle_pause(self, _: Any) -> None:
        assert threading.current_thread() is threading.main_thread()
        self._paused = not self._paused
        if self._paused:
            self._pause_event.set()
            self._btn_pause.label.set_text("Resume")
            self._add_log("Paused")
        else:
            self._pause_event.clear()
            self._btn_pause.label.set_text("Pause")
            self._add_log("Resumed")

    def _on_mode_changed(self, val: float) -> None:
        assert threading.current_thread() is threading.main_thread()
        self._vis_mode = int(round(val))
        lbl = self.VIS_MODES[self._vis_mode]
        self._lbl_mode.set_text(f"View: {lbl}")
        self._add_log(f"View → {lbl}")

    def _on_path_submit(self, text: str) -> None:
        assert threading.current_thread() is threading.main_thread()
        path = Path(text.strip())
        if not path.is_dir():
            self._add_log(f"WARN: not a directory: {text}")
            return
        found = discover_agents(path)
        if not found:
            self._add_log(f"WARN: no agents found in {path.name}")
            return
        self.agent_dirs = found
        self._agent_idx = 0
        self._add_log(f"Found {len(found)} agent(s) in {path.name}")
        self._lbl_agent.set_text(f"Agent: {found[0].name}")
        self._switch_agent(found[0])

    def _on_window_close(self, _: Any) -> None:
        assert threading.current_thread() is threading.main_thread()
        LOG.info("Window close — shutting down")
        self._shutdown()

    def _on_quit(self, _: Any) -> None:
        assert threading.current_thread() is threading.main_thread()
        LOG.info("Quit button clicked")
        plt.close("all")

    # ══════════════════════════════════════════════════════════════════ run

    def run(self) -> None:
        assert threading.current_thread() is threading.main_thread()
        if IN_COLAB:
            self._run_colab()
        else:
            self._run_interactive()

    def _run_interactive(self) -> None:
        self._anim = animation.FuncAnimation(
            self._fig, self._update_render,
            interval=50, cache_frame_data=False, blit=False,
        )
        try:
            plt.show(block=True)
        except KeyboardInterrupt:
            LOG.info("KeyboardInterrupt — shutting down")
        finally:
            self._shutdown()

    def _run_colab(self) -> None:
        from IPython import display as ipd  # type: ignore
        try:
            while not self._shutdown_event.is_set():
                self._update_render(0)
                self._fig.canvas.draw()
                ipd.clear_output(wait=True)
                ipd.display(self._fig)
                time.sleep(0.1)
        except KeyboardInterrupt:
            LOG.info("KeyboardInterrupt (Colab) — shutting down")
        finally:
            self._shutdown()

    def _shutdown(self) -> None:
        assert threading.current_thread() is threading.main_thread()
        LOG.info("Graceful shutdown initiated")
        self._stop_inference()
        LOG.info("Graceful shutdown completed")
        plt.close("all")


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="World Model Interpretability Visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python visualizer.py --agent_dir ../world-models-carracing/checkpoints/\n"
            "  python visualizer.py --agent_dir agents/ --games CarRacing-v2\n"
            "  python visualizer.py --agent_dir ckpts/ --no_loop --device cpu\n"
            "\n"
            "Environment variables:\n"
            "  WM_PROJECT_DIR  path to world-models-carracing source tree\n"
        ),
    )
    ap.add_argument(
        "--agent_dir", required=True,
        help=(
            "Directory with vae_final.pt + mdnrnn_final.pt + controller_best.npz, "
            "or parent of several such directories."
        ),
    )
    ap.add_argument(
        "--games", nargs="+", default=["CarRacing-v2"], metavar="ENV_ID",
        help="Gymnasium env IDs for the game selector (default: CarRacing-v2)",
    )
    ap.add_argument(
        "--no_loop", action="store_true",
        help="Stop after one episode rather than looping",
    )
    ap.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
        help="Torch device — 'auto' picks mps > cuda > cpu",
    )
    ap.add_argument(
        "--world_models_dir", default=None,
        help=(
            "Path to the world-models-carracing source tree. "
            "Defaults to $WM_PROJECT_DIR or ../world-models-carracing"
        ),
    )
    args = ap.parse_args()

    # Device
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    LOG.info(f"Device: {device}  |  world-models source: {_WM_DIR}")

    # Discover agents
    agent_path = Path(args.agent_dir)
    if not agent_path.is_dir():
        LOG.error(f"Not a directory: {agent_path}")
        sys.exit(1)

    agent_dirs = discover_agents(agent_path)
    if not agent_dirs:
        LOG.error(
            f"No valid agents found in '{agent_path}'.  "
            "Each agent directory must contain: "
            "vae_final.pt, mdnrnn_final.pt, controller_best.npz"
        )
        sys.exit(1)
    LOG.info(f"Discovered {len(agent_dirs)} agent(s): {[d.name for d in agent_dirs]}")

    app = VisualizerApp(
        agent_dirs=agent_dirs,
        env_ids=args.games,
        device=device,
        loop=not args.no_loop,
    )
    try:
        app.run()
    except Exception:
        LOG.error(f"Fatal:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
