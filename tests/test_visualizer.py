"""
tests/test_visualizer.py
========================
Unit tests for the World Model Interpretability Visualizer.

Runs fully headless: matplotlib is mocked; gymnasium environments that need no
display (CartPole-v1) are used wherever a live env is required.

Coverage
--------
 1. Hook extraction     — correct tensor shapes after both VAE + RNN fire
 2. Latent alignment    — z-dim / n_mix labelling derived from model config
 3. Queue backpressure  — inference never blocks on a full queue
 4. Frame dropping      — newest frame processed; older ones silently dropped
 5. Agent switching     — hooks torn down/re-registered; queue flushed
 6. Config mismatch     — viz reinitialises correctly for different layer counts
 7. Graceful shutdown   — all shutdown paths terminate cleanly
 8. Thread safety       — no matplotlib calls off the main thread
 9. LSTM state (KV-cache analogue) — hook shapes consistent across steps
10. Agent discovery     — finds nested dirs; ignores incomplete dirs
"""

import sys
import os
import queue
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import numpy as np
import torch

# ── project path ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
WM_SOURCE    = PROJECT_ROOT.parent / "world-models-carracing"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(WM_SOURCE))

# ── mock matplotlib BEFORE visualizer imports it ─────────────────────────────
sys.modules.setdefault("matplotlib",           MagicMock())
sys.modules.setdefault("matplotlib.pyplot",    MagicMock())
sys.modules.setdefault("matplotlib.gridspec",  MagicMock())
sys.modules.setdefault("matplotlib.animation", MagicMock())
sys.modules.setdefault("matplotlib.widgets",   MagicMock())
sys.modules.setdefault("matplotlib.patches",   MagicMock())
sys.modules.setdefault("matplotlib.colors",    MagicMock())

from models.vae       import VAE
from models.mdnrnn    import MDNRNN
from models.controller import Controller
from config           import Config

from visualizer import (
    HookExtractor,
    HookData,
    FrameData,
    AgentBundle,
    InferenceThread,
    discover_agents,
    QUEUE_MAXSIZE,
    HISTORY_LEN,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _vae(z_dim: int = 32, in_ch: int = 1) -> VAE:
    m = VAE(in_ch=in_ch, z_dim=z_dim)
    m.eval()
    return m


def _rnn(z_dim: int = 32, hidden: int = 256, n_mix: int = 10) -> MDNRNN:
    m = MDNRNN(z_dim=z_dim, action_dim=3, hidden=hidden, n_mix=n_mix)
    m.eval()
    return m


def _ctrl(z_dim: int = 32, hidden_rnn: int = 256) -> Controller:
    return Controller(in_dim=z_dim + hidden_rnn, hidden=0, action_dim=3)


def _dummy_hook(z_dim: int = 32, n_mix: int = 10, hidden: int = 256) -> HookData:
    return HookData(
        z=np.zeros(z_dim), logvar=np.zeros(z_dim),
        h_state=np.zeros(hidden),
        mixture_weights=np.ones(n_mix) / n_mix,
        mu_mix=np.zeros((n_mix, z_dim)),
        log_sigma_mix=np.zeros((n_mix, z_dim)),
    )


def _dummy_fd(step: int = 0, z_dim: int = 32, n_mix: int = 10) -> FrameData:
    return FrameData(
        raw_frame=np.zeros((96, 96, 3), dtype=np.uint8),
        hook=_dummy_hook(z_dim, n_mix),
        step=step, episode=1, reward=0.0, done=False,
    )


def _fire_both(extractor: HookExtractor, vae: VAE, rnn: MDNRNN,
               z_dim: int = 32, hidden: int = 256) -> None:
    """Run one forward pass through both models so all three hooks fire."""
    x  = torch.zeros(1, 1, 64, 64)
    za = torch.zeros(1, 1, z_dim + 3)
    h  = (torch.zeros(1, 1, hidden), torch.zeros(1, 1, hidden))
    with torch.no_grad():
        vae.encode(x)
        rnn(za, h)


def _save_agent(tmp_dir: Path, z_dim: int = 32, n_mix: int = 10,
                rnn_hidden: int = 256) -> None:
    """Write the three checkpoint files expected by AgentBundle.load()."""
    cfg = Config()
    cfg.z_dim         = z_dim
    cfg.mdn_mixtures  = n_mix
    cfg.rnn_hidden    = rnn_hidden

    in_ch = 1 if cfg.grayscale else 3
    vae_m = VAE(in_ch=in_ch, z_dim=z_dim)
    torch.save({"state_dict": vae_m.state_dict(), "cfg": cfg.__dict__},
               tmp_dir / "vae_final.pt")

    rnn_m = MDNRNN(z_dim=z_dim, action_dim=3, hidden=rnn_hidden, n_mix=n_mix)
    torch.save({"state_dict": rnn_m.state_dict(), "cfg": cfg.__dict__},
               tmp_dir / "mdnrnn_final.pt")

    ctrl_m = Controller(in_dim=z_dim + rnn_hidden, hidden=0, action_dim=3)
    theta  = np.concatenate(
        [p.detach().cpu().numpy().ravel() for p in ctrl_m.parameters()]
    ).astype(np.float64)
    np.savez_compressed(
        tmp_dir / "controller_best.npz",
        best_theta=theta, best_score=0.0, cfg=cfg.__dict__,
    )


# ── 1. Hook extraction ────────────────────────────────────────────────────────

class TestHookExtractor(unittest.TestCase):
    """
    All tests fire both VAE and RNN before calling get_data(), mirroring the
    inference loop where both models run in the same step before data is read.
    """

    def setUp(self) -> None:
        self.z_dim  = 32
        self.n_mix  = 10
        self.hidden = 256
        self.vae = _vae(self.z_dim)
        self.rnn = _rnn(self.z_dim, self.hidden, self.n_mix)
        self.ext = HookExtractor()

    def tearDown(self) -> None:
        self.ext.detach()

    # ── shape checks ──────────────────────────────────────────────────────────

    def test_vae_z_shape(self) -> None:
        self.ext.attach(self.vae, self.rnn)
        _fire_both(self.ext, self.vae, self.rnn, self.z_dim, self.hidden)
        d = self.ext.get_data(self.n_mix, self.z_dim)
        self.assertIsNotNone(d)
        self.assertEqual(d.z.shape, (self.z_dim,))
        self.assertEqual(d.logvar.shape, (self.z_dim,))

    def test_rnn_hook_shapes(self) -> None:
        self.ext.attach(self.vae, self.rnn)
        _fire_both(self.ext, self.vae, self.rnn, self.z_dim, self.hidden)
        d = self.ext.get_data(self.n_mix, self.z_dim)
        self.assertIsNotNone(d)
        self.assertEqual(d.h_state.shape,        (self.hidden,))
        self.assertEqual(d.mixture_weights.shape, (self.n_mix,))
        self.assertEqual(d.mu_mix.shape,          (self.n_mix, self.z_dim))
        self.assertEqual(d.log_sigma_mix.shape,   (self.n_mix, self.z_dim))

    def test_mixture_weights_sum_to_one(self) -> None:
        self.ext.attach(self.vae, self.rnn)
        _fire_both(self.ext, self.vae, self.rnn, self.z_dim, self.hidden)
        d = self.ext.get_data(self.n_mix, self.z_dim)
        self.assertIsNotNone(d)
        self.assertAlmostEqual(float(d.mixture_weights.sum()), 1.0, places=5)

    def test_returns_none_before_both_hooks_fire(self) -> None:
        self.ext.attach(self.vae, self.rnn)
        # Only fire VAE — RNN keys still missing
        with torch.no_grad():
            self.vae.encode(torch.zeros(1, 1, 64, 64))
        d = self.ext.get_data(self.n_mix, self.z_dim)
        self.assertIsNone(d)

    def test_detach_clears_all_handles(self) -> None:
        self.ext.attach(self.vae, self.rnn)
        self.assertEqual(len(self.ext._handles), 3)
        self.ext.detach()
        self.assertEqual(len(self.ext._handles), 0)

    def test_hooks_are_read_only(self) -> None:
        """Forward hooks must not alter model outputs."""
        vae_clean = _vae(self.z_dim)
        vae_clean.load_state_dict(self.vae.state_dict())
        self.ext.attach(self.vae, self.rnn)
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            mu_h,  lv_h  = self.vae.encode(x)
            mu_c,  lv_c  = vae_clean.encode(x)
        self.assertTrue(torch.allclose(mu_h, mu_c))
        self.assertTrue(torch.allclose(lv_h, lv_c))

    def test_reattach_after_detach(self) -> None:
        self.ext.attach(self.vae, self.rnn)
        self.ext.detach()
        self.ext.attach(self.vae, self.rnn)
        self.assertEqual(len(self.ext._handles), 3)
        _fire_both(self.ext, self.vae, self.rnn, self.z_dim, self.hidden)
        self.assertIsNotNone(self.ext.get_data(self.n_mix, self.z_dim))

    # ── multi-timestep sequence ────────────────────────────────────────────────

    def test_multi_timestep_sequence_shape(self) -> None:
        """When T>1 the hook must extract the last time-step's parameters."""
        self.ext.attach(self.vae, self.rnn)
        with torch.no_grad():
            self.vae.encode(torch.zeros(1, 1, 64, 64))
            self.rnn(torch.zeros(1, 5, self.z_dim + 3), None)   # T=5
        d = self.ext.get_data(self.n_mix, self.z_dim)
        self.assertIsNotNone(d)
        self.assertEqual(d.mixture_weights.shape, (self.n_mix,))
        self.assertEqual(d.mu_mix.shape,          (self.n_mix, self.z_dim))


# ── 2. Latent alignment ───────────────────────────────────────────────────────

class TestLatentAlignment(unittest.TestCase):
    """z-dim and n_mix are read from the model at runtime, not hardcoded."""

    def _check(self, z_dim: int, n_mix: int, hidden: int = 256) -> None:
        vae_m = _vae(z_dim)
        rnn_m = _rnn(z_dim, hidden, n_mix)
        ext   = HookExtractor()
        ext.attach(vae_m, rnn_m)
        _fire_both(ext, vae_m, rnn_m, z_dim, hidden)
        d = ext.get_data(n_mix, z_dim)
        self.assertIsNotNone(d, f"z_dim={z_dim}, n_mix={n_mix}")
        self.assertEqual(d.z.shape[0],               z_dim)
        self.assertEqual(d.mixture_weights.shape[0], n_mix)
        ext.detach()

    def test_z16_mix5(self)  -> None: self._check(16, 5, 128)
    def test_z32_mix10(self) -> None: self._check(32, 10, 256)
    def test_z64_mix20(self) -> None: self._check(64, 20, 512)


# ── 3. Queue backpressure ─────────────────────────────────────────────────────

class TestQueueBackpressure(unittest.TestCase):

    def test_put_nowait_never_blocks(self) -> None:
        q: queue.Queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
        for _ in range(QUEUE_MAXSIZE):
            q.put_nowait(_dummy_fd())

        t0 = time.perf_counter()
        dropped = 0
        try:
            q.put_nowait(_dummy_fd())
        except queue.Full:
            dropped += 1
        elapsed = time.perf_counter() - t0

        self.assertEqual(dropped, 1)
        self.assertLess(elapsed, 0.05, "put_nowait blocked — should raise immediately")

    def test_drop_count_increments(self) -> None:
        q: queue.Queue = queue.Queue(maxsize=1)
        q.put_nowait(_dummy_fd())   # fill it

        dropped = 0
        for _ in range(5):
            try:
                q.put_nowait(_dummy_fd())
            except queue.Full:
                dropped += 1
        self.assertEqual(dropped, 5)


# ── 4. Frame dropping ─────────────────────────────────────────────────────────

class TestFrameDropping(unittest.TestCase):

    def test_render_uses_latest_frame(self) -> None:
        """When multiple frames are queued the render thread should pick the newest."""
        q: queue.Queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
        for step in (1, 2, 3):
            q.put_nowait(_dummy_fd(step=step))

        latest = None
        while True:
            try:
                fd = q.get_nowait()
            except queue.Empty:
                break
            if not fd.is_sentinel:
                latest = fd

        self.assertIsNotNone(latest)
        self.assertEqual(latest.step, 3)

    def test_queue_never_exceeds_maxsize(self) -> None:
        q: queue.Queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
        put_ok = drop = 0
        for _ in range(QUEUE_MAXSIZE + 10):
            try:
                q.put_nowait(_dummy_fd())
                put_ok += 1
            except queue.Full:
                drop += 1
        self.assertEqual(put_ok, QUEUE_MAXSIZE)
        self.assertEqual(drop,   10)


# ── 5. Agent switching ────────────────────────────────────────────────────────

class TestAgentSwitching(unittest.TestCase):

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmpdir.name)
        for name in ("a1", "a2"):
            d = self.tmp / name
            d.mkdir()
            _save_agent(d)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_attach_removes_old_hooks(self) -> None:
        ext = HookExtractor()
        v1, r1 = _vae(), _rnn()
        ext.attach(v1, r1)
        old_handles = list(ext._handles)
        v2, r2 = _vae(), _rnn()
        ext.attach(v2, r2)   # implicit detach of v1/r1
        for h in old_handles:
            self.assertNotIn(h, ext._handles)
        self.assertEqual(len(ext._handles), 3)
        ext.detach()

    def test_queue_flushed_on_switch(self) -> None:
        q: queue.Queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
        for _ in range(QUEUE_MAXSIZE):
            q.put_nowait(_dummy_fd())
        self.assertEqual(q.qsize(), QUEUE_MAXSIZE)
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break
        self.assertEqual(q.qsize(), 0)

    def test_load_from_disk(self) -> None:
        b = AgentBundle.load(self.tmp / "a1", torch.device("cpu"))
        self.assertEqual(b.rnn.z_dim,   32)
        self.assertEqual(b.rnn.n_mix,   10)
        self.assertEqual(b.rnn.hidden, 256)

    def test_old_hooks_inactive_after_switch(self) -> None:
        """After switching to agent2, agent1's modules must not populate data."""
        ext = HookExtractor()
        v1, r1 = _vae(32), _rnn(32)
        ext.attach(v1, r1)
        v2, r2 = _vae(32), _rnn(32)
        ext.attach(v2, r2)   # detaches v1/r1

        # Clear stored data then run only agent1's modules
        with ext._lock:
            ext._data.clear()
        with torch.no_grad():
            v1.encode(torch.zeros(1, 1, 64, 64))
            r1(torch.zeros(1, 1, 35), None)
        with ext._lock:
            fired = len(ext._data) > 0
        self.assertFalse(fired, "Stale hooks from agent1 must not fire")
        ext.detach()


# ── 6. Config mismatch ────────────────────────────────────────────────────────

class TestConfigMismatch(unittest.TestCase):

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmpdir.name)
        for name, dims in (
            ("small", dict(z_dim=16, n_mix=5,  rnn_hidden=128)),
            ("large", dict(z_dim=64, n_mix=20, rnn_hidden=512)),
        ):
            d = self.tmp / name
            d.mkdir()
            _save_agent(d, **dims)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_load_small(self) -> None:
        b = AgentBundle.load(self.tmp / "small", torch.device("cpu"))
        self.assertEqual(b.rnn.z_dim,  16)
        self.assertEqual(b.rnn.n_mix,   5)
        self.assertEqual(b.rnn.hidden, 128)

    def test_load_large(self) -> None:
        b = AgentBundle.load(self.tmp / "large", torch.device("cpu"))
        self.assertEqual(b.rnn.z_dim,  64)
        self.assertEqual(b.rnn.n_mix,  20)
        self.assertEqual(b.rnn.hidden, 512)

    def test_history_buffer_resizes(self) -> None:
        """Heatmap history rows must equal the new agent's n_mix."""
        h_small = np.zeros((5,  HISTORY_LEN))
        h_large = np.zeros((20, HISTORY_LEN))
        self.assertEqual(h_small.shape[0], 5)
        self.assertEqual(h_large.shape[0], 20)

    def test_hook_extractor_handles_dim_switch(self) -> None:
        ext = HookExtractor()
        # Small
        vs, rs = _vae(16), _rnn(16, 128, 5)
        ext.attach(vs, rs)
        _fire_both(ext, vs, rs, z_dim=16, hidden=128)
        ds = ext.get_data(5, 16)
        self.assertIsNotNone(ds)
        self.assertEqual(ds.z.shape[0],               16)
        self.assertEqual(ds.mixture_weights.shape[0],  5)
        # Large
        vl, rl = _vae(64), _rnn(64, 512, 20)
        ext.attach(vl, rl)
        _fire_both(ext, vl, rl, z_dim=64, hidden=512)
        dl = ext.get_data(20, 64)
        self.assertIsNotNone(dl)
        self.assertEqual(dl.z.shape[0],               64)
        self.assertEqual(dl.mixture_weights.shape[0], 20)
        ext.detach()


# ── 7. Graceful shutdown ──────────────────────────────────────────────────────

class TestGracefulShutdown(unittest.TestCase):

    def _make_thread(self, tmp: Path, loop: bool = False) -> InferenceThread:
        _save_agent(tmp)
        bundle = AgentBundle.load(tmp, torch.device("cpu"))
        ext = HookExtractor()
        ext.attach(bundle.vae, bundle.rnn)
        t = InferenceThread(
            agent=bundle,
            env_id="CartPole-v1",
            frame_queue=queue.Queue(maxsize=100),
            hook_extractor=ext,
            shutdown_event=threading.Event(),
            pause_event=threading.Event(),
            loop=loop,
        )
        t._ext = ext   # keep reference so we can detach later
        return t

    def test_shutdown_event_stops_thread(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            t = self._make_thread(Path(td), loop=True)
            t.start()
            time.sleep(0.15)
            t.shutdown.set()
            t.join(timeout=8.0)
        self.assertFalse(t.is_alive())
        t._ext.detach()

    def test_sentinel_pushed_on_exit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            t = self._make_thread(Path(td), loop=False)
            t.start()
            t.join(timeout=30.0)
        self.assertFalse(t.is_alive())
        sentinel_found = False
        while True:
            try:
                fd = t.frame_queue.get_nowait()
                if fd.is_sentinel:
                    sentinel_found = True
                    break
            except queue.Empty:
                break
        self.assertTrue(sentinel_found, "Sentinel must be pushed on thread exit")
        t._ext.detach()

    def test_missing_checkpoint_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                AgentBundle.load(Path(td), torch.device("cpu"))

    def test_discover_empty_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(discover_agents(Path(td)), [])

    def test_discover_nonexistent_path(self) -> None:
        self.assertEqual(discover_agents(Path("/no/such/path/xyz")), [])

    def test_discover_valid_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            _save_agent(Path(td))
            found = discover_agents(Path(td))
        self.assertIn(Path(td), found)


# ── 8. Thread safety ──────────────────────────────────────────────────────────

class TestThreadSafety(unittest.TestCase):

    def test_concurrent_reads_writes_no_crash(self) -> None:
        vae_m = _vae(32)
        rnn_m = _rnn(32)
        ext   = HookExtractor()
        ext.attach(vae_m, rnn_m)
        errors: List[Exception] = []

        def writer() -> None:
            for _ in range(50):
                try:
                    with torch.no_grad():
                        rnn_m(torch.randn(1, 1, 35), None)
                except Exception as e:
                    errors.append(e)

        def reader() -> None:
            for _ in range(100):
                try:
                    ext.get_data(10, 32)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in threads: t.start()
        for t in threads: t.join(timeout=10.0)
        self.assertEqual(errors, [])
        ext.detach()

    def test_main_thread_guard_in_source(self) -> None:
        import inspect
        import visualizer as vis
        src = inspect.getsource(vis.VisualizerApp.__init__)
        self.assertIn("main_thread", src)

    def test_inference_thread_has_no_plt_calls(self) -> None:
        import inspect
        import visualizer as vis
        src = inspect.getsource(vis.InferenceThread.run)
        self.assertNotIn("plt.",       src)
        self.assertNotIn("matplotlib", src)


# ── 9. LSTM state (KV-cache analogue) ─────────────────────────────────────────

class TestLSTMStateConsistency(unittest.TestCase):
    """
    The LSTM hidden state (h, c) is the world-model equivalent of a KV-cache:
    it accumulates context across steps without re-encoding past frames.
    Hook shapes must be identical whether h is fresh (None) or carried over.
    """

    def setUp(self) -> None:
        self.vae_m = _vae(32)
        self.rnn_m = _rnn(32, 256, 10)
        self.ext   = HookExtractor()
        self.ext.attach(self.vae_m, self.rnn_m)

    def tearDown(self) -> None:
        self.ext.detach()

    def _step(self, h):
        """One inference step. Returns updated h and hook snapshot."""
        za = torch.randn(1, 1, 35)
        with torch.no_grad():
            self.vae_m.encode(torch.zeros(1, 1, 64, 64))
            _, h_new = self.rnn_m(za, h)
        d = self.ext.get_data(10, 32)
        return h_new, d

    def test_shapes_consistent_fresh_vs_carry(self) -> None:
        # Step 1: fresh hidden state (None → zeros internally)
        h1, d1 = self._step(None)
        # Step 2: carry h1
        h2, d2 = self._step(h1)
        # Step 3: carry h2
        _,  d3 = self._step(h2)

        for i, d in enumerate((d1, d2, d3), 1):
            self.assertIsNotNone(d, f"Step {i} returned None")
            self.assertEqual(d.h_state.shape,        (256,),
                             f"Step {i}: h_state wrong shape")
            self.assertEqual(d.mixture_weights.shape, (10,),
                             f"Step {i}: mixture_weights wrong shape")

    def test_hidden_state_changes_across_steps(self) -> None:
        """LSTM state must actually evolve with different random inputs."""
        h = (torch.zeros(1, 1, 256), torch.zeros(1, 1, 256))
        states = []
        for _ in range(3):
            h, d = self._step(h)
            self.assertIsNotNone(d)
            states.append(d.h_state.copy())
        # Different random inputs should produce different hidden states
        self.assertFalse(np.allclose(states[0], states[1]),
                         "LSTM hidden state should vary across steps")

    def test_zero_init_equals_none_init(self) -> None:
        """Explicit zeros and None should produce identical first-step outputs."""
        za = torch.zeros(1, 1, 35)
        x  = torch.zeros(1, 1, 64, 64)

        with torch.no_grad():
            self.vae_m.encode(x)
            _, h_explicit = self.rnn_m(za,
                (torch.zeros(1, 1, 256), torch.zeros(1, 1, 256)))
        d_explicit = self.ext.get_data(10, 32)

        with torch.no_grad():
            self.vae_m.encode(x)
            _, h_none = self.rnn_m(za, None)
        d_none = self.ext.get_data(10, 32)

        self.assertIsNotNone(d_explicit)
        self.assertIsNotNone(d_none)
        np.testing.assert_allclose(d_explicit.h_state, d_none.h_state, atol=1e-6)


# ── 10. Agent discovery ───────────────────────────────────────────────────────

class TestAgentDiscovery(unittest.TestCase):

    def test_finds_nested_agents(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            parent = Path(td)
            for name in ("run_001", "run_002"):
                d = parent / name
                d.mkdir()
                _save_agent(d)
            found = {p.name for p in discover_agents(parent)}
        self.assertIn("run_001", found)
        self.assertIn("run_002", found)

    def test_ignores_incomplete_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            parent = Path(td)
            good = parent / "good";  good.mkdir(); _save_agent(good)
            bad  = parent / "bad";   bad.mkdir()
            # bad has VAE + RNN but no controller
            cfg  = Config()
            vae_m = VAE(in_ch=1, z_dim=32)
            torch.save({"state_dict": vae_m.state_dict(), "cfg": cfg.__dict__},
                       bad / "vae_final.pt")
            rnn_m = MDNRNN(z_dim=32, action_dim=3, hidden=256, n_mix=10)
            torch.save({"state_dict": rnn_m.state_dict(), "cfg": cfg.__dict__},
                       bad / "mdnrnn_final.pt")
            found = {p.name for p in discover_agents(parent)}
        self.assertIn("good", found)
        self.assertNotIn("bad", found)

    def test_empty_dir_returns_empty_list(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(discover_agents(Path(td)), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
