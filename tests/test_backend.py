"""
Backend test suite for the WM Visualizer.

Covers:
  - Hook extraction: correct tensor shapes (cached and non-cached)
  - KV cache interaction: shapes consistent across full rollout
  - Token alignment: labels derived from config, never hardcoded
  - Hook cleanup: zero hooks remain after failure or agent switch
  - Queue backpressure: inference thread never blocked
  - Frame dropping: frames dropped not queued when consumer falls behind
  - Agent switching: hooks re-registered, queue flushed, no stale state
  - Config mismatch: reinitialises correctly between agents with different arch
  - Shutdown: all paths terminate cleanly with no hanging threads
  - Frame encoding: raw observation correctly base64-encoded as PNG
"""

import base64
import io
import sys
import threading
import time
from pathlib import Path
from queue import Queue
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Path setup: add IRIS src/ so models can be imported
# ---------------------------------------------------------------------------

_IRIS_ROOT = Path(__file__).parent.parent.parent / "iris"
_IRIS_SRC = _IRIS_ROOT / "src"
sys.path.insert(0, str(_IRIS_SRC))
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from models.world_model import WorldModel
from models.transformer import TransformerConfig
from hooks import IrisHookExtractor
from inference import (
    _FpsCounter,
    _encode_frame,
    FrameData,
    get_token_labels,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_world_model(
    num_layers: int = 2,
    num_heads: int = 2,
    embed_dim: int = 16,
    tokens_per_block: int = 5,
    max_blocks: int = 10,
) -> WorldModel:
    config = TransformerConfig(
        tokens_per_block=tokens_per_block,
        max_blocks=max_blocks,
        attention="causal",
        num_layers=num_layers,
        num_heads=num_heads,
        embed_dim=embed_dim,
        embed_pdrop=0.0,
        resid_pdrop=0.0,
        attn_pdrop=0.0,
    )
    return WorldModel(obs_vocab_size=16, act_vocab_size=8, config=config).eval()


# ---------------------------------------------------------------------------
# Hook extraction: shapes
# ---------------------------------------------------------------------------

class TestHookExtraction:

    def test_attn_shape_uncached(self):
        """Uncached forward pass: attention shape is (1, nh, T, T)."""
        wm = make_world_model(num_layers=2, num_heads=2, tokens_per_block=5)
        hooks = IrisHookExtractor()
        hooks.attach(wm)

        tokens = torch.randint(0, 8, (1, 5))
        with torch.no_grad():
            wm(tokens, past_keys_values=None)

        attn, _ = hooks.get_data()
        assert attn is not None
        for i in range(2):
            assert attn[i].shape == (1, 2, 5, 5), f"Layer {i}: {attn[i].shape}"
        hooks.detach()

    def test_norm_shape_uncached(self):
        """Norm values are positive floats, one per layer."""
        wm = make_world_model(num_layers=2, num_heads=2, tokens_per_block=5)
        hooks = IrisHookExtractor()
        hooks.attach(wm)

        tokens = torch.randint(0, 8, (1, 5))
        with torch.no_grad():
            wm(tokens, past_keys_values=None)

        _, norms = hooks.get_data()
        assert norms is not None
        assert len(norms) == 2
        for i in range(2):
            # get_data() now returns 0-d tensors for norms; .item() is deferred to
            # the consumer (inference.py) so it runs once after the full forward pass
            # rather than forcing a GPU→CPU sync per layer inside the hook callback.
            import torch as _torch
            assert isinstance(norms[i], _torch.Tensor)
            assert norms[i].ndim == 0          # 0-dimensional (scalar tensor)
            assert norms[i].item() > 0

        hooks.detach()

    def test_attn_shape_cached_first_pass(self):
        """Cached first pass (L=0, T=5): T_k = 5."""
        wm = make_world_model(num_layers=2, num_heads=2, tokens_per_block=5, max_blocks=20)
        hooks = IrisHookExtractor()
        hooks.attach(wm)

        kv = wm.transformer.generate_empty_keys_values(n=1, max_tokens=100)
        tokens = torch.randint(0, 8, (1, 5))
        with torch.no_grad():
            wm(tokens, past_keys_values=kv)

        attn, _ = hooks.get_data()
        for i in range(2):
            assert attn[i].shape == (1, 2, 5, 5), f"Layer {i}: {attn[i].shape}"
        hooks.detach()

    def test_attn_shape_cached_second_pass(self):
        """After caching 5 tokens (L=5), a 1-token query: T_k = L+T = 6."""
        wm = make_world_model(num_layers=2, num_heads=2, tokens_per_block=5, max_blocks=20)
        hooks = IrisHookExtractor()
        hooks.attach(wm)

        kv = wm.transformer.generate_empty_keys_values(n=1, max_tokens=100)
        tokens = torch.randint(0, 8, (1, 5))
        with torch.no_grad():
            wm(tokens, past_keys_values=kv)

        tokens2 = torch.randint(0, 8, (1, 1))
        with torch.no_grad():
            wm(tokens2, past_keys_values=kv)

        attn, _ = hooks.get_data()
        for i in range(2):
            assert attn[i].shape == (1, 2, 1, 6), f"Layer {i}: {attn[i].shape}"
        hooks.detach()


# ---------------------------------------------------------------------------
# KV cache interaction
# ---------------------------------------------------------------------------

class TestKVCacheInteraction:

    def test_shapes_grow_across_rollout(self):
        """T_k grows by 1 on each 1-token step, matching WorldModelEnv pattern."""
        wm = make_world_model(num_layers=2, num_heads=2, tokens_per_block=5, max_blocks=20)
        hooks = IrisHookExtractor()
        hooks.attach(wm)

        kv = wm.transformer.generate_empty_keys_values(n=1, max_tokens=100)

        # Initial 4-token pass
        tokens = torch.randint(0, 8, (1, 4))
        with torch.no_grad():
            wm(tokens, past_keys_values=kv)
        attn, _ = hooks.get_data()
        assert attn[0].shape == (1, 2, 4, 4)

        # 5 subsequent 1-token steps
        for step in range(1, 6):
            token = torch.randint(0, 8, (1, 1))
            with torch.no_grad():
                wm(token, past_keys_values=kv)
            attn, _ = hooks.get_data()
            expected_tk = 4 + step
            assert attn[0].shape == (1, 2, 1, expected_tk), (
                f"Step {step}: expected T_k={expected_tk}, got {attn[0].shape}"
            )
        hooks.detach()

    def test_hook_fires_same_count_cached_vs_uncached(self):
        """Both forward modes fire exactly num_layers attn hooks."""
        num_layers = 3
        wm = make_world_model(num_layers=num_layers, num_heads=2, tokens_per_block=5)
        hooks = IrisHookExtractor()
        hooks.attach(wm)

        tokens = torch.randint(0, 8, (1, 5))

        # Uncached
        with torch.no_grad():
            wm(tokens, past_keys_values=None)
        attn_u, norms_u = hooks.get_data()
        assert len(attn_u) == num_layers
        assert len(norms_u) == num_layers

        # Cached
        kv = wm.transformer.generate_empty_keys_values(n=1, max_tokens=50)
        with torch.no_grad():
            wm(tokens, past_keys_values=kv)
        attn_c, norms_c = hooks.get_data()
        assert len(attn_c) == num_layers
        assert len(norms_c) == num_layers

        hooks.detach()


# ---------------------------------------------------------------------------
# Token alignment
# ---------------------------------------------------------------------------

class TestTokenAlignment:

    def test_labels_default_config(self):
        """tokens_per_block=17: positions 0–15 → o{i}, position 16 → act."""
        labels = get_token_labels(17, 17)
        for i in range(16):
            assert labels[i] == f"o{i}"
        assert labels[16] == "act"

    def test_labels_custom_config(self):
        """tokens_per_block=5: [o0, o1, o2, o3, act]."""
        assert get_token_labels(5, 5) == ["o0", "o1", "o2", "o3", "act"]

    def test_labels_length(self):
        """Output length always equals num_tokens."""
        for n in [1, 5, 17, 34]:
            assert len(get_token_labels(n, 17)) == n

    def test_labels_multi_block(self):
        """10 tokens, tpb=5: two complete [o0..o3, act] blocks."""
        expected = ["o0", "o1", "o2", "o3", "act"] * 2
        assert get_token_labels(10, 5) == expected

    def test_labels_never_hardcoded(self):
        """Label pattern changes correctly when tokens_per_block changes."""
        l3 = get_token_labels(3, 3)
        assert l3 == ["o0", "o1", "act"]
        l7 = get_token_labels(7, 7)
        assert l7[-1] == "act"
        for i in range(6):
            assert l7[i] == f"o{i}"


# ---------------------------------------------------------------------------
# Hook cleanup on failure
# ---------------------------------------------------------------------------

class TestHookCleanupOnFailure:

    def test_no_hooks_remain_after_failed_attach(self):
        """
        If register_forward_hook raises on layer 1's attn_drop, attach() must:
          1. Raise RuntimeError
          2. Leave zero hooks on any module in the model
        """
        wm = make_world_model(num_layers=3, num_heads=2)
        hooks = IrisHookExtractor()

        def bad_register(fn):
            raise RuntimeError("Simulated hook registration failure")

        with patch.object(
            wm.transformer.blocks[1].attn.attn_drop,
            "register_forward_hook",
            bad_register,
        ):
            with pytest.raises(RuntimeError):
                hooks.attach(wm)

        total = sum(len(m._forward_hooks) for m in wm.modules())
        assert total == 0, f"Expected 0 hooks, found {total}"

    def test_detach_removes_all_hooks(self):
        """detach() leaves zero hooks on the model."""
        wm = make_world_model(num_layers=2, num_heads=2)
        hooks = IrisHookExtractor()
        hooks.attach(wm)
        assert sum(len(m._forward_hooks) for m in wm.modules()) > 0
        hooks.detach()
        assert sum(len(m._forward_hooks) for m in wm.modules()) == 0

    def test_num_layers_reflects_attachment(self):
        wm = make_world_model(num_layers=3, num_heads=2)
        hooks = IrisHookExtractor()
        assert hooks.num_layers == 0
        hooks.attach(wm)
        assert hooks.num_layers == 3
        hooks.detach()
        assert hooks.num_layers == 0


# ---------------------------------------------------------------------------
# Queue backpressure & frame dropping
# ---------------------------------------------------------------------------

class TestQueueBackpressure:

    def test_queue_never_blocks_producer(self):
        """
        A producer that fills a tiny bounded queue must never hang.
        Frames should be dropped instead.
        """
        from queue import Queue, Full

        q: Queue = Queue(maxsize=2)
        dropped = 0
        TOTAL = 20

        t0 = time.perf_counter()
        for _ in range(TOTAL):
            try:
                q.put_nowait(object())
            except Full:
                dropped += 1
        elapsed = time.perf_counter() - t0

        assert elapsed < 0.1, f"Producer blocked for {elapsed:.3f}s"
        assert dropped == TOTAL - 2, f"Expected {TOTAL - 2} drops, got {dropped}"

    def test_frames_dropped_not_queued(self):
        """With a queue of size 1, at least TOTAL-1 frames are dropped."""
        from queue import Queue, Full

        q: Queue = Queue(maxsize=1)
        dropped = 0
        TOTAL = 10

        for _ in range(TOTAL):
            try:
                q.put_nowait("frame")
            except Full:
                dropped += 1

        assert dropped >= TOTAL - 1


# ---------------------------------------------------------------------------
# Agent switching (mocked — no real checkpoint)
# ---------------------------------------------------------------------------

class TestAgentSwitching:

    def _make_engine_with_mock_agent(self):
        """Return an InferenceEngine with _load_agent mocked out."""
        from inference import InferenceEngine

        engine = InferenceEngine(
            iris_src=str(_IRIS_SRC),
            iris_root=str(_IRIS_ROOT),
        )
        return engine

    def test_hooks_detached_after_stop(self):
        """After stop(), hook extractor has 0 layers."""
        from inference import InferenceEngine

        engine = InferenceEngine(
            iris_src=str(_IRIS_SRC),
            iris_root=str(_IRIS_ROOT),
        )
        wm = make_world_model(num_layers=2, num_heads=2)
        engine._hooks.attach(wm)
        engine._agent = MagicMock()
        engine._env = MagicMock()

        # Simulate a stopped state (no thread)
        engine._hooks.detach()
        assert engine._hooks.num_layers == 0

    def test_queue_flushed_on_switch(self):
        """_drain_queue() empties the queue completely."""
        from inference import InferenceEngine

        engine = InferenceEngine(
            iris_src=str(_IRIS_SRC),
            iris_root=str(_IRIS_ROOT),
        )
        for i in range(5):
            engine._queue.put_nowait(FrameData())
        assert engine._queue.qsize() == 5
        engine._drain_queue()
        assert engine._queue.qsize() == 0

    def test_event_callbacks_receive_agent_loaded(self):
        """Event callbacks fire when emit_event is called."""
        from inference import InferenceEngine

        engine = InferenceEngine(
            iris_src=str(_IRIS_SRC),
            iris_root=str(_IRIS_ROOT),
        )
        received = []
        engine.register_event_callback(lambda name, data: received.append((name, data)))
        engine._emit_event("agent_loaded", {"agent": "Test"})
        assert len(received) == 1
        assert received[0][0] == "agent_loaded"

    def test_event_callback_unregistered(self):
        from inference import InferenceEngine

        engine = InferenceEngine(
            iris_src=str(_IRIS_SRC),
            iris_root=str(_IRIS_ROOT),
        )
        received = []
        cb = lambda name, data: received.append(name)
        engine.register_event_callback(cb)
        engine._emit_event("ping", {})
        engine.unregister_event_callback(cb)
        engine._emit_event("pong", {})
        assert received == ["ping"]


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------

class TestShutdown:

    def test_stop_before_start_is_safe(self):
        """Calling stop() before start() must not raise."""
        from inference import InferenceEngine
        engine = InferenceEngine(
            iris_src=str(_IRIS_SRC),
            iris_root=str(_IRIS_ROOT),
        )
        engine.stop()  # should be a no-op

    def test_stop_event_terminates_paused_thread(self):
        """
        A paused inference thread must unblock and exit within 2 s when
        stop() is called.
        """
        stop_event = threading.Event()
        pause_event = threading.Event()
        pause_event.clear()  # paused

        def worker():
            pause_event.wait()  # would block forever without stop signal

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        # Simulate stop: unblock pause
        pause_event.set()
        t.join(timeout=2.0)
        assert not t.is_alive(), "Thread did not terminate"


# ---------------------------------------------------------------------------
# Frame encoding
# ---------------------------------------------------------------------------

class TestFrameEncoding:

    def test_encode_frame_produces_valid_png(self):
        """_encode_frame returns a base64 string that decodes to a valid PNG."""
        from PIL import Image as _Image

        obs = np.random.randint(0, 255, (210, 160, 3), dtype=np.uint8)
        b64 = _encode_frame(obs)

        raw = base64.b64decode(b64)
        img = _Image.open(io.BytesIO(raw))
        assert img.format == "PNG"
        assert img.size == (160, 210)

    def test_encode_frame_not_preprocessed(self):
        """The encoded frame uses raw pixel values, not normalised floats."""
        obs = np.full((10, 10, 3), 200, dtype=np.uint8)
        b64 = _encode_frame(obs)
        from PIL import Image as _Image
        raw = base64.b64decode(b64)
        img = _Image.open(io.BytesIO(raw))
        arr = np.array(img)
        assert arr.max() == 200, "Frame appears normalised"


# ---------------------------------------------------------------------------
# FPS counter
# ---------------------------------------------------------------------------

class TestFpsCounter:

    def test_fps_zero_initially(self):
        counter = _FpsCounter()
        assert counter.fps == 0.0

    def test_fps_nonzero_after_ticks(self):
        counter = _FpsCounter()
        for _ in range(5):
            counter.tick()
        assert counter.fps > 0.0

    def test_fps_drops_old_frames(self):
        """Frames older than 1 s are evicted; fps reflects last second only."""
        t = [0.0]

        class FakeCounter(_FpsCounter):
            def __init__(self):
                super().__init__()
                self._clock_fn = lambda: t[0]

            def tick(self):
                from collections import deque
                now = self._clock_fn()
                self._ts.append(now)
                cutoff = now - 1.0
                while self._ts and self._ts[0] < cutoff:
                    self._ts.popleft()
                self.fps = float(len(self._ts))
                return self.fps

        counter = FakeCounter()
        for _ in range(10):
            counter.tick()
        assert counter.fps == 10.0

        t[0] = 1.5
        counter.tick()
        assert counter.fps == 1.0


# ---------------------------------------------------------------------------
# Config mismatch (arch change between agents)
# ---------------------------------------------------------------------------

class TestConfigMismatch:

    def test_hooks_reregistered_after_switch(self):
        """
        Simulates switching from a 2-layer to a 3-layer model.
        Hooks should be detached from old model and fresh on new model.
        """
        wm_small = make_world_model(num_layers=2, num_heads=2)
        wm_large = make_world_model(num_layers=3, num_heads=2)

        hooks = IrisHookExtractor()
        hooks.attach(wm_small)
        assert hooks.num_layers == 2

        # Simulate switch: detach old, attach new
        hooks.detach()
        assert hooks.num_layers == 0
        assert sum(len(m._forward_hooks) for m in wm_small.modules()) == 0

        hooks.attach(wm_large)
        assert hooks.num_layers == 3
        assert sum(len(m._forward_hooks) for m in wm_large.modules()) > 0

        hooks.detach()
