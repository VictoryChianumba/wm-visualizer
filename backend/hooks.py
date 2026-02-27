"""
PyTorch forward hook extraction for IRIS world model interpretability.

Hook targets:
  - block.attn.attn_drop  → inp[0]: (B, nh, T_q, T_k) post-softmax attention
  - block                 → out:    (B, T, E)  residual stream; norm of last token

KV cache shape notes:
  Uncached:      T_q = T_k = T  (full block, symmetric lower-triangular attention)
  Cached (L≥0):  T_q = new tokens,  T_k = L + T_q  (grows each inference step)

Hook extraction runs in the inference thread only; attach/detach must be called
from the same thread that owns the world model.
"""

import logging
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

HOOK_LATENCY_WARN_MS = 10.0


class IrisHookExtractor:
    """
    Registers and manages forward hooks on a WorldModel transformer.

    Thread-safety: get_data() returns shallow copies of the internal dicts,
    which is safe for single-writer (inference thread) / single-reader (WS thread)
    usage.  attach() and detach() must be called from the same thread.
    """

    def __init__(self) -> None:
        self._handles: list = []
        self._attn_data: Dict[int, torch.Tensor] = {}
        self._norms_data: Dict[int, torch.Tensor] = {}  # 0-d tensors; .item() deferred to consumer

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def attach(self, world_model) -> None:
        """
        Register hooks on every transformer Block.

        On any failure the method removes all already-registered hooks before
        re-raising so the model is always left in a clean state.
        """
        handles: list = []
        try:
            for i, block in enumerate(world_model.transformer.blocks):
                h_attn = block.attn.attn_drop.register_forward_hook(
                    self._make_attn_hook(i)
                )
                handles.append(h_attn)
                h_norm = block.register_forward_hook(self._make_norm_hook(i))
                handles.append(h_norm)
        except Exception as exc:
            for h in handles:
                h.remove()
            raise RuntimeError(f"Hook registration failed: {exc}") from exc

        self._handles = handles
        logger.info(
            "Attached %d hook pairs to %d transformer layers",
            len(handles) // 2,
            len(world_model.transformer.blocks),
        )

    def detach(self) -> None:
        """Remove all registered hooks and clear cached data."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._attn_data.clear()
        self._norms_data.clear()
        logger.info("Detached all hooks")

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def get_data(
        self,
    ) -> Tuple[Optional[Dict[int, torch.Tensor]], Optional[Dict[int, torch.Tensor]]]:
        """
        Return shallow copies of the latest attention and norm data.

        Returns (None, None) if no forward pass has run yet.
        """
        if not self._attn_data or not self._norms_data:
            return None, None
        return dict(self._attn_data), dict(self._norms_data)

    def clear(self) -> None:
        """Discard cached data without removing hooks."""
        self._attn_data.clear()
        self._norms_data.clear()

    @property
    def num_layers(self) -> int:
        """Number of layers currently hooked (0 if not attached)."""
        return len(self._handles) // 2

    # ------------------------------------------------------------------
    # Hook factories
    # ------------------------------------------------------------------

    def _make_attn_hook(self, layer_idx: int):
        def hook(module: nn.Module, inp: tuple, out: torch.Tensor) -> None:
            t0 = time.perf_counter()
            # inp[0]: (B, nh, T_q, T_k) — post-softmax attention before dropout.
            # attn_drop is nn.Dropout (inplace=False), so it never mutates inp[0]
            # after this hook returns — .detach() alone is safe, no clone needed.
            self._attn_data[layer_idx] = inp[0].detach()
            ms = (time.perf_counter() - t0) * 1000.0
            if ms > HOOK_LATENCY_WARN_MS:
                logger.warning("Attn hook layer %d: %.1f ms (threshold %.0f ms)",
                               layer_idx, ms, HOOK_LATENCY_WARN_MS)
        return hook

    def _make_norm_hook(self, layer_idx: int):
        def hook(module: nn.Module, inp: tuple, out: torch.Tensor) -> None:
            # out: (B, T, embed_dim) — residual stream after this block.
            # Store a 0-d tensor; do NOT call .item() here — that would force a
            # GPU→CPU sync on every layer inside the forward pass.  The consumer
            # (inference.py) calls .item() once, after the full forward completes.
            self._norms_data[layer_idx] = out[0, -1].norm().detach()
        return hook
