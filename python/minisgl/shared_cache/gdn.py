"""
Async-reasoning support for Qwen3.5 Gated DeltaNet (GDN) linear-attention layers.

Full-attention layers reuse ``SharedCacheAttention`` (token-addressable KV blocks).
GDN layers instead keep a monolithic recurrent state, so a worker's chain of blocks
is composed via the **affine summary** ``(A_hat, B_hat)`` stored on each ``CacheBlock``
(see ``gdn_affine``): the initial recurrent state for a chain is
``S0 = 0 @ A_chain + B_chain = B_chain`` (block convention), transposed to the HF
kernel convention ``[B, H, d_k, d_v]`` at the model interface.

``SharedCacheGDN`` is created once by the session (it knows the GDN dims) and handed the
current forward's ``(cache_structure, write_to)`` via ``set_context``.  The patched
``Qwen3_5GatedDeltaNet.forward`` reads it off ``get_global_ctx().gdn_ar``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Sequence

import torch

from .gdn_affine import compose_gdn_affines, init_gdn_affine, update_affine_summary

if TYPE_CHECKING:
    from .shared_block import CacheBlock


class SharedCacheGDN:
    """Per-forward composer/capturer of GDN affine summaries over worker chains."""

    def __init__(
        self,
        *,
        num_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        conv_dim: int,
        conv_kernel: int,
        device: torch.device,
        state_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.device = device
        if state_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError('Shared GDN state dtype must be float32 or bfloat16')
        self.state_dtype = state_dtype
        self.num_heads = num_heads  # H (post GQA-repeat, = linear_num_value_heads)
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.conv_dim = conv_dim
        self.conv_kernel = conv_kernel
        # Set per forward pass.
        self.cache_structure: List[List[CacheBlock]] = []
        self.write_to: List[CacheBlock] = []
        # Prefill only: per-request token counts, in row order, when the forward
        # batches several prefills.  ``None`` for decode (one token per worker)
        # and for a single-request prefill.
        self.prefill_segments: Optional[List[int]] = None
        self.decode_buffers = None
        self._prepared_decode = False
        self.prefill_buffers = None
        self._prepared_prefill = False

    def set_context(
        self,
        cache_structure: Sequence[Sequence["CacheBlock"]],
        write_to: Sequence["CacheBlock"],
        prefill_segments: Optional[Sequence[int]] = None,
    ) -> None:
        self.cache_structure = [list(c) for c in cache_structure]
        self.write_to = list(write_to)
        self.prefill_segments = None if prefill_segments is None else list(prefill_segments)
        self._prepared_decode = False
        self._prepared_prefill = False

    def prepare_prefill(self, layers: int, dtype: torch.dtype, rows: int, *, buffers=None):
        from .gdn_prefill import GDNPrefillBuffers

        depth = max(1, max(map(len, self.cache_structure), default=0))
        if buffers is None:
            buffers = self.prefill_buffers
            if (buffers is None or buffers.workers != self.num_workers or buffers.rows != rows
                    or buffers.depth < depth or buffers.layers != layers or buffers.dtype != dtype):
                buffers = GDNPrefillBuffers(self, layers, self.num_workers, depth, dtype, rows)
        self.prefill_buffers = buffers
        buffers.prepare(self.cache_structure, self.write_to, self.prefill_segments or [rows])
        self._prepared_prefill = True

    def finish_prefill(self, success: bool):
        if self._prepared_prefill:
            self.prefill_buffers.publish(success)
            self._prepared_prefill = False

    def prepare_decode(self, layers: int, dtype: torch.dtype, *, buffers=None) -> None:
        """Prepare addresses before entering the existing model forward."""
        from .gdn_decode import GDNDecodeBuffers

        depth = max(1, max(map(len, self.cache_structure), default=0))
        if buffers is None:
            buffers = self.decode_buffers
            if (buffers is None or buffers.workers != self.num_workers
                    or buffers.depth < depth or buffers.layers != layers or buffers.dtype != dtype):
                buffers = GDNDecodeBuffers(self, layers, self.num_workers, depth, dtype)
        self.decode_buffers = buffers
        buffers.prepare(self.cache_structure, self.write_to)
        self._prepared_decode = True

    def finish_decode(self, success: bool) -> None:
        if self._prepared_decode:
            self.decode_buffers.publish(success)
            self._prepared_decode = False

    @property
    def num_workers(self) -> int:
        return len(self.cache_structure)

    # ------------------------------------------------------------------
    # Reads: compose prior state for the current forward
    # ------------------------------------------------------------------

    def has_previous_affine(self, lin_idx: int) -> bool:
        for chain in self.cache_structure:
            for block in chain:
                if lin_idx in block.linear_affine:
                    return True
        return False

    def compose_initial_recurrent_state(
        self, lin_idx: int, dtype: torch.dtype
    ) -> Optional[torch.Tensor]:
        """Compose each worker's chain into an initial recurrent state.

        Returns ``[num_workers, H, d_k, d_v]`` in HF convention (or ``None`` if no
        block in any chain has an affine for this layer).
        """
        if self._prepared_decode:
            return self.decode_buffers.compose(lin_idx).to(dtype=dtype)
        if not self.has_previous_affine(lin_idx):
            return None

        # Worker chains typically share leading blocks (e.g. [prompt, thinker] is a
        # prefix of [prompt, thinker, writer]).  Memoize each composed prefix by its
        # block-id tuple so a shared prefix is composed once per call, not per worker.
        # Bit-identical to composing each chain independently (compose is deterministic).
        prefix_memo: dict = {}

        def compose_chain(chain):
            acc = None  # (A, B) once we hit the first block with an affine
            key: tuple = ()
            for block in chain:
                key = key + (id(block),)
                pair = block.linear_affine.get(lin_idx)
                if pair is None:
                    continue  # block has no affine for this layer -> acc unchanged
                if key in prefix_memo:
                    acc = prefix_memo[key]
                    continue
                A_b = pair[0].to(dtype=self.state_dtype, device=self.device)
                B_b = pair[1].to(dtype=self.state_dtype, device=self.device)
                if acc is None:
                    acc = (A_b, B_b)  # first real block: no identity compose needed
                else:
                    acc = compose_gdn_affines(
                        A_first=acc[0], B_first=acc[1], A_second=A_b, B_second=B_b
                    )
                prefix_memo[key] = acc
            if acc is None:
                acc = init_gdn_affine(
                    batch_size=1,
                    num_heads=self.num_heads,
                    d_k=self.head_k_dim,
                    d_v=self.head_v_dim,
                    dtype=self.state_dtype,
                    device=self.device,
                )
            return acc

        per_worker = [compose_chain(chain)[1] for chain in self.cache_structure]
        S_block = torch.cat(per_worker, dim=0)  # [W, H, d_v, d_k]
        S_hf = S_block.transpose(-1, -2).contiguous()  # [W, H, d_k, d_v]
        return S_hf.to(dtype=dtype)

    def prior_conv_states(self, lin_idx: int) -> Optional[torch.Tensor]:
        """Per-worker most-recent conv window along the chain, ``[W, conv_dim, k]``
        (zeros for a worker whose chain has none), or ``None`` if all are empty."""
        if self._prepared_decode:
            return self.decode_buffers.conv(lin_idx)
        per_worker: List[Optional[torch.Tensor]] = []
        present: Optional[torch.Tensor] = None
        for chain in self.cache_structure:
            found: Optional[torch.Tensor] = None
            for block in reversed(chain):
                c = block.linear_conv_state.get(lin_idx)
                if c is not None:
                    found = c
                    break
            if found is not None:
                present = found
            per_worker.append(found)
        if present is None:
            return None
        # Match the stored states' dtype: stacking a float32 filler with bf16
        # states would silently promote the whole window and break the conv.
        zeros = torch.zeros(
            self.conv_dim, self.conv_kernel, device=self.device, dtype=present.dtype
        )
        filled = [(c if c is not None else zeros).to(device=self.device) for c in per_worker]
        return torch.stack(filled, dim=0)  # [W, conv_dim, k]

    # ------------------------------------------------------------------
    # Writes: capture this forward's tokens into the write-target blocks
    # ------------------------------------------------------------------

    def capture_token_affines(
        self,
        lin_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        l2norm_eps: float = 1e-6,
        workers: Optional[Sequence[int]] = None,
    ) -> None:
        """Accumulate per-token affine updates into each worker's write block.

        ``key/value`` are ``[W, seq, H, d]``; ``alpha/beta`` are ``[W, seq, H]``.
        The key is L2-normed to match the kernel's ``use_qk_l2norm_in_kernel=True``.

        *workers* selects which write blocks the ``W`` rows correspond to
        (default: all of them, in order).  A batched prefill has a different
        token count per request, so it captures one request at a time.

        The rank-1 update is batched over workers (one call per token, not per
        worker), so the only Python loop is the inherently-sequential token scan
        (length 1 for decode; the block length for a prefill).
        """
        if self._prepared_decode:
            assert workers is None and key.shape[1] == 1
            self.decode_buffers.capture(lin_idx, key, value, alpha, beta, l2norm_eps)
            return
        targets = self.write_to if workers is None else [self.write_to[w] for w in workers]
        W, seq, H, dk = key.shape
        dv = value.shape[-1]
        key_f = key.float()
        key_f = key_f * torch.rsqrt((key_f * key_f).sum(dim=-1, keepdim=True) + l2norm_eps)
        value_f, alpha_f, beta_f = value.float(), alpha.float(), beta.float()

        # Gather each worker's prior affine into one batched [W, H, ...] pair.
        A, B = init_gdn_affine(
            batch_size=W, num_heads=H, d_k=dk, d_v=dv, dtype=torch.float32, device=key.device
        )
        for w, target in enumerate(targets):
            pair = target.linear_affine.get(lin_idx)
            if pair is not None:
                A[w] = pair[0][0].to(dtype=torch.float32, device=key.device)
                B[w] = pair[1][0].to(dtype=torch.float32, device=key.device)

        for t in range(seq):
            A, B = update_affine_summary(
                A_hat=A,
                B_hat=B,
                k=key_f[:, t],
                v=value_f[:, t],
                alpha=alpha_f[:, t],
                beta=beta_f[:, t],
            )

        for w, target in enumerate(targets):
            target.linear_affine[lin_idx] = (A[w : w + 1].to(self.state_dtype),
                                             B[w : w + 1].to(self.state_dtype))

    def set_conv_states(
        self, lin_idx: int, conv: torch.Tensor, workers: Optional[Sequence[int]] = None
    ) -> None:
        """Store per-worker conv windows ``[W, conv_dim, k]`` into write blocks
        (``workers`` selects which, as in :meth:`capture_token_affines`)."""
        if self._prepared_decode:
            assert workers is None
            self.decode_buffers.store_conv(lin_idx, conv)
            return
        targets = self.write_to if workers is None else [self.write_to[w] for w in workers]
        for w, target in enumerate(targets):
            target.linear_conv_state[lin_idx] = conv[w].detach().clone()


__all__ = ["SharedCacheGDN"]
