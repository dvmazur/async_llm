from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from minisgl.kvcache import BaseCacheHandle
from minisgl.utils import div_ceil


@dataclass(frozen=True)
class _NullCacheHandle(BaseCacheHandle):
    """Placeholder handle for shared-cache requests that bypass the prefix cache."""

    def get_matched_indices(self) -> torch.Tensor:
        return torch.empty(0, dtype=torch.int32)


NULL_CACHE_HANDLE = _NullCacheHandle(cached_len=0)


class CacheBlock:
    """
    A reusable, paged block of KV cache that can be shared across multiple
    workers.

    The block owns a list of **pages** borrowed from the engine's page
    allocator; each page holds ``page_size`` contiguous token slots.  Token
    ``i`` (block-relative, ``0 <= i < num_tokens``) lives at physical slot
    ``page_starts[i // page_size] + (i % page_size)``.  The last page may be
    partially filled.

    Keys are stored at block-relative RoPE positions (0..num_tokens-1). Normal
    cache views rotate queries instead of keys (see ``shared_cache.attention``);
    merging blocks shifts copied keys into the merged block's coordinate frame.
    """

    _next_id: int = 0

    def __init__(self, device: torch.device, page_size: int = 1):
        self.block_id = CacheBlock._next_id
        CacheBlock._next_id += 1
        self.device = device
        self.page_size = page_size
        # Page-start token slots (multiples of page_size), one per owned page.
        self.page_starts: List[int] = []
        self.num_tokens: int = 0
        # Host-side copy of the token ids stored in this block, in block order.
        # Kept in sync by whoever writes the block (prefill extends it, the
        # async engine appends decoded tokens); consumers use it for probes and
        # end-of-step detection without decoding KV.
        self.token_ids: List[int] = []

        # mRoPE span (Qwen3.5 multimodal): how much the running mRoPE position
        # advances over this block.  For text blocks it equals num_tokens (default,
        # via ``mrope_span``); an image compresses positions, so an image-bearing
        # prefill sets ``mrope_span_override`` to its ``get_rope_index`` span.
        self.mrope_span_override: Optional[int] = None

        # --- Gated DeltaNet (Qwen3.5) per-linear-layer state ---
        # Block-level affine summary (A_hat, B_hat) of this block's GDN token
        # trajectory, keyed by linear-layer index; fp32, block convention
        # (A_hat [1,H,d_k,d_k], B_hat [1,H,d_v,d_k]).  Composing a worker's chain
        # of these folds into an initial recurrent state (see shared_cache.gdn).
        self.linear_affine: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        # Production Qwen blocks keep all square per-layer A/B summaries in one
        # block-owned slab.  Ownership follows the block lifetime instead of a
        # heterogeneous prefill batch, so freeing one block never remains pinned
        # by an unrelated worker that happened to share that batch.
        self.linear_affine_storage: Optional[torch.Tensor] = None
        # Monotonic per-layer revisions make persistent compose-cache keys safe
        # across writes, clear/reuse, allocator pointer reuse, and block merges.
        self.linear_affine_revision: Dict[int, int] = {}
        # Rolling causal-conv window (last conv_kernel columns) per linear layer,
        # [conv_dim, conv_kernel].  Standard full-attention blocks leave these empty.
        self.linear_conv_state: Dict[int, torch.Tensor] = {}

    @property
    def mrope_span(self) -> int:
        """Running-mRoPE advance over this block (== num_tokens unless overridden)."""
        return self.num_tokens if self.mrope_span_override is None else self.mrope_span_override

    @property
    def num_pages(self) -> int:
        return len(self.page_starts)

    @property
    def last_page_len(self) -> int:
        """Valid token count in the final page (1..page_size), 0 if empty."""
        if self.num_tokens == 0:
            return 0
        return self.num_tokens - (self.num_pages - 1) * self.page_size

    @property
    def has_capacity(self) -> bool:
        """Whether the current last page has room for one more token."""
        return self.num_tokens < self.num_pages * self.page_size

    @property
    def free_tail(self) -> int:
        """Free token slots left in the current last page (0 if the block is empty)."""
        return self.num_pages * self.page_size - self.num_tokens

    def pages_needed(self, num_new_tokens: int) -> int:
        """How many fresh pages appending ``num_new_tokens`` requires: the last
        page's free slots are filled first, the rest go to new pages."""
        return div_ceil(max(0, num_new_tokens - self.free_tail), self.page_size)

    def page_starts_tensor(self) -> torch.Tensor:
        """Page-start token slots as a device tensor ``[num_pages]``."""
        from minisgl.kernel.metadata import device_metadata
        return device_metadata(self.page_starts, dtype=torch.int32, device=self.device)

    def page_numbers_tensor(self) -> torch.Tensor:
        """Physical page numbers (= page_start // page_size) for paged kernels."""
        starts = self.page_starts_tensor()
        return starts if self.page_size == 1 else starts // self.page_size

    def token_slots_tensor(self) -> torch.Tensor:
        """Per-token physical slots ``[num_tokens]`` (flattened paged layout)."""
        if self.num_tokens == 0:
            return torch.empty(0, dtype=torch.int32, device=self.device)
        starts = self.page_starts_tensor()
        if self.page_size == 1:
            return starts[: self.num_tokens]
        offsets = torch.arange(self.page_size, dtype=torch.int32, device=self.device)
        return (starts[:, None] + offsets[None, :]).flatten()[: self.num_tokens]

    def grow_pages(self, page_starts: torch.Tensor, num_new_tokens: int) -> None:
        """Record a prefill write: ``num_new_tokens`` tokens appended to the
        block, filling the current last page's free slots first and then the
        freshly-allocated ``page_starts`` (``pages_needed(num_new_tokens)`` of
        them).  A non-empty block is *extended*, exactly as ``append_token``
        does one token at a time."""
        assert num_new_tokens > 0, "grow_pages needs at least one token"
        expected = self.pages_needed(num_new_tokens)
        assert page_starts.numel() == expected, (
            f"grow_pages got {page_starts.numel()} pages for {num_new_tokens} tokens "
            f"(free tail {self.free_tail}, page_size {self.page_size}); expected {expected}"
        )
        self.page_starts.extend(int(p) for p in page_starts.tolist())
        self.num_tokens += num_new_tokens

    def append_token(self, new_page_start: Optional[int]) -> None:
        """Record a single decode write.  ``new_page_start`` is the page-start
        slot of a freshly-allocated page when this token starts a new page, else
        ``None`` (the token fit in the current last page)."""
        if new_page_start is not None:
            assert not self.has_capacity, "append_token got a new page but last page has room"
            self.page_starts.append(int(new_page_start))
        else:
            assert self.has_capacity, "append_token needs a new page but none was given"
        self.num_tokens += 1
        if self.mrope_span_override is not None:
            # A decoded token is text: it advances the mRoPE frame by one, exactly
            # like the token count.  Tracked explicitly because an image earlier in
            # the block has decoupled the span from ``num_tokens``.
            self.mrope_span_override += 1

    def clear(self) -> List[int]:
        """Reset the block and return the page-start slots the caller should free."""
        pages = list(self.page_starts)
        self.page_starts.clear()
        self.num_tokens = 0
        self.mrope_span_override = None
        self.token_ids.clear()
        for layer_idx in self.linear_affine:
            self.linear_affine_revision[layer_idx] = (
                self.linear_affine_revision.get(layer_idx, 0) + 1
            )
        self.linear_affine.clear()
        self.linear_affine_storage = None
        self.linear_conv_state.clear()
        return pages

    def affine_storage_pair(
        self,
        layer_idx: int,
        *,
        num_layers: int,
        num_heads: int,
        d_k: int,
        d_v: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return contiguous block-owned output views for one GDN layer."""

        if d_k != d_v:
            raise ValueError("block-owned affine slabs currently require square GDN state")
        expected = (num_layers, 2, num_heads, d_k, d_k)
        if self.linear_affine_storage is None:
            self.linear_affine_storage = torch.empty(
                expected, dtype=torch.float32, device=self.device
            )
        elif self.linear_affine_storage.shape != expected:
            raise ValueError(
                "existing GDN affine slab has shape "
                f"{tuple(self.linear_affine_storage.shape)}, expected {expected}"
            )
        A = self.linear_affine_storage[layer_idx, 0].unsqueeze(0)
        B = self.linear_affine_storage[layer_idx, 1].unsqueeze(0)
        assert A.is_contiguous() and B.is_contiguous()
        return A, B

    def set_linear_affine(self, layer_idx: int, pair: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """Replace one affine summary and advance its persistent cache revision."""
        self.linear_affine[layer_idx] = pair
        self.linear_affine_revision[layer_idx] = self.linear_affine_revision.get(layer_idx, 0) + 1

    def __repr__(self) -> str:
        return (
            f"CacheBlock(id={self.block_id}, tokens={self.num_tokens}, "
            f"pages={self.num_pages}, page_size={self.page_size})"
        )


# Historical name, kept so existing tests/scripts keep working.
SharedBlock = CacheBlock
