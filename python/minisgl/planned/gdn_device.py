"""Stable metadata/workspace bindings, not a session or allocator implementation."""
import torch
import triton

from .forward_plan import PhasePlan, RowPlan
from . import gdn_kernels as kernels


def _table_shape(values):
    # Metadata is either a flat tuple or a small number of coordinate axes.
    # Inspect dimensions, not each scalar; NumPy copies nested tuples in C.
    if values and isinstance(values[0], (tuple, list)):
        width=len(values[0])
        if any(len(axis)!=width for axis in values):raise ValueError("ragged metadata axes")
        return len(values),width
    return (len(values),)


class _DeviceTables:
    def __init__(self, plan, device):
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.host = {}
        self._copy_done = None
        for name in self.fields:
            values = getattr(plan, name)
            dtype = torch.bool if name in ("active", "write_fresh") else torch.int64
            self.host[name] = torch.empty(_table_shape(values), dtype=dtype, pin_memory=self.device.type == "cuda")
            setattr(self, name, torch.empty_like(self.host[name],device=device))
        self.upload(plan)

    def upload(self, plan):
        values={name:getattr(plan,name) for name in self.fields}
        for name in self.fields:
            if _table_shape(values[name]) != tuple(self.host[name].shape):
                raise ValueError(f"metadata shape changed: {name}")
        if self._copy_done is not None:
            # Protect pinned host staging until its previous DMA completed.
            self._copy_done.synchronize()
        for name in self.fields:
            self.host[name].numpy()[:] = values[name]
            getattr(self, name).copy_(self.host[name], non_blocking=True)
        if self.device.type == "cuda":
            if self._copy_done is None:
                self._copy_done = torch.cuda.Event()
            self._copy_done.record(torch.cuda.current_stream(self.device))


class DevicePhase(_DeviceTables):
    """Upload once before a forward; all GDN layers consume the same addresses."""
    fields = ("active", "write_slots", "write_fresh", "prior_conv_slots", "level_counts",
              "node_slots", "parent_rows", "terminal_nodes", "sink_offsets", "sink_workers")

    def __init__(self, plan: PhasePlan, device):
        self.width, self.depth = len(plan.active), len(plan.level_counts)
        super().__init__(plan, device)


class DeviceRows(_DeviceTables):
    fields = ("active", "request", "local_token", "prefill_offsets", "output_rows",
              "chunk_indices", "chunk_offsets", "prefill_recipe")

    def __init__(self, plan: RowPlan, device):
        super().__init__(plan, device)


class BoundGDN:
    """One phase's GPU recipe; bind once per profile, reuse across layers.

    The full runner will own/pin pool and workspace. No block objects, tensors
    list construction, allocations or metadata upload occur in gpu methods.
    """
    def __init__(self, pool, metadata: DevicePhase, *, workspace=None):
        if not pool.is_cuda or pool.dtype != torch.float32 or pool.ndim != 6 or not pool.is_contiguous():
            raise ValueError("expected contiguous CUDA FP32 [L,slots,2,H,D,D] pool")
        self.layers, self.slots, pair, self.heads, dim, other = pool.shape
        if pair != 2 or dim != other or min(pool.shape) < 1 or metadata.device != pool.device:
            raise ValueError("incompatible pool layout or metadata device")
        if metadata.depth == 0:
            raise ValueError("GPU recipe requires positive depth capacity")
        self.pool, self.meta, self.dim = pool, metadata, dim
        shape = (metadata.width, self.heads, dim, dim)
        if workspace is None:
            workspace = torch.empty((3, *shape), device=pool.device, dtype=torch.float32)
        if (workspace.shape != (3, *shape) or workspace.device != pool.device
                or workspace.dtype != torch.float32 or not workspace.is_contiguous()):
            raise ValueError("invalid shared compose workspace")
        self.workspace = workspace
        self.frontiers = [workspace[0], workspace[1]]
        self.initial = workspace[2]
        self.bm = 64 if dim >= 64 else max(16, triton.next_power_of_2(dim))
        self.bn = 32 if dim >= 32 else max(16, triton.next_power_of_2(dim))

    def compose(self, layer):
        if not 0 <= layer < self.layers:
            raise ValueError("layer index outside pool")
        m, d = self.meta, self.dim
        if m.width == 0:
            return self.initial
        kernels.compose_first[(m.width, self.heads, triton.cdiv(d*d, 256))](
            self.pool, m.node_slots, m.level_counts, m.terminal_nodes, m.sink_offsets, m.sink_workers,
            self.frontiers[0], self.initial, LAYER=layer, SLOTS=self.slots,
            H=self.heads, D=d, BLOCK=256)
        for depth in range(1, m.depth):
            kernels.compose_level[(triton.cdiv(d, self.bm)*triton.cdiv(d, self.bn), m.width*self.heads)](
                self.pool, m.node_slots, m.parent_rows, m.level_counts, m.sink_offsets, m.sink_workers,
                self.frontiers[(depth-1)%2], self.frontiers[depth%2], self.initial,
                LAYER=layer, DEPTH=depth, WIDTH=m.width, SLOTS=self.slots, H=self.heads, D=d,
                BM=self.bm, BN=self.bn, BK=16, num_warps=4, num_stages=1)
        return self.initial

    def decode(self, layer, query, key, value, g, beta, alpha, output, *, conv=None):
        # Only static shapes/types are inspected; no values read from GPU.
        if not 0 <= layer < self.layers:
            raise ValueError("layer index outside pool")
        w, d = self.meta.width, self.dim
        if (query.ndim != 4 or query.shape[:2] != (w, 1) or query.shape[-1] != d
                or key.shape != query.shape or query.shape[2] < 1
                or self.heads % query.shape[2] or value.shape != (w, 1, self.heads, d)
                or output.shape != value.shape
                or any(x.shape != (w, 1, self.heads) for x in (g, beta, alpha))):
            raise ValueError("invalid bound recurrent shapes")
        inputs = (query, key, value, g, beta, alpha, output)
        if any(x.device != self.pool.device for x in inputs):
            raise ValueError("bound inputs must be on the pool device")
        if any(x.stride(-1) != 1 or x.stride(-2) != d for x in (query,key,value)):
            raise ValueError("qkv must have contiguous head rows")
        if any(not x.is_contiguous() for x in (g,beta,alpha,output)):
            raise ValueError("gates/output must be contiguous")
        if any(x.dtype not in (torch.float32, torch.float16, torch.bfloat16) for x in inputs):
            raise ValueError("unsupported recurrent dtype")
        if conv is not None and (conv.phase is not self.meta or conv.prefill
                                 or conv.slots != self.slots or conv.layers != self.layers):
            raise ValueError("conv publication must use the same decode phase and pool slots")
        if self.meta.width:
            kernels.recurrent_capture[(triton.cdiv(self.dim, 8), self.meta.width*self.heads)](
                query, key, value, g, beta, alpha, self.initial, self.pool,
                self.meta.write_slots, self.meta.write_fresh, self.meta.active, output,
                self.pool if conv is None else conv.pool,
                self.initial if conv is None else conv.new_window,
                LAYER=layer, SLOTS=self.slots, H=query.shape[2], HV=self.heads,
                D=self.dim, KD=triton.next_power_of_2(self.dim), ROWS=8, num_warps=1,
                HAS_CONV=conv is not None, C=0 if conv is None else conv.channels,
                CK=0 if conv is None else conv.window_size,
                QSTRIDE=query.stride(0), KSTRIDE=key.stride(0), VSTRIDE=value.stride(0),
                prefill_recipe=self.meta.active, PREFILL_SPECIAL=False,
                num_stages=3)
