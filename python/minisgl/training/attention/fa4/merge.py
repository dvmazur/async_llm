"""CuTe DSL reduction of partial attention outputs and their LSE normalizers.

The CSR plan is static trace metadata. Each partial row belongs to exactly one
original query, and ``indices`` must list every partial row exactly once. The
attention backend, rather than this operator, constructs and validates that plan.
"""

from functools import lru_cache

import torch
from torch.autograd.function import once_differentiable

_WARP_SIZE = 32
_WARPS_PER_BLOCK = 4
_THREADS_PER_BLOCK = _WARP_SIZE * _WARPS_PER_BLOCK


@lru_cache(maxsize=1)
def _kernel_types():
    # Keep the ordinary torch replay usable without the optional FA4/CuTe stack.
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass import Float32

    class MergeForward:
        def __init__(self, width):
            self.width = width
            self.values_per_lane = (width + _WARP_SIZE - 1) // _WARP_SIZE

        @cute.jit
        def __call__(
            self,
            parts: cute.Tensor,
            lse: cute.Tensor,
            indptr: cute.Tensor,
            indices: cute.Tensor,
            output: cute.Tensor,
            output_fp32: cute.Tensor,
            total_lse: cute.Tensor,
            stream: cuda.CUstream,
        ):
            self.kernel(parts, lse, indptr, indices, output, output_fp32, total_lse).launch(
                grid=[cute.ceil_div(output.shape[0] * output.shape[1], _WARPS_PER_BLOCK), 1, 1],
                block=[_THREADS_PER_BLOCK, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            parts: cute.Tensor,
            lse: cute.Tensor,
            indptr: cute.Tensor,
            indices: cute.Tensor,
            output: cute.Tensor,
            output_fp32: cute.Tensor,
            total_lse: cute.Tensor,
        ):
            thread, _, _ = cute.arch.thread_idx()
            block, _, _ = cute.arch.block_idx()
            lane = thread % _WARP_SIZE
            row_head = block * _WARPS_PER_BLOCK + thread // _WARP_SIZE
            heads = output.shape[1]
            if row_head < output.shape[0] * heads:
                row, head = row_head // heads, row_head % heads
                start, end = indptr[row], indptr[row + 1]
                maximum = Float32(-Float32.inf)
                for position in cutlass.range(start, end):
                    part = indices[position]
                    maximum = cute.arch.fmax(maximum, lse[head, part])

                accum = cute.make_rmem_tensor((self.values_per_lane,), Float32)
                accum.fill(0.0)
                denominator = Float32(0.0)
                if maximum != Float32(-Float32.inf):
                    for position in cutlass.range(start, end):
                        part = indices[position]
                        weight = cute.math.exp(lse[head, part] - maximum)
                        denominator += weight
                        for item in cutlass.range_constexpr(self.values_per_lane):
                            dimension = lane + item * _WARP_SIZE
                            if dimension < self.width:
                                accum[item] += weight * Float32(parts[part, head, dimension])

                normalizer = Float32(-Float32.inf)
                if denominator > 0.0:
                    normalizer = maximum + cute.math.log(denominator)
                if lane == 0:
                    total_lse[row, head] = normalizer
                for item in cutlass.range_constexpr(self.values_per_lane):
                    dimension = lane + item * _WARP_SIZE
                    if dimension < self.width:
                        value = Float32(0.0)
                        if denominator > 0.0:
                            value = accum[item] / denominator
                        output[row, head, dimension] = output.element_type(value)
                        output_fp32[row, head, dimension] = value

    class MergeBackward:
        def __init__(self, width):
            self.width = width
            self.values_per_lane = (width + _WARP_SIZE - 1) // _WARP_SIZE

        @cute.jit
        def __call__(
            self,
            parts: cute.Tensor,
            lse: cute.Tensor,
            indptr: cute.Tensor,
            indices: cute.Tensor,
            output_fp32: cute.Tensor,
            total_lse: cute.Tensor,
            grad: cute.Tensor,
            grad_parts: cute.Tensor,
            grad_lse: cute.Tensor,
            stream: cuda.CUstream,
        ):
            self.kernel(
                parts, lse, indptr, indices, output_fp32, total_lse, grad, grad_parts, grad_lse
            ).launch(
                grid=[
                    cute.ceil_div(output_fp32.shape[0] * output_fp32.shape[1], _WARPS_PER_BLOCK),
                    1,
                    1,
                ],
                block=[_THREADS_PER_BLOCK, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            parts: cute.Tensor,
            lse: cute.Tensor,
            indptr: cute.Tensor,
            indices: cute.Tensor,
            output_fp32: cute.Tensor,
            total_lse: cute.Tensor,
            grad: cute.Tensor,
            grad_parts: cute.Tensor,
            grad_lse: cute.Tensor,
        ):
            thread, _, _ = cute.arch.thread_idx()
            block, _, _ = cute.arch.block_idx()
            lane = thread % _WARP_SIZE
            row_head = block * _WARPS_PER_BLOCK + thread // _WARP_SIZE
            heads = output_fp32.shape[1]
            if row_head < output_fp32.shape[0] * heads:
                row, head = row_head // heads, row_head % heads
                normalizer = total_lse[row, head]
                start, end = indptr[row], indptr[row + 1]
                for position in cutlass.range(start, end):
                    part = indices[position]
                    weight = Float32(0.0)
                    if normalizer != Float32(-Float32.inf):
                        weight = cute.math.exp(lse[head, part] - normalizer)
                    dot = Float32(0.0)
                    for item in cutlass.range_constexpr(self.values_per_lane):
                        dimension = lane + item * _WARP_SIZE
                        if dimension < self.width:
                            upstream = Float32(grad[row, head, dimension])
                            difference = (
                                Float32(parts[part, head, dimension])
                                - output_fp32[row, head, dimension]
                            )
                            dot += upstream * difference
                            grad_parts[part, head, dimension] = grad_parts.element_type(
                                weight * upstream
                            )
                    dot = cute.arch.warp_reduction_sum(dot)
                    if lane == 0:
                        grad_lse[head, part] = weight * dot

    return MergeForward, MergeBackward


@lru_cache(maxsize=64)
def _compile_kernel(backward, dtype, heads, width, capability):
    import cutlass
    import cutlass.cute as cute

    del capability  # Separates kernels compiled for different CUDA architectures.
    element = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
    }[dtype]
    n, p = cute.sym_int(), cute.sym_int()

    def rows(kind, count):
        return cute.runtime.make_fake_tensor(
            kind, (count, heads, width), stride=(heads * width, width, 1), assumed_align=16
        )

    parts = rows(element, p)
    lse = cute.runtime.make_fake_tensor(
        cutlass.Float32,
        (heads, p),
        stride=(cute.sym_int64(), cute.sym_int64()),
        assumed_align=4,
    )
    indptr = cute.runtime.make_fake_tensor(cutlass.Int32, (cute.sym_int(),), stride=(1,))
    indices = cute.runtime.make_fake_tensor(cutlass.Int32, (p,), stride=(1,))
    output = rows(element, n)
    output_fp32 = rows(cutlass.Float32, n)
    total_lse = cute.runtime.make_fake_tensor(
        cutlass.Float32, (n, heads), stride=(heads, 1), assumed_align=4
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    forward_type, backward_type = _kernel_types()
    if backward:
        grad_parts = rows(element, p)
        grad_lse = cute.runtime.make_fake_tensor(
            cutlass.Float32, (heads, p), stride=(cute.sym_int64(), 1), assumed_align=4
        )
        return cute.compile(
            backward_type(width),
            parts,
            lse,
            indptr,
            indices,
            output_fp32,
            total_lse,
            output,
            grad_parts,
            grad_lse,
            stream,
            options="--enable-tvm-ffi",
        )
    return cute.compile(
        forward_type(width),
        parts,
        lse,
        indptr,
        indices,
        output,
        output_fp32,
        total_lse,
        stream,
        options="--enable-tvm-ffi",
    )


class _MergeAttentionParts(torch.autograd.Function):
    @staticmethod
    def forward(ctx, parts, lse, indptr, indices):
        parts = parts.contiguous()
        rows, heads, width = indptr.numel() - 1, parts.shape[1], parts.shape[2]
        output = parts.new_empty((rows, heads, width))
        output_fp32 = parts.new_empty(output.shape, dtype=torch.float32)
        total_lse = parts.new_empty((rows, heads), dtype=torch.float32)
        with torch.cuda.device(parts.device):
            _compile_kernel(
                False, parts.dtype, heads, width, torch.cuda.get_device_capability(parts.device)
            )(parts, lse, indptr, indices, output, output_fp32, total_lse)
        ctx.save_for_backward(parts, lse, indptr, indices, output_fp32, total_lse)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        parts, lse, indptr, indices, output_fp32, total_lse = ctx.saved_tensors
        grad = grad.contiguous()
        grad_parts = torch.empty_like(parts)
        grad_lse = torch.empty(lse.shape, device=lse.device, dtype=lse.dtype)
        with torch.cuda.device(parts.device):
            _compile_kernel(
                True,
                parts.dtype,
                parts.shape[1],
                parts.shape[2],
                torch.cuda.get_device_capability(parts.device),
            )(parts, lse, indptr, indices, output_fp32, total_lse, grad, grad_parts, grad_lse)
        return grad_parts, grad_lse, None, None


def merge_attention_parts(parts, lse, indptr, indices):
    """Merge FA4 ``parts[P,H,D]`` / natural-log ``lse[H,P]`` using a CSR plan.

    ``indptr[N+1]`` and ``indices[P]`` are contiguous CUDA int32 metadata. The
    operation supports FP16/BF16/FP32 outputs and first-order gradients through
    both outputs of the partial-attention kernel.
    """
    if parts.ndim != 3 or lse.shape != (parts.shape[1], parts.shape[0]):
        raise ValueError("Expected attention parts[P,H,D] and lse[H,P]")
    if parts.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("Attention parts must be FP16, BF16, or FP32")
    if lse.dtype != torch.float32:
        raise TypeError("Attention LSE must be FP32")
    if not all(t.is_cuda and t.device == parts.device for t in (parts, lse, indptr, indices)):
        raise ValueError("Attention merge inputs must be on the same CUDA device")
    if any(
        t.ndim != 1 or t.dtype != torch.int32 or not t.is_contiguous() for t in (indptr, indices)
    ):
        raise ValueError("Attention merge CSR metadata must be contiguous int32 vectors")
    if indptr.numel() < 2 or indices.numel() != parts.shape[0]:
        raise ValueError("Attention merge CSR metadata has incompatible sizes")
    return _MergeAttentionParts.apply(parts, lse, indptr, indices)
