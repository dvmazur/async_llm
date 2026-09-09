"""Tensor-only Qwen FP32 gate/rotation; CUDA fusion preserves reference rounding."""
import torch


def rotate_fp32_eager(x,cos,sin):
    xf=x.float();left,right=xf.chunk(2,dim=-1)
    return (xf*cos+torch.cat((-right,left),dim=-1)*sin).to(x.dtype)


def output_gate_eager(output,gate):
    return (output.float()*gate.float().sigmoid()).to(output.dtype)


_compiled_rotate=torch.compile(rotate_fp32_eager,fullgraph=True,dynamic=True)
_compiled_output_gate=torch.compile(output_gate_eager,fullgraph=True,dynamic=True)


def rotate_fp32(x,cos,sin):
    return (_compiled_rotate if x.is_cuda else rotate_fp32_eager)(x,cos,sin)


def output_gate(output,gate):
    return (_compiled_output_gate if output.is_cuda else output_gate_eager)(output,gate)
