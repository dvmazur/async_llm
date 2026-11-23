"""Local compiler policy for the tensor regions of replay attention."""

from functools import wraps

import torch

# Different chunk shapes/read layouts need separate compiled graphs.
# Allow more than Dynamo's default 8 cached variants per function.
_COMPILE_CACHE_LIMIT = 1024


def _inductor_backend(graph_module, example_inputs):
    from torch._functorch import config as aot_autograd_config
    from torch._inductor import compile as inductor_compile

    # Training calls backward outside the forward's optional autocast context.
    # AOTAutograd must trace that same policy; patch only this compilation.
    with aot_autograd_config.patch(backward_pass_autocast="off"):
        return inductor_compile(graph_module, example_inputs)


def compile_attention(function):
    """Lazily compile CUDA calls, retaining ordinary eager execution on CPU.

    Trace planning stays outside the compiled region. Static full graphs avoid
    symbolic-shape failures on historical views; compiler errors and exhausted
    cache limits propagate instead of silently changing execution mode.
    """
    compiled_function = None

    @wraps(function)
    def dispatch(query, *args, **kwargs):
        nonlocal compiled_function
        if query.device.type != "cuda":
            return function(query, *args, **kwargs)
        if compiled_function is None:
            compiled_function = torch.compile(
                function,
                backend=_inductor_backend,
                fullgraph=True,
                dynamic=False,
                recompile_limit=_COMPILE_CACHE_LIMIT,
            )

        from torch._dynamo import config as dynamo_config

        # Torch 2.13 keeps these overrides local to the current context. The
        # cumulative cap is separate from the per-region torch.compile limit.
        with dynamo_config.patch(
            accumulated_recompile_limit=max(
                dynamo_config.accumulated_recompile_limit, _COMPILE_CACHE_LIMIT
            )
        ):
            return compiled_function(query, *args, **kwargs)

    return dispatch
