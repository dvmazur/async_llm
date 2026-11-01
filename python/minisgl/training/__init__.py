"""Record asynchronous inference and replay it with gradients."""

import warnings
from functools import wraps

from .forward import replay_trace
from .recording import RecordedAsyncLLM
from .trace import Trace


@wraps(replay_trace)
def replay_from_trace(*args, **kwargs):
    """Deprecated alias for replay_trace."""
    warnings.warn(
        "replay_from_trace is deprecated; use replay_trace instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return replay_trace(*args, **kwargs)


__all__ = ["RecordedAsyncLLM", "Trace", "replay_trace", "replay_from_trace"]
