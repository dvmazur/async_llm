"""Record asynchronous inference and replay it with gradients."""

from .forward import replay_trace
from .recording import RecordedAsyncLLM
from .trace import Trace, record

replay_from_trace = replay_trace

__all__ = ["RecordedAsyncLLM", "Trace", "record", "replay_trace", "replay_from_trace"]
