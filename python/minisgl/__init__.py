"""Asynchronous inference and differentiable replay of recorded execution."""

from importlib import import_module

__all__ = ["AsyncLLM", "training"]


def __getattr__(name):
    if name == "AsyncLLM":
        value = import_module(".llm", __name__).AsyncLLM
    elif name == "training":
        value = import_module(".training", __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
