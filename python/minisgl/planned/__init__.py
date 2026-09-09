"""New planned runtime, built independently of the legacy session.

Importing this namespace must not import torch, initialize CUDA, or activate
a runner. The explicitly requested session class loads the GPU adapters lazily.
"""

def __getattr__(name):
    if name=="PlannedSession":
        from .session import PlannedSession
        return PlannedSession
    if name=="RuntimeProfile":
        from .catalogue import RuntimeProfile
        return RuntimeProfile
    raise AttributeError(name)
