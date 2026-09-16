"""Telemetry must also support the upstream shared session without graph_runner."""
import asyncio
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

from experiment_runner.logs import read_jsonl


def test_backend_with_no_shared_graph_runner(tmp_path, monkeypatch):
    # This test exercises telemetry/session lifecycle, not tensor math. Load an
    # isolated adapter module with just its memory counters stubbed; no Torch or
    # GPU installation is required and the normal engine module is not cached.
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(
        memory_allocated=lambda: 0, memory_reserved=lambda: 0)))
    spec = importlib.util.spec_from_file_location('experiment_runner._telemetry_test',
        Path(__file__).resolve().parents[1]/'experiment_runner/engine.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    Backend = module.Backend
    output = object()
    original = lambda batch: output
    session = SimpleNamespace(_forward=original)

    class LLM:
        async_engine = SimpleNamespace(session=session)
        closed = False

        async def close(self):
            self.closed = True

    async def run():
        llm = LLM()
        backend = Backend(llm, tmp_path)
        assert session._forward(SimpleNamespace(is_decode=True, size=2)) is output
        await backend.close()
        assert session._forward is original and llm.closed
    asyncio.run(run())
    rows = list(read_jsonl(tmp_path/'forwards.jsonl'))
    assert len(rows) == 1
    assert rows[0]['phase'] == 'decode'
    assert rows[0]['decode_requests'] == 2 and rows[0]['graph_replay'] is False
