"""CPU stub model, real AsyncLLM scheduler/recorder and real ViZDoom environment.

These are plumbing/gradient checks, explicitly not successful teacher rollouts.
"""

import asyncio
from contextlib import nullcontext
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vizdoom")
from minisgl.training import RecordedAsyncLLM, Trace
from minisgl_examples.training.doom_health.environment import Setup, make_env
from minisgl_examples.training.doom_health.experiment import run_episode
from minisgl_examples.training.doom_health.recording import TargetRecorder

from tests.core.test_async_cache_engine import StubSession
from tests.core.test_async_llm import _make_llm
from tests.training.fixtures import torch_reference_kernels as torch_reference_kernels
from tests.training.helpers import copy_operations
from tests.training.models import tiny_image_model


class Tokenizer:
    vocab = {"continue": 3, "wait": 4, "forward": 5, "right": 6, "left": 7, "<|im_end|>": 28}

    def encode(self, text, **kwargs):
        if text in self.vocab:
            return [self.vocab[text]]
        if text.endswith("Action: `"):
            return [1, 8]
        return [1, 10]

    def __call__(self, text, **kwargs):
        return {"input_ids": torch.tensor([self.encode(text)], dtype=torch.long)}

    def decode(self, tokens, **kwargs):
        return "".join("<|im_end|>" if int(t) == 28 else "." for t in tokens)


class Processor:
    def apply_chat_template(self, messages, **kwargs):
        if isinstance(messages[0]["content"], str):
            return messages[0]["content"]
        ids = torch.tensor([[1, 30, 29, 31, 30, 29, 31, 28, 2]])
        return {
            "input_ids": ids,
            "mm_token_type_ids": (ids == 29).long(),
            "attention_mask": torch.ones_like(ids),
            "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 2]]),
            "pixel_values": torch.arange(96).float().reshape(8, 12) * 0.001,
        }


class Session(StubSession):
    def __init__(self, always_continue=False):
        super().__init__()
        self.probes = 0
        self.always_continue = always_continue

    def _mrope_rel(self, ids, types, grid):
        return torch.arange(len(ids)).expand(3, -1)

    def prefill_batch(self, jobs):
        result = super().prefill_batch(jobs)
        for i, job in enumerate(jobs):
            result[i] = torch.zeros(1, 32)
            result[i][0, 2] = 10
            if int(job.input_ids[-1]) == 8:
                self.probes += 1
                result[i][0, 5] = 9
                result[i][0, 3] = 12 if self.always_continue or self.probes % 2 else 1
        return result

    def decode_step(self, group, input_ids):
        result = super().decode_step(group, input_ids)
        result.zero_()
        result[:, 2] = 10
        return result


@pytest.fixture(autouse=True)
def no_cuda_nvtx(monkeypatch):
    monkeypatch.setattr(torch.cuda.nvtx, "range", lambda *args, **kwargs: nullcontext())


def record(tmp_path, always_continue=False, *, control="normal"):
    async def run():
        llm, _ = _make_llm()
        session = Session(always_continue)
        copy_operations(session)
        llm.async_engine.session = session
        llm.tokenizer = Tokenizer()
        llm.processor = Processor()
        cfg = replace(
            Setup(),
            max_actions=2,
            min_actions=2,
            consolidation_steps=1,
            thought_length=6,
            description_length=3,
            readiness_interval=2,
        )
        env = make_env(cfg)
        try:
            with RecordedAsyncLLM(llm) as rec:
                port = TargetRecorder(rec)
                data = await run_episode(port, cfg, env, tmp_path, control=control)
                data.update(port.export())
            data["runtime"] = {"kind": "cpu_scheduler_stub"}
            return data
        finally:
            env.close()
            await llm.close()

    return asyncio.run(run())


def test_hdelta_port_real_scheduler_and_environment(tmp_path):
    data = record(tmp_path)
    assert len(data["steps"]) == 2
    assert all(len(row["probes"]) >= 2 for row in data["steps"])
    assert all(row["consolidated"] and row["delta_after"] == 0 for row in data["steps"])
    trace = Trace.from_dict(data["trace"])
    assert trace.to("cpu").num_tokens == len(data["targets"])
    assert any(u.refs for u in trace.updates)
    assert any(len(b.requests) >= 2 for b in trace.batches)
    assert {i for i, t in enumerate(data["targets"]) if t != -100} == {
        s["row"] for s in data["samples"]
    }
    assert all(data["targets"][s["row"]] == s["token"] for s in data["samples"])
    assert data["steps"][0]["current_frame_hash"] == data["steps"][0]["previous_frame_hash"]
    assert data["steps"][1]["current_frame_hash"] == data["steps"][0]["next_frame_hash"]


def test_continue_is_bounded_by_force_action(tmp_path):
    data = record(tmp_path, True)
    assert all(row["probes"][-1]["forced_action"] for row in data["steps"])
    assert len(data["steps"]) == 2


def test_blank_control_replaces_model_frames_but_keeps_live_game(tmp_path, monkeypatch):
    original = Processor.apply_chat_template
    seen = []

    def check(self, messages, **kwargs):
        content = messages[0]["content"]
        if isinstance(content, list):
            assert len(content) == 2
            assert all(not row["image"].any() for row in content)
            seen.append(True)
        return original(self, messages, **kwargs)

    monkeypatch.setattr(Processor, "apply_chat_template", check)
    data = record(tmp_path, control="blank")
    assert len(seen) == 2
    assert data["steps"][1]["game_tick"] > data["steps"][0]["game_tick"]
    observations = [e for e in data["events"] if e["event"] == "observation"]
    assert all(e["current_frame_hash"] not in e["model_pixel_hashes"] for e in observations)


def test_cpu_toy_multimodal_replay_backprop(tmp_path, torch_reference_kernels):
    from minisgl_examples.training.loss import compute_supervised_loss

    data = record(tmp_path)
    trace = Trace.from_dict(data["trace"]).to("cpu")
    model = tiny_image_model()
    before = model.lm_head.weight.detach().clone()
    targets = torch.tensor(data["targets"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = compute_supervised_loss(model, trace, targets, 32) / int((targets != -100).sum())
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.model.visual.parameters()
    )
    optimizer.step()
    assert not torch.equal(before, model.lm_head.weight)
