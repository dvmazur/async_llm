"""Test-only FP8 serving adapter for the isolated model worker.

Unlike its shared-cache adapter, this uses the upstream LLM scheduler and ordinary
per-request KV/GDN state. One fresh prompt arrives per scheduler tick, alongside
already decoding requests. No scheduler or model math is replaced. The sampler
records original logits, then supplies fixture tokens so both schedules see the
same histories. This is a correctness test, NOT a throughput benchmark.

Use the same text fixtures with ``--mini-scheduling mixed`` and ``sequential``.
The output format also works with the existing SGLang/Transformers adapters and
``comparison.py``. Images and shared-cache mixed GDN are NOT covered: the
upstream offline LLM interface only accepts text/token IDs.
"""

from pathlib import Path
import sys

import torch


class TeacherRecorder:
    """Route logits by request UID, never by its changing position in a batch."""

    def __init__(self, cases, label="decode"):
        self.cases = cases
        self.label = label
        self.values = [[] for _ in cases]
        self.forwards = []
        self.current = None

    def before_forward(self, batch):
        from minisgl.scheduler.prefill import ChunkedReq

        rows = []
        for i, req in enumerate(batch.reqs):
            assert not isinstance(req, ChunkedReq), "GDN serving does not support chunked prefill"
            case = self.cases[req.uid]
            step = len(self.values[req.uid])
            assert step < len(case["teacher_tokens"]), "extra forward after the last teacher token"
            is_prefill = i < batch.num_prefill
            # These are snapshotted before Engine.complete_one() advances Req.
            assert is_prefill == (step == 0), "missing/duplicated prefill or wrong request mapping"
            assert req.device_len == len(case["prompt_ids"]) + step
            assert req.extend_len == (len(case["prompt_ids"]) if is_prefill else 1)
            rows.append(dict(uid=req.uid, case=case["name"], step=step,
                             phase="prefill" if is_prefill else "decode",
                             input_tokens=req.extend_len))
        assert len({r["uid"] for r in rows}) == len(rows), "duplicate request in a batch"
        self.current = dict(label=self.label, mixed=batch.is_mixed, rows=rows)

    def sample(self, logits, _sample_args):
        assert self.current is not None, "sampler called without a forward snapshot"
        rows = self.current["rows"]
        assert logits.ndim == 2 and logits.shape[0] == len(rows)
        # Save before forcing the next tokens; never overwrite the model logits.
        saved = logits.detach().cpu().clone()
        assert torch.isfinite(saved).all(), "non-finite model logits"
        tokens = []
        for i, row in enumerate(rows):
            uid, step = row["uid"], row["step"]
            token = self.cases[uid]["teacher_tokens"][step]
            assert 0 <= token < logits.shape[1]
            self.values[uid].append(saved[i])
            tokens.append(token)
        self.forwards.append(self.current)
        self.current = None
        return torch.tensor(tokens, dtype=torch.int32, device=logits.device)

    def finish(self):
        assert self.current is None
        assert all(len(rows) == len(case["teacher_tokens"])
                   for rows, case in zip(self.values, self.cases)), "incomplete teacher histories"
        return [torch.stack(rows) for rows in self.values]


def check_schedule(forwards, mode):
    """Fail rather than call an all-prefill run a successful mixed test."""
    rollout = [f for f in forwards if f["label"] == "decode"]
    assert rollout, "no rollout forwards recorded"
    mixed = [f for f in rollout if f["mixed"]]
    if mode == "mixed":
        assert mixed, "no mixed batch formed"
        for batch in mixed:
            phases = [r["phase"] for r in batch["rows"]]
            assert "prefill" in phases and "decode" in phases
        # We also want ordinary decode, not only the decode rows of mixed batches.
        assert any(not f["mixed"] and all(r["phase"] == "decode" for r in f["rows"])
                   for f in rollout), "no pure decode forward formed"
    else:
        assert mode == "sequential"
        assert not mixed and all(len(f["rows"]) == 1 for f in rollout)
    return dict(mixed_batches=len(mixed), total_batches=len(rollout),
                mixed_prefill_rows=sum(r["phase"] == "prefill" for f in mixed for r in f["rows"]),
                mixed_decode_rows=sum(r["phase"] == "decode" for f in mixed for r in f["rows"]))


@torch.inference_mode()
def mini_serving(args, cases):
    from .common import ROOT, save_json

    assert args.quantization == "fp8", "serving parity must explicitly enable FP8"
    assert cases and all("image_tensors" not in case for case in cases), (
        "Serving mixed parity is text-only; select --cases text_0 text_1 ... explicitly"
    )
    assert len({c["name"] for c in cases}) == len(cases)
    if args.mini_scheduling == "mixed":
        assert len(cases) >= 2 and all(len(c["teacher_tokens"]) >= 2 for c in cases)
    assert not any((args.audit_linear, args.audit_moe, args.reference_quant_outputs,
                    args.capture_prefill)), "module observers belong to the shared-cache adapter"
    repo = (args.mini_repo or ROOT).resolve()
    sys.path.insert(0, str(repo / "python"))
    from minisgl.core import SamplingParams
    from minisgl.llm import LLM
    from minisgl.env import ENV

    class ArrivingLLM(LLM):
        def offline_receive_msg(self, blocking=False):
            # Only emulate arrivals; leave admission, budget, mixing and execution
            # to the upstream scheduler. Preserve the original UID/status handling.
            deferred = self.pending_requests[1:]
            self.pending_requests = self.pending_requests[:1]
            try:
                return super().offline_receive_msg(blocking)
            finally:
                self.pending_requests.extend(deferred)

        def _forward(self, forward_input):
            self.recorder.before_forward(forward_input.batch)
            return super()._forward(forward_input)

    assert Path(sys.modules[LLM.__module__].__file__).resolve().is_relative_to(repo)
    max_length = max(len(c["prompt_ids"]) + len(c["teacher_tokens"]) for c in cases)
    # No radix prefix reuse or chunking: ordinary upstream GDN supports neither.
    # This still uses each request's real KV and recurrent cache during decode.
    llm = ArrivingLLM(
        args.model, quantization="fp8", max_running_req=len(cases), cache_type="naive",
        max_extend_tokens=max_length + len(cases), max_seq_len_override=max_length + 16,
        num_page_override=(max_length + 16) * (len(cases) + 1),
        attention_backend="fi", use_pynccl=False, cuda_graph_bs=[], cuda_graph_max_bs=0,
        distributed_addr=f"tcp://127.0.0.1:{args.port}",
    )
    # FLA import may initialize CUDA; Engine must own the first initialization.
    import minisgl.models.qwen3_5_delta as gdn

    original_sample = llm.engine.sampler.sample
    original_fla = gdn._fla_chunk, gdn._fla_recurrent
    forwards = []
    try:
        assert llm.engine.ctx.gdn_state is not None, "expected a hybrid Qwen GDN checkpoint"
        if args.disable_fla:
            gdn._fla_chunk = gdn._fla_recurrent = None
        weights = llm.engine.model.state_dict()
        fp8_count = sum(w.dtype == torch.float8_e4m3fn for w in weights.values())
        assert fp8_count > 0, "test accidentally loaded unquantized weights"
        save_json(args.output / "storage.json", dict(
            fp8_tensors=fp8_count, bytes=sum(w.numel() * w.element_size() for w in weights.values()),
            source=str(repo), fla_recurrent=gdn._fla_recurrent is not None))

        def generate(group, label):
            llm.recorder = TeacherRecorder(group, label)
            llm.engine.sampler.sample = llm.recorder.sample
            params = [SamplingParams(temperature=0, ignore_eos=True,
                                     max_tokens=len(c["teacher_tokens"])) for c in group]
            outputs = llm.generate([c["prompt_ids"] for c in group], params)
            values = llm.recorder.finish()
            for case, output in zip(group, outputs):
                assert output["token_ids"] == case["teacher_tokens"], "teacher forcing did not reach scheduler output"
            forwards.extend(llm.recorder.forwards)
            return values

        groups = [cases] if args.mini_scheduling == "mixed" else [[c] for c in cases]
        for group in groups:
            for case, values in zip(group, generate(group, "decode")):
                torch.save(values, args.output / f"{case['name']}_decode.pt")
        coverage = check_schedule(forwards, args.mini_scheduling)
        for case in cases:
            for step in case["cold_steps"]:
                cold = dict(case, prompt_ids=case["prompt_ids"] + case["teacher_tokens"][:step],
                            teacher_tokens=case["teacher_tokens"][step:step + 1])
                values = generate([cold], f"cold{step}")[0]
                torch.save(values[0], args.output / f"{case['name']}_cold{step}.pt")
        save_json(args.output / "schedule.json", dict(
            mode=args.mini_scheduling, path="ordinary-serving", text_only=True,
            overlap_scheduling=not bool(ENV.DISABLE_OVERLAP_SCHEDULING),
            **coverage, forwards=forwards))
    finally:
        llm.engine.sampler.sample = original_sample
        gdn._fla_chunk, gdn._fla_recurrent = original_fla
        llm.shutdown()
