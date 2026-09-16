"""Small multi-block FP8 parity, using one model load for all three layouts.

Run only this profile with pytest tests/e2e/fp8/test_parity.py -k chain
--fp8-model /path/to/small-Qwen-FP8. Three text prompts, eight teacher-forced
positions, three cold-prefill checkpoints. SGLang/Transformers see exactly the
concatenated causal history. No sibling cross-reading/reordering of cached
history: that has different semantics and cannot use a flat HF reference.
Use -k bf16_chain --chain-bf16-model /path/to/Qwen-BF16 for the identical
non-quantized control. No FP8 tensors/quantization option are allowed there.
"""
import hashlib
import json
from pathlib import Path
import sys
from contextlib import contextmanager

from .common import ROOT, save_json

LAYOUTS = ("chain-flat", "chain-split", "chain-shared")


def check_moe_execution(storage, *, require_mixed=False):
    assert storage["num_experts"] > 0, "expected MoE, not dense"
    assert storage["fp8_expert_tensors"] > 0, "expected serialized FP8 expert weights"
    assert storage["moe_fp8_prefill_calls"] > 0, "no FP8 experts executed during prefill"
    assert storage["moe_fp8_decode_calls"] > 0, "no FP8 experts executed during decode"
    if require_mixed:
        assert storage["moe_fp8_mixed_calls"] > 0, "no FP8 experts executed in mixed batches"


@contextmanager
def count_moe_calls(enabled):
    """Observe native expert execution, without changing routes or outputs."""
    counts = dict(prefill=0, decode=0, mixed=0)
    if not enabled:
        yield counts
        return
    import torch
    from minisgl.core import get_global_ctx
    import minisgl.moe.fused as moe
    original = moe.fused_experts_impl
    def observed(x, w1, w2, *args, **kwargs):
        assert w1.dtype == w2.dtype == torch.float8_e4m3fn, "experts are not FP8"
        assert kwargs.get("w1_scale") is not None and kwargs.get("w2_scale") is not None
        batch = get_global_ctx().batch
        phase = "mixed" if getattr(batch, "is_mixed", False) else "prefill" if batch.is_prefill else "decode"
        counts[phase] += 1
        return original(x, w1, w2, *args, **kwargs)
    moe.fused_experts_impl = observed
    try:
        yield counts
    finally:
        moe.fused_experts_impl = original


def fixtures(tokenizer, tokens):
    prefix = ("Use the observations below to distinguish facts from assumptions. "
              "An agent is exploring an unfamiliar room and can inspect it again. ")
    questions = ["A door is closed. What should the agent verify?",
                 "The camera moved but the box stayed in the same place. What does this show?",
                 "A blue marker disappeared behind a wall. The corridor has two exits. "
                 "Explain what remains uncertain before choosing an exit."]
    teacher = tokenizer.encode(("The observation supports only what is visible. " * 4), add_special_tokens=False)[:tokens]
    assert len(teacher) == tokens and tokens >= 4
    cases = []
    for i, question in enumerate(questions):
        prompt = tokenizer.apply_chat_template([dict(role="user", content=prefix + question)],
            add_generation_prompt=True, enable_thinking=False, tokenize=False)
        cases.append(dict(name=f"chain_{i}", prompt_ids=tokenizer.encode(prompt, add_special_tokens=False),
                          teacher_tokens=teacher, cold_steps=[0, tokens // 2 - 1, tokens - 1]))
    return cases


def _parts(ids, count):
    assert len(ids) >= count
    return [ids[len(ids)*i//count:len(ids)*(i+1)//count] for i in range(count)]


def plan(cases, layout, step=0):
    """CPU-only topology. Each node's parents are its EXACT causal prefix."""
    assert layout in LAYOUTS and len(cases) == 3
    histories = [c["prompt_ids"] + c["teacher_tokens"][:step] for c in cases]
    nodes, chains = {}, []
    common = 0
    if layout == "chain-shared":
        # Never share generated answers; only a genuinely identical prompt prefix.
        for row in zip(*(c["prompt_ids"] for c in cases)):
            if len(set(row)) != 1:
                break
            common += 1
        common = min(common, 32)
        assert common >= 4 and all(len(h) > common + 2 for h in histories)
        nodes["common"] = dict(ids=histories[0][:common], parents=[])
    for i, history in enumerate(histories):
        chain = ["common"] if common else []
        count = 3 if layout == "chain-split" else 2 if common and i == 2 else 1
        for j, ids in enumerate(_parts(history[common:], count)):
            name = f"worker{i}.{j}"
            nodes[name] = dict(ids=ids, parents=list(chain))
            chain.append(name)
        if common and i > 0:
            name = f"worker{i}.answer"
            nodes[name] = dict(ids=[], parents=list(chain))
            chain.append(name)
        assert [t for name in chain for t in nodes[name]["ids"]] == history
        chains.append(chain)
    return nodes, chains


def _digest(ids):
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def materialize(session, cases, layout, step, blocks, job_type):
    """Construct real blocks by dependency depth, never prefill dependent jobs together."""
    import torch

    nodes, chains = plan(cases, layout, step)
    for name in nodes:
        blocks[name] = session.create_block()
    logits_by_node = {}
    for depth in range(max(len(n["parents"]) for n in nodes.values()) + 1):
        names = [name for name, n in nodes.items() if len(n["parents"]) == depth and n["ids"]]
        if not names:
            continue
        jobs = [job_type(blocks[name], torch.tensor(nodes[name]["ids"], dtype=torch.int32),
                         context=[blocks[parent] for parent in nodes[name]["parents"]]) for name in names]
        # Preserve returned logits before the next forward reuses graph buffers.
        for name, value in zip(names, session.prefill_batch(jobs)):
            logits_by_node[name] = value.detach().cpu().clone()
    last = [next(name for name in reversed(chain) if nodes[name]["ids"]) for chain in chains]
    return chains, torch.cat([logits_by_node[name] for name in last])


def snapshot(cases, blocks, chains, step, order):
    rows = []
    for i in order:
        live = [blocks[name] for name in chains[i]]
        ids = [t for block in live for t in block.token_ids]
        expected = cases[i]["prompt_ids"] + cases[i]["teacher_tokens"][:step]
        assert ids == expected, "block history differs from the flat reference"
        rows.append(dict(case=cases[i]["name"], step=step,
            block_ids=[id(b) for b in live], lengths=[b.num_tokens for b in live],
            write_id=id(live[-1]), history_sha256=_digest(ids)))
    return rows


def check_coverage(coverage, cases, layout):
    assert coverage["layout"] == layout
    assert coverage["decode_forward_calls"] == len(cases[0]["teacher_tokens"]) - 1
    assert len(coverage["forwards"]) == len(cases[0]["teacher_tokens"]), "missing/extra forward snapshots"
    expected = {c["name"]: c for c in cases}
    rows_by_case = {name: [] for name in expected}
    for forward in coverage["forwards"]:
        assert len(forward) == len(cases) and {r["case"] for r in forward} == set(expected)
        for row in forward:
            case, rows = expected[row["case"]], rows_by_case[row["case"]]
            step = len(rows)
            assert row["step"] == step
            assert row["history_sha256"] == _digest(case["prompt_ids"] + case["teacher_tokens"][:step])
            assert len(row["block_ids"]) == len(row["lengths"]) and len(set(row["block_ids"])) == len(row["block_ids"])
            assert row["write_id"] == row["block_ids"][-1]
            assert sum(row["lengths"]) == len(case["prompt_ids"]) + step
            if rows:
                previous = rows[-1]
                assert row["block_ids"] == previous["block_ids"], "unexpected block replacement"
                assert row["lengths"][:-1] == previous["lengths"][:-1], "immutable prefix was modified"
                assert row["lengths"][-1] == previous["lengths"][-1] + 1
            rows.append(row)
    assert all(len(rows) == len(expected[name]["teacher_tokens"]) for name, rows in rows_by_case.items())
    first = [rows_by_case[c["name"]][0] for c in cases]
    depths = [len(r["block_ids"]) for r in first]
    if layout == "chain-flat":
        assert depths == [1, 1, 1]
    elif layout == "chain-split":
        assert depths == [3, 3, 3] and all(min(r["lengths"]) > 0 for r in first)
    else:
        assert layout == "chain-shared" and depths == [2, 3, 4]
        assert len({r["block_ids"][0] for r in first}) == 1, "prefix is copied, not shared"
        assert first[0]["lengths"][-1] > 0 and [r["lengths"][-1] for r in first[1:]] == [0, 0]
    # No private tail may alias another worker's tail.
    private_ids = [block for r in first for block in r["block_ids"][int(layout == "chain-shared"):]]
    assert len(private_ids) == len(set(private_ids))
    assert len({tuple(r["case"] for r in f) for f in coverage["forwards"]}) > 1, "worker order never changed"


def mini_chains(args, cases):
    import torch
    from minisgl.engine import Engine, EngineConfig
    from minisgl.distributed import DistributedInfo
    from minisgl.shared_cache import SharedCacheSession, PrefillJob, WorkerGroup
    from transformers import GenerationConfig

    assert args.quantization in (None, "fp8") and all("image_tensors" not in c for c in cases)
    source = args.mini_repo or ROOT
    assert Path(sys.modules[Engine.__module__].__file__).resolve().is_relative_to(source)
    engine = Engine(EngineConfig(model_path=args.model, tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16, quantization=args.quantization, max_running_req=4,
        num_page_override=4096, max_seq_len_override=1024, attention_backend="fi",
        use_pynccl=False, cuda_graph_bs=[], cuda_graph_max_bs=0, distributed_addr=f"tcp://127.0.0.1:{args.port}",
        generation_config=GenerationConfig(do_sample=False, temperature=None, top_k=None, top_p=None)))
    import minisgl.models.qwen3_5_delta as gdn
    original_fla = gdn._fla_chunk, gdn._fla_recurrent
    if args.disable_fla:
        gdn._fla_chunk = gdn._fla_recurrent = None
    try:
        with torch.inference_mode(), count_moe_calls(args.require_moe) as moe_calls:
            session = SharedCacheSession(engine)
            assert session.graph_runner is None, "chain profile explicitly selects eager execution"
            weights = engine.model.state_dict()
            storage = dict(source=str(source), fp8_tensors=sum(w.dtype == torch.float8_e4m3fn for w in weights.values()),
                           bf16_tensors=sum(w.dtype == torch.bfloat16 for w in weights.values()),
                           num_experts=engine.model.model.config.num_experts,
                           fp8_expert_tensors=sum(w.ndim == 3 and w.dtype == torch.float8_e4m3fn for w in weights.values()),
                           execution="eager", fla_recurrent=gdn._fla_recurrent is not None)
            if args.quantization == "fp8":
                assert storage["fp8_tensors"] > 0
            else:
                assert storage["fp8_tensors"] == 0 and storage["bf16_tensors"] > 0

            for layout in LAYOUTS:
                output = args.output.parent / layout
                output.mkdir(exist_ok=False)
                start_moe_calls = dict(moe_calls)
                histories = [[] for _ in cases]
                coverage = dict(layout=layout, forwards=[])
                blocks = {}
                try:
                    chains, logits = materialize(session, cases, layout, 0, blocks, PrefillJob)
                    decode_calls = 0
                    coverage["forwards"].append(snapshot(cases, blocks, chains, 0, list(range(len(cases)))))
                    for i, row in enumerate(logits):
                        histories[i].append(row.clone())
                    for step in range(1, args.tokens):
                        # Rotate request order; histories are indexed by worker, not batch row.
                        order = [(i + step) % len(cases) for i in range(len(cases))]
                        group = WorkerGroup(cache_structure=[[blocks[n] for n in chains[i]] for i in order],
                                            write_to=[blocks[chains[i][-1]] for i in order])
                        logits = session.decode_step(group, torch.tensor(
                            [cases[i]["teacher_tokens"][step - 1] for i in order], dtype=torch.int32,
                            device=engine.device)).detach().cpu()
                        decode_calls += 1
                        coverage["forwards"].append(snapshot(cases, blocks, chains, step, order))
                        for i, row in zip(order, logits):
                            histories[i].append(row.clone())
                    coverage["decode_forward_calls"] = decode_calls
                    check_coverage(coverage, cases, layout)
                    save_json(output / "chain_coverage.json", coverage)
                    for case, history in zip(cases, histories):
                        torch.save(torch.stack(history), output / f"{case['name']}_decode.pt")
                finally:
                    for block in blocks.values():
                        session.free_block(block)
                for step in cases[0]["cold_steps"]:
                    blocks = {}
                    try:
                        chains, logits = materialize(session, cases, layout, step, blocks, PrefillJob)
                        snapshot(cases, blocks, chains, step, list(range(len(cases))))
                        for case, row in zip(cases, logits):
                            torch.save(row, output / f"{case['name']}_cold{step}.pt")
                    finally:
                        for block in blocks.values():
                            session.free_block(block)
                observed_storage = dict(storage,
                    moe_fp8_prefill_calls=moe_calls["prefill"] - start_moe_calls["prefill"],
                    moe_fp8_decode_calls=moe_calls["decode"] - start_moe_calls["decode"],
                    moe_fp8_mixed_calls=moe_calls["mixed"] - start_moe_calls["mixed"])
                if args.require_moe:
                    check_moe_execution(observed_storage)
                save_json(output / "storage.json", observed_storage)
                print("DONE", layout, flush=True)
    finally:
        gdn._fla_chunk, gdn._fla_recurrent = original_fla
        engine.shutdown()
