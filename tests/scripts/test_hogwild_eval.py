"""CPU checks for evaluation validity and reproducible experiment artifacts."""

import json
from decimal import Decimal

import pytest
from minisgl_examples.training.parallel_thinking import eval as evaluation

GPU_UUID = "GPU-95b1d0e0-af9a-b75e-ac8d-2acee2a25add"
OTHER_GPU_UUID = "GPU-b472d02d-14a6-40dc-83a8-79a5b400033d"


def mock_gpu(monkeypatch, index, *, uuid=GPU_UUID, memory=0, utilization=0, processes=""):
    def check_output(args, *, text):
        assert args[0] == "nvidia-smi" and "-i" in args
        if "--query-compute-apps=pid" in args:
            return processes
        return f"{index}, {uuid}, NVIDIA B200, {memory}, 183359, {utilization}\n"

    monkeypatch.setattr(evaluation.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(evaluation.subprocess, "check_output", check_output)


@pytest.mark.parametrize("selected,index", [(None, 7), ("0", 0), ("5", 5)])
@pytest.mark.parametrize("memory", [0, 5, 8])
def test_gpu_preflight_accepts_selected_idle_device(monkeypatch, selected, index, memory):
    if selected is None:
        monkeypatch.delenv("HOGWILD_GPU", raising=False)
    else:
        monkeypatch.setenv("HOGWILD_GPU", selected)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU_UUID)
    mock_gpu(monkeypatch, index, memory=memory)

    metadata = evaluation.gpu_preflight()

    assert int(metadata["index"]) == index
    assert metadata["uuid"] == GPU_UUID
    assert int(metadata["memory_used_mib"]) == memory


@pytest.mark.parametrize("visible", ["", "5", GPU_UUID[:12], GPU_UUID + "," + OTHER_GPU_UUID])
def test_gpu_preflight_requires_one_complete_visible_uuid(monkeypatch, visible):
    monkeypatch.setenv("HOGWILD_GPU", "5")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    mock_gpu(monkeypatch, 5)
    with pytest.raises(RuntimeError):
        evaluation.gpu_preflight()


@pytest.mark.parametrize("index,uuid", [(7, GPU_UUID), (5, OTHER_GPU_UUID)])
def test_gpu_preflight_refuses_mismatched_physical_identity(monkeypatch, index, uuid):
    monkeypatch.setenv("HOGWILD_GPU", "5")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU_UUID)
    mock_gpu(monkeypatch, index, uuid=uuid)
    with pytest.raises(RuntimeError):
        evaluation.gpu_preflight()


@pytest.mark.parametrize("occupancy", [
    {"memory": -1}, {"memory": 9}, {"memory": 128},
    {"memory": 5, "utilization": 1}, {"memory": 5, "processes": "72142\n"},
])
def test_gpu_preflight_refuses_occupied_device(monkeypatch, occupancy):
    monkeypatch.setenv("HOGWILD_GPU", "5")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU_UUID)
    mock_gpu(monkeypatch, 5, **occupancy)
    with pytest.raises(RuntimeError, match="occupied"):
        evaluation.gpu_preflight()


@pytest.mark.parametrize("value", [None, True, "", "answer is 42", r"\frac{84}{2}",
                                 "42 or 43", "4,200", "NaN", "inf", "1e2"])
def test_numeric_grading_does_not_recover_digits_from_nonanswers(value):
    assert evaluation.numeric_answer(value) is None


def test_numeric_grading_preserves_precision_and_sign():
    assert evaluation.numeric_answer(" $-0.125$ ") == Decimal("-0.125")
    assert evaluation.numeric_answer("+0042.000") == evaluation.numeric_answer(42)
    assert evaluation.numeric_answer("9007199254740993") != evaluation.numeric_answer(
        "9007199254740992"
    )


@pytest.mark.parametrize("dataset", ["math500", "limo"])
def test_selection_is_explicit_ordered_and_validated(dataset):
    rows = [
        {"problem": f"Problem {i}", "answer": r"\frac{3}{7}", "unique_id": f"test/{i}"}
        if dataset == "math500" else {"question": f"Question {i}", "answer": str(i)}
        for i in range(4)
    ]
    assert evaluation.select_rows(rows, [3, 1], start=0, limit=1, dataset=dataset) == [
        (3, rows[3]), (1, rows[1])
    ]
    assert evaluation.select_rows(rows, None, start=2, limit=9, dataset=dataset) == [
        (2, rows[2]), (3, rows[3])
    ]
    for indices in ([], [1, 1], [-1], [4]):
        with pytest.raises(ValueError):
            evaluation.select_rows(rows, indices, start=0, limit=1, dataset=dataset)


def test_limo_requires_a_numeric_gold_answer():
    with pytest.raises(ValueError):
        evaluation.select_rows(
            [{"question": "Valid?", "answer": "answer: 42"}], None, 0, 1, dataset="limo"
        )


def test_model_thinking_defaults_require_no_unimplemented_penalties():
    params = evaluation.sampling_defaults({
        "do_sample": True, "temperature": 1.0, "top_p": 0.95, "top_k": 20,
        "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0,
    })
    assert (params.temperature, params.top_p, params.top_k) == (1.0, 0.95, 20)
    assert not params.is_greedy
    assert params.max_tokens == 1


@pytest.mark.parametrize("name,value", [
    ("presence_penalty", 1.5), ("repetition_penalty", 1.1), ("min_p", 0.05),
])
def test_non_neutral_unsupported_sampling_is_not_silently_ignored(name, value):
    with pytest.raises(ValueError, match=name):
        evaluation.sampling_defaults({name: value})


@pytest.mark.parametrize("dataset,workers,expected_context,cuda_graphs,custom_finisher,causal,publication_arg", [
    ("math500", [1, 2], 40960, False, None, False, None),
    ("limo", [16], 262144, False, None, False, "--no-paragraph-publication"),
    ("math500", [1], 24576, True, 23, True, "--paragraph-publication"),
])
def test_eval_cli_grades_dataset_answers_and_saves_per_task_metrics(
    tmp_path, monkeypatch, dataset, workers, expected_context, cuda_graphs, custom_finisher,
    causal, publication_arg,
):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({
        "text_config": {"max_position_embeddings": 262144},
    }))
    (model / "generation_config.json").write_text(json.dumps({
        "do_sample": True, "temperature": 1.0, "top_p": 0.95, "top_k": 20,
        "eos_token_id": [248046, 248044],
    }))
    dataset_path = tmp_path / f"{dataset}.jsonl"
    if dataset == "math500":
        rows = [
            {"unique_id": "test/first", "problem": "First", "answer": r"\frac{14}{3}"},
            {"unique_id": "test/second", "problem": "Second", "answer": r"\sqrt{7}"},
        ]
    else:
        rows = [{"question": "First", "answer": "42"}, {"question": "Second", "answer": "9"}]
    dataset_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    downloads, engines, rollouts = [], [], []

    def snapshot_download(repo, **kwargs):
        downloads.append((repo, kwargs))
        return str(model)

    def hf_hub_download(repo, filename, **kwargs):
        assert dataset == "limo"
        downloads.append((repo, {"filename": filename, **kwargs}))
        return str(dataset_path)

    def load_math500():
        assert dataset == "math500"
        return dataset_path, rows

    class FakeLLM:
        tokenizer = object()

        def __init__(self, path, **options):
            self.options, self.closed = options, False
            engines.append(self)

        async def close(self):
            self.closed = True

    async def run_agents(llm, formatter, **options):
        rollouts.append(options)
        first = options["problem"].endswith("First")
        prediction = (r"\dfrac{28}{6}" if dataset == "math500" else "42.0") if first else None
        per_worker = dict.fromkeys(formatter, 5)
        published = 2 if options["publish_paragraphs"] and first else 0
        return {
            "boxed_answer": prediction,
            "transcript": "Synthetic test transcript",
            "answer_text": prediction or "",
            "stop_reason": "eos" if first else "context_limit",
            "eos_worker": formatter[0] if first else None,
            "eos_token_id": 248046 if first else None,
            "completed_steps": published,
            "first_publication_round": 2 if published else None,
            "decode_rounds": 5,
            "total_generated_tokens": 5 * len(formatter) + 2,
            "reasoning_generated_tokens": 5 * len(formatter),
            "answer_generated_tokens": 2,
            "generated_tokens": per_worker,
            "peak_context_tokens": 100,
            "used_finisher": True,
            "finisher_mode": "cached",
            "finisher_stop_reason": "answer" if first else "eos",
            "finisher_eos_token_id": None if first else 248044,
        }

    monkeypatch.setattr(evaluation, "snapshot_download", snapshot_download)
    monkeypatch.setattr(evaluation, "hf_hub_download", hf_hub_download)
    monkeypatch.setattr(evaluation.math500, "load_math500", load_math500)
    monkeypatch.setattr(evaluation, "AsyncLLM", FakeLLM)
    monkeypatch.setattr(evaluation, "MathFormatting", lambda tokenizer, names, **kwargs: names)
    monkeypatch.setattr(evaluation, "run_agents", run_agents)
    monkeypatch.setattr(evaluation, "gpu_preflight", lambda: {"index": "7", "uuid": "test"})
    monkeypatch.setattr(evaluation.torch, "manual_seed", lambda seed: None)
    monkeypatch.setattr(evaluation.torch.cuda, "manual_seed_all", lambda seed: None)
    monkeypatch.setattr(evaluation.subprocess, "check_output", lambda command, **kwargs:
                        "hogwild-multi-workers\n" if "branch" in command else "test-commit\n")
    output = tmp_path / "output"
    argv = ["--workers", *map(str, workers), "--single-worker-layout", "legacy", "--limit", "2", "--output", str(output)]
    if dataset == "limo":
        argv += ["--dataset", "limo"]
    if not cuda_graphs:
        argv.append("--no-cuda-graphs")
    if custom_finisher is not None:
        argv += ["--finisher-tokens", str(custom_finisher)]
    if causal:
        argv.append("--causal-headers")
    if publication_arg:
        argv.append(publication_arg)
    evaluation.main(argv)

    manifest = json.loads((output / "config.json").read_text())
    records = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    completed = json.loads((output / "completed.json").read_text())
    assert downloads[0][1]["revision"] == manifest["model_revision"]
    if dataset == "math500":
        assert manifest["dataset_revision"] == evaluation.math500.DATASET_REVISION
        assert manifest["dataset"] == evaluation.math500.DATASET
        assert [record["unique_id"] for record in records] == ["test/first", "test/second"] * len(workers)
    else:
        assert downloads[1][1]["revision"] == manifest["dataset_revision"]
        assert manifest["dataset"] == "GAIR/LIMO"
    assert manifest["indices"] == [0, 1]
    assert manifest["context_cap"] == expected_context
    assert manifest["budget_unit"] == "synchronized_decode_rounds_per_worker"
    expected_finisher = custom_finisher or (128 if dataset == "math500" else 16)
    assert manifest["finisher_tokens_outside_budget"] == expected_finisher
    assert manifest["enable_thinking"] is True
    multiple = max(workers) > 1
    assert manifest["multi_worker_layout"] == ("native" if multiple else None)
    assert manifest["multi_worker_answer_cue"] == ("qwen" if multiple else None)
    assert manifest["pending_eos_in_finisher"] is False
    assert [opt["native_workers"] for opt in rollouts] == [n > 1 for n in workers for _ in range(2)]
    assert [opt["native_answer_cue"] for opt in rollouts] == [n > 1 for n in workers for _ in range(2)]
    publishes = publication_arg != "--no-paragraph-publication"
    assert manifest["paragraph_publication"] is publishes
    assert manifest["sampling"]["temperature"] == 1.0
    assert manifest["sampling"]["top_p"] == 0.95
    assert len(engines) == 1 and engines[0].closed
    assert engines[0].options["max_seq_len_override"] == expected_context
    if cuda_graphs:
        # Even a single reasoner initializes two header blocks concurrently.
        # Shared prefill graph request capacity is max(cuda_graph_bs).
        assert max(engines[0].options["cuda_graph_bs"]) >= 2
        assert engines[0].options["cuda_graph_max_bs"] >= 2
    assert all(options["max_context_tokens"] == expected_context for options in rollouts)
    assert all(options["eos_token_ids"] == [248046, 248044] for options in rollouts)
    assert all(options["answer_steps"] == expected_finisher for options in rollouts)
    assert all(options["symbolic_answers"] == (dataset == "math500") for options in rollouts)
    assert all(options["cached_finisher"] is True for options in rollouts)
    assert all(options["causal_headers"] == causal for options in rollouts)
    assert all(options["publish_paragraphs"] is publishes for options in rollouts)
    assert [record["correct"] for record in records] == [True, False] * len(workers)
    if dataset == "math500":
        assert [record["grading_status"] for record in records] == [
            "correct", "unparsed_prediction"
        ] * len(workers)
    assert [record["generated_tokens"] for record in records] == [
        5 * count + 2 for count in workers for _ in range(2)
    ]
    assert all(record["decoding_steps"] == 5 for record in records)
    assert all(record["total_decoding_steps"] == 7 for record in records)
    assert [record["published_paragraphs"] for record in records] == [
        2 if publishes else 0, 0
    ] * len(workers)
    assert [record["first_publication_round"] for record in records] == [
        2 if publishes else None, None
    ] * len(workers)
    assert [record["eos_token_id"] for record in records] == [248046, None] * len(workers)
    assert [record["finisher_stop_reason"] for record in records] == ["answer", "eos"] * len(workers)
    assert [record["finisher_eos_token_id"] for record in records] == [None, 248044] * len(workers)
    assert all("transcript" not in record and "trace" not in record for record in records)
    assert not list(output.glob("*.txt"))
    assert completed["cases"] == 2 * len(workers)
    for count in workers:
        summary = completed["summary"][str(count)]
        assert (summary["problems"], summary["correct"], summary["accuracy"]) == (2, 1, 0.5)
        assert summary["context_limited"] == 1
        assert summary["generated_tokens"] == 2 * (5 * count + 2)
        assert summary["decoding_steps"] == 10
        assert summary["total_decoding_steps"] == 14
