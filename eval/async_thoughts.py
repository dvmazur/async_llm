"""Gemma framing and metrics around the repository's actual async_thoughts demo.

Runs the canonical training demo, including its sampled-target recorder.
"""
from contextlib import nullcontext
from pathlib import Path
import time

from minisgl_examples.demos.async_thoughts.demo import DemoConfig, Prompting, _run_loop

UPSTREAM_COMMIT = "be911914b9f113cd75fb15907bfb9718cbfeeac9"
UPSTREAM_PATH = "scripts/minisgl_examples/demos/async_thoughts"


class GemmaPrompting(Prompting):
    """Keep the demo's prose/probe unchanged; replace Qwen's control markers."""

    def __init__(self, problem, tokenizer, stop_ids):
        super().__init__(problem)
        close = "<turn|>\n"

        def user_prefix(text, thinking):
            rendered = tokenizer.apply_chat_template([dict(role="user", content=text)],
                tokenize=False, add_generation_prompt=False, enable_thinking=thinking)
            if not rendered.endswith(close):
                raise ValueError("Unsupported Gemma chat template ending")
            return rendered[:-len(close)].rstrip("\n") + "\n"

        self.input_prompt = user_prefix(problem, True)
        self.thinker_output_prefix = "<turn|>\n<|turn>model\n<|channel>thought"
        self.writer_output_prefix = self.writer_output_prefix.replace("</think>", "<channel|>")
        probe_text = self.mode_switching_prompt.removeprefix("<|im_start|>user\n")
        self.mode_switching_prompt = user_prefix(probe_text, False)
        self.forbidden_tokens = ("<|turn>", "<|channel>", "<|tool_call>")
        self.end_think_token = "<channel|>"
        self.stop_ids = set(stop_ids)
        vocab = tokenizer.get_vocab()
        for marker in ("<turn|>", "<|turn>", "<|channel>", "<channel|>"):
            if marker not in vocab:
                raise ValueError(f"Gemma tokenizer lacks {marker}")


def live_token_bound(tokenizer, problem, config):
    prompting = GemmaPrompting(problem, tokenizer, set())
    texts = (prompting.input_prompt, prompting.thinker_output_prefix, prompting.writer_output_prefix,
             prompting.mode_switching_prompt, prompting.mode_switching_question)
    prefix = sum(len(tokenizer.encode(text, add_special_tokens=False)) for text in texts)
    # Five block tails may round up to page boundaries. Two seeded newline inputs
    # and the last pending thinker flush are conservative additional reservations.
    return prefix + config["max_new_tokens"] + 3 + 5 * (config["engine"]["page_size"] - 1)


def trace_summary(trace):
    """Session batches, not physical forwards: prefill can be chunked internally."""
    result = {"steps": trace.steps, "input_rows": trace.num_tokens}
    for kind in ("prefill", "decode"):
        batches = [batch for batch in trace.batches if batch.kind == kind]
        requests = sum(len(batch.requests) for batch in batches)
        result[kind] = dict(batches=len(batches), requests=requests,
            input_rows=sum(len(r.input_ids) for b in batches for r in b.requests),
            mean_requests=requests / len(batches) if batches else 0)
    return result


async def solve(llm, problem, config, sampling_params, stop_ids, *, on_progress=None,
                context_factory=None, trace_path=None):
    prompting = GemmaPrompting(problem, llm.tokenizer, stop_ids)
    demo = DemoConfig(problem=problem, max_steps=config["max_think_tokens"],
                      max_answer_tokens=config["max_answer_tokens"], probe_period=config["probe_period"])
    metrics = {}
    if trace_path is not None:
        from minisgl.training import RecordedAsyncLLM
        from minisgl_examples.training.recording import save_training_example

        trace_path = Path(trace_path)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        if trace_path.exists():
            raise FileExistsError(trace_path)
    scope = RecordedAsyncLLM(llm) if trace_path is not None else nullcontext(llm)
    samples = {} if trace_path is not None else None
    started = time.perf_counter()
    with scope as recorded:
        thought, answer = await _run_loop(demo, recorded, llm.tokenizer, prompting, samples,
            sampling_params=sampling_params, metrics=metrics, on_progress=on_progress,
            verbose=False, context_factory=context_factory)
    elapsed = time.perf_counter() - started
    training_trace = None
    if trace_path is not None:
        export_started = time.perf_counter()
        save_training_example(recorded, samples, trace_path, model=config["model_path"],
            problem=problem, thinker_tokens=thought, writer_tokens=answer)
        training_trace = dict(path=str(trace_path), **trace_summary(recorded.get_trace()),
                              export_s=time.perf_counter() - export_started)
    raw = llm.tokenizer.decode(answer, skip_special_tokens=False)
    return dict(completion_text=raw, answer_text=raw, token_ids=answer,
                thinker_token_ids=thought, thinker_text=llm.tokenizer.decode(thought, skip_special_tokens=False),
                generated_tokens=sum(metrics["sampled"].values()),
                prompt_tokens=metrics["prefix_tokens"], probe_tokens=metrics["probe_tokens"],
                stop_reason=metrics["writer_stop_reason"], elapsed_s=elapsed,
                async_thoughts=metrics, training_trace=training_trace)
