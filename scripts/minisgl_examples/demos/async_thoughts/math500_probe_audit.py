"""Reconstruct writer-gate behavior from a completed native-chat AsyncReasoning trace.

The input is the recorded primary trace, or its unmodified list of forward batches.
Verify the compressed trace's checksum against its row before calling this helper.
It only reads the trace. No model forwards, logits or counterfactual votes are used.

A standalone admission question has a private one-block view. Its repeated use
means the preceding question kept the writer waiting. Later questions use a
four-block view and a separately cached static prefix. A writer step before the
next question, while the thinker is still decoding, witnesses permission. No
writer step between consecutive questions is an operational wait. When writing
only resumes after the last thinker decode, thinker termination can release the
writer independently of the vote. Such intervals are unresolved. A WRITE can
also reaffirm an open gate. These counts measure executed questions, since timer
and paragraph triggers can coalesce. They do not recover scores or confidence.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from statistics import median


def audit_probe_schedule(
    primary: Mapping | Sequence,
    *,
    include_events: bool = False,
    writer_ids: Sequence[int] | None = None,
    tokenizer=None,
) -> dict:
    """Audit one native primary trace. Optionally check paragraph boundaries from actual writer output."""
    batches = primary["batches"] if isinstance(primary, Mapping) else primary
    thinker, writer, writer_inputs, prefills = [], [], [], []
    thinker_views, writer_views = set(), set()
    for step, batch in enumerate(batches):
        if batch["step"] != step:
            raise ValueError("Trace steps must be consecutive and zero based")
        for request in batch["requests"]:
            view = tuple(int(block) for block, _ in request["cache_view"])
            if not view or view[-1] != request["write_to"]:
                raise ValueError("Invalid trace view")
            if batch["kind"] == "prefill":
                prefills.append((step, view, len(request["input_ids"])))
            elif batch["kind"] == "decode" and len(view) == 2:
                thinker_views.add(view)
                thinker.append(step)
            elif batch["kind"] == "decode" and len(view) == 3:
                writer_views.add(view)
                writer.append(step)
                writer_inputs.extend(request["input_ids"])
            else:
                raise ValueError("Expected native thinker/writer decodes and probe prefills")
    if len(thinker_views) != 1 or len(writer_views) != 1:
        raise ValueError("Expected single persistent thinker and writer views")
    actor_blocks = writer_views.pop()
    if thinker_views.pop() != actor_blocks[:2] or len(set(actor_blocks)) != 3:
        raise ValueError("The thinker and writer do not have the native one-way visibility")
    prompt, thought, reply = actor_blocks
    static_prefixes = {view[0] for _, view, _ in prefills if len(view) == 4}
    questions = []
    static_prefills = 0
    for step, view, length in prefills:
        block = view[-1]
        if len(view) == 4 and view[1:3] == (thought, reply):
            questions.append((step, "live", length))
        elif len(view) == 1 and block not in (*actor_blocks, *static_prefixes):
            questions.append((step, "start", length))
        elif len(view) == 1 and block in static_prefixes:
            static_prefills += 1
        elif view not in ((prompt,), actor_blocks):
            raise ValueError("Unexpected prefill for the native-chat protocol")
    if any(a[0] >= b[0] for a, b in zip(questions, questions[1:])):
        raise ValueError("Expected separately ordered gate checks")
    if any(step >= writer[0] for step, kind, _ in questions if kind == "start"):
        raise ValueError("Standalone admission occurred after first writer decode")

    start, live = Counter(), Counter()
    events = []
    for pos, (step, kind, length) in enumerate(questions):
        following = questions[pos + 1] if pos + 1 < len(questions) else None
        end = following[0] if following is not None else len(batches)
        left, right = bisect_right(writer, step), bisect_left(writer, end)
        next_write = writer[left] if left < right else None
        if next_write is not None and next_write <= thinker[-1]:
            outcome = "write_while_thinking"
        elif next_write is not None:
            outcome = "unresolved_terminal_release"
        elif following is not None:
            outcome = "wait_until_next_probe"
        else:
            outcome = "unresolved_trace_end"
        counts = start if kind == "start" else live
        counts[outcome] += 1
        counts["calls"] += 1
        counts["question_input_tokens"] += length
        if kind == "start" and following is not None and following[1] == "start":
            counts["followed_by_repeated_standalone"] += 1
        if include_events:
            events.append(
                {
                    "step": step,
                    "kind": kind,
                    "outcome": outcome,
                    "next_probe_step": None if following is None else end,
                    "first_writer_step_in_interval": next_write,
                    "writer_steps_in_interval": right - left,
                    "thinker_steps_in_interval": bisect_left(thinker, end)
                    - bisect_right(thinker, step),
                }
            )
    thinker_set = set(thinker)
    joint = sum(step in thinker_set for step in writer)
    # Recorded views describe pre-step lengths. The same-batch thinker input is
    # visible to the writer. Native thinker's first input is the final prompt token.
    first_visible = bisect_right(thinker, writer[0]) - 1
    first_check = questions[0][0] if questions else None
    result = {
        "start": dict(start),
        "live": dict(live),
        "static_live_prefix_prefills": static_prefills,
        "first_probe_step": first_check,
        "first_writer_step": writer[0],
        "first_writer_while_thinking": writer[0] <= thinker[-1],
        "first_writer_joint": writer[0] in thinker_set,
        "first_writer_visible_generated_thinker_tokens": max(0, first_visible),
        "admitted_at_first_standalone": start["calls"] == 1 and start["write_while_thinking"] == 1,
        "decode": {
            "thinker": len(thinker),
            "writer": len(writer),
            "joint": joint,
            "writer_only_before_final_thinker": bisect_right(writer, thinker[-1]) - joint,
            "writer_only_after_final_thinker": len(writer) - bisect_right(writer, thinker[-1]),
        },
    }
    if include_events:
        result["events"] = events
    if (writer_ids is None) != (tokenizer is None):
        raise ValueError("Provide both actual writer IDs and their tokenizer for paragraph counts")
    if writer_ids is not None:
        if len(writer) - len(writer_ids) not in (0, 1) or writer_inputs[1:] != list(
            writer_ids[: len(writer_inputs) - 1]
        ):
            raise ValueError("Writer samples disagree with recorded decoder inputs")
        boundaries = Counter()
        for i in range(1, len(writer_ids)):
            if tokenizer.decode(list(writer_ids[i - 1 : i + 1])).endswith("\n\n"):
                # On a length stop the loop breaks before applying a paragraph pause.
                if i == len(writer) - 1:
                    boundaries["terminal_budget_before_pause"] += 1
                elif writer[i] < thinker[-1]:
                    boundaries["while_thinker_active"] += 1
                elif writer[i] == thinker[-1]:
                    boundaries["at_last_thinker_step_unresolved"] += 1
                else:
                    boundaries["after_last_thinker_step"] += 1
        result["writer_paragraph_boundaries"] = dict(boundaries)
    return result


def summarize_probe_schedules(audits: Iterable[dict]) -> dict:
    """Aggregate counts and event-weighted operational rates across completed traces."""
    rows = list(audits)
    out = {"problems": len(rows)}
    for kind in ("start", "live", "decode", "writer_paragraph_boundaries"):
        if kind == "writer_paragraph_boundaries" and not any(kind in row for row in rows):
            continue
        total = Counter()
        for row in rows:
            total.update(row.get(kind, {}))
        out[kind] = dict(total)
        if kind in ("start", "live"):
            yes = total["write_while_thinking"]
            resolved = yes + total["wait_until_next_probe"]
            unresolved = total["calls"] - resolved
            out[kind].update(
                resolved_calls=resolved,
                unresolved_calls=unresolved,
                operational_write_fraction=yes / resolved if resolved else None,
                all_call_write_fraction_bounds=(
                    [yes / total["calls"], (yes + unresolved) / total["calls"]]
                    if total["calls"]
                    else None
                ),
                median_calls_per_problem=median(row[kind].get("calls", 0) for row in rows)
                if rows
                else None,
            )
    for key in (
        "admitted_at_first_standalone",
        "first_writer_while_thinking",
        "first_writer_joint",
        "static_live_prefix_prefills",
    ):
        out[key] = sum(row[key] for row in rows)
    out["median_first_writer_visible_generated_thinker_tokens"] = (
        median(row["first_writer_visible_generated_thinker_tokens"] for row in rows)
        if rows
        else None
    )
    return out
