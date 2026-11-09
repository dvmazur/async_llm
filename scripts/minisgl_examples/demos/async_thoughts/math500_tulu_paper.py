"""Render linked, verified Tülu comparison and content-budget JSON as a paper Markdown table."""

import argparse
import json
import math
from pathlib import Path

ARMS = {
    "archived-ordinary": "Ordinary (archived)",
    "direct": "Direct answer",
    "serial-thinking": "Serial private thinking",
    "archived-native-chat": "Native chat (archived)",
    "evidence-turns": "Evidence-turn native chat",
}
CANDIDATE = "evidence-turns"
ORDINARY = "archived-ordinary"


def check(value, message):
    if not value:
        raise ValueError(message)


def validate(comparison, budget):
    n, target = comparison["coverage"], comparison["target"]
    cases = comparison["cases"]
    check(0 < n <= target and target in (64, 500), "Invalid paper cohort")
    check(
        (budget["coverage"], budget["expected"], budget["partial"]) == (n, target, n < target),
        "Coverage or completion state differs",
    )
    ids = [c["unique_id"] for c in cases]
    check(
        [c["rank"] for c in cases] == list(range(n))
        and budget["rank_ids"] == ids
        and len(set(ids)) == n,
        "The reports use different rank IDs or order",
    )
    check(
        set(comparison["summary"])
        == set(budget["full_observed"])
        == set(budget["equal_aggregate_content"])
        == set(ARMS),
        "Arm inventory differs",
    )
    for arm in ARMS:
        s, f = comparison["summary"][arm], budget["full_observed"][arm]
        correct = sum(c["outcomes"][arm] for c in cases)
        check(
            s["coverage"] == n and correct == s["correct"] == f["correct"],
            f"{arm} full score differs",
        )
        for name, total in [
            ("writer_tokens", "writer_content"),
            ("thinker_tokens", "thinker_content"),
            ("content_tokens", "content"),
            ("decode_batches", "recorded_decode_batches"),
            ("forward_batches", "recorded_forward_batches"),
        ]:
            if name in s:
                d = s[name]
                check(
                    d["n"] == n and d["sum"] == f[total] and math.isclose(d["mean"], d["sum"] / n),
                    f"{arm} {name} differs",
                )
        check(
            f["writer_content"] + f["thinker_content"] == f["content"]
            and f["recorded_forward_batches"] >= f["recorded_decode_batches"] > 0,
            f"{arm} content or batch accounting differs",
        )
    ordinary = budget["full_observed"][ORDINARY]
    target_tokens = ordinary["content"]
    check(
        target_tokens == budget["ordinary_content_target"]
        and ordinary["correct"] == budget["full_ordinary_correct"],
        "Ordinary target differs",
    )
    for arm, e in budget["equal_aggregate_content"].items():
        natural = budget["full_observed"][arm]["content"]
        expected_status = (
            "below_target_at_natural_completion"
            if natural < target_tokens
            else ("exact" if e["content"] == target_tokens else "closest_uniform_cap")
        )
        check(
            0 <= e["content"] <= natural
            and e["content"] == e["writer_content"] + e["thinker_content"]
            and e["residual_from_target"] == e["content"] - target_tokens
            and e["status"] == expected_status
            and (natural >= target_tokens or e["content"] == natural)
            and 0 <= e["correct"] <= n
            and e["correct"]
            == ordinary["correct"] + e["only_this_arm_vs_full_ordinary"] - e["only_full_ordinary"],
            f"{arm} aggregate budget differs",
        )
    paired = comparison["paired_against_evidence_turns"]
    for arm in ARMS.keys() - {CANDIDATE}:
        p = paired[arm]
        gains = [c["rank"] for c in cases if c["outcomes"][CANDIDATE] and not c["outcomes"][arm]]
        losses = [c["rank"] for c in cases if c["outcomes"][arm] and not c["outcomes"][CANDIDATE]]
        discordant = len(gains) + len(losses)
        sign_p = min(
            1.0,
            2
            * sum(math.comb(discordant, k) for k in range(min(len(gains), len(losses)) + 1))
            / 2**discordant,
        )
        check(math.isclose(sign_p, p["exact_two_sided_sign_p"]), f"{arm} paired sign p differs")
        check(
            (
                p["n"],
                p["baseline_correct"],
                p["candidate_correct"],
                p["gained"],
                p["lost"],
                p["gained_count"],
                p["lost_count"],
            )
            == (
                n,
                comparison["summary"][arm]["correct"],
                comparison["summary"][CANDIDATE]["correct"],
                gains,
                losses,
                len(gains),
                len(losses),
            )
            and math.isclose(p["delta_percentage_points"], 100 * (len(gains) - len(losses)) / n),
            f"{arm} paired result differs",
        )
    s = comparison["summary"][CANDIDATE]
    votes = [s[f"{phase}_votes"] for phase in ("admission", "continuation")]
    check(
        all(0 <= v["yes"] <= v["applied"] for v in votes)
        and sum(v["yes"] for v in votes) == s["positive_probe_count"]["sum"]
        and sum(v["applied"] for v in votes) <= s["probe_count"]["sum"],
        "Probe counts differ",
    )


def cost_lines(receipts):
    lines = []
    seen = set()
    for receipt in receipts:
        pod = receipt.get("pod") or receipt.get("pod_id")
        check(
            isinstance(pod, str) and pod and pod not in seen, "Missing or duplicate pod in receipts"
        )
        seen.add(pod)
        meta = (receipt.get("billing") or {}).get("metadata", {})
        check(
            meta.get("query", {}).get("podId", pod) == pod, "Billing query belongs to another pod"
        )
        parts = []
        posted = receipt.get("posted_usd", receipt.get("posted_total_usd"))
        if meta.get("recordCount") == 0 or (not meta and posted == 0):
            check(posted in (None, 0), "Zero billing records contradict the posted subtotal")
            parts.append("no charge had posted at the recorded lookup")
        else:
            billed = meta.get("totals", {}).get("totalAmount")
            if billed is not None and posted is not None:
                check(
                    math.isclose(billed, posted, abs_tol=1e-8),
                    "Billing subtotal differs from receipt",
                )
            posted = billed if posted is None else posted
            if posted is not None:
                parts.append(f"posted ${posted:.4f}")
        for key, label in (
            ("estimated_usd", "receipt estimate"),
            ("upper_estimate_usd", "receipt upper estimate"),
        ):
            if receipt.get(key) is not None:
                parts.append(f"{label} ${receipt[key]:.4f}")
        check(parts, "Receipt has no supported billing evidence")
        date = receipt.get("as_of_utc")
        lines.append(f"- `{pod}` — " + ", ".join(parts) + (f" ({date})" if date else "") + ".")
    return lines


def fraction(correct, n):
    return f"{correct}/{n} ({100 * correct / n:.1f}%)"


def render(comparison, budget, receipts=()):
    validate(comparison, budget)
    n, target = comparison["coverage"], comparison["target"]
    status = (
        f"Complete frozen {target}-problem cohort"
        if n == target
        else f"**Partial frozen cohort, {n} of {target} problems**"
    )
    out = [
        "# Tülu 3 8B on MATH-500",
        "",
        status + f". Every arm uses the same ranks 0–{n - 1}. "
        "Accuracy is strict equality of the last balanced boxed answer with the reference, after trimming whitespace.",
        "",
        "## Full observed runs",
        "",
        "| Arm | Strict exact | Writer tokens, mean | Writer tokens, median | Thinker tokens, mean | Decode batches, mean | All forward batches, mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, label in ARMS.items():
        s = comparison["summary"][arm]
        f = budget["full_observed"][arm]
        out.append(
            f"| {label} | {fraction(s['correct'], n)} | {s['writer_tokens']['mean']:.1f} | "
            f"{s['writer_tokens']['median']:g} | {s['thinker_tokens']['mean']:.1f} | "
            f"{f['recorded_decode_batches'] / n:.1f} | {f['recorded_forward_batches'] / n:.1f} |"
        )
    out += [
        "",
        "Means and medians are per problem. Direct answer and serial private thinking share a 16,384-sample limit per problem. "
        "The serial private draft can use up to 8,192 of those samples. Async runs allow 16,384 samples per role. "
        "Direct answer requests only the final result. Serial thinking uses a private draft followed by an answer. "
        "The two AsyncLLM native-chat arms are Tülu adaptations that share the original benchmark task prompt. "
        "Recorded scheduler batches count a joint decode once. "
        "Prefill batches can be internally chunked. Latency and hardware cost require separate measurements.",
        "",
        "## Paired evidence-turn results",
        "",
        "Gains and losses are relative to each listed comparator on these same problems. Each p value is an unadjusted paired sign test.",
        "",
        "| Comparator | Gained | Lost | Evidence-turn delta, pp | Exact two-sided sign p |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm, label in ARMS.items():
        if arm == CANDIDATE:
            continue
        p = comparison["paired_against_evidence_turns"][arm]
        out.append(
            f"| {label} | {p['gained_count']} | {p['lost_count']} | {p['delta_percentage_points']:+.1f} | {p['exact_two_sided_sign_p']:.4g} |"
        )
    s = comparison["summary"][CANDIDATE]
    applied = sum(s[f"{phase}_votes"]["applied"] for phase in ("admission", "continuation"))
    out += [
        "",
        "## Evidence-turn probe",
        "",
        "| Phase | Applied votes | WRITE | WAIT |",
        "|---|---:|---:|---:|",
    ]
    for phase in ("admission", "continuation"):
        v = s[f"{phase}_votes"]
        out.append(
            f"| {phase.capitalize()} | {v['applied']} | {v['yes']} | {v['applied'] - v['yes']} |"
        )
    out += [
        "",
        f"The run scored {s['probe_count']['sum']:,} probes and discarded {s['probe_count']['sum'] - applied:,} late results. "
        f"The roles ran jointly in {s['joint_decode_batches']['sum']:,} decode batches. "
        f"The writer generated {s['writer_tokens_after_last_thinker_decode']['sum']:,} content tokens after the final thinker decode.",
        "",
        "## Generated content at ordinary's aggregate spend",
        "",
        f"The target is ordinary generation's {budget['ordinary_content_target']:,} content tokens. Each arm uses a uniform cap "
        "chosen by token spend alone. Whole joint batches can leave a residual. A run that naturally ends below the target is reported at its observed length.",
        "",
        "| Arm | Uniform cap | Actual content | Difference from target | Strict exact | Match |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for arm, label in ARMS.items():
        e = budget["equal_aggregate_content"][arm]
        match = {
            "exact": "Exact",
            "closest_uniform_cap": "Closest uniform cap",
            "below_target_at_natural_completion": "Natural output below target",
        }[e["status"]]
        out.append(
            f"| {label} | {e['uniform_cap']:,} | {e['content']:,} | {e['residual_from_target']:+,} | {fraction(e['correct'], n)} | {match} |"
        )
    out += [
        "",
        "Content counts accepted thinker and writer tokens. EOS, prompts, probe inputs and flush-only steps are excluded. "
        "These grades use prefixes of the observed outputs. A shorter cap in an actual rerun could change scheduling and answers. "
        "Runs at equal content spend can have different decode counts, elapsed time and FLOPs.",
    ]
    if receipts:
        out += ["", "## Rental receipts", "", *cost_lines(receipts)]
    return "\n".join(out) + "\n"


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--comparison", required=True, type=Path)
    cli.add_argument("--content-budget", required=True, type=Path)
    cli.add_argument("--receipt", action="append", default=[], type=Path)
    args = cli.parse_args()

    def load(path):
        return json.loads(path.read_bytes())

    print(
        render(load(args.comparison), load(args.content_budget), [load(p) for p in args.receipt]),
        end="",
    )


if __name__ == "__main__":
    main()
