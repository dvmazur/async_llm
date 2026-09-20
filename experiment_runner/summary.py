"""Pure CPU analysis of saved logs, reusable after downloading results."""
import argparse
import json
from pathlib import Path

from .logs import atomic, read_jsonl


def divide(a, b):
    return a / b if b else None


def batch_stats(rows):
    result = {}
    for phase in ("decode", "prefill", "mixed"):
        selected = [r for r in rows if r["phase"] == phase and r.get("status", "completed") == "completed"]
        n = len(selected)
        result[phase] = dict(forwards=n,
            decode_requests=sum(r["decode_requests"] for r in selected),
            prefill_requests=sum(r["prefill_requests"] for r in selected),
            prefill_rows=sum(r["prefill_rows"] for r in selected))
        for field in ("decode_requests", "prefill_requests", "prefill_rows"):
            result[phase]["mean_" + field] = divide(result[phase][field], n)
    return result


class Summary:
    def __init__(self, directory):
        self.directory = Path(directory)

    def compute(self):
        workers, episodes, all_forwards = [], [], []
        for gpu in sorted(self.directory.glob("gpu-*")):
            local = []
            for path in sorted(gpu.glob("slot-*/repeat-*/completion.json")):
                data = json.loads(path.read_text())
                context = json.loads((path.parent / "context.json").read_text())
                steps = list(read_jsonl(path.parent / "steps.jsonl"))
                events = list(read_jsonl(path.parent / "events.jsonl"))
                tokens = sum(e.get("sampled_tokens", 0) for e in events if e["kind"] == "stream")
                # Client already records each completed action readout through
                # this episode's Recorder, before asking World to execute it.
                readouts = sum(e['kind'] == 'decision' for e in events)
                actions = sum(s["kind"] == "action" for s in steps)
                row = dict(episode_id=context["episode_id"], status=data["status"],
                    model_seed=context["model_seed"], world_seed=context["world_seed"],
                    tokens=tokens, actions=actions, action_readouts=readouts, start=data.get("workload_start"),
                    end=data.get("workload_end"), tokens_per_action=divide(tokens, actions),
                    heights=[dict(step=s["step"], height=s.get("height")) for s in steps])
                episodes.append(row)
                local.append(row)
            valid = [r for r in local if r["status"] == "completed"]
            bounds = [r for r in local if r["start"] is not None and r["end"] is not None]
            start = min((r["start"] for r in bounds), default=None)
            end = max((r["end"] for r in bounds), default=None)
            forwards = list(read_jsonl(gpu / "forwards.jsonl")) if (gpu / "forwards.jsonl").exists() else []
            forwards = [r for r in forwards if start is not None and start <= r["monotonic"] <= end]
            # No fake "successful" interval when one slot failed alongside others.
            status_file = gpu / "status.json"
            complete = status_file.exists() and json.loads(status_file.read_text())["status"] == "completed"
            seconds = end - start if start is not None else None
            tokens, actions = sum(r["tokens"] for r in valid), sum(r["actions"] for r in valid)
            generated = list(read_jsonl(gpu / "generation.jsonl")) if (gpu / "generation.jsonl").exists() else []
            readouts = sum(r['action_readouts'] for r in local)
            totals_path = gpu/'engine-totals.json'
            totals = json.loads(totals_path.read_text()) if totals_path.exists() else {}
            forward_telemetry = totals.get('forward_telemetry_available', True)
            engine_readouts = totals.get('restricted_readouts', totals.get('action_readouts'))
            if complete and engine_readouts is not None and readouts != engine_readouts:
                raise ValueError(f'pipeline/engine action readout accounting mismatch in {gpu.name}')
            if complete and tokens != sum(r["kind"] == "sample" for r in generated):
                raise ValueError(f"role/engine token accounting mismatch in {gpu.name}")
            row = dict(gpu=gpu.name, status="completed" if complete else "partial", start=start, end=end,
                workload_seconds=seconds, generated_tokens=tokens, actions=actions, action_readouts=readouts,
                generated_tps=divide(tokens, seconds) if complete else None,
                # Historical launcher included one action readout in output tokens.
                output_tokens_including_readouts_tps=divide(tokens + readouts, seconds) if complete else None,
                tokens_per_action=divide(tokens, actions),
                forward_telemetry_available=forward_telemetry,
                batches=batch_stats(forwards) if forward_telemetry else None,
                decode_rows_per_second=divide(sum(r["decode_requests"] for r in forwards), seconds)
                    if complete and forward_telemetry else None)
            samples = list(read_jsonl(gpu/'gpu-samples.jsonl')) if (gpu/'gpu-samples.jsonl').exists() else []
            samples = [r for r in samples if start is not None and start <= r['monotonic'] <= end]
            utilization = [r['utilization_percent'] for r in samples if r.get('utilization_percent') is not None]
            row['gpu_utilization_sample_mean_percent'] = divide(sum(utilization), len(utilization))
            row['gpu_utilization_samples'] = len(utilization)
            row['gpu_sample_errors'] = sum('error' in r for r in samples)
            workers.append(row)
            if complete:
                all_forwards.extend(forwards)
        complete = bool(workers) and all(w["status"] == "completed" for w in workers)
        starts = [w["start"] for w in workers if w["start"] is not None]
        ends = [w["end"] for w in workers if w["end"] is not None]
        seconds = max(ends) - min(starts) if starts and ends else None
        tokens = sum(w["generated_tokens"] for w in workers)
        actions = sum(w["actions"] for w in workers)
        return dict(schema_version=1, status="completed" if complete else "partial",
            metric_contract="Full workload including first actions and role drain; role tokens include EOS; no padding in batches.",
            workload_seconds=seconds, generated_tokens=tokens, actions=actions,
            generated_tps=divide(tokens, seconds) if complete else None,
            tokens_per_action=divide(tokens, actions), batches=batch_stats(all_forwards)
                if all(w['forward_telemetry_available'] for w in workers) else None,
            workers=workers, episodes=episodes)

    def write(self):
        report = self.compute()
        atomic(self.directory / "analysis/summary.json", report)
        return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    args = parser.parse_args()
    print(json.dumps(Summary(args.directory).write(), indent=2))


if __name__ == "__main__":
    main()
