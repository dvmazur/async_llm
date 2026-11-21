"""Build a reviewable index from all evaluated episodes, including failures."""

import argparse
import html
import json
import os
from pathlib import Path


def build(root, sources=None):
    """Write one report, optionally linking episodes from several suite roots."""
    root = root.resolve()
    sources = [Path(source).resolve() for source in (sources or [root])]
    root.mkdir(parents=True, exist_ok=True)
    runs = []
    cards = []
    seen = set()
    for source in sources:
        for directory in sorted(source.iterdir()):
            if not directory.is_dir() or not (directory / "summary.json").exists():
                continue
            model = directory / "model_metadata.json"
            metadata = json.loads(model.read_text()) if model.exists() else {}
            # Infrastructure smoke tests remain available, but never enter the 27B benchmark.
            if "Qwen3.8-27B" not in metadata.get("model", ""):
                continue
            if directory.name in seen:
                raise ValueError(f"Duplicate episode in combined report: {directory.name}")
            seen.add(directory.name)
            summary = json.loads((directory / "summary.json").read_text())
            summary["episode"] = directory.name
            summary["artifact_directory"] = str(directory)
            runs.append(summary)
            name = html.escape(directory.name)
            link = html.escape(os.path.relpath(directory, root), quote=True)
            passed = summary["passed"]
            agent = summary.get("agent", {})
            cards.append(f'''<article><h2>{name} · {"PASS" if passed else "NOT COMPLETED"}</h2>
<p>Reason: {html.escape(summary["reason"])} · path {summary["truth_path_m"]:.2f} m · contacts {summary["contacts"]} · decisions {agent.get("decisions", 0)}</p>
<video controls preload="metadata" src="{link}/replay.mp4"></video>
<p><a href="{link}/summary.json">Metrics</a> · <a href="{link}/decisions.jsonl">LLM choices</a> · <a href="{link}/policy_inputs.json">Policy inputs</a> · <a href="{link}/artifact_audit.json">Artifact audit</a> · <a href="{link}/source_hashes.json">Source hashes</a></p></article>''')
    runs.sort(key=lambda item: item["episode"])
    (root / "benchmark.json").write_text(json.dumps(runs, indent=2))
    page = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qwen 27B · online rover navigation</title><style>body{margin:0;background:#101923;color:#edf4fa;font:16px/1.6 system-ui}main{max-width:1180px;margin:auto;padding:28px}article{background:#1b2b3b;padding:24px;margin:28px 0;border-radius:14px}video{width:100%;background:black}a{color:#7edcca}h1{font-size:32px}</style><main>
<h1>Qwen3.8-27B: online SLAM and rover navigation</h1>
<p>The map starts unknown. RGB + measured lidar geometry + IMU/odometry feed the model. The native AsyncLLM observer and reasoner run concurrently; the selector reads their live shared cache and publishes its choice before analysis finishes. Episodes labelled direct use forward/left/right/stop commands without Nav2. Geometric and qwen episodes use model-selected viewpoints executed by Nav2.</p>
<p>Yellow circle: model waypoint where applicable. Red X: evaluator-only target, added after recording and never shown to the policy. Object hypotheses from RGB/lidar are magenta. Unknown space stays gray. These are simulation experiments, with approximate rover and sensor parameters.</p>
<p>All evaluated 27B episodes below are included, including failures. Small-model infrastructure smoke tests are excluded.</p>"""
    (root / "index.html").write_text(page + "\n".join(cards) + "</main></html>")
    lines = [
        "# Qwen 27B rover navigation — measured results",
        "",
        "| Episode | Pass | Reason | Path, m | Contacts | Decisions | Median LLM, s | Wall, s |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        agent = run.get("agent", {})
        latency = agent.get("latency_median_s")
        latency_text = f"{latency:.2f}" if latency is not None else "—"
        lines.append(
            f"| {run['episode']} | {run['passed']} | {run['reason']} | {run['truth_path_m']:.2f} | {run['contacts']} | {agent.get('decisions', 0)} | {latency_text} | {run['wall_seconds']:.1f} |"
        )
    lines += [
        "",
        "Videos contain real simulated front-camera frames and sensor-built SLAM maps. The red target overlay is evaluator-only.",
        "No prebuilt map, hidden target coordinates, scripted route or automatic choice between candidates is supplied to the policy.",
        "Success requires model completion, true proximity, sufficient travelled distance for each scenario, and no body contact.",
        "This small development suite does not establish general navigation reliability.",
    ]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    return runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path, help="Report output, or suite root without --source")
    p.add_argument(
        "--source",
        action="append",
        type=Path,
        help="Suite root to include; repeat to combine independently executed suites",
    )
    args = p.parse_args()
    print(json.dumps(build(args.root, args.source), indent=2))


if __name__ == "__main__":
    main()
