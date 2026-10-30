"""Parse saved model responses and build a source-image proposal review packet."""

import argparse
import html
import json
import os
from pathlib import Path

from pipeline import records, save
from image_edit_test import parse_solve


def collect(root, output):
    jobs = json.loads((output / "jobs.json").read_text())
    indexed = {r["id"]: r for r in records(root)}
    audit_path = output / "review_notes.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    rows, cards = [], []
    for job in jobs["jobs"]:
        record = indexed[job["id"]]
        source = record["source"]
        path = output / f"{job['id']}.response.json"
        row = {"id": job["id"], "source_id": source["source_id"],
               "category": source["category"], "question": source["question"],
               "source_answer": source["answer"], "status": "missing_response"}
        if path.exists():
            response = json.loads(path.read_text())
            row.update({"service_tier": response.get("service_tier"),
                        "provider": response.get("provider"), "usage": response.get("usage", {})})
            try:
                choice = response["choices"][0]
                if choice["finish_reason"] != "stop":
                    raise ValueError("Incomplete generation")
                content = choice["message"]["content"].strip()
                if content.startswith("```"):
                    content = content.split("\n", 1)[1].rsplit("```", 1)[0]
                suggestion = parse_solve(content)[0]
                if suggestion.get("eligible") is False:
                    row.update(status="model_rejected", suggestion=suggestion)
                elif suggestion.get("eligible") is True:
                    proposal = suggestion["proposal"]
                    required = ("edit_type", "changed_fact_before", "changed_fact_after",
                                "expected_answer_before", "solution_before", "solution_after",
                                "editing_method", "reasoning_change")
                    if not all(isinstance(proposal.get(k), str) and proposal[k].strip() for k in required):
                        raise ValueError("Missing proposal fields")
                    if proposal["expected_answer_before"].strip() == source["answer"].strip():
                        raise ValueError("Before answer equals source answer")
                    row.update(status="pending_review", suggestion=suggestion)
                    save(output / f"{job['id']}.proposal.json", proposal)
                else:
                    raise ValueError("Missing boolean eligibility")
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                row.update(status="invalid_response", error=str(exc))
        if source["source_id"] in audit:
            row["review_notes"] = audit[source["source_id"]]
        rows.append(row)
        image = os.path.relpath(root / source["image"], output)
        cards.append(f'<article><h2>MathVista {html.escape(source["source_id"])} — '
                     f'{html.escape(row["status"])}</h2><p>{html.escape(source["question"])}</p>'
                     f'<img src="{html.escape(image)}"><pre>{html.escape(json.dumps(row, indent=2))}</pre>'
                     '</article>')
    summary = {"count": len(rows), "status_counts": {status: sum(r["status"] == status for r in rows)
                for status in sorted({r["status"] for r in rows})},
               "reported_cost": sum(r.get("usage", {}).get("cost", 0) or 0 for r in rows),
               "items": rows}
    save(output / "summary.json", summary)
    (output / "review.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>MathVista edit proposals</title>'
        '<style>body{font:16px sans-serif;max-width:1100px;margin:30px auto}'
        'img{max-width:100%;max-height:500px}pre{white-space:pre-wrap}'
        'article{border-top:1px solid #aaa;margin:30px 0}</style>'
        '<h1>Real-source proposals — no edited images yet</h1>' + ''.join(cards))
    print(json.dumps({k: v for k, v in summary.items() if k != "items"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    args = parser.parse_args()
    collect(args.workspace.resolve(), args.responses.resolve())
