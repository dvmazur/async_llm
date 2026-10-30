"""Assemble a reviewed plan: edit -> normalize -> blind solve -> pair audit -> export.

All requests use gateway's DATASET_BUDGET_LEDGER. Final acceptance remains an explicit
review command; model checks alone never approve a pair.
"""

import argparse
import json
import os
from pathlib import Path

import gateway
from image_edit_test import data_url, generate, parse_solve
from pipeline import (digest, export, import_sources, normalize_record, read,
                      record_path, records, require, save, transition)


def model_job(path, text, images):
    settings = {"model": "google/gemini-3.8-flash", "service_tier": "flex",
                "max_tokens": 4096, "reasoning": {"effort": "medium"}, "seed": 42}
    request = {**settings, "prompt": text,
        "images": [{"path": str(p), "sha256": digest(p)} for p in images]}
    request_path = path.with_suffix('.request.json')
    if path.exists():
        require(request_path.exists() and read(request_path)==request,
                'Cached model request or image changed; use a new version.')
        retry = path.with_suffix('.retry.json')
        if retry.exists():
            retry_request = read(retry.with_suffix('.request.json'))
            require(retry_request == {**request, 'max_tokens':8192}, 'Retry settings changed')
            return read(retry)
        return read(path)
    require(not request_path.exists(),
            'A request exists without a saved response; cost/outcome unknown. Investigate before explicit retry.')
    save(request_path, request)
    response = gateway.request({**settings, "messages": [{"role": "user", "content": [
        {"type": "text", "text": text}, *[{"type": "image_url", "image_url": {"url": data_url(p)}}
                                          for p in images]]}]})
    save(path, response)
    return response


def parsed(response):
    require(response["choices"][0]["finish_reason"] == "stop", "Incomplete model response")
    return parse_solve(response["choices"][0]["message"]["content"])[0]


def canonical(answer, source):
    value = str(answer).strip().strip("$").replace("\\boxed{", "").rstrip("}").strip()
    choices = source.get("original", {}).get("choices") or []
    letter = value.strip("() .")
    if len(letter) == 1 and "A" <= letter <= "Z" and ord(letter) - 65 < len(choices):
        value = str(choices[ord(letter) - 65])
    try:
        return float(value.replace("°", ""))
    except ValueError:
        return value.casefold()


def build(plan_path, root):
    require(os.environ.get("DATASET_BUDGET_LEDGER"), "Set DATASET_BUDGET_LEDGER before building.")
    plan = read(plan_path)
    root.mkdir(parents=True, exist_ok=True)
    (root / "records").mkdir(exist_ok=True)
    if (root / "plan.json").exists():
        require(read(root / "plan.json") == plan, "Plan changed; use a new build version.")
    else:
        save(root / "plan.json", plan)
    for item in plan["items"]:
        source_root = Path(item["workspace"]).resolve()
        source_record = next(r for r in records(source_root) if r["source"]["source_id"] == item["source_id"])
        sid = source_record["id"]
        work = root / "evidence" / sid
        work.mkdir(parents=True, exist_ok=True)
        source = {**source_record["source"], "image": str(source_root / source_record["source"]["image"])}
        source["difficulty_group"] = item["difficulty_group"]
        source["construction_notes"] = item["notes"]
        source_manifest = work / "source.jsonl"
        source_manifest.write_text(json.dumps(source) + "\n")
        import_sources(root, source_manifest)
        proposal = read(Path(item["proposal"]))
        current = read(record_path(root, sid))
        if current["status"] == "imported":
            transition(root, sid, "propose", proposal)
        current = read(record_path(root, sid))
        if current["status"] == "edit_proposed":
            if item.get("reuse_before"):
                before = Path(item["reuse_before"])
            else:
                image_work = work / "editing"
                pair_file = image_work / item["source_id"] / "pair.json"
                if not pair_file.exists():
                    generate(source_root, image_work, [item["source_id"]],
                             {item["source_id"]: proposal["editing_method"]})
                pair = read(pair_file)
                before = pair_file.parent / pair["before"]
            transition(root, sid, "attach", {"image": str(before),
                "approval_note": f"User approved OpenRouter image-edit workflow; construction plan budget ${plan.get('budget_usd', 2)}."})
        current = read(record_path(root, sid))
        if current["status"] == "rendered":
            normalize_record(root, sid)
        current = read(record_path(root, sid))
        images = {k: root / v["path"] for k, v in current["normalization"]["images"].items()}
        checks = {}
        for stage in ("before", "after"):
            prompt = ("Solve the question using the attached image. No answer key is supplied. "
                      "Return only JSON: answer (string; use the option VALUE, not letter), "
                      "visual_facts (list), solution (string), ambiguity (string or null).\nQuestion: "
                      + source["question"])
            response = model_job(work / f"solve_{stage}.json", prompt, [images[stage]])
            result = parsed(response)
            expected = proposal["expected_answer_before"] if stage == "before" else source["answer"]
            checks[stage] = {"result": result, "expected": expected,
                              "matches": canonical(result["answer"], source) == canonical(expected, source),
                              "image_sha256": digest(images[stage])}
            if not checks[stage]['matches'] and plan.get('semantic_answer_grading', False):
                grading_prompt = (
                    'Compare two answer strings for the following question. Do not re-solve the problem. '
                    'Return JSON: equivalent (boolean), explanation (string). Accept ONLY synonymous '
                    'names, equivalent mathematics, units, formatting, or rounding explicitly required '
                    'by the question. A different number, different curve, different ranking or subset '
                    'is NOT equivalent. If uncertain return false.\n'+json.dumps({
                        'question':source['question'],'reference':expected,'prediction':result['answer']}))
                grade = parsed(model_job(work / f'grade_{stage}.json', grading_prompt, []))
                checks[stage]['semantic_grade'] = grade
                checks[stage]['matches'] = grade.get('equivalent') is True
            equivalence_path = work / 'answer_equivalence_review.json'
            if not checks[stage]['matches'] and equivalence_path.exists():
                decision = read(equivalence_path).get(stage)
                if decision:
                    require(decision.get('reference') == expected and decision.get('prediction') == result['answer']
                            and decision.get('reviewer') and decision.get('notes'), 'Invalid answer equivalence review')
                    checks[stage]['reviewed_equivalence'] = decision
                    checks[stage]['matches'] = decision.get('equivalent') is True
        audit_prompt = (
            "Audit a controlled image edit. Image 1 is erroneous BEFORE; image 2 is original AFTER. "
            "Check whether ONLY the requested semantic edit occurred. Ignore minor resampling/JPEG "
            "differences, but flag changed equations, other labels, geometry, missing text or regions. "
            "Also flag inconsistent redundant encodings in BEFORE: changed numeric labels with "
            "unchanged bar/point/slice geometry, invalid percentage totals or contradictory captions. "
            "An intended value change must update its corresponding geometry consistently. "
            "Return JSON: intended_edit_present (boolean), unrelated_semantic_changes (list), "
            "readable (boolean), notes (string). Do not solve the task or assume the edit succeeded.\n"
            "Intended BEFORE fact: " + proposal["changed_fact_before"] + "\nAFTER fact: "
            + proposal["changed_fact_after"])
        audit = parsed(model_job(work / "pair_audit.json", audit_prompt, [images["before"], images["after"]]))
        checks["pair_audit"] = audit
        checks["passes_model_checks"] = (all(checks[s]["matches"] for s in ("before", "after"))
            and audit.get("intended_edit_present") is True and audit.get("readable") is True
            and audit.get("unrelated_semantic_changes") == [])
        save(work / "checks.json", checks)
        print(json.dumps({"source_id": item["source_id"], "checks_passed": checks["passes_model_checks"],
                          "answers": {k: checks[k]["result"]["answer"] for k in ("before", "after")}}), flush=True)


def finalize(root, reviews_path, output):
    reviews = read(reviews_path)
    for record in records(root):
        sid = record["id"]
        decision = reviews[record["source"]["source_id"]]
        checks = read(root / "evidence" / sid / "checks.json")
        if decision["decision"] == "accept":
            require(checks["passes_model_checks"], "Model checks failed; cannot accept automatically.")
        if record["status"] == "normalized":
            review = {**decision, "answer_before": record["proposal"]["expected_answer_before"],
                      "answer_after": record["source"]["answer"]}
            transition(root, sid, "review", review)
    export(root, output)
    import shutil
    shutil.copytree(root / "evidence", output / "evidence")
    save(output / "construction_plan.json", read(root / "plan.json"))
    save(output / "reviews.json", reviews)
    ledger_path = os.environ.get("DATASET_BUDGET_LEDGER")
    if ledger_path:
        ledger = read(Path(ledger_path))
        save(output / "budget.json", ledger)
    else:
        ledger = {"requests": []}
    accepted = [r for r in records(root) if r["status"] == "accepted"]
    gallery = ["# Image and text shards\n", "Pic 1 is erroneous; pic 2 is the original corrected source, both normalized.\n"]
    overview = ["# MathVista async image corrections v1\n",
        "Small reviewed construction pilot, not a model benchmark result.\n",
        "| Source ID | Group | Before answer | After answer |\n|---|---|---|---|\n"]
    for record in accepted:
        source = record["source"]
        normalized = record["normalization"]["images"]
        overview.append(f"| {source['source_id']} | {source['difficulty_group']} | "
                        f"{record['proposal']['expected_answer_before']} | {source['answer']} |\n")
        gallery.extend([f"\n## MathVista {source['source_id']}\n\n",
            "| Pic 1 | Pic 2 |\n|---|---|\n",
            f"| ![Before]({normalized['before']['path']}) | ![After]({normalized['after']['path']}) |\n\n",
            f"Text shard 1: {source['question']}\n\n",
            "Text shard 2: The earlier picture contained an error. It has now been corrected. "
            "Recheck your reasoning using the current picture.\n"])
    spend = sum(r.get("cost", 0) for r in ledger["requests"])
    overview.extend([f"\nNew reported API spend: ${spend:.8f}; budget: $2. Earlier pilot edits were reused.\n",
        "\nInputs: `inputs.jsonl` and normalized PNGs. Answers/provenance: `annotations.jsonl`. "
        "`evidence/` holds normalized-image blind solves and semantic pair audits. "
        "`examples.md` displays both image and text shards.\n",
        "\nEvery accepted pair was visually reviewed and its normalized before/after states solved "
        "separately without reference answers. The same Gemini 3.8 Flash family was used for "
        "proposals, solves and audits; these checks are not independent human labels. "
        "Codex additionally checked the image evidence and calculations.\n",
        "\nReport calibration and reveal subsets separately. The pilot is intentionally selected "
        "for valid edits and is not a random representative sample. Original images may contain "
        "printed question text. Questions and choices are preserved.\n",
        "\nImage dimensions, aspect ratio, RGB PNG mode and metadata match within each pair. "
        "Generated edits can change unrelated pixels; only semantic preservation was reviewed. "
        "Exact Qwen processor grid/token parity is pending model selection. No GPU or async "
        "inference experiment was run.\n",
        "\nProvenance: AI4Math/MathVista testmini at the pinned revision in annotations. "
        "MathVista contributions are CC-BY-SA-4.0; original image/question rights are retained "
        "by their sources. This is a local derived evaluation set, not training data.\n"])
    (output / "README.md").write_text(''.join(overview))
    (output / "examples.md").write_text(''.join(gallery))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("build", "finalize"))
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.stage == "build":
        require(args.plan, "Provide --plan")
        build(args.plan, args.workspace.resolve())
    else:
        require(args.reviews and args.output, "Provide --reviews and --output")
        finalize(args.workspace.resolve(), args.reviews, args.output.resolve())
