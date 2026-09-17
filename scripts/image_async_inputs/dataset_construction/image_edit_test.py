"""Small, explicit OpenRouter image-edit pilot with separate blind solving.

Uses the user-approved Eliza chat endpoint; no local model or GPU is loaded.
"""

import argparse
import base64
import copy
import io
import html
import json
import mimetypes
import os
import re
from pathlib import Path

from PIL import Image

import gateway
from pipeline import NOTICE, digest, records, require, save


EDITS = {
    "88": "Remove only the digit '2' from the right-hand angle label '2x°', leaving 'x°'. "
          "Keep the left-hand x° label, square, line, P, l, and printed question unchanged.",
    "926": "Extend the existing light-teal shading to fill the previously white left half "
           "of the semicircle, between x=0 and x=3. The entire area under the semicircle "
           "from x=0 to x=6 must now be shaded uniformly. Preserve the curve, axes, "
           "tick labels, and the exact equation; add no new lines or text.",
    "45": "Replace only '14' in the September waiting-time cell with '20'. Preserve "
          "August 17, October 26, November 17, December 25, every month name, all "
          "headings, table borders, colors, and alignment.",
}


def data_url(path):
    return f"data:{mimetypes.guess_type(path.name)[0]};base64," + base64.b64encode(path.read_bytes()).decode()


def extract_image(response):
    """Support chat images and dedicated Image API results, without URL downloads."""
    if response.get("data") and "b64_json" in response["data"][0]:
        return base64.b64decode(response["data"][0]["b64_json"], validate=True)
    message = response.get("choices", [{}])[0].get("message", {})
    for item in message.get("images", []):
        url = item.get("image_url", {}).get("url", "")
        if url.startswith("data:image/") and ";base64," in url:
            return base64.b64decode(url.split(",", 1)[1], validate=True)
    raise ValueError("Response contains no supported inline image; inspect saved metadata.")


def metadata_only(response):
    value = copy.deepcopy(response)
    for choice in value.get("choices", []):
        message = choice.get("message", {})
        if "images" in message:
            message["images"] = [{"saved_separately": True} for _ in message["images"]]
    for item in value.get("data", []):
        if "b64_json" in item:
            item["b64_json"] = "[saved separately]"
    return value


def parse_solve(content):
    content = content.strip()
    fenced = content.startswith("```") and content.endswith("```")
    if fenced:
        content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(content), fenced
    except json.JSONDecodeError:
        # Preserve literal LaTeX backslashes that are illegal JSON escapes.
        repaired = re.sub(r'\\(?!["\\/bfnrtu])', lambda match: "\\\\", content)
        return json.loads(repaired), True


def generate(root, output, ids, edits=None):
    edits = edits or EDITS
    indexed = {r["source"]["source_id"]: r for r in records(root)}
    for pid in ids:
        source = indexed[pid]["source"]
        image_path = root / source["image"]
        destination = output / pid
        destination.mkdir(parents=True, exist_ok=True)
        require(not (destination / "generation.response.json").exists(),
                f"Generation response exists for {pid}; do not repeat billable calls silently.")
        with Image.open(image_path) as image:
            size = image.size
        prompt = (
            "Use case: precise scientific-diagram edit. Image 1 is the edit target.\n"
            "This is a controlled dataset corruption, not a request to solve the problem.\n"
            f"Change only: {edits[pid]}\n"
            f"Preserve the original {size[0]} by {size[1]} canvas dimensions and framing. "
            "Keep all unrelated pixels, typography, colors, symbols and whitespace unchanged. "
            "Do not beautify, redraw, add explanations, solve, annotate, crop or add a watermark. "
            "Return exactly one edited image."
        )
        settings = {"model": "google/gemini-3.1-flash-image", "modalities": ["image", "text"],
                    "max_tokens": 8192, "service_tier": "flex", "seed": 42}
        save(destination / "generation.request.json", {**settings, "prompt": prompt,
             "source_image": str(image_path), "source_sha256": digest(image_path)})
        payload = {**settings, "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url(image_path)}}]}]}
        response = gateway.request(payload)
        save(destination / "generation.response.json", metadata_only(response))
        raw = extract_image(response)
        with Image.open(io.BytesIO(raw)) as generated:
            generated.load()
            suffix = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}[generated.format]
            generated_size = generated.size
        before = destination / ("before" + suffix)
        before.write_bytes(raw)
        save(destination / "pair.json", {"source_id": pid, "before": before.name,
             "after": str(image_path), "before_sha256": digest(before),
             "after_sha256": digest(image_path), "source_size": size,
             "generated_size": generated_size, "same_dimensions": size == generated_size,
             "question": source["question"], "status": "pending_visual_review"})
        print(json.dumps({"id": pid, "source_size": size, "generated_size": generated_size,
                          "tier": response.get("service_tier"), "usage": response.get("usage")}), flush=True)


def verify(output, ids):
    for pid in ids:
        destination = output / pid
        pair = json.loads((destination / "pair.json").read_text())
        for stage in ("before", "after"):
            response_path = destination / f"solve_{stage}.response.json"
            if response_path.exists():
                print(f"Skipping saved solve: {pid} {stage}", flush=True)
                continue
            image = destination / pair[stage] if stage == "before" else Path(pair[stage])
            prompt = (
                "Solve the question using only the attached image and question. "
                "Return JSON with answer (string), visual_facts (list), solution (string), "
                "and any ambiguity. Do not infer missing labels. No Markdown fences.\nQuestion: "
                + pair["question"]
            )
            settings = {"model": "google/gemini-3.8-flash", "service_tier": "flex",
                        "reasoning": {"effort": "medium"}, "max_tokens": 4096, "seed": 42}
            save(destination / f"solve_{stage}.request.json", {**settings, "prompt": prompt,
                 "image_sha256": digest(image)})
            response = gateway.request({**settings, "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url(image)}}]}]})
            save(response_path, response)
            print(json.dumps({"id": pid, "stage": stage,
                              "tier": response.get("service_tier"),
                              "usage": response.get("usage"),
                              "content": response.get("choices", [{}])[0].get("message", {}).get("content")}), flush=True)


def report(output, ids):
    expected = {"88": {"before": "45", "after": "30"},
                "926": {"before": "14.14", "after": "7.07"},
                "45": {"before": "3", "after": "-3"}}
    rows, cards = [], []
    for pid in ids:
        destination = output / pid
        pair = json.loads((destination / "pair.json").read_text())
        generation = json.loads((destination / "generation.response.json").read_text())
        row = {**pair, "generation_tier": generation.get("service_tier"),
               "generation_cost": generation.get("usage", {}).get("cost", 0), "solves": {}}
        columns = []
        for stage in ("before", "after"):
            response = json.loads((destination / f"solve_{stage}.response.json").read_text())
            choice = response["choices"][0]
            require(choice["finish_reason"] == "stop", "Incomplete blind solve")
            answer, repaired = parse_solve(choice["message"]["content"])
            row["solves"][stage] = {"result": answer, "expected": expected[pid][stage],
                "answer_matches": answer["answer"].strip() == expected[pid][stage],
                "latex_json_escape_repaired": repaired,
                "cost": response.get("usage", {}).get("cost", 0),
                "service_tier": response.get("service_tier")}
            path = destination / pair[stage] if stage == "before" else Path(pair[stage])
            url = os.path.relpath(path, output)
            text = pair["question"] if stage == "before" else NOTICE
            columns.append(f'<section><h3>{stage}: pic_{1 if stage == "before" else 2}</h3>'
                           f'<img src="{html.escape(url)}"><p>{html.escape(text)}</p>'
                           f'<p>Blind answer (evaluator-only): {html.escape(answer["answer"])}</p></section>')
        row["status"] = "semantic_check_passed_size_mismatch" if all(
            s["answer_matches"] for s in row["solves"].values()) and not pair["same_dimensions"] else "review_required"
        row["total_cost"] = row["generation_cost"] + sum(s["cost"] for s in row["solves"].values())
        rows.append(row)
        cards.append(f'<article><h2>MathVista {pid}</h2><p>{html.escape(EDITS[pid])}</p>'
                     '<div>' + ''.join(columns) + '</div>'
                     f'<p>Source size {pair["source_size"]}; generated size {pair["generated_size"]}. '
                     'Images displayed at comparable widths; raw files are unchanged.</p></article>')
    summary = {"pairs": rows, "total_reported_cost": sum(r["total_cost"] for r in rows),
               "semantic_checks_passed": sum(all(s["answer_matches"] for s in r["solves"].values()) for r in rows),
               "same_dimension_pairs": sum(r["same_dimensions"] for r in rows)}
    save(output / "summary.json", summary)
    (output / "review.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Flash Image edit test</title>'
        '<style>body{font:16px sans-serif;max-width:1200px;margin:30px auto}'
        'article{border-top:1px solid #aaa;padding:20px 0}div{display:flex;gap:24px}'
        'section{width:50%;min-width:0}img{width:100%;object-fit:contain;height:380px}</style>'
        '<h1>Real-source image edits and blind solves</h1><p>Shard 1 remains in context; '
        'pic_1 is replaced by pic_2 and shard 2 is appended. No async experiment has run.</p>'
        + ''.join(cards))
    print(json.dumps({k: v for k, v in summary.items() if k != "pairs"}, indent=2))


def normalize_test_pairs(output, ids):
    from normalize_pair import normalize

    for pid in ids:
        directory = output / pid
        pair = json.loads((directory / "pair.json").read_text())
        result = normalize(directory / pair["before"], Path(pair["after"]), directory / "normalized")
        print(json.dumps({"id": pid, "target_size": result["target_size"],
                          "checks": result["checks"], "processor_parity": result["processor_parity"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("generate", "verify", "report", "normalize"))
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ids", nargs="+", choices=tuple(EDITS), default=list(EDITS))
    args = parser.parse_args()
    if args.stage == "generate":
        generate(args.workspace.resolve(), args.output.resolve(), args.ids)
    elif args.stage == "verify":
        verify(args.output.resolve(), args.ids)
    elif args.stage == "normalize":
        normalize_test_pairs(args.output.resolve(), args.ids)
    else:
        report(args.output.resolve(), args.ids)
