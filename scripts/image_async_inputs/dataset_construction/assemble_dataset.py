"""Export explicitly reviewed pairs across datasets; never auto-accept model outputs."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil

from pipeline import read, save, records, transition, record_path, export, require, NOTICE, digest


def assemble(config_path, staging, output):
    config=read(config_path)
    require(not staging.exists() and not output.exists(), 'Choose new staging/export directories')
    (staging/'records').mkdir(parents=True)
    seen=set()
    for item in config['samples']:
        root=Path(item['workspace']).resolve(); sid=item['id']
        require(sid not in seen,'Duplicate sample'); seen.add(sid)
        record=read(record_path(root,sid))
        checks=read(root/'evidence'/sid/'checks.json')
        require(checks['passes_model_checks'],'Model checks did not pass')
        shutil.copytree(root/'assets'/sid,staging/'assets'/sid)
        save(record_path(staging,sid),record)
        if record['status']=='normalized':
            require(item.get('visually_reviewed') is True and item.get('notes'), 'Explicit visual review required')
            review={'decision':'accept','reviewer':'Codex visual/semantic review, not independent human annotation',
                'notes':item['notes'],'answer_before':record['proposal']['expected_answer_before'],
                'answer_after':record['source']['answer'],
                **{k:True for k in ('isolated_edit','readable','both_solvable','visual_required',
                                    'answers_distinct','source_terms_checked','same_dimensions')}}
            transition(staging,sid,'review',review)
        else:
            require(record['status']=='accepted','Record is not reviewable')
        shutil.copytree(root/'evidence'/sid,staging/'evidence'/sid)
    export(staging,output)
    shutil.copytree(staging/'evidence',output/'evidence')
    save(output/'assembly_review.json',config)
    ledger=read(Path(os.environ['DATASET_BUDGET_LEDGER']))
    from budget import charged
    limit=ledger['limit_usd']
    require(charged(ledger)<=limit,'Budget limit exceeded')
    save(output/'budget.json',ledger)
    for path in config.get('supporting_evidence',[]):
        source=Path(path)
        require(not source.is_absolute() and '..' not in source.parts, 'Evidence paths must be workspace-relative')
        dest=output/'construction_evidence'/source
        dest.parent.mkdir(parents=True,exist_ok=True)
        if source.is_dir(): shutil.copytree(source,dest)
        else: shutil.copyfile(source,dest)
    accepted=records(staging)
    counts=Counter(r['source']['dataset'] for r in accepted)
    spend=sum(r.get('cost',0) for r in ledger['requests'])
    summary={'count':len(accepted),'datasets':dict(counts),'categories':dict(Counter(r['source']['category'] for r in accepted)),
        'groups':dict(Counter(r['source'].get('difficulty_group','reasoning') for r in accepted)),
        'new_reported_cost_usd':spend,'accounted_cost_including_unknown_reservations_usd':charged(ledger),
        'unknown_cost_requests':sum('cost' not in r for r in ledger['requests']),
        'budget_usd':limit,'processor_parity':'pending_model_selection','async_inference_run':False}
    save(output/'summary.json',summary)
    gallery=['# Image and text shards\n\nPic 1 is edited; pic 2 is the original source. Both are normalized.\n']
    for r in accepted:
        src=r['source']; images=r['normalization']['images']
        gallery.extend([f"\n## {src['dataset']} / {src['source_id']}\n\n",
            f"| Pic 1 | Pic 2 |\n|---|---|\n| ![Before]({images['before']['path']}) | ![After]({images['after']['path']}) |\n\n",
            f"Text shard 1: {src['question']}\n\nText shard 2: {NOTICE}\n"])
    (output/'examples.md').write_text(''.join(gallery))
    lines=['# Diverse async image corrections\n\n',f"{len(accepted)} reviewed pairs. New reported API spend: ${spend:.6f} / ${limit:g}.\n\n",
        '| Dataset | Pairs |\n|---|---|\n']
    lines.extend(f'| {name} | {count} |\n' for name,count in sorted(counts.items()))
    lines.extend(['\n[Image/text shards](examples.md). Model inputs are `inputs.jsonl`; answers, edit descriptions, source metadata and reviewer notes are separate in `annotations.jsonl`.\n',
        f"\nConservative accounted cost, including unknown-cost request reservations: ${charged(ledger):.6f}. Unknown costs retain their reservations rather than being assumed free; see budget.json.\n",
        '\nThese are source-derived questions, not wholly synthetic problems. Pic 1 is AI-edited; pic 2 preserves source content. Prior pilot assets may be reused (see assembly review). New costs exclude those prior calls.\n',
        '\nEach pair passed separate blind solves of both normalized image states and a semantic edit audit, followed by Codex visual review. Proposals, solves and audits use Gemini 3.8 Flash; image edits use Gemini 3.1 Flash Image. This is not independent human ground truth. Non-exact equivalent answers have explicit semantic grading evidence.\n',
        '\nAll pairs match dimensions, aspect ratio, RGB/PNG mode and empty embedded metadata. Original raw files are retained. Minor global rasterization changes are possible. Any relaxed raw-aspect tolerance is explicitly recorded; no blanket alignment override is used. Exact Qwen processor grid/token parity is deferred. No GPU or async inference was run.\n',
        '\nSelection is supervised and intentionally biased toward unambiguous editable samples. Report calibration/reveal controls separately. Keep common-source images grouped across splits; this is a local evaluation pilot, not a representative benchmark score.\n',
        '\nSources and terms: [MathVista](https://huggingface.co/datasets/AI4Math/MathVista), [MathVision](https://huggingface.co/datasets/MathLLMs/MathVision), [CharXiv](https://huggingface.co/datasets/princeton-nlp/CharXiv). Revisions and original metadata are retained per sample. MathVista contributions and CharXiv questions use CC-BY-SA-4.0; MathVision card lists MIT. Original image rights remain with their sources. CharXiv is evaluation-only, not training data. Preserve attribution and check rights before redistribution.\n'])
    lines.append('\nAdditional sources, when present: ChartQA (dataset card GPL-3.0; original chart rights retained), TabMWP (CC-BY-NC-SA-4.0), MapQA-U (CC-BY-SA-4.0; retain KFF map rights), CLEVR (CC-BY-4.0). Source images are reused, not newly rerendered; original CLEVR scenes are synthetic. This mixed-license artifact is for local evaluation; public redistribution needs source-specific rights review.\n')
    (output/'README.md').write_text(''.join(lines))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--staging',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); assemble(a.config,a.staging,a.output)
