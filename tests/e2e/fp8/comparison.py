"""Compare teacher-forced logits; initial prefill is NOT counted as decode."""
import json

import torch

from .serving import check_schedule


def distances(actual, reference):
    actual=actual.float().reshape(-1,actual.shape[-1])
    reference=reference.float().reshape(-1,reference.shape[-1])
    assert actual.shape==reference.shape
    assert torch.isfinite(actual).all() and torch.isfinite(reference).all()
    tv=(actual.softmax(-1)-reference.softmax(-1)).abs().sum(-1)*.5
    a=actual-actual.mean(-1,keepdim=True)
    b=reference-reference.mean(-1,keepdim=True)
    relative=(a-b).norm(dim=-1)/b.norm(dim=-1).clamp_min(1e-12)
    agreement=actual.argmax(-1)==reference.argmax(-1)
    return tv,relative,agreement


def aggregate(rows):
    tv,relative,agreement=[torch.cat([row[i] for row in rows]) for i in range(3)]
    return dict(positions=len(tv),mean_tv=float(tv.mean()),p95_tv=float(tv.quantile(.95)),
        max_tv=float(tv.max()),mean_centered_relative_l2=float(relative.mean()),
        p95_centered_relative_l2=float(relative.quantile(.95)),
        top1_agreement=float(agreement.float().mean()))


def compare(mini,sglang,transformers):
    manifests=[json.loads((p/'complete.json').read_text()) for p in (mini,sglang,transformers)]
    assert len({m['fixtures_sha256'] for m in manifests})==1
    assert all(m['cases']==manifests[0]['cases'] for m in manifests)
    assert len({m['arguments']['model'] for m in manifests})==1
    assert len({m['arguments']['tokens'] for m in manifests})==1
    scheduling=manifests[0]['arguments'].get('mini_scheduling','shared-cache')
    results={name:{phase:[] for phase in ('prefill','decode')} for name in ('mini','transformers')}
    per_case={}
    files=sorted(mini.glob('*.pt'))
    assert files
    for path in files:
        ref=torch.load(sglang/path.name,weights_only=True,map_location='cpu')
        for name,root in [('mini',mini),('transformers',transformers)]:
            values=torch.load(root/path.name,weights_only=True,map_location='cpu')
            if path.stem.endswith('_decode'):
                if scheduling in ('mixed','sequential'):
                    # In mixed serving the first row is the actual mixed-prefill
                    # output. Cold-prefill reruns alone would miss that coverage.
                    initial=distances(values[:1],ref[:1])
                    results[name]['prefill'].append(initial)
                    per_case.setdefault(path.stem+'_initial_prefill',{})[name]=aggregate([initial])
                phase='decode'
                values,reference=values[1:],ref[1:]
            else:
                phase='prefill';reference=ref
            row=distances(values,reference)
            results[name][phase].append(row)
            per_case.setdefault(path.stem,{})[name]=aggregate([row])
    metrics={name:{phase:aggregate(rows) for phase,rows in phases.items()}
             for name,phases in results.items()}
    # Reference-relative quality limits. Set before examining model results.
    limits=dict(mean_tv=.005,p95_tv=.01,mean_centered_relative_l2=.005,
                p95_centered_relative_l2=.01)
    gates={phase:{key:metrics['mini'][phase][key]<=metrics['transformers'][phase][key]+margin
            for key,margin in limits.items()} for phase in ('prefill','decode')}
    schedule=None
    if scheduling in ('mixed','sequential'):
        schedule=json.loads((mini/'schedule.json').read_text())
        assert schedule['mode']==scheduling
        check_schedule(schedule['forwards'],scheduling)
    return dict(metrics=metrics,gates=gates,passed=all(v for p in gates.values() for v in p.values()),
        mini_scheduling=scheduling,mixed_schedule=schedule,
        additive_margins=limits,per_case=per_case,fixtures_sha256=manifests[0]['fixtures_sha256'],
        quant_output_ablation=manifests[0]['arguments'].get('reference_quant_outputs',False),
        sglang_native_mrope_override=manifests[1]['arguments'].get('sglang_native_mrope',False),
        notes='Distances to SGLang, not task accuracy. See mini_scheduling/mixed_schedule for mixed coverage; '
              'the SGLang/Transformers references remain sequential, with identical teacher histories. '
              'Check quant_output_ablation: an explicit benchmark override is not production validation. '
              'Check sglang_native_mrope_override: if true, the reference uses SGLang native mRoPE '
              'to bypass its installed 1D-only fused CUDA preparation; default results are separate. '
              'Decode excludes logits after initial prefill. FP8 config correction for tied '
              'Transformers lm_head is documented in its reference output directory.')
