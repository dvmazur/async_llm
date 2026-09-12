"""Compare teacher-forced logits; initial prefill is NOT counted as decode."""
import json
import math

import torch

from .serving import check_schedule


ERROR_METRICS = ('mean_tv', 'p95_tv', 'mean_centered_relative_l2', 'p95_centered_relative_l2')
RELATIVE_ERROR_TOLERANCE = 0.05


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


def relative_error_summary(metrics):
    """Compare two errors to SGLang, not mini-to-Transformers distance.

    User-approved 5% relative allowance replaces the old additive margins:
    error(mini, SG) <= 1.05 * error(Transformers, SG). For example, prefill
    p95 L2 0.207365 / 0.201603 - 1 is about +2.86%, not +2.86 probability pp.
    Top1 agreement is diagnostic only and never participates in these gates.
    """
    gates, deltas = {}, {}
    for phase in ('prefill', 'decode'):
        gates[phase], deltas[phase] = {}, {}
        for key in ERROR_METRICS:
            actual, baseline = metrics['mini'][phase][key], metrics['transformers'][phase][key]
            valid = math.isfinite(actual) and math.isfinite(baseline) and actual >= 0 and baseline >= 0
            gates[phase][key] = valid and actual <= baseline * (1 + RELATIVE_ERROR_TOLERANCE)
            # No epsilon/additive floor: exact TF/SG equality permits only zero
            # error. Report an undefined ratio as JSON null, never as NaN/Inf.
            delta = (100 * (actual / baseline - 1) if baseline > 0 else
                     0.0 if actual == 0 else None) if valid else None
            deltas[phase][key] = delta if delta is None or math.isfinite(delta) else None
    return dict(gates=gates, passed=all(v for phase in gates.values() for v in phase.values()),
                relative_tolerance_percent=100 * RELATIVE_ERROR_TOLERANCE,
                relative_error_delta_percent=deltas)


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
    quality = relative_error_summary(metrics)
    schedule=None
    if scheduling in ('mixed','sequential'):
        schedule=json.loads((mini/'schedule.json').read_text())
        assert schedule['mode']==scheduling
        check_schedule(schedule['forwards'],scheduling)
    return dict(metrics=metrics,**quality,
        mini_scheduling=scheduling,mixed_schedule=schedule,
        per_case=per_case,fixtures_sha256=manifests[0]['fixtures_sha256'],
        quant_output_ablation=manifests[0]['arguments'].get('reference_quant_outputs',False),
        sglang_native_mrope_override=manifests[1]['arguments'].get('sglang_native_mrope',False),
        notes='Distances to SGLang, not task accuracy. Numerical errors may exceed Transformers errors by '
              'at most 5% relatively, not by an additive 0.05; top1 agreement is diagnostic only. '
              'See mini_scheduling/mixed_schedule for mixed coverage; '
              'the SGLang/Transformers references remain sequential, with identical teacher histories. '
              'Check quant_output_ablation: an explicit benchmark override is not production validation. '
              'Check sglang_native_mrope_override: if true, the reference uses SGLang native mRoPE '
              'to bypass its installed 1D-only fused CUDA preparation; default results are separate. '
              'Decode excludes logits after initial prefill. FP8 config correction for tied '
              'Transformers lm_head is documented in its reference output directory.')
