"""Compare teacher-forced logits; initial prefill is NOT counted as decode."""
import json
import math

import torch

from .serving import check_schedule
from tests.gemma_tolerances import GEMMA_REFERENCE_ERROR_FACTOR

ERROR_METRICS = ('mean_tv', 'p95_tv', 'mean_centered_relative_l2', 'p95_centered_relative_l2')
# User-approved 2026-09-15: temporarily relax 1.05x to 2x TF-to-SGLang error.
# FRAGILE: batch/checkpoint/reference-sensitive, not calibrated task accuracy.
# Keep visible in reports; reconsider or remove this provisional gate later.
RELATIVE_ERROR_TOLERANCE = 1.0  # +100% relative error => factor 2, NOT additive 1.
# User-approved 2026-09-15 follow-up: the unquantized BF16 chain control had
# prefill p95 TV / TF error = 2.1603. Temporarily allow 2.2 ONLY for this metric
# in BF16 controls (all chain layouts); FP8 and every other metric stay at 2x.
# FRAGILE, chosen after observing this small panel, not an accuracy repair.
BF16_PREFILL_P95_TV_FACTOR = 2.2


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


def relative_error_summary(metrics, *, bf16_control=False, strict=False, reference="sglang",
                           error_factor=None):
    """Compare both engines to one explicitly selected reference.

    Temporary, fragile user-approved rule: error(mini, SG) <= 2 * error(TF, SG).
    This is a tolerance-policy change, NOT a numerical fix or quality guarantee.
    Apply to mean/p95 TV and centered L2 in both phases. Max TV and top1 remain
    diagnostics. BF16 controls alone have a fragile 2.2x prefill p95 TV exception.
    Zero baseline error still permits only zero Mini error. Gemma instead uses
    Transformers as reference and defaults to zero tolerance; Gemma tests explicitly
    supply GEMMA_REFERENCE_ERROR_FACTOR. An explicit factor applies uniformly.
    """
    assert reference in ("sglang", "transformers")
    baseline_engine = "sglang" if reference == "transformers" else "transformers"
    if error_factor is not None:
        if not math.isfinite(error_factor) or error_factor < 1:
            raise ValueError("error_factor must be finite and at least 1")
        if strict and error_factor != 1:
            raise ValueError("strict=True cannot be combined with a relaxed error_factor")
    strict = strict or (reference == "transformers" and error_factor is None)
    gates, strict_gates, deltas, factors = {}, {}, {}, {}
    for phase in ('prefill', 'decode'):
        gates[phase], strict_gates[phase], deltas[phase], factors[phase] = {}, {}, {}, {}
        for key in ERROR_METRICS:
            actual, baseline = metrics['mini'][phase][key], metrics[baseline_engine][phase][key]
            valid = math.isfinite(actual) and math.isfinite(baseline) and actual >= 0 and baseline >= 0
            factor = (BF16_PREFILL_P95_TV_FACTOR if bf16_control and phase == 'prefill' and key == 'p95_tv'
                      else 1 + RELATIVE_ERROR_TOLERANCE)
            if strict:
                factor = 1.0
            if error_factor is not None:
                factor = error_factor
            factors[phase][key] = factor
            gates[phase][key] = valid and actual <= baseline * factor
            strict_gates[phase][key] = valid and actual <= baseline
            # No epsilon/additive floor: exact TF/SG equality permits only zero
            # error. Report an undefined ratio as JSON null, never as NaN/Inf.
            delta = (100 * (actual / baseline - 1) if baseline > 0 else
                     0.0 if actual == 0 else None) if valid else None
            deltas[phase][key] = delta if delta is None or math.isfinite(delta) else None
    return dict(gates=gates, passed=all(v for phase in gates.values() for v in phase.values()),
                strict_gates=strict_gates,
                strict_passed=all(v for phase in strict_gates.values() for v in phase.values()),
                relative_tolerance_percent=(100 * (error_factor - 1) if error_factor is not None else
                                            0 if strict else 100 * RELATIVE_ERROR_TOLERANCE),
                error_factors=factors, bf16_control=bf16_control,
                relative_error_delta_percent=deltas,
                reference=reference, baseline_engine=baseline_engine,
                acceptance_policy=('calibrated_reference_error_factor' if error_factor is not None else
                                   'no_worse_than_sglang_to_transformers' if reference == 'transformers' else
                                   'no_worse_than_transformers_to_sglang' if strict else
                                   'temporary_fragile_2x_with_bf16_prefill_p95_tv_2_2x' if bf16_control
                                   else 'temporary_fragile_2x_transformers_error'),
                acceptance_policy_fragile=(error_factor > 1 if error_factor is not None else not strict))


def compare(mini,sglang,transformers, *, strict=False, reference="sglang", error_factor=None):
    assert reference in ("sglang", "transformers")
    baseline = "sglang" if reference == "transformers" else "transformers"
    compared = ("mini", baseline)
    manifests=[json.loads((p/'complete.json').read_text()) for p in (mini,sglang,transformers)]
    assert len({m['fixtures_sha256'] for m in manifests})==1
    assert all(m['cases']==manifests[0]['cases'] for m in manifests)
    assert len({m['arguments']['model'] for m in manifests})==1
    assert len({m['arguments']['tokens'] for m in manifests})==1
    scheduling=manifests[0]['arguments'].get('mini_scheduling','shared-cache')
    results={name:{phase:[] for phase in ('prefill','decode')} for name in compared}
    per_case={}
    direct={phase:[] for phase in ('prefill','decode')}
    files=sorted(mini.glob('*.pt'))
    assert files
    for path in files:
        actuals={name:torch.load(root/path.name,weights_only=True,map_location='cpu')
                 for name,root in [('mini',mini),('sglang',sglang),('transformers',transformers)]}
        ref = actuals[reference]
        for name in compared:
            values=actuals[name]
            if path.stem.endswith('_decode'):
                if scheduling in ('mixed','sequential'):
                    # In mixed serving the first row is the actual mixed-prefill
                    # output. Cold-prefill reruns alone would miss that coverage.
                    initial=distances(values[:1],ref[:1])
                    results[name]['prefill'].append(initial)
                    per_case.setdefault(path.stem+'_initial_prefill',{})[name]=aggregate([initial])
                phase='decode'
                values,reference_values=values[1:],ref[1:]
            else:
                phase='prefill';reference_values=ref
            row=distances(values,reference_values)
            results[name][phase].append(row)
            per_case.setdefault(path.stem,{})[name]=aggregate([row])
        a,b=actuals['mini'],actuals['transformers']
        if path.stem.endswith('_decode'):
            if scheduling in ('mixed','sequential'):
                direct['prefill'].append(distances(a[:1],b[:1]))
            a,b=a[1:],b[1:]
        direct[phase].append(distances(a,b))
    metrics={name:{phase:aggregate(rows) for phase,rows in phases.items()}
             for name,phases in results.items()}
    bf16_control = (scheduling in ('chain-flat', 'chain-split', 'chain-shared')
                    and manifests[0]['arguments'].get('quantization', 'fp8') is None)
    quality = relative_error_summary(metrics, bf16_control=bf16_control, strict=strict,
                                     reference=reference, error_factor=error_factor)
    schedule=None
    if scheduling in ('mixed','sequential'):
        schedule=json.loads((mini/'schedule.json').read_text())
        assert schedule['mode']==scheduling
        check_schedule(schedule['forwards'],scheduling)
    return dict(metrics=metrics,**quality,
        mini_vs_transformers={phase:aggregate(rows) for phase,rows in direct.items()},
        mini_scheduling=scheduling,mixed_schedule=schedule,
        per_case=per_case,fixtures_sha256=manifests[0]['fixtures_sha256'],
        quant_output_ablation=manifests[0]['arguments'].get('reference_quant_outputs',False),
        sglang_native_mrope_override=manifests[1]['arguments'].get('sglang_native_mrope',False),
        notes=(f'Calibrated acceptance: error(Mini, reference) <= {error_factor:g} * error(baseline, reference). '
               'Same multiplier for every prefill/decode mean/p95 TV and centered L2. '
               'This is a tolerance relaxation, NOT an accuracy improvement. '
               'strict_gates and strict_passed retain the original zero-tolerance result.' if error_factor is not None else
               'Strict: error(Mini, TF) <= error(SGLang, TF), every prefill/decode mean/p95 '
               'TV and centered L2. Both L2 distances are normalized by the SAME Transformers logits. '
               'No added tolerance. Max TV and top1 are diagnostics.' if reference == 'transformers' else
               'Strict: every prefill/decode mean/p95 TV and centered-L2 error to SGLang must be '
               'no larger than Transformers error to the SAME SGLang reference. No added tolerance. '
               'Max TV and top1 are diagnostics.' if strict else
              'Distances to SGLang, not task accuracy. Numerical errors may exceed Transformers errors by '
              'a factor of 2 by default (BF16-control prefill p95 TV only: 2.2). '
              'See error_factors for effective per-metric limits. TEMPORARY FRAGILE tolerance, '
              'not a task-quality guarantee. '
              'Max TV and top1 agreement are diagnostic only. '
              'See mini_scheduling/mixed_schedule for mixed coverage; '
              'the SGLang/Transformers references remain sequential, with identical teacher histories. '
              'Check quant_output_ablation: an explicit benchmark override is not production validation. '
              'Check sglang_native_mrope_override: if true, the reference uses SGLang native mRoPE '
              'to bypass its installed 1D-only fused CUDA preparation; default results are separate. '
              'Decode excludes logits after initial prefill. FP8 config correction for tied '
              'Transformers lm_head is documented in its reference output directory.'))
