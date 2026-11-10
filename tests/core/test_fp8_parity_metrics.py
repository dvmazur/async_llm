import json
import math

import pytest
import torch

from tests.e2e.fp8 import comparison as metrics


def test_metrics_identical_and_offset():
    x=torch.tensor([[1.,2.,3.],[-1.,0.,1.]])
    tv,l2,agree=metrics.distances(x+10,x)
    torch.testing.assert_close(tv,torch.zeros(2),rtol=0,atol=0)
    torch.testing.assert_close(l2,torch.zeros(2),rtol=0,atol=0)
    assert agree.all()


def test_metrics_detect_change_and_invalid():
    x=torch.tensor([[1.,2.,3.]])
    tv,l2,agree=metrics.distances(x.flip(-1),x)
    assert tv.item()>.5 and l2.item()>1 and not agree.item()
    with pytest.raises(AssertionError):metrics.distances(x*float('nan'),x)


def test_transformers_reference_is_strict_and_uses_its_own_l2_denominator(tmp_path):
    # Relative L2 is asymmetric: recompute both errors to TF, not to SG.
    logits = {'mini': torch.tensor([0., 2., 4.]),
              'transformers': torch.tensor([0., 1., 2.]),
              'sglang': torch.tensor([0., 4., 8.])}
    for engine, values in logits.items():
        root = tmp_path / engine
        root.mkdir()
        (root / 'complete.json').write_text(json.dumps(dict(
            fixtures_sha256='same', cases=['case'], arguments=dict(
                model='same-checkpoint', tokens=3, mini_scheduling='shared-cache', quantization='fp8'))))
        torch.save(values, root / 'case_cold0.pt')
        torch.save(values.repeat(3, 1), root / 'case_decode.pt')
    result = metrics.compare(*(tmp_path / name for name in ('mini', 'sglang', 'transformers')),
                             reference='transformers')
    assert result['reference'] == 'transformers' and result['baseline_engine'] == 'sglang'
    assert result['relative_tolerance_percent'] == 0 and result['passed']
    assert result['metrics']['mini']['prefill']['mean_centered_relative_l2'] == pytest.approx(1.)
    assert result['metrics']['sglang']['prefill']['mean_centered_relative_l2'] == pytest.approx(3.)
    assert result['metrics']['mini']['prefill']['positions'] == 1
    assert result['metrics']['mini']['decode']['positions'] == 2
    errors = result['metrics']
    errors['mini']['decode']['p95_tv'] = math.nextafter(errors['sglang']['decode']['p95_tv'], math.inf)
    assert not metrics.relative_error_summary(errors, reference='transformers')['passed']


def _errors(actual, baseline):
    return {engine: {phase: dict.fromkeys(metrics.ERROR_METRICS, value)
                     for phase in ('prefill', 'decode')}
            for engine, value in (('mini', actual), ('transformers', baseline))}


@pytest.mark.parametrize('reference', ['sglang', 'transformers'])
@pytest.mark.parametrize('phase', ['prefill', 'decode'])
@pytest.mark.parametrize('key', metrics.ERROR_METRICS)
def test_gemma_factor_is_uniform_and_retains_strict_diagnostics(reference, phase, key):
    factor = metrics.GEMMA_REFERENCE_ERROR_FACTOR
    errors = _errors(.1, .1)
    errors['sglang'] = errors['transformers']
    errors['mini'][phase][key] = .1 * factor
    result = metrics.relative_error_summary(errors, reference=reference,
                                           bf16_control=True, error_factor=factor)
    assert result['passed'] and not result['strict_passed']
    assert not result['strict_gates'][phase][key]
    assert result['relative_tolerance_percent'] == 50
    assert all(v == factor for row in result['error_factors'].values() for v in row.values())
    errors['mini'][phase][key] = math.nextafter(.1 * factor, math.inf)
    assert not metrics.relative_error_summary(errors, reference=reference, error_factor=factor)['passed']


@pytest.mark.parametrize('actual,baseline,passed', [
    (0., 0., True), (1e-12, 0., False), (float('nan'), .1, False),
    (.1, float('inf'), False), (float('inf'), .1, False), (-.1, .1, False),
])
def test_gemma_factor_does_not_hide_zero_or_invalid_errors(actual, baseline, passed):
    result = metrics.relative_error_summary(_errors(actual, baseline),
                                           error_factor=metrics.GEMMA_REFERENCE_ERROR_FACTOR)
    assert result['passed'] is passed
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('factor', [0., .9, float('nan'), float('inf')])
def test_invalid_explicit_factor_is_rejected(factor):
    with pytest.raises(ValueError, match='error_factor'):
        metrics.relative_error_summary(_errors(.1, .1), error_factor=factor)


def test_explicit_factor_one_is_strict_and_conflicting_options_are_rejected():
    result = metrics.relative_error_summary(_errors(.1, .1), error_factor=1.)
    assert result['passed'] and result['strict_passed']
    assert not result['acceptance_policy_fragile']
    with pytest.raises(ValueError, match='strict=True'):
        metrics.relative_error_summary(_errors(.1, .1), strict=True,
                                       error_factor=metrics.GEMMA_REFERENCE_ERROR_FACTOR)


@pytest.mark.parametrize('phase', ['prefill', 'decode'])
@pytest.mark.parametrize('key', metrics.ERROR_METRICS)
def test_strict_gate_rejects_any_regressed_metric(phase, key):
    errors = _errors(.1, .1)
    assert metrics.relative_error_summary(errors, strict=True)['passed']
    errors['mini'][phase][key] = math.nextafter(.1, math.inf)
    result = metrics.relative_error_summary(errors, strict=True, bf16_control=True)
    assert not result['passed'] and not result['gates'][phase][key]
    assert result['relative_tolerance_percent'] == 0
    assert all(f == 1 for row in result['error_factors'].values() for f in row.values())


@pytest.mark.parametrize('baseline', [1e-8, .02, .2, 10.0])
@pytest.mark.parametrize('factor,passed', [(.8, True), (1., True), (1.0286, True),
                                          (1.05, True), (1.1, True), (2., True), (2.000001, False)])
def test_temporary_fragile_allowance_is_twice_reference_error(baseline, factor, passed):
    result = metrics.relative_error_summary(_errors(baseline * factor, baseline))
    assert result['passed'] is passed
    assert result['relative_tolerance_percent'] == 100
    assert result['acceptance_policy_fragile'] is True
    assert result['acceptance_policy'] == 'temporary_fragile_2x_transformers_error'
    for phase in ('prefill', 'decode'):
        for key in metrics.ERROR_METRICS:
            assert result['gates'][phase][key] is passed
            assert result['relative_error_delta_percent'][phase][key] == pytest.approx(100*(factor-1))


@pytest.mark.parametrize('phase', ['prefill', 'decode'])
@pytest.mark.parametrize('key', metrics.ERROR_METRICS)
def test_each_metric_has_its_own_gate(phase, key):
    errors = _errors(.2, .2)
    errors['mini'][phase][key] = math.nextafter(.2 * 2, math.inf)
    result = metrics.relative_error_summary(errors)
    assert not result['passed']
    assert sum(not passed for values in result['gates'].values() for passed in values.values()) == 1
    assert not result['gates'][phase][key]


@pytest.mark.parametrize('actual,baseline,passed,delta', [
    (0., 0., True, 0.), (1e-12, 0., False, None), (0., .2, True, -100.),
    (float('nan'), .2, False, None), (.2, float('inf'), False, None),
    (float('inf'), .2, False, None), (-.1, .2, False, None), (.2, -.1, False, None),
])
def test_zero_and_invalid_reference_errors_are_not_hidden(actual, baseline, passed, delta):
    result = metrics.relative_error_summary(_errors(actual, baseline))
    assert result['passed'] is passed
    assert result['relative_error_delta_percent']['prefill']['mean_tv'] == delta
    json.dumps(result, allow_nan=False)


def test_top1_is_not_a_numerical_acceptance_gate():
    errors = _errors(.2, .2)
    for phase in ('prefill', 'decode'):
        errors['mini'][phase]['top1_agreement'] = 0
        errors['transformers'][phase]['top1_agreement'] = 1
    result = metrics.relative_error_summary(errors)
    assert result['passed']
    assert all('top1_agreement' not in phase for phase in result['gates'].values())


@pytest.mark.parametrize('phase', ['prefill', 'decode'])
@pytest.mark.parametrize('key', metrics.ERROR_METRICS)
def test_bf16_exception_is_confined_to_prefill_p95_tv(phase, key):
    errors = _errors(.1, .1)
    errors['mini'][phase][key] = .21603
    assert not metrics.relative_error_summary(errors)['passed']  # FP8 is not relaxed
    result = metrics.relative_error_summary(errors, bf16_control=True)
    assert result['passed'] is (phase == 'prefill' and key == 'p95_tv')
    assert result['error_factors'][phase][key] == (2.2 if result['passed'] else 2.)


@pytest.mark.parametrize('actual,baseline,passed', [
    (.1 * 2.2, .1, True), (math.nextafter(.1 * 2.2, math.inf), .1, False),
    (0., 0., True), (1e-12, 0., False), (float('nan'), .1, False), (.1, float('inf'), False),
])
def test_bf16_exception_keeps_boundary_and_invalid_error_checks(actual, baseline, passed):
    errors = _errors(.1, .1)
    errors['mini']['prefill']['p95_tv'] = actual
    errors['transformers']['prefill']['p95_tv'] = baseline
    result = metrics.relative_error_summary(errors, bf16_control=True)
    assert result['passed'] is passed
    assert result['acceptance_policy_fragile']
    json.dumps(result, allow_nan=False)
