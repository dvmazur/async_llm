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


def _errors(actual, baseline):
    return {engine: {phase: dict.fromkeys(metrics.ERROR_METRICS, value)
                     for phase in ('prefill', 'decode')}
            for engine, value in (('mini', actual), ('transformers', baseline))}


@pytest.mark.parametrize('baseline', [1e-8, .02, .2, 10.0])
@pytest.mark.parametrize('factor,passed', [(.8, True), (1., True), (1.0286, True),
                                          (1.05, True), (1.050001, False), (1.1, False)])
def test_allowance_is_five_percent_of_reference_error(baseline, factor, passed):
    result = metrics.relative_error_summary(_errors(baseline * factor, baseline))
    assert result['passed'] is passed
    assert result['relative_tolerance_percent'] == 5
    for phase in ('prefill', 'decode'):
        for key in metrics.ERROR_METRICS:
            assert result['gates'][phase][key] is passed
            assert result['relative_error_delta_percent'][phase][key] == pytest.approx(100*(factor-1))


@pytest.mark.parametrize('phase', ['prefill', 'decode'])
@pytest.mark.parametrize('key', metrics.ERROR_METRICS)
def test_each_metric_has_its_own_gate(phase, key):
    errors = _errors(.2, .2)
    errors['mini'][phase][key] = math.nextafter(.2 * 1.05, math.inf)
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
