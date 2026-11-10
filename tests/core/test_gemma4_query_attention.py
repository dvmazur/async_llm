"""Chunking the submitted queries keeps the same causal/window semantics."""
import pytest
import torch

from test_gemma4_text import models, run
from tests.gemma_tolerances import GEMMA_SMALL_LOGIT_ATOL

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('stops', [(1, 7, 8, 9, 25), (8, 16, 24, 25), (25,)])
def test_query_chunks_match_full_context(models, stops):
    reference, actual, context = models
    ids = list(range(3, 28))
    start = 0
    for stop in stops:
        result = run(actual, context, ids[start:stop], cached=start)
        with torch.inference_mode():
            expected = reference(torch.tensor([ids[:stop]], device='cuda'),
                                 use_cache=False).logits[:, -1]
        torch.testing.assert_close(result, expected, atol=GEMMA_SMALL_LOGIT_ATOL, rtol=0)
        start = stop
