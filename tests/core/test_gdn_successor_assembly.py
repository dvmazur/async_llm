"""One terminal assembly with successor hits; old code is an independent oracle."""
import gc
import sys

import pytest
import torch

from minisgl.shared_cache.gdn import _materialize_state_parts
from minisgl.shared_cache.shared_block import CacheBlock
from reference_gdn_successor_assembly import begin_decode_state as old_begin
from test_gdn_successor import world, save


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("layout", ["reorder", "duplicate", "mixed_depth", "empty", "all_empty_misses"])
@torch.inference_mode()
def test_partial_hits_one_assembly_matches_literal_old(monkeypatch, device, cache, layout):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    # Rectangular states exercise both layout dimensions; CPU also covers no Triton.
    torch.manual_seed(253)
    gdn, common, tails = world(5, device, dk=16, dv=11)
    state = save(gdn)
    for i in range(5):
        state[i].fill_(i + 1.)
    # Keep workers 0/4 as actual successor hits. Others require composition.
    for i in (1, 2, 3):
        tails[i].set_linear_affine(0, tuple(x.clone() for x in common.linear_affine[0]))
    chains = list(gdn.cache_structure)
    if layout == "duplicate":
        chains[2] = list(chains[1])
    elif layout == "mixed_depth":
        extra = CacheBlock(gdn.device)
        extra.set_linear_affine(0, tuple(x.clone() for x in common.linear_affine[0]))
        chains[2] = [common, extra, tails[2]]
    elif layout in ("empty", "all_empty_misses"):
        for i in ((2,) if layout == "empty" else (1, 2, 3)):
            chains[i] = [CacheBlock(gdn.device)]
    order = [4, 2, 0, 3, 1]
    gdn.set_context([chains[i] for i in order], [tails[i] for i in order])
    original_cat = torch.cat
    counts = []

    def counted(parts, *args, **kwargs):
        # The CPU fallback also concatenates computed frontier rows; that is
        # evaluation, not the redundant terminal assembly targeted here.
        if sys._getframe(1).f_code.co_name in {
            "_materialize_state_parts", "_terminal_states",
            "_evaluate_affine_prefix_plan_cached", "assemble_state_rows", "begin_decode_state",
        }:
            counts.append(sum(x.numel() * x.element_size() for x in parts))
        return original_cat(parts, *args, **kwargs)

    outputs, copies, tickets = [], [], []
    for fn in (old_begin, type(gdn).begin_decode_state):
        gdn.configure_compose_cache(1.6 if cache else 0)
        # Exercise resident prefixes as well as cold/no-cache layouts.
        if cache:
            for _ in range(3):
                gdn.compose_initial_recurrent_state(0, torch.float32, state_v_first=True)
        counts.clear()
        with monkeypatch.context() as patch:
            patch.setattr(torch, "cat", counted)
            output, ticket = fn(gdn, 0)
        outputs.append(output)
        copies.append(list(counts))
        tickets.append(ticket)
    torch.testing.assert_close(outputs[1], outputs[0], rtol=0, atol=0)
    assert tickets[1] == tickets[0]
    # One assembly of the entire worker batch, and no intermediate miss cat.
    assert copies[1] == [outputs[1].numel() * outputs[1].element_size()]
    assert len(copies[0]) >= len(copies[1])
    if layout in ("duplicate", "mixed_depth", "empty"):
        assert len(copies[0]) == 2


def test_deferred_rows_own_their_lifetime_without_retaining_new_cache():
    gdn, common, tails = world(3)
    save(gdn)
    gdn.set_context([gdn.cache_structure[2], [], gdn.cache_structure[0]],
                    [tails[2], tails[1], tails[0]])
    parts = gdn._compose_initial_recurrent_parts(0)
    assert isinstance(parts, list)
    expected = _materialize_state_parts(parts).clone()
    gdn.configure_successor_cache(0)
    gdn.configure_compose_cache(0)
    common.clear()
    for tail in tails:
        tail.clear()
    del gdn, common, tails
    gc.collect()
    torch.testing.assert_close(_materialize_state_parts(parts), expected, rtol=0, atol=0)
