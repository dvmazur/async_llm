import pytest

from minisgl_examples.training.parallel_thinking import budget_long_verify as audit
from minisgl_examples.training.parallel_thinking.budget_curve_eval import token_sha256


class Tokenizer:
    def decode(self, tokens, **kwargs):
        return "".join({1: "\\boxed{", 2: "3", 3: "}", 4: "x", 99: "<eos>"}[t] for t in tokens)


@pytest.mark.parametrize("stop", ["max_steps", "context_limit"])
def test_no_observed_eos_at_limit(stop):
    rollout = {"worker_token_ids": {"A": [7, 4]}, "eos_worker": None, "eos_token_id": None}
    assert audit.verify_source_eos(rollout, stop, {99}, "case") == {99}


@pytest.mark.parametrize("stream,stop", [([99, 4], "max_steps"), ([4, 99], "context_limit"), ([4], "eos")])
def test_missing_eos_metadata_cannot_hide_invalid_stop(stream, stop):
    with pytest.raises(ValueError):
        audit.verify_source_eos({"worker_token_ids": {"A": stream}}, stop, {99}, "case")


def test_simultaneous_worker_eos():
    rollout = {"worker_token_ids": {"A": [4, 99], "B": [7, 98]}, "eos_worker": "A",
               "eos_token_id": 99, "eos_token_ids": [98, 99]}
    assert audit.verify_source_eos(rollout, "eos", {98, 99}, "case") == {98, 99}


def test_worker_view_follows_publication_order_and_excludes_eos():
    rollout = dict(layout="native_workers", workers=["A", "B"], prompt_ids=[100, 7], decode_rounds=4,
                   publication_policy="encoded_separator_v1", worker_token_ids={"A": [10,11,12,99],"B":[20,21,22,23]},
                   publications=[dict(worker="A",round=3,committed_token_count=2),
                                 dict(worker="B",round=3,committed_token_count=2)])
    expected = {2: [100,7,20,21,10,11], 3: [100,7,10,11,20,21,22,12], 9000: [100,7,10,11,20,21,22,23,12]}
    for budget, tokens in expected.items():
        view = audit.reconstruct(rollout,budget,{99},7,8,{True:[8,101],False:[101]})
        assert view["context_sha256"] == token_sha256(tokens+[8,101])
        assert view["view_sha256"] == token_sha256(tokens)
        assert view["effective_rounds"] == min(4,budget)
        assert view["publication_count"] == (0 if budget<3 else 2)


@pytest.mark.parametrize("layout", ["causal","native"])
def test_single_worker_omits_future_tokens(layout):
    rollout = dict(layout=layout, workers=["A"], prompt_ids=[2], decode_rounds=4,
                   worker_token_ids={"A":[7,10,8,99]}, publications=[])
    cues = {True:[8,5], False:[5]}
    assert audit.reconstruct(rollout, 2, {99},7,8,cues)["cue_token_ids"] == [8,5]
    assert audit.reconstruct(rollout, 4, {99},7,8,cues)["context_sha256"] == token_sha256([2,7,10,8,5])


@pytest.mark.parametrize("tokens,result", [([2,3],("answer","3")),([99],("eos",None)),
                                           ([2]*8,("max_tokens",None))])
def test_first_complete_box_cap_or_eos(tokens,result):
    assert audit.answer_from_tokens(tokens,[1],Tokenizer(),{99}) == result


@pytest.mark.parametrize("tokens", [[99,4],[2,3,4],[2],[],[4]*9])
def test_invalid_sample_stop(tokens):
    with pytest.raises(ValueError):
        audit.answer_from_tokens(tokens,[1],Tokenizer(),{99})
