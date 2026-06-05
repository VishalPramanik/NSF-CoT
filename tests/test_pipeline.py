"""End-to-end pipeline tests on the bundled offline examples.

These verify the qualitative behaviour reported in the paper: the confabulated
step is flagged Unfaithful (and pruned), every evidence-grounded step is
Faithful, and attribution recovers an empty relied-upon set for the
confabulation.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nsfcot import (  # noqa: E402
    NSFCoT,
    NSFCoTConfig,
    DictionaryParser,
    MockLM,
    OverlapJudge,
    steps_to_prune,
    prune_chain,
)
from nsfcot.data import Example  # noqa: E402

EXAMPLES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples"
)


def _audit(name: str):
    example = Example.load(os.path.join(EXAMPLES_DIR, name))
    nsf = NSFCoT(
        parser=DictionaryParser(example.parse_table()),
        lm=MockLM(example.relevance),
        judge=OverlapJudge(),
        config=NSFCoTConfig(),
    )
    return example, nsf.audit(example)


@pytest.mark.parametrize("name", ["qasc_evaporation.json", "hummingbird.json"])
def test_runs_without_error(name):
    example, result = _audit(name)
    assert len(result.steps) == len(example.steps)


@pytest.mark.parametrize("name", ["qasc_evaporation.json", "hummingbird.json"])
def test_confabulated_step_is_unfaithful(name):
    # In both bundled examples, step index 3 (s4) is the confabulation.
    _, result = _audit(name)
    s4 = result.steps[3]
    assert s4.verdict == "Unfaithful"
    assert s4.F < 0.5
    assert s4.relied_upon == []          # I_k = empty for an unsupported claim
    assert min(s4.scores.G, s4.scores.V) < 0.5


@pytest.mark.parametrize("name", ["qasc_evaporation.json", "hummingbird.json"])
def test_evidence_steps_are_faithful(name):
    _, result = _audit(name)
    for k in (0, 1, 2, 4):               # s1, s2, s3, s5
        step = result.steps[k]
        assert step.verdict == "Faithful", f"s{k + 1} should be faithful"
        assert step.F >= 0.5
        assert step.relied_upon, f"s{k + 1} should rely on at least one fact"


@pytest.mark.parametrize("name", ["qasc_evaporation.json", "hummingbird.json"])
def test_pruning_removes_only_the_confabulation(name):
    example, result = _audit(name)
    drop = steps_to_prune(result)
    assert drop == [3]
    pruned = prune_chain([s.text for s in example.steps], result)
    assert pruned[3] == "[REMOVED]"
    assert all(pruned[k] != "[REMOVED]" for k in (0, 1, 2, 4))


def test_multi_hop_proof_support_in_qasc():
    # s5 (Causes(sunlight, water_to_vapor)) is only derivable by chaining
    # Causes-transitivity over f1, f2, f3 -- check it is grounded and that its
    # proof support draws on multiple facts.
    _, result = _audit("qasc_evaporation.json")
    s5 = result.steps[4]
    assert s5.scores.G_smt == 1.0
    assert len(s5.scores.proof_support) >= 2


def test_scores_lie_in_unit_interval():
    for name in ("qasc_evaporation.json", "hummingbird.json"):
        _, result = _audit(name)
        for s in result.steps:
            for val in (s.scores.G, s.scores.V, s.scores.U, s.F):
                assert 0.0 <= val <= 1.0
