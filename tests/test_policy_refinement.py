from itertools import count

import pytest

from rwt.policy_refinement import refine_policy


def run(propose, objective, **kwargs):
    settings = dict(off="off", evidence_sha256="a"*64, supported=True,
                    max_steps=3, proposal_limit=2)
    return refine_policy("a", propose=propose, objective=objective, **(settings | kwargs))


def test_monotonic_acceptance_and_ancestor_proposal():
    measured = []
    def measure(key):
        measured.append(key)
        return {"a": 5., "b": 2., "c": 2.}[key]
    result = run(lambda k: ["c", "b"] if k == "a" else ["a"], measure)
    assert result.policy == "b" and result.stop_reason == "proposal_cycle"
    assert measured == ["a", "c", "b"] and result.objective_calls == 3
    assert [s.accepted.policy if s.accepted else None for s in result.steps] == ["b", None]
    other = run(lambda k: [], measure, evidence_sha256="b"*64)
    assert other.state_id != result.steps[0].incumbent.state_id


def test_budget_unchanged_and_off():
    run_forever = lambda k: (str(i) for i in count(int(k) + 1 if k != "a" else 1))
    capped = run(run_forever, lambda k: -int(k) if k != "a" else 0.)
    assert capped.stop_reason == "step_budget" and capped.objective_calls == 7
    assert run(lambda k: ["b"], lambda k: 0.).stop_reason == "no_improvement"
    assert run(lambda k: ["a"], lambda k: 0.).stop_reason == "stationary"
    def forbidden(_):
        raise AssertionError("unsupported evidence reached the planner")
    off = run(forbidden, forbidden, supported=False)
    assert (off.policy, off.objective_calls, off.stop_reason) == ("off", 0, "missing_support")


def test_invalid_state_and_objective_fail_closed():
    with pytest.raises(ValueError):
        run(lambda k: [" b"], lambda k: 0.)
    with pytest.raises(ValueError):
        run(lambda k: "abc", lambda k: 0.)
    with pytest.raises(ValueError):
        run(lambda k: [], lambda k: float("nan"))
    with pytest.raises(ValueError):
        run(lambda k: [], lambda k: 0., max_steps=0)
