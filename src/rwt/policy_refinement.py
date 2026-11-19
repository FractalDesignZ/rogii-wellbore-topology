"""Bounded, observed-objective refinement of a finite policy identifier."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from hashlib import sha256
from itertools import islice
import json
import math
from numbers import Real


@dataclass(frozen=True)
class PolicyTrial:
    policy: str
    state_id: str
    objective: float


@dataclass(frozen=True)
class RefinementStep:
    incumbent: PolicyTrial
    proposals: tuple[PolicyTrial, ...]
    accepted: PolicyTrial | None


@dataclass(frozen=True)
class RefinementResult:
    policy: str
    state_id: str
    objective: float | None
    stop_reason: str
    objective_calls: int
    steps: tuple[RefinementStep, ...]


def _key(value: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("policy identifiers must be nonempty canonical strings")
    return value


def refine_policy(initial: str, *, off: str, evidence_sha256: str, supported: bool,
                  propose: Callable[[str], Iterable[str]], objective: Callable[[str], float],
                  max_steps: int, proposal_limit: int) -> RefinementResult:
    """Evaluate a bounded ordered proposal prefix and accept strict decreases.

    The caller binds observations, their roles, scoring semantics and policy
    definitions to the supplied digest. The objective must be deterministic
    on that evidence. Each policy is measured at most once. The step trace
    includes revisited proposals. The objective-call bound is
    1 + max_steps * proposal_limit. Stationarity and ancestor-only proposed
    cycles describe the inspected prefix. A strictly improving accepted path
    cannot cycle on fixed evidence. Callers requiring wall-time or resource
    limits should run their callbacks in a bounded worker.
    """
    _key(initial)
    _key(off)
    if (not isinstance(evidence_sha256, str) or len(evidence_sha256) != 64
            or any(char not in "0123456789abcdef" for char in evidence_sha256)):
        raise ValueError("expected a lowercase SHA-256 evidence digest")
    if (type(supported) is not bool or type(max_steps) is not int or max_steps < 1
            or type(proposal_limit) is not int or proposal_limit < 1):
        raise ValueError("expected boolean support and positive integer bounds")

    def state_id(policy: str) -> str:
        data = json.dumps([evidence_sha256, policy], separators=(",", ":")).encode()
        return sha256(data).hexdigest()

    if not supported:
        return RefinementResult(off, state_id(off), None, "missing_support", 0, ())

    observed: dict[str, PolicyTrial] = {}

    def observe(policy: str) -> PolicyTrial:
        if policy not in observed:
            value = objective(policy)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError("expected a finite numerical objective")
            try:
                value = float(value)
            except OverflowError as exc:
                raise ValueError("expected a finite numerical objective") from exc
            if not math.isfinite(value):
                raise ValueError("expected a finite numerical objective")
            observed[policy] = PolicyTrial(policy, state_id(policy), value)
        return observed[policy]

    current = observe(initial)
    ancestors: set[str] = set()
    steps: list[RefinementStep] = []
    reason = "step_budget"
    for _ in range(max_steps):
        suggestions = propose(current.policy)
        if isinstance(suggestions, (str, bytes)):
            raise ValueError("expected an iterable of individual policy identifiers")
        keys = tuple(dict.fromkeys(_key(x) for x in islice(suggestions, proposal_limit)))
        trials = tuple(observe(key) for key in keys)
        best = min(trials, key=lambda v: (v.objective, v.policy), default=current)
        accepted = best if best.objective < current.objective else None
        steps.append(RefinementStep(current, trials, accepted))
        if accepted is not None:
            ancestors.add(current.policy)
            current = accepted
            continue
        if not keys or keys == (current.policy,):
            reason = "stationary"
        elif all(key in ancestors for key in keys):
            reason = "proposal_cycle"
        else:
            reason = "no_improvement"
        break
    return RefinementResult(current.policy, current.state_id, current.objective, reason,
                            len(observed), tuple(steps))
