from dataclasses import FrozenInstanceError, replace

import pytest

from rwt.merit_weights import (
    FixedWeight, Footprint, Given, HeldMerit, PairedWellLoss, WeightNode, WeightScope, resolve_weights,
)
from rwt.policy_refinement import refine_policy


FP = Footprint(12, 'a' * 64)
WELLS = tuple(f'train-{i}' for i in range(8))
SCOPE = WeightScope('outer-0', WELLS, ('query',))
LOSSES = tuple(PairedWellLoss(w, 1., 4., FP, FP) for w in WELLS)


def node(key, rule, parents=(), evidence=None):
    return WeightNode(key, SCOPE.scope_id, 'survey', evidence or key, rule, parents)


def test_recursive_merit_counts_a_shared_factor_once():
    role = node('role', Given())
    q = node('merit', HeldMerit('skill', LOSSES), ('role',))
    duplicate = replace(q, node_id='second-view')
    left = node('left', FixedWeight('gain', .5), ('merit',))
    right = node('right', Given(), ('merit', 'second-view'))
    action = node('action', Given(), ('left', 'right'))
    graph = resolve_weights(SCOPE, [role, q, duplicate, left, right, action])
    decision = graph.decision('action')
    assert decision.weight == .375 and decision.factors == ('gain', 'skill')
    assert decision.parent_hashes == tuple((n, graph.decision(n).sha256) for n in ('left', 'right'))
    assert graph == resolve_weights(SCOPE, [action, right, left, duplicate, q, role])
    with pytest.raises(FrozenInstanceError):
        decision.weight = 1.
    with pytest.raises(ValueError):
        resolve_weights(SCOPE, [q, role, replace(duplicate, rule=HeldMerit('another-skill', LOSSES))])
    with pytest.raises(ValueError):
        resolve_weights(SCOPE, [q, role, replace(duplicate, evidence_id='different-proof')])


def test_merit_rejects_role_aliases_and_unpaired_footprints():
    bad_rows = [replace(LOSSES[0], well_id=key) for key in ('query', 'QUERY', ' query', 'train-0 ', 'other')]
    bad_rows += [replace(LOSSES[0], baseline_footprint=Footprint(11, FP.sha256)),
                 replace(LOSSES[0], baseline_footprint=Footprint(12, 'b' * 64)),
                 replace(LOSSES[0], baseline=-1.), replace(LOSSES[0], candidate=float('nan'))]
    for bad in bad_rows:
        with pytest.raises(ValueError):
            resolve_weights(SCOPE, [node('q', HeldMerit('skill', (bad,) + LOSSES[1:]))])
    with pytest.raises(ValueError):
        resolve_weights(replace(SCOPE, query_wells=('TRAIN-0',)), [])
    with pytest.raises(ValueError, match='canonical Unicode'):
        resolve_weights(WeightScope('fold', ('Café',), ('Cafe\u0301',)), [node('a', Given())])
    with pytest.raises(ValueError):
        resolve_weights(SCOPE, [node('q', HeldMerit('skill', LOSSES + (LOSSES[0],)))])
    with pytest.raises(ValueError, match='complete declared'):
        resolve_weights(replace(SCOPE, calibration_wells=WELLS + ('hard-well',)),
                        [node('q', HeldMerit('skill', LOSSES))])


def test_zero_unsupported_and_insufficient_merit_turn_children_off():
    rules = [HeldMerit('q', ()), HeldMerit('q', LOSSES, 12),
             HeldMerit('q', tuple(replace(r, baseline=0.) for r in LOSSES)),
             HeldMerit('q', tuple(replace(r, candidate=4.) for r in LOSSES)),
             HeldMerit('q', tuple(replace(r, candidate=5.) for r in LOSSES)), FixedWeight('zero', 0.)]
    for rule in rules:
        graph = resolve_weights(SCOPE, [node('measure', rule), node('action', Given(), ('measure',))])
        assert graph.decision('measure').weight == 0. and not graph.decision('action').enabled
    no_data = replace(node('source', Given()), supported=False)
    child = node('action', FixedWeight('declared', .8), ('source',))
    d = resolve_weights(SCOPE, [no_data, child]).decision('action')
    assert d.reason == 'parent_off' and d.local_weight == .8 and d.weight == 0.


def test_rejects_invalid_edges_provenance_and_fixed_values():
    a = node('A', FixedWeight('half', .5), evidence='receipt')
    for b in (replace(a, node_id='a'), replace(a, node_id='B', parents=('a',)),
              replace(a, node_id='B', source_id='SURVEY'), replace(a, node_id='B', evidence_id='Receipt'),
              replace(a, node_id='B', rule=FixedWeight('HALF', .5)), replace(a, node_id='B', scope_id='outer-1')):
        with pytest.raises(ValueError):
            resolve_weights(SCOPE, [a, b])
    with pytest.raises(ValueError):
        resolve_weights(SCOPE, [node('a', Given(), ('b',)), node('b', Given(), ('a',))])
    for v in (-.1, 1.01, True, float('inf')):
        with pytest.raises(ValueError):
            resolve_weights(SCOPE, [node('bad', FixedWeight('factor', v))])


def test_weights_cannot_admit_a_worse_fixed_objective():
    graph = resolve_weights(SCOPE, [node('proposal', HeldMerit('skill', LOSSES))])
    assert graph.decision('proposal').enabled
    result = refine_policy('incumbent', off='off', evidence_sha256=graph.sha256, supported=True,
                           propose=lambda _: ('proposal',), objective={'incumbent': 3., 'proposal': 4.}.__getitem__,
                           max_steps=1, proposal_limit=1)
    assert result.policy == 'incumbent' and result.objective == 3.
