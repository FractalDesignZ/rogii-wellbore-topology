# Hidden-label ancestry through the artifact graph

`rwt.label_lineage` carries declared hidden-label well identities through an artifact's complete parent graph. A derived value retains the union of its parents' label dependencies. Filtering its rows or declaring no additional labels cannot erase inherited exposure. [API](../src/rwt/label_lineage.py) · [Behavior checks](../tests/test_label_lineage.py)

This addresses a concrete selection problem. A visible-heel calibration scalar can be target-free while the roster used to pool that scalar depends on a fitted depth model. If that model trained on the current held wells, the roster and pooled value inherit those labels. Selecting references directly from the permitted visible calibration removes that model dependency. The selected well IDs can happen to be identical under both rules; equality of IDs does not establish purity of the calculation.

## Contract

Each immutable `LabelArtifact` supplies an artifact ID, an evidence SHA-256, parent artifact IDs and any directly consumed hidden-label well IDs. The caller binds source, AST, execution, roles and row/column footprints into that evidence. `build_label_lineage` returns immutable topological decisions and a graph hash. Each decision records its direct labels, cumulative labels and parent decision hashes. Shared ancestors contribute each well once.

`require_exclusion` checks a selected artifact against the specified held-well IDs. Success returns a hashed exclusion proof. An overlap raises `LabelOverlapError` carrying the corresponding proof and exact overlapping IDs. Changes to source evidence propagate to descendant hashes even when the label set stays the same. Input ordering does not change the receipts.

Unknown parents, cycles, duplicate nodes or parents, malformed digests and ambiguous case or noncanonical Unicode spellings fail. The guard propagates declarations supplied by the caller. It does not discover reads, authenticate a checkpoint's training history or establish completeness of the declared parent edges. Those facts require source, role and execution evidence.

| Artifact | Direct hidden-label declaration |
| --- | --- |
| Permitted visible heel or paired typewell observations | Empty; the evidence identifies the permitted channels and rows. |
| Fitted depth model | All wells whose hidden labels entered its fit or label-derived inputs. |
| Derived curve, selector or aggregate | Any additional labels consumed, plus every parent dependency. |
| Calibrated outer-fold policy | Its own authorized calibration labels; its ancestry must still exclude the outer queries. |

A bank forecaster must exclude both its outer queries and its own held bank. A later policy may learn from that bank and must exclude the outer queries. The requested exclusion set follows the stage's role contract.

## Example

These synthetic hashes illustrate declarations, not field evidence.

```python
from hashlib import sha256
from rwt.label_lineage import LabelArtifact, LabelOverlapError, build_label_lineage

def node(name, parents=(), labels=()):
    return LabelArtifact(name, sha256(name.encode()).hexdigest(), parents, labels)

graph = build_label_lineage([
    node("visible-heel"),
    node("other-fold-model", labels=("held-well",)),
    node("old-roster", ("visible-heel", "other-fold-model")),
    node("filtered-mean", ("old-roster",)),
    node("observed-mean", ("visible-heel",)),
])
graph.require_exclusion("observed-mean", ("held-well",))
try:
    graph.require_exclusion("filtered-mean", ("held-well",))
except LabelOverlapError as rejected:
    assert rejected.proof.overlap == ("held-well",)
```

## Real role replay and verification scope

An independent metadata replay authenticated eighteen original model fit/prediction receipts and constructed all 81 original-model/held-bank combinations. The nine matching predictors passed exclusion of their outer queries and bank. Eighteen same-outer siblings retained the entire prohibited bank after filtering. All 54 models from other outer contexts retained both outer and bank exposure. The declared graph therefore rejected all 72 inappropriate combinations. No numerical field labels or forecasts were decoded during this check.

That independent replay root is `6b03d62e5f2dde4e3f525d2c677571127b5f269a70c6e3d83b07a809f7e65e3b`. A broader integration rehashed 90 artifacts and tested eighteen original/forward checkpoints across all nine held contexts. Eighteen of 162 pairings passed; all 36 sibling and 108 cross-outer pairings retained prohibited labels both before and after filtering. Three geometry-selected training rosters inherited their complete outer holdout. The revised visible-only projections excluded those labels while selecting the same reference IDs. Two fresh executions reproduced receipt bytes. This integration root is `27729ac12457cbad87239c755fa65eeea3a5237bfe160fe5246f370aa56a7aa3`.

These checks verify declared lineage against authenticated metadata. Predictive merit still requires the independently scored native-station comparison. The checks opened no numerical field labels or forecasts; all 154 confirmation wells remain sealed. [Native measurements and role history](MASTER_ITERATION.md)
