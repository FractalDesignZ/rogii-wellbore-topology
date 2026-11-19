# Searching for a depth on a shared graph

A query starts at a known station and asks which true vertical thickness (TVT) value can be reached at a later station. Each node represents one well, station, layer and candidate depth in feet. A measured relation connects two stations. A boundary map receives that relation, a reachable source node and a proposed destination. Its decision records which measurements support, contradict or leave open that transition.

Several maps can examine the same edge. Every map receives the same three objects and contributes its own verdict, evidence and provenance. The supplied overlay admits a transition when every map accepts, rejects it when any map rejects, and otherwise leaves it unresolved. Each admitted destination can become the source of the next step. This implements the user's observation-to-candidate boundary as a finite search in public RWT code. The optional native Network Engine integration has a separate measurement interface.

## Try the two-map query

```bash
python scripts/demo_relation_search.py
```

The example starts at 1,000 ft with two synthetic observations of depth change. It asks for a reachable depth at station 1. Each map uses the discrepancy between the proposed and observed change as the initial coordinate of a quadratic orbit. The query supplies no expected answer.

| View | Reachable terminal TVTs (ft) | Search result |
| --- | --- | --- |
| First quadratic map | 996, 1,000 | Ambiguous |
| Second quadratic map | 1,000, 1,004 | Ambiguous |
| Both on the same transitions | 1,000 | Found |

The script also prints the iteration at which each map accepted or rejected each transition. Iterations count applications of the map. Survey stations advance in measured-depth order.

Add `--phase-stack` for a second constructed example with 30 possible depths. Three sensors report phase at periods of 2, 3 and 5 ft. Each phase constrains a different set of depths. Applying the three maps in order leaves 15, then five, then one candidate at **1,017 ft**. Repeating the first reading three times leaves 15. The measured periods and exact phases make this a controlled example of resolving aliases.

With the separately installed Network Engine, add `--native-lift` to the command. The adapter passes the admitted nodes and integer edge endpoints to the native provider. Each level-one node represents an admitted transition. Two such nodes are adjacent if their transitions share an endpoint. The local receipt maps each edge back to its observation and binds the input topology to a SHA-256 digest. It preserves the search status and any native pair-cap cutoff. Shared endpoints describe incidence. The search retains the original directions and decides reachability.

## Read a search result

`search_relations` records the observed relations, proposed crossings, decisions and parents of admitted nodes. Exact equal endpoints merge. Multiple ways to reach one depth can establish a unique value while leaving its route ambiguous. The reported path count saturates at two. The search returns `found` for a completely checked unique depth, `ambiguous` for two or more accepted terminal depths, `exhausted` when a complete search finds none, and `inconclusive` when unresolved evidence or an execution limit prevents a conclusion. These conclusions apply to the supplied candidates and relations.

The two synthetic maps iterate `z ← z² + c` with `c = 0` and `c = −1`. A radius greater than two witnesses escape for these parameters. A radius at most one quarter witnesses attraction to zero or the zero/minus-one cycle, respectively. A finite orbit without either witness remains unresolved. The geological meaning of a coordinate comes from its measurement model and requires a separate accuracy check.

## A 12-well development check

The pilot started at each well's last visible depth and traversed its first eight toe block centers. Each center offered the same three offsets, −4, 0 and +4 ft from the heel depth. The geometry map measured disagreement with the layer trend at the heel. The gamma-ray (GR) map measured disagreement with the calibrated typewell response. Each used a complex coordinate so that two measured discrepancies could contribute to one orbit. The control applied a unit-radius threshold to each of those same coordinates. Every arm saved its graph and predictions before evaluation opened the terminal TVTs.

| Boundary | Unique | Ambiguous | Exhausted | Inconclusive | Terminal RMSE (ft) |
| --- | ---: | ---: | ---: | ---: | ---: |
| All transitions accepted | 0 | 12 | 0 | 0 | 1.4235 |
| Geometry Julia map | 7 | 5 | 0 | 0 | 1.4235 |
| GR Julia map | 2 | 1 | 7 | 2 | 1.7606 |
| Both Julia maps | 2 | 0 | 8 | 2 | 1.4235 |
| Two radius thresholds | 7 | 1 | 2 | 2 | 1.0830 |

RMSE includes one terminal station per well, with last visible TVT as the prespecified fallback. The two depths resolved by the Julia overlay both equaled the heel depth. Its 135 checked transitions included 36 joint accepts, 97 vetoes and two unresolved outcomes. The radius control's gain came from one correct directional change. The pilot provides evidence for retaining the shared graph and per-map trace. The hard Julia veto showed no placement improvement in this experiment. Full-toe development scores use a different evaluation scope. [Full-toe comparisons](MASTER_ITERATION.md)

Replaying both rules on **all 792 possible transitions**, including ones pruned early, located their difference. The geometry Julia map matched its radius control on all 792. The GR Julia map rejected 218 edges its radius control accepted, accepted 14 the control rejected and left one additional edge unresolved. This comparison holds the proposed transitions fixed across methods.

All 60 retained search graphs also replayed through the native lift. On one well, the unfiltered search had 25 nodes and 66 edges. The next native level had 66 relation nodes and 300 shared-endpoint pairs. The joint Julia search on that well had nine nodes and eight edges in a single path. Its next level contained eight relation nodes with seven shared-endpoint pairs.
