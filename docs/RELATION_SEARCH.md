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
