# Searching for a depth on a shared graph

A query starts at a known station and asks which true vertical thickness (TVT) value can be reached at a later station. Each node represents one well, station, layer and candidate depth in feet, with an optional activity state. A measured relation connects two stations. A boundary map receives that relation, a reachable source node and a proposed destination. Its decision records which measurements support, contradict or leave open that transition.

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

## Following a desired depth through the graph

When an observation-based model supplies a desired depth at every station, `rwt.offset_path.select_offset_path` finds the reachable trajectory with the smallest weighted squared error to those depths. Each station has the same increasing list of offsets around its own center. A path may stay in the current bin or move one bin at each step. Run it once per well in measured-depth order.

```python
from rwt.offset_path import select_offset_path

indices = select_offset_path(
    center_ft=[1000, 1002, 1004],
    offsets_ft=[-5, 0, 5],
    desired_ft=[995, 1007, 1009],
    row_weights=[8, 8, 3],
)
# indices [0, 1, 2] select depths [995, 1002, 1009].
```

### Weighting relationships inside the search

An edge can carry several measurements of the same proposed move. With evidence cost `e_m` for measurement `m` and local reliability `w_m`, its cost is `sum(w_m * e_m)`. A weight can change by station, source bin or destination bin. For example, agreement between a visible-heel gradient and nearby well gradients can increase the weight on geometry for that crossing. A missing reading contributes zero.

```python
import numpy as np
from rwt.offset_path import minimum_cost_offset_path, relation_cost

# Start in bin 0. The next station admits a stay or a move to bin 1.
nodes = [[0, np.inf], [0, 0]]
shape = (1, 3, 2)  # station transition, bin change (-1, 0, +1), destination
geo, gamma = np.zeros(shape), np.zeros(shape)
geo[0, 1, 0], gamma[0, 1, 0] = 0.15, 2.0  # Stay
geo[0, 2, 1], gamma[0, 2, 1] = 2.0, 0.3   # Move

def route(geometry_weight):
    edges = relation_cost({"geo": geo, "gamma": gamma},
                          {"geo": geometry_weight, "gamma": 1.0})
    return minimum_cost_offset_path(nodes, edges).tolist()

assert route(0.5) == [0, 1]
assert route(3.0) == [0, 0]

# Only the move has independent heel/donor agreement about its geometry.
agreement = np.zeros(shape)
agreement[0, 2, 1] = 1.0
assert route(0.5 + 2.5 * agreement) == [0, 0]
```

This example changes the weight while holding observations and allowed transitions fixed. Learned weights can enter the same interface after fitting on training wells. Station costs and transition costs both contribute to the sum over the entire path.

Two routes with equal station costs and identical evidence and context at every crossing receive the same score. The solver can return a deterministic route in that case. Distinguishing them from measurements requires a relationship on which they differ.

## Read a search result

`search_relations` records the observed relations, proposed crossings, decisions and parents of admitted nodes. Exact equal endpoints merge. Multiple ways to reach one depth can establish a unique value while leaving its route ambiguous. The reported path count saturates at two. The search returns `found` for a completely checked unique depth, `ambiguous` for two or more accepted terminal depths, `exhausted` when a complete search finds none, and `inconclusive` when unresolved evidence or an execution limit prevents a conclusion. These conclusions apply to the supplied candidates and relations.

The two synthetic maps iterate `z ← z² + c` with `c = 0` and `c = −1`. A radius greater than two witnesses escape for these parameters. A radius at most one quarter witnesses attraction to zero or the zero/minus-one cycle, respectively. A finite orbit without either witness remains unresolved. The geological meaning of a coordinate comes from its measurement model and requires a separate accuracy check.

### What the graph saves

Let `R[t,m]` be the set of transitions admitted from station `t` by map `m`. With fully decided evidence, the joint relation is the intersection of those sets. Starting with the known heel, the next reachable set contains each destination with an incoming joint transition from the current reachable set. A singleton at the requested station identifies one value in the candidate model. A finite unresolved witness or execution cutoff is recorded separately.

Eight layers of three candidates can describe 6,561 complete paths. The unfiltered pilot needs only 66 distinct transition checks per well, three from the heel and nine at each of the seven later crossings. This reduction holds because histories reaching an equal state have the same available next decisions. A model that uses path history must encode the relevant memory in its state before merging histories.

Observed conditions travel on the relation. An optional `CandidateState.activity` label carries the inferred mode that matters to the next decision. For example, two routes may reach 100 ft from opposite directions. Both give the same answer to a depth query at that station, but an observation of continuing increase can distinguish their next transitions. A test preserves both states and recovers the single compatible route to 101 ft. Omitting activity reproduces the original depth-only node identities and serialized traces.

The label can carry a proposed activity regime. Establishing such a regime requires evidence that relationships behave differently over a sustained interval. The direction example tests preservation of memory. It supplies no evidence by itself for a geological regime change.

`result.supporting_steps()` traces backward from the reached goals and returns the original observed crossings with their map evidence. Pass one reached terminal state to inspect just its supporting routes. The trace includes every admitted parent on those routes, and skips branches that lead elsewhere. This walk is linear in the number of recorded crossings and avoids enumerating individual paths.

### Related work

- Constraint propagation on networks of relations and message passing on factor graphs give precedents for assembling a global answer from local compatibility conditions. The periodic example resolves residues at coprime periods, a direct application of the Chinese remainder theorem. [Mackworth (1977)](https://www.cs.ubc.ca/~mack/Publications/AI77.pdf) · [Kschischang, Frey and Loeliger (2001)](https://www.cise.ufl.edu/~arunava/Teaching/cap6617sp21/Readings/frey2001factor.pdf) · [Chinese remainder theorem](https://crypto.stanford.edu/pbc/notes/numbertheory/crt.html)
- Graph-directed complex dynamics studies sequences in which previously selected maps constrain the next map. The wellbore overlay asks several maps to judge the same proposed transition. [Sumi and Watanabe (2019)](https://arxiv.org/abs/1810.09922)
- Using edges as higher-level nodes has an applied geometric precedent in ALIGNN. It uses the line graph of an atomic bond graph to represent angles between bonds. [Choudhary and DeCost (2021)](https://arxiv.org/abs/2106.01829)
- In geosteering, particle filters maintain competing boundary configurations as gamma-ray observations arrive. The competition's winning team describes geographic priors and particle filtering in its published account. [Djecta and colleagues (2025)](https://www.iccs-meeting.org/archive/iccs2025/papers/159070192.pdf) · [Winner's published overview](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/writeups/1st-place-solution)
- Linear-chain conditional random fields use observation-dependent features to score transitions between hidden states. This is a precedent for weighting competing routes with local relational evidence. Our public minimum-cost selector returns a path directly. A conditional random field also defines a normalized distribution over paths. [Lafferty, McCallum and Pereira (2001)](https://nlp.cs.nyu.edu/nycnlp/lafferty01conditional.pdf)

The project-specific question is whether orbit evidence derived independently from survey geometry and gamma-ray response on a shared TVT transition improves placement beyond a matched conventional graph. These precedents establish the surrounding techniques. Priority for that exact combination remains open.

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

## Does a correct candidate lie near the basin boundary?

A separate diagnostic froze the next 48 development wells and saved evidence for all 3,168 transitions before reading their target depths. It used the same three offsets and eight stations as the pilot. For the geometry map, distance to the unit circle is exact. For the GR map, the nearest tested perturbation with the opposite certified fate gives a numerical proximity estimate. Missing evidence, unresolved starts and searches without an opposite witness are tracked separately.

The primary comparison asked how often the candidate nearest the measured TVT had a smaller mean boundary distance than either alternative, with half credit for ties. The evaluator used the previous station's nearest true candidate to select among already saved transitions. This tests the representation under a known previous state. A second calculation held the source at the visible heel depth throughout.

Of 384 stations, 363 had common support across all three proposed depths. The combined proximity measure scored **18.04%** against a 50% chance reference. A well-level bootstrap gave a 95% interval of **12.18% to 24.44%**. Using the fixed heel source gave **19.83%**. The prespecified initial-radius residual control scored **98.48%** on the primary comparison. On 25 of the 363 eligible stations, the nearest candidate differed from the heel depth. These are short-span ranking measurements. They give no support to choosing a candidate by closeness to these two basin boundaries.

## Using a boundary signal to allocate search

A separate experiment asked where an extra candidate would earn its cost. It used 96 further development wells, each with eight near-heel block centers. Every station started with one candidate at the last visible depth. A fixed budget allowed two additional depths at two stations per well. The three budgeted policies differed only in which stations they expanded. The chooser scored the resulting candidates against the same geometry and gamma-ray observations.

| Expansion policy | Candidate checks | RMSE on 768 centers (ft) |
| --- | ---: | ---: |
| Heel only | 768 | 1.0071 |
| Uniform placement on supported stations | 1,152 | 0.9761 |
| Shift in the Julia boundary statistic | 1,152 | 0.9574 |
| Direct disagreement in observed geometry and GR | 1,152 | 0.9415 |
| Every supported station | 2,240 | 0.9076 |

The Julia policy varied its quadratic parameter with the observed discrepancies. It counted adjacent seed-grid locations with differing finite escape outcomes, then ranked the change from a reference parameter. The 769 parameter settings, including the reference, produced identical integer counts on CPU and GPU. A warmed float64 batch on the RTX 5070 took 57 ms including transfers, versus 605 ms for the vectorized CPU calculation. These timings cover the boundary calculation. Each policy also incurs its own screening and candidate-evaluation costs.

The Julia policy beat uniform allocation on one well, lost on one and tied on 94. Direct discrepancy beat uniform on two and tied on 94. At this budget direct discrepancy had lower pooled error than the Julia statistic. The limited number of changed well outcomes confines the interpretation to this pilot.
