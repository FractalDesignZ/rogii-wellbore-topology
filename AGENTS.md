# First-party design rule

This rule applies throughout this repository, including private experiments, agents and pull requests.

- External solutions, including Ruby, serve only as examples of typical approaches and as references for results their authors or organizers published. Cite the source, population and metric when making a comparison.
- Design our methods from the user's Network Engine framework, existing first-party work, authorized well measurements and independently chosen mathematics. Use standard libraries through their documented interfaces.
- Copying, porting, translating, reproducing or running another team's solution is prohibited. Its implementation, feature recipe, configuration, weights, predictions and generated artifacts are prohibited inputs to our designs, training, selection and validation. This applies even when the material is public or permissively licensed.
- Before running a candidate, record the first-party origin of its observations, relationship encodings, parameters and dependencies. Verify the same provenance before admitting a result. An excluded input also excludes candidates that rely on it.
- Keep historical receipts identifiable as historical. A source-informed experiment cannot establish independent development merely because its code was rewritten or its score reproduced.

The N2 learned-alignment and 334-feature own-heel branches are excluded because their design records identify competitor implementation input. The private Forward–Backward port of the supplied v0.2.1 package and the comparator that used its kernel are also excluded. Apply the same rule to any candidate that consumes their features or predictions. [Measurement history](docs/MASTER_ITERATION.md)
