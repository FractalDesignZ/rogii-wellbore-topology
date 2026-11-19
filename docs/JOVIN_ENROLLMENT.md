# Jovin source enrollment

The Jovin project manifest covers the Python launcher, worker, statically resolved first-party modules and targeted tests for the nodal comparison. It also enrolls a C++17 verifier for the minimum-cost depth path. The configured gates are file inventory, Python AST analysis and native CPU checks. The existing run harness separately measures worker execution, model comparisons and predictive error. The launcher records the design reference at [mission-lab, revision dfdf8b7](https://github.com/ZuluYokohama/mission-lab/tree/dfdf8b7dded59f06d0fc2bc581a9292de305e977).

## Original Python-only check

On November 19, 2026 UTC, the installed Jovin CLI enrolled 19 Python files (213,370 bytes) from commit `807c76b54d81cf794b0be1a8edea7bf9b0ca331d`. Every source hash matched its Git blob. The normalized source manifest, which also binds the project configuration, has SHA-256 `24125388b05916aa2bda74b4bbfa970b888de35f2de78fd29c11a380de5b6878`. Seven local working-copy files had CRLF line endings. The enrollment used the committed bytes.

| Operation | Receipt | Result |
| --- | --- | --- |
| Project baseline | `20261119T025419Z-9d7de237a2e4` | Enrolled, exit 0 |
| Project check | `20261119T025420Z-59fcfd39d766` | Overall blocked, exit 1 |
| Receipt chain verification | Head `d664c26a801dc1ad0607e74d7a69da5b7ec9c2c67b3e65435d921bc25cf54eef` | Valid, two receipts, zero problems |

All three check gates passed. `baseline_binding` found zero additions, removals or modifications. `inventory` verified the 19 scoped file identities and their partition assignments. `python_ast` parsed the sources and generated static symbol and call projections. The local CLI requires a successful supervised native CPU receipt bound to this exact baseline and source hash for overall project acceptance.

The committed comparator uses Python and external scientific libraries. The commit contains no native source or native build plan. The separate Network Engine bridge exports MD, Z and GR for offline diagnostics. Its earlier receipts pertain to that provider and do not bind a native check of the four nodal readouts. The Jovin check therefore retains its overall blocked status. Model score and process evidence are documented in the [nodal harness report](NODAL_HARNESS.md) and the bridge's scope in [Network Engine integration](NETWORK_ENGINE_INTEGRATION.md).

The local enrollment root is `artifacts/jovin_nodal_enrollment_001/source`. The adjacent `receipts_data`, `scoped_git_inventory.json`, `package_comparison.json`, `baseline.json`, `check.json` and `chain.json` retain the receipt chain and supporting checks.

## Independent native path check

The shared synthetic corpus contains 1,309 depth lattices. Thirteen target empty, singleton, boundary, tie, signed, disabled and unreachable cases. The other 1,296 exhaust all four-station-cell assignments drawn from −1, 0 and 1 on a two-station, two-bin lattice and all 16 admissible-edge masks. The signed costs on enabled crossings are fixed. Each row stores the path returned by the production Python solver and its computed total cost, or an unreachable marker. A Python test invokes the production solver on every frozen row and checks exact path and cost. Two independent C++ procedures solve each row using dynamic programming and exhaustive complete-path enumeration, including the specified tie preference for the lowest terminal bin and then lowest predecessors. [Corpus](../qa/path_vectors.txt) · [Python comparison](../tests/test_native_path_vectors.py) · [Native verifier](../qa/path_verify.cpp)

The Windows CMake build with MSVC 19.44.35229 passed CTest on all 1,309 lattices and enumerated 6,562 complete paths. Production Python verification and the existing offset-path tests passed seven tests. These cases check the combinatorial solver on small, exactly representable numbers. They establish neither field accuracy nor parity at arbitrary floating-point magnitudes.

## Reproducing the source scope

Use the installed Jovin CLI. Materialize a fresh directory containing the Git blob bytes of `.jovin/project.json` and exactly the relative paths in its `partitions[].paths` at one commit. Compare each file SHA-256 with the corresponding Git blob's SHA-256. Run project enrollment and the later check on that same directory. The project scanner inventories every eligible file under the supplied root. This restricted checkout keeps research, raw measurements, credentials, Git metadata, virtual environments and earlier receipts outside the scanner's input.

Set `JOVIN_HARNESS_DATA_DIR` to a separate local directory before recording the baseline and retain it for the check and chain verification.

```powershell
jovin-harness project baseline --root $sourceSnapshot
jovin-harness native check --root $sourceSnapshot --baseline-receipt $baselineReceipt --plan qa/native_path_plan.json
jovin-harness project check --root $sourceSnapshot --baseline-receipt $baselineReceipt --native-receipt $nativeReceipt
jovin-harness receipts verify
```

The supplied archive has SHA-256 `a93a990e183730915a78be40bdf46a217da80e56a3a4d2155b9b848a10dcf8a7`. Of its 27 package files outside bytecode caches, 24 match the installed package byte for byte. The installed CLI module, native sandbox module and native Dockerfile differ, and the installation contains six additional benchmark files. The static project implementation, AST analyzer, configuration, receipt store and project-service methods match. The CLI's project dispatch branches also match under Python AST comparison. Enrollment uses the installed CLI executable.
