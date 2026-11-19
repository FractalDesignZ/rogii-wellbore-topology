# Jovin source enrollment

Jovin checks the enrolled source snapshot, Python AST, native CPU parity, compiler AST and governed native process and socket/FD-I/O traces. The native checker covers minimum-cost depth paths on 1,309 synthetic lattices. Worker artifact reads and field scores have their own evidence in the nodal harness. [Observed wellbore runs and field scores](NODAL_HARNESS.md)

The project manifest covers the Python launcher, worker, first-party modules, targeted tests and the independent C++ depth-path verifier. The controller follows [Mission Lab at revision `dfdf8b7`](https://github.com/ZuluYokohama/mission-lab/tree/dfdf8b7dded59f06d0fc2bc581a9292de305e977).

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

## Final native and project receipts

Jovin enrolled the revised report viewer and its regression test with the native checker at commit `17cf7559b9d8e7245cd0ee3820da79d5f944dadf`. The scoped inventory contained 26 files (308,667 bytes). Every inventoried file matched its Git blob. The source manifest SHA-256 was `e2fdfc9ff0c8f47ce674267a3494677b42b802a84c320c7b3c1f4fa0e53b101a`. A separate invocation of the production Python solver passed both corpus tests against this same checkout.

| Operation | Final local receipt | Result |
| --- | --- | --- |
| Baseline | `20261119T030555Z-70009ff410f3` | Enrolled |
| Native CPU check | `20261119T030604Z-546a69dd043b` | Passed `path_parity` |
| Project check | `20261119T030604Z-ff81a4fcfc71` | Passed with zero source drift |

All four project gates passed. `baseline_binding` and `inventory` verified the working snapshot against the baseline, `python_ast` recorded the Python syntax and call projections, and `native_cpu` verified the supervised native phase receipts. The final chain contained 16 linked receipts, zero verification problems and head SHA-256 `00b8bf2518c5d7dc1834b28ac5484c003bf5a3252cb4ad42ffa9bf80aaccb76b`.

The installed Jovin CLI built and ran the checker with cached Linux toolchain image `sha256:9e1fa3183ce71070a422e13121a49eab4ada0ce6d0be22a3a57e4439b090ad4a` on rootless Podman. Compilation produced a 33,888-byte ELF executable with SHA-256 `415cd7083426d4357cecae54d774fe386af1c06142e88ae256999c57aa912f37`. The supervised command reported all 1,309 lattices and 6,562 complete paths, exited zero and confirmed cleanup. The 9.12 seconds for the native CLI invocation included the supervised build, artifact packaging, execution and receipt creation. That time measures verification of the small solver fixtures. Field-model runtime and accuracy have separate receipts.

The final run evidence resides under `artifacts/jovin_nodal_native_002/`, including frozen inventory, production Python replay, native status, baseline, native result, project check and chain verification. The shared receipt store remains under `artifacts/jovin_nodal_enrollment_001/receipts_data/`. The first successful native enrollment at commit `644a5057c71e721a286da8e25174e0e20d5a723f` preceded the report fix and remains under `artifacts/jovin_nodal_native_001/` with baseline `20261119T030339Z-aeabd6f153cf`, native result `20261119T030348Z-abda56fe9153` and project check `20261119T030349Z-3d135226129a`.

## Compiler and runtime evidence

Jovin accepted the exact scoped Git blobs from `d8a48c963359d1241f303f5bbd3554687437dfd3`. All seven project checks passed, including compiler AST, native parity and governed process and socket/FD-I/O traces. Clang 19.1.7 produced 450 graph nodes and 1,023 edges for the enrolled C++20 `NDEBUG` context. The trace covered configure, build and the 1,309-lattice check. It recorded 40 process or thread lifecycles across the three commands, all with observed exit, and 25,950 syscall entries. The collector reported zero network-category syscalls in those commands.

| Check | Local receipt |
| --- | --- |
| Scoped baseline | `20261119T044905Z-32c53c13a92f` |
| Compiler AST | `20261119T044910Z-14d999befb0f` |
| Native CPU parity | `20261119T044926Z-2cef69ba543f` |
| Governed process and socket/FD-I/O trace | `20261119T044928Z-b5c81ed1ce56` |
| Final project check | `20261119T044934Z-85d8124bb185` |

A second project check omitted the trace receipt. The process and network gates blocked while the other five passed. The source manifest is `483ecca5880b506bffd8e4ada7c57b5d6e650b3b854a5b047f99a4d4d03fa0d6`. The 13-receipt chain verified at `a26f8de8314d6d030b35ae5f2cee217c5e9c0ef417e2dc615a1fd55ec13df493`. Evidence remains local under `artifacts/jovin_native_trace_002/`. The trace plan associates the compiled executable with its enrolled AST context. Individual C++ function execution, GPU activity and field predictions fall outside this native trace. [Build plan](../qa/native_path_plan.json) · [AST plan](../qa/native_path_ast.json) · [Trace plan](../qa/native_path_trace.json)

The station correction API and its tests entered the source scope in commit `ff90958921ea4a59fc3693af147df0d2377a37c5`. The fresh 30-file enrollment passed all seven checks under project receipt `20261119T050301Z-76541df80707`, with source manifest `078a6aae0fea879f7dd12101e5fa30a1e8f038125fb2a26b6c8c0d7e02232507`. The current native parity and trace receipts are `20261119T050256Z-0ca36e9b2769` and `20261119T050258Z-c48b6c72dca5`. The receipt chain verified at `63b9f3b76b015482c769c1e696d5a900043d153561a3a9906471d9c8d5253d59`. The missing-trace check again blocked only the two trace gates. Evidence remains local under `artifacts/jovin_native_trace_003/`. [Station correction API](../src/rwt/station_corrections.py) · [Tests](../tests/test_station_corrections.py)

The keyed member-correction API entered the 33-file enrollment at `06afebfc9cfddaa0d3107b5debed6f6437d77697`. All seven checks passed under `20261119T052128Z-b878ecfe2144`, with source manifest `5cdc367f194126b8e0b004b708442f5c568f49916a3e349aa34b3073fa004d98`. The pair of native receipts is `20261119T052123Z-f1c1b558fa20` for parity and `20261119T052124Z-ce1e4fa69202` for the trace. Omitting the trace blocked its two gates. The linked chain verified at `f1c8b460a213e190a029b365fd6bfec85ac25944c49aa05d97bb2348cda15656`. Evidence remains local under `artifacts/jovin_native_trace_004/`. [Member correction API](../src/rwt/member_corrections.py) · [Tests](../tests/test_member_corrections.py)

Typed observations and their tests entered the 35-file enrollment at `77586a4dadfdd2a9a3cfc426ee8b353a8dd54428`. All seven checks passed under `20261119T053244Z-82ed182edb0f`, with source manifest `d9b420de0b8ee5bc38c531a982d6b562ddf2cecbda1a7fcafa8f845b834c85b3`. The native parity and trace receipts are `20261119T053239Z-e5e1b90a46b7` and `20261119T053241Z-a33cf823d632`. The linked chain verified at `32ca3a40b86ba08aa907b97a7d51fabea2ae51c8f66c28b4a91aa092fd40acbf`. Omitting the trace blocked its two checks. Evidence remains local under `artifacts/jovin_native_trace_005/`. [Observation tensor API](../src/rwt/observation_tensors.py) · [Original-value and relationship audit](NODAL_HARNESS.md#typed-observations-and-neighbor-axes)

The method-reference verifier and its registry entered the 38-file enrollment at `4f6c6d50272070b0ad56c54265378fde9128ccc8`. All seven checks passed under `20261119T055511Z-6978423fc2aa`, with source manifest `b978f42440f294c5ebc0e2d1f32a10a285e925bd65950560abc9fd34a2156543`. Native parity and trace receipts are `20261119T055506Z-9a10e5acbb2e` and `20261119T055508Z-4d42e44489a5`. The receipt chain verified at `dacce3fcd624106e9d4efc55449a49d1f3b66228b383d178d94a98f97308cd2d`. Omitting the trace again blocked its two checks. Evidence remains local under `artifacts/jovin_native_trace_006/`. [Method reference graph](METHOD_REFERENCE_GRAPH.md)

The component-span and operator-selection APIs entered the 42-file enrollment at `a5a132c8bb607bce7b993cb6e82d391481667906`. All seven checks passed under `20261119T061253Z-446868039e8d`, with source manifest `dd259996e3c41b60e74e349a28526db63f1ebe4cabb1f9433ff729b1b637ecf3`. Native parity and trace receipts are `20261119T061248Z-92467c4eee74` and `20261119T061250Z-feeda3404d27`. The receipt chain verified at `132d11843dee3aea35abf082fc21a2876dcdb00a3f717f9ca973a963119a616d`. The missing-trace check blocked its two gates. Evidence remains local under `artifacts/jovin_native_trace_007/`. [Component spans](../src/rwt/relation_history.py) · [Operator selection](../src/rwt/operator_selection.py)

The bounded policy adapter entered the 44-file enrollment at `26e04375a001facd258a26446648ec8b64ef807f`. All seven checks passed under `20261119T063257Z-e86505405eb8`, with source manifest `84925c4a6c0e4688314746f137d3d012b87a919ea036787491c7e8bcce150aac`. Native parity and trace receipts are `20261119T063252Z-cfdd1712b59d` and `20261119T063254Z-1731210a3de0`. The receipt chain verified at `db276a8a0437d13a36cec38ea5240e5161b7067fb0ba6446cad4662cb952add4`. Omitting the trace again blocked its two checks. Evidence remains local under `D:/rwt_private_artifacts/jovin_native_trace_008/`. The native checker retains its 1,309-lattice scope. Python policy behavior is separately covered by the test suite. [Policy adapter and control semantics](POLICY_REFINEMENT.md)

## Reproducing the source scope

Use the installed Jovin CLI. Materialize a fresh directory containing the Git blob bytes of `.jovin/project.json` and exactly the relative paths in its `partitions[].paths` at one commit. Compare each file SHA-256 with the corresponding Git blob's SHA-256. Run project enrollment and the later check on that same directory. The project scanner inventories every eligible file under the supplied root. This restricted checkout keeps research, raw measurements, credentials, Git metadata, virtual environments and earlier receipts outside the scanner's input.

Set `JOVIN_HARNESS_DATA_DIR` to a separate local directory before recording the baseline and retain it for the check and chain verification.

```powershell
$baseline = jovin-harness project baseline --root $sourceSnapshot | ConvertFrom-Json
$ast = jovin-harness ast check --root $sourceSnapshot --baseline-receipt $baseline.receipt_id --plan qa/native_path_ast.json | ConvertFrom-Json
$trace = jovin-harness trace check --root $sourceSnapshot --baseline-receipt $baseline.receipt_id --ast-receipt $ast.receipt_id --plan qa/native_path_trace.json | ConvertFrom-Json
jovin-harness project check --root $sourceSnapshot --baseline-receipt $baseline.receipt_id --ast-receipt $ast.receipt_id --trace-receipt $trace.receipt_id --native-receipt $trace.native_receipt_id
jovin-harness receipts verify
```

The supplied archive has SHA-256 `a93a990e183730915a78be40bdf46a217da80e56a3a4d2155b9b848a10dcf8a7`. Of its 27 package files outside bytecode caches, 24 match the installed package byte for byte. The installed CLI module, native sandbox module and native Dockerfile differ, and the installation contains six additional benchmark files. The static project implementation, AST analyzer, configuration, receipt store and project-service methods match. The CLI's project dispatch branches also match under Python AST comparison. Enrollment uses the installed CLI executable.
