# Jovin source enrollment

The Jovin project manifest covers the Python launcher, worker, statically resolved first-party modules and targeted tests for the nodal comparison. The configured gates are file inventory and Python AST analysis. The existing run harness separately measures worker execution, model comparisons and predictive error. The launcher records the design reference at [mission-lab, revision dfdf8b7](https://github.com/ZuluYokohama/mission-lab/tree/dfdf8b7dded59f06d0fc2bc581a9292de305e977).

## Reproducing the source scope

Use the installed Jovin CLI. Materialize a fresh directory containing the Git blob bytes of `.jovin/project.json` and exactly the relative paths in its `partitions[].paths` at one commit. Compare each file SHA-256 with the corresponding Git blob's SHA-256. Run project enrollment and the later check on that same directory. The project scanner inventories every eligible file under the supplied root. This restricted checkout keeps research, raw measurements, credentials, Git metadata, virtual environments and earlier receipts outside the scanner's input.

Set `JOVIN_HARNESS_DATA_DIR` to a separate local directory before recording the baseline and retain it for the check and chain verification.

```powershell
jovin-harness project baseline --root $sourceSnapshot
jovin-harness project check --root $sourceSnapshot --baseline-receipt $baselineReceipt
jovin-harness receipts verify
```

The supplied archive has SHA-256 `a93a990e183730915a78be40bdf46a217da80e56a3a4d2155b9b848a10dcf8a7`. Of its 27 package files outside bytecode caches, 24 match the installed package byte for byte. The installed CLI module, native sandbox module and native Dockerfile differ, and the installation contains six additional benchmark files. The static project implementation, AST analyzer, configuration, receipt store and project-service methods match. The CLI's project dispatch branches also match under Python AST comparison. Enrollment uses the installed CLI executable.
