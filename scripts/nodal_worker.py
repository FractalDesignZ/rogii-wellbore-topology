"""Single-stage worker for the source-bound nodal comparison."""
from pathlib import Path
import argparse
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rwt.nodal_study import ARMS, _write, audit_run, predict_run, prepare_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", *ARMS, "audit"])
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    reads = []
    if args.stage == "prepare":
        prepare_run(args.run)
    elif args.stage == "audit":
        _write(args.run / "metrics.json", audit_run(args.run, reads=reads))
    else:
        predict_run(args.run, args.stage, reads=reads)
    _write(args.run / f"{args.stage}.reads.json", {"schema": "rwt.worker-artifact-reads.v1",
           "stage": args.stage, "pid": os.getpid(), "parent_pid": os.getppid(), "reads": reads})


if __name__ == "__main__":
    main()
