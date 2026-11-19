"""Cohort falsifiers: label independence, group isolation, frozen copies, safe paths."""
from __future__ import annotations

from pathlib import Path
import tempfile
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prepare_master_cohort import assign_wells, hash_file, prepare_master_cohort  # noqa: E402


def make_data(root: Path, n: int = 40) -> Path:
    data = root / "data"
    (data / "train").mkdir(parents=True)
    for i in range(n):
        # Arbitrary bytes prove the preparation path does not parse labels/CSVs.
        (data / "train" / f"W{i:03d}__horizontal_well.csv").write_bytes(b"opaque\xff\x00" + bytes([i]))
        (data / "train" / f"W{i:03d}__typewell.csv").write_bytes(b"book\xff" + bytes([i]))
    return data


class TestMasterCohort(unittest.TestCase):
    def test_labels_cannot_change_membership_or_pilot_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = make_data(root)
            a = prepare_master_cohort(data, root / "art_a", root / "pilot_a", pilot_wells=6)
            for path in (data / "train").glob("*__horizontal_well.csv"):
                path.write_bytes(b"completely changed target labels, never parsed")
            b = prepare_master_cohort(data, root / "art_b", root / "pilot_b", pilot_wells=6)
            self.assertEqual(a["assignment_sha256"], b["assignment_sha256"])
            self.assertEqual(a["development_well_ids"], b["development_well_ids"])
            self.assertEqual(a["confirmation_well_ids"], b["confirmation_well_ids"])
            self.assertEqual(a["pilot_well_ids"], b["pilot_well_ids"])
            self.assertNotEqual(a["input_sha256"], b["input_sha256"])

    def test_pilot_contains_only_development_pairs_and_source_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = make_data(root)
            before = {p.name: hash_file(p) for p in (data / "train").iterdir()}
            pilot = root / "pilot"
            manifest = prepare_master_cohort(data, root / "art", pilot, pilot_wells=12)
            expected = {f"{wid}{suffix}" for wid in manifest["pilot_well_ids"] for suffix in ("__horizontal_well.csv", "__typewell.csv")}
            self.assertEqual({p.name for p in (pilot / "train").iterdir()}, expected)
            self.assertFalse(set(manifest["pilot_well_ids"]) & set(manifest["confirmation_well_ids"]))
            self.assertEqual(set(manifest["all_well_ids"]), set(manifest["development_well_ids"]) | set(manifest["confirmation_well_ids"]))
            for path in (pilot / "train").iterdir():
                self.assertEqual(hash_file(path), before[path.name])
            self.assertEqual(before, {p.name: hash_file(p) for p in (data / "train").iterdir()})
            self.assertTrue(manifest["confirmation_unmeasured"])
            self.assertEqual(manifest["trainer_data_argument"], str(pilot.resolve()))
            self.assertTrue((root / "art" / "cohort_manifest.json").is_file())

    def test_missing_and_orphan_typewells_fail_before_creating_outputs(self) -> None:
        for defect in ("missing", "orphan"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                data = make_data(root)
                if defect == "missing":
                    (data / "train" / "W000__typewell.csv").unlink()
                else:
                    (data / "train" / "ORPHAN__typewell.csv").write_bytes(b"orphan")
                with self.assertRaisesRegex(ValueError, "matching typewell"):
                    prepare_master_cohort(data, root / "art", root / "pilot")
                self.assertFalse((root / "art").exists())
                self.assertFalse((root / "pilot").exists())

    def test_fresh_and_disjoint_paths_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = make_data(root)
            existing = root / "existing"
            existing.mkdir()
            keep = existing / "keep.txt"
            keep.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                prepare_master_cohort(data, existing, root / "pilot")
            self.assertEqual(keep.read_text(), "keep")
            for out, pilot in ((data / "art", root / "pilot"), (root / "art", data / "pilot"), (root / "art", root / "art" / "pilot")):
                with self.subTest(out=out, pilot=pilot), self.assertRaisesRegex(ValueError, "disjoint"):
                    prepare_master_cohort(data, out, pilot)

    def test_empty_duplicate_and_degenerate_groups_are_refused(self) -> None:
        for ids in ([], ["A", "a"], ["only_one"]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                assign_wells(ids)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = make_data(root)
            for n in (0, -1, 1000, True):
                with self.subTest(pilot_wells=n), self.assertRaises(ValueError):
                    prepare_master_cohort(data, root / "art", root / "pilot", pilot_wells=n)
            self.assertFalse((root / "art").exists())

    def test_input_enumeration_order_does_not_change_assignment(self) -> None:
        ids = [f"W{i:03d}" for i in range(40)]
        self.assertEqual(assign_wells(ids), assign_wells(list(reversed(ids))))


if __name__ == "__main__":
    unittest.main()
