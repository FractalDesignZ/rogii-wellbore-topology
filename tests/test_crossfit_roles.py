from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rwt.crossfit_roles import nested_well_roles


def test_nested_roles_preserve_original_order_and_exclude_outer_query():
    outer = {"b": 1, "d": 0, "a": 1, "c": 2}
    roles = nested_well_roles(outer, {"c": 1, "b": 0, "a": 0}, 0, 1)
    assert roles.fit_wells == ("b", "a")
    assert roles.residual_wells == ("c",)
    assert roles.query_wells == ("d",)


@pytest.mark.parametrize(
    "inner",
    [{"b": 1}, {"b": 0, "c": 1, "a": 1}, {"b": 0, "c": 0},
     {"b": 0, "C": 1}, {"b": True, "c": 0}, {"b": 0, "c": -1}],
)
def test_invalid_inner_scope_is_rejected(inner):
    with pytest.raises(ValueError):
        nested_well_roles({"a": 0, "b": 1, "c": 2}, inner, 0, 0)


def test_case_alias_in_outer_scope_is_rejected():
    with pytest.raises(ValueError, match="case folding"):
        nested_well_roles({"a": 0, "A": 1, "b": 1}, {"A": 1, "b": 0}, 0, 0)
