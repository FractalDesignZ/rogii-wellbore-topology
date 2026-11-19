"""Well-level partition checks for nested out-of-fold inference."""
from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import NamedTuple


class NestedWellRoles(NamedTuple):
    """Original-order well IDs for fit, inner residual and outer query roles."""

    fit_wells: tuple[str, ...]
    residual_wells: tuple[str, ...]
    query_wells: tuple[str, ...]


def _validate_assignment(assignment: Mapping[str, int]) -> None:
    if not assignment or any(not isinstance(well, str) or not well or well != well.strip() for well in assignment):
        raise ValueError("fold assignments require nonempty well IDs")
    if len({well.casefold() for well in assignment}) != len(assignment):
        raise ValueError("well IDs must be unique under case folding")
    if any(isinstance(fold, bool) or not isinstance(fold, Integral) or fold < 0 for fold in assignment.values()):
        raise ValueError("fold assignments require nonnegative integer fold IDs")


def nested_well_roles(
    outer_by_well: Mapping[str, int],
    inner_by_well: Mapping[str, int],
    outer_fold: int,
    inner_fold: int,
) -> NestedWellRoles:
    """Check that an inner assignment covers exactly one outer training set.

    Access to observed query measurements may follow a separate policy. An
    inner estimator may use fit-well targets, and residuals may be computed
    for its inner holdout after predictions are fixed. Outer query targets
    belong to final evaluation only. This checks identities before data I/O.
    """
    _validate_assignment(outer_by_well)
    _validate_assignment(inner_by_well)
    if (isinstance(outer_fold, bool) or not isinstance(outer_fold, Integral) or
            isinstance(inner_fold, bool) or not isinstance(inner_fold, Integral)):
        raise ValueError("requested folds must be integer fold IDs")
    query = tuple(well for well, fold in outer_by_well.items() if fold == outer_fold)
    outer_train = tuple(well for well in outer_by_well if outer_by_well[well] != outer_fold)
    if not query or set(inner_by_well) != set(outer_train):
        raise ValueError("inner assignments must cover exactly the outer training wells")
    fit = tuple(well for well in outer_train if inner_by_well[well] != inner_fold)
    residual = tuple(well for well in outer_train if inner_by_well[well] == inner_fold)
    if not fit or not residual:
        raise ValueError("inner fit and residual partitions must both be nonempty")
    return NestedWellRoles(fit, residual, query)
