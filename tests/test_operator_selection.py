import numpy as np
import pytest

from rwt.operator_selection import select_supported_operator


def choose(errors, *, held=("held",), observed=("a","b","c"), query=0, counts=(5,4,2)):
    return select_supported_operator(("a","b","c","d"), held, ("a","b","c"), observed,
                                     errors, counts, np.zeros(3, int), query, min_wells=3)


def test_comparison_requires_whole_bank_and_disjoint_roles():
    errors = np.tile([8., 2., 5.], (3, 1))
    assert choose(errors) == 1
    for observed in (("a","b","held"), ("a","a","b"), ("a","c")):
        with pytest.raises(ValueError):
            choose(errors, observed=observed)
    with pytest.raises(ValueError):
        choose(errors, held=("a",))
    with pytest.raises(ValueError):
        choose(errors, held=("A",))
    with pytest.raises(ValueError):
        choose(errors, held=(" a",))
    with pytest.raises(ValueError):
        select_supported_operator(("a","A","b","c"), ("held",), ("a","b","c"),
                                  ("a","b","c"), errors, (5,4,2), (0,0,0), 0)
    with pytest.raises(ValueError):
        choose(errors, counts=(5,0,2))


def test_missing_evidence_weak_support_uncertainty_and_ties_choose_off():
    assert choose(np.tile([5.,1.,3.], (3,1)), query=-1) == 0
    assert choose(np.tile([5.,1.,3.], (3,1)), query=1) == 0
    assert choose(np.tile([5.,1.,1.], (3,1))) == 0
    assert choose([[4.,0.,8.],[4.,7.,8.],[4.,4.,8.]]) == 0


def test_exact_one_standard_error_and_large_values():
    original = np.array([[4.,4.,8.],[4.,4.,8.],[4.,1.,8.]])
    unchanged = original.copy()
    assert choose(original) == 0
    np.testing.assert_array_equal(original, unchanged)
    assert choose(np.tile([9e307, 5e307, 8e307], (3,1))) == 1
    assert choose(np.zeros((3,3))) == 0
