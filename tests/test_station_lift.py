"""Station-row reconstruction from known heel measurements and native block centers."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rwt.station_lift import lift_block_predictions


def survey():
    md = np.arange(15, dtype=float)**2
    z = -50. + (np.arange(15) % 5)**2
    tvt_input = np.full(15, np.nan)
    tvt_input[:2] = [141., 100. - z[1]]
    centers = np.array([5, 12])
    tvt = 100. - z[centers]
    return md, z, tvt_input, centers, tvt


def test_original_md_knots_heel_and_partial_tail_for_every_readout():
    md, z, visible, centers, tvt = survey()
    plain = lift_block_predictions(md, z, visible, centers, tvt, method="md_tvt")
    broadcast = lift_block_predictions(md, z, visible, centers, tvt, method="constant")
    height = lift_block_predictions(md, z, visible, centers, tvt)
    np.testing.assert_equal(broadcast[2:], np.repeat(tvt, [8, 5]))
    np.testing.assert_allclose(height[2:13], 100.-z[2:13], rtol=0, atol=1e-12)
    expected_plain = np.interp(md[2:], md[[1, 5, 12]], [visible[1], *tvt])
    np.testing.assert_equal(plain[2:], expected_plain)
    assert plain[3] != pytest.approx(height[3])
    for result in (height, plain, broadcast):
        assert result.dtype == np.float64 and result.shape == md.shape
        np.testing.assert_equal(result[:2], visible[:2])
        np.testing.assert_equal(result[centers], tvt)
        np.testing.assert_equal(result[13:], np.full(2, tvt[-1]))
    assert np.isnan(visible[2:]).all()


def test_missing_elevations_fall_back_only_where_unsupported():
    md, z, visible, centers, tvt = survey()
    reference = lift_block_predictions(md, z, visible, centers, tvt)
    ordinary = lift_block_predictions(md, z, visible, centers, tvt, method="md_tvt")
    z[[3, 12]] = np.nan
    actual = lift_block_predictions(md, z, visible, centers, tvt)
    np.testing.assert_equal(actual[[2, 4, 5]], reference[[2, 4, 5]])
    np.testing.assert_equal(actual[3], ordinary[3])
    np.testing.assert_equal(actual[6:], ordinary[6:])
    z[1] = np.nan
    np.testing.assert_equal(lift_block_predictions(md, z, visible, centers, tvt), ordinary)


def test_valid_coordinate_survives_curve_null_decoding_and_anchor_only_cases():
    md, z, visible, centers, tvt = survey()
    visible[2:] = -999.25
    z[3] = -9999.25
    assert lift_block_predictions(md, z, visible, centers, tvt)[3] == pytest.approx(100.-z[3])
    np.testing.assert_equal(lift_block_predictions([0, 1], [np.nan, 0], [14, 13], [], []), [14, 13])
    np.testing.assert_equal(lift_block_predictions([0, 1], [np.nan, 0], [14, np.nan], [1], [12]), [14, 12])


@pytest.mark.parametrize(("index","bad","message"), [
    (0, [1., 2.], "aligned"), (0, np.zeros(15), "increasing"),
    (0, np.full(15, np.nan), "finite"), (1, np.zeros((15, 1)), "aligned"),
    (2, np.full(15, np.nan), "heel"), (2, [1., np.nan, 1.] + [np.nan]*12, "heel"),
    (3, [6, 12], "eight-row"), (3, [12, 5], "eight-row"), (3, [5., 12.], "integer"),
    (4, [10.], "each center"), (4, [0., np.inf], "finite"),
])
def test_mismatched_stations_knots_and_hidden_values_rejected(index, bad, message):
    args = list(survey())
    args[index] = bad
    with pytest.raises(ValueError, match=message):
        lift_block_predictions(*args)


def test_unknown_readout_rejected():
    with pytest.raises(ValueError, match="method"):
        lift_block_predictions(*survey(), method="fit")
