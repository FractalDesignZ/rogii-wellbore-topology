import numpy as np
import pandas as pd
import pytest

from rwt.observation_tensors import gather_md_neighbors, pack_well_observations


def sample():
    lateral = pd.DataFrame({"TVT": [900, 901, 902, 903], "MD": [0., 5., 10., 16.],
                            "X": [1., 2., 3., 4.], "Y": [4., 4., 4., 5.], "Z": [-999.25, -8., -9., -10.],
                            "GR": [20., -999.25, np.nan, 30.], "TVT_input": [13., 12., np.nan, np.nan]})
    reference = pd.DataFrame({"TVT": [99., 91.], "GR": [21., -9999.25]})
    return lateral, reference


def test_inbound_tensor_keeps_exact_source_values_schema_and_null_mask():
    lateral, reference = sample()
    packed = pack_well_observations("A", lateral, reference)
    assert packed.lateral.well_id == "A"
    assert packed.lateral.channels == ("MD", "X", "Y", "Z", "GR", "TVT_input")
    assert packed.lateral.valid[0, 3]
    np.testing.assert_array_equal(packed.lateral.valid[:, 4], [True, False, False, True])
    np.testing.assert_array_equal(packed.lateral.valid[:, 5], [True, True, False, False])
    for tensor, frame in zip(packed, (lateral, reference)):
        values = frame[list(tensor.channels)].to_numpy(dtype=np.float64)
        np.testing.assert_array_equal(tensor.values.view(np.uint64), values.view(np.uint64))
        rows = tensor.to_structured()
        np.testing.assert_array_equal(rows["source_station"], np.arange(len(rows)))
        for i, name in enumerate(tensor.channels):
            np.testing.assert_array_equal(rows[name].view(np.uint64), values[:, i].view(np.uint64))
            np.testing.assert_array_equal(rows[f"{name}__present"], tensor.valid[:, i])
    assert not packed.typewell.valid[1, 1]


def test_relationship_tensor_keeps_sender_rows_missingness_and_depth_offset():
    packed = pack_well_observations("A", *sample())
    neighbors = gather_md_neighbors(packed.lateral, [1], [0, 2.5, 6., 20., -4.], tolerance_ft=2.5)
    assert neighbors.values.shape == (1, 5, 6)
    np.testing.assert_array_equal(neighbors.source_station, [[1, 1, 2, -1, 0]])
    np.testing.assert_array_equal(neighbors.actual_offset_ft, [[0., 0., 5., np.nan, -5.]])
    assert neighbors.values[0, 0, 4] == -999.25 and not neighbors.valid[0, 0, 4]
    assert not neighbors.valid[0, 3].any() and np.isnan(neighbors.values[0, 3]).all()


@pytest.mark.parametrize("rows,offsets,tolerance", [([1.0], [0.], 1.), ([4], [0.], 1.),
                                                     ([0], [0., 0.], 1.), ([0], [0.], -1.)])
def test_invalid_neighbor_axes_are_rejected(rows, offsets, tolerance):
    packed = pack_well_observations("A", *sample())
    with pytest.raises(ValueError):
        gather_md_neighbors(packed.lateral, rows, offsets, tolerance_ft=tolerance)
