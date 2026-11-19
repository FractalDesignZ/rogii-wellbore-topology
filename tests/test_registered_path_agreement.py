import numpy as np
import pytest

from rwt.registered_views import compare_registered_path


def test_both_directions_expose_spurious_ink_and_keep_station_denominator():
    station = [[1, 1], [2, 1], [np.nan, np.nan], [11, 1]]
    pixels = np.array([[1, 1], [2, 1], [1, 7], [1, 8]])
    arrays, report = compare_registered_path(station, pixels, (0, 0, 10, 10),
        radius_px=1, exclude_rois=((0, 7, 2, 8),))
    station_result = report["stations_to_retained_pixels"]
    assert station_result["fraction_within_radius"] == 2/4
    assert station_result["measured"] == 3
    assert report["projected_stations_inside_path_roi"] == 2
    assert report["retained_pixels_to_stations"]["fraction_within_radius"] == 2/3
    np.testing.assert_array_equal(arrays["nearest_candidate_row"], [0, 1, -1, 1])
    np.testing.assert_array_equal(arrays["nearest_station_row"], [0, 1, -1, 0])
    assert arrays["station_to_pixel_distance_px"][-1] == 9


def test_empty_support_and_invalid_exclusion():
    station = np.array([[3., 4.]])
    pixels = np.array([[3, 4]])
    roi = (0, 0, 10, 10)
    arrays, report = compare_registered_path(station, pixels, roi, radius_px=1, exclude_rois=(roi,))
    assert report["stations_to_retained_pixels"]["fraction_within_radius"] == 0
    assert report["retained_pixels_to_stations"]["fraction_within_radius"] is None
    assert np.isnan(arrays["station_to_pixel_distance_px"]).all()
    with pytest.raises(ValueError, match="contained"):
        compare_registered_path(station, pixels, roi, radius_px=1, exclude_rois=((-1, 0, 2, 1),))
