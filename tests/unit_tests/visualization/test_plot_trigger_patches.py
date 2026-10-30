"""Tests for offline trigger-patch configuration analysis."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.table import Table
from matplotlib import pyplot as plt

from simtools.visualization import plot_trigger_patches as patch_plot
from simtools.visualization.plot_trigger_patches import (
    DISTANCE_STATUS,
    _format_pixel_ids,
    analyze_trigger_patch_configuration,
    format_trigger_patch_summary,
    write_trigger_patch_report,
)

GROUP_ID = "group_id"
PIXEL_ID = "pixel_id"
PIXEL_IDS = "pix_id"
PATCH_COUNT = "patch_count"
TYPICAL_DISTANCE = "typical_distance_cm"
COMPONENTS = "connected_components"
MEDIAN_PAIR_DISTANCE = "median_pair_distance_cm"
UNIQUE_ERROR = "must be unique"
ANALOG_SUM = "analog_sum"
MAJORITY = "majority"
CONFIGURED = "configured"
AVAILABLE = "available"
CAMERA_AXIS_BOUNDS = "camera_axis_bounds"


def _camera():
    return SimpleNamespace(
        telescope_name="LSTN-test",
        pixels={
            PIXEL_IDS: [10, 11, 12, 13, 14],
            "x": [0.0, 1.0, 2.0, 10.0, 11.0],
            "y": [0.0, 0.0, 0.0, 0.0, 0.0],
            "pixel_shape": 1,
            "pixel_diameter": 0.8,
            "rotate_angle": 0.0,
            "orientation": 0.0,
            "pix_on": [True, True, True, True, True],
        },
        get_neighbor_pixels=lambda: [[1], [0, 2], [1], [4], [3]],
    )


def _group_tables():
    groups = Table(
        rows=[(1, ANALOG_SUM), (2, ANALOG_SUM)],
        names=[GROUP_ID, "kind"],
    )
    members = Table(
        rows=[
            (1, 10),
            (1, 11),
            (1, 12),
            (2, 11),
            (2, 12),
            (2, 13),
        ],
        names=[GROUP_ID, PIXEL_ID],
    )
    return groups, members


def test_analyze_trigger_patch_configuration_measures_coverage_and_distances():
    groups, members = _group_tables()

    pixels, group_summary = analyze_trigger_patch_configuration(_camera(), groups, members)

    assert pixels[PATCH_COUNT].tolist() == [1, 2, 2, 1, 0]
    assert pixels["patch_ids"].tolist() == ["1", "1,2", "1,2", "2", ""]
    assert pixels["trigger_status"].tolist() == [
        CONFIGURED,
        CONFIGURED,
        CONFIGURED,
        CONFIGURED,
        "unclassified",
    ]
    assert pixels[DISTANCE_STATUS].tolist() == [
        AVAILABLE,
        AVAILABLE,
        AVAILABLE,
        AVAILABLE,
        "no patch membership",
    ]
    assert np.isclose(pixels[TYPICAL_DISTANCE][0], 1.5)
    assert np.isclose(pixels[TYPICAL_DISTANCE][1], 3.0)
    assert np.isclose(pixels[TYPICAL_DISTANCE][2], 3.0)
    assert np.isclose(pixels[TYPICAL_DISTANCE][3], 8.5)
    assert np.isnan(pixels[TYPICAL_DISTANCE][4])
    assert group_summary[MEDIAN_PAIR_DISTANCE].tolist() == [1.0, 8.0]
    assert group_summary["max_pair_distance_cm"].tolist() == [2.0, 9.0]
    assert group_summary[COMPONENTS].tolist() == [1, 2]


def test_analyze_trigger_patch_configuration_rejects_members_for_unknown_group():
    groups, members = _group_tables()
    members[GROUP_ID][0] = 99

    with pytest.raises(ValueError, match="unknown trigger group"):
        analyze_trigger_patch_configuration(_camera(), groups, members)


def test_analyze_trigger_patch_configuration_rejects_members_for_unknown_pixel():
    groups, members = _group_tables()
    members[PIXEL_ID][0] = 99

    with pytest.raises(ValueError, match="not in the camera"):
        analyze_trigger_patch_configuration(_camera(), groups, members)


def test_analyze_trigger_patch_configuration_rejects_empty_group():
    groups, members = _group_tables()
    groups.add_row((3, MAJORITY))

    with pytest.raises(ValueError, match="no members"):
        analyze_trigger_patch_configuration(_camera(), groups, members)


def test_analyze_trigger_patch_configuration_rejects_duplicate_camera_ids():
    camera = _camera()
    camera.pixels[PIXEL_IDS][1] = 10
    groups, members = _group_tables()

    with pytest.raises(ValueError, match=UNIQUE_ERROR):
        analyze_trigger_patch_configuration(camera, groups, members)


def test_analyze_trigger_patch_configuration_rejects_coordinate_length_mismatch():
    camera = _camera()
    camera.pixels["x"].pop()
    groups, members = _group_tables()

    with pytest.raises(ValueError, match="matching lengths"):
        analyze_trigger_patch_configuration(camera, groups, members)


def test_analyze_trigger_patch_configuration_rejects_nonfinite_coordinates():
    camera = _camera()
    camera.pixels["x"][0] = np.nan
    groups, members = _group_tables()

    with pytest.raises(ValueError, match="coordinates must be finite"):
        analyze_trigger_patch_configuration(camera, groups, members)


def test_analyze_trigger_patch_configuration_rejects_duplicate_group_ids():
    groups, members = _group_tables()
    groups[GROUP_ID][1] = 1

    with pytest.raises(ValueError, match="group IDs " + UNIQUE_ERROR):
        analyze_trigger_patch_configuration(_camera(), groups, members)


def test_analyze_trigger_patch_configuration_rejects_no_groups():
    groups = Table(names=[GROUP_ID, "kind"], dtype=[int, str])
    members = Table(names=[GROUP_ID, PIXEL_ID], dtype=[int, int])

    with pytest.raises(ValueError, match="no trigger groups"):
        analyze_trigger_patch_configuration(_camera(), groups, members)


def test_analyze_trigger_patch_configuration_rejects_missing_group_table_column():
    groups = Table(rows=[(1,)], names=[GROUP_ID])
    members = Table(rows=[(1, 10)], names=[GROUP_ID, PIXEL_ID])

    with pytest.raises(ValueError, match="missing required column"):
        analyze_trigger_patch_configuration(_camera(), groups, members)


def test_analyze_trigger_patch_configuration_rejects_neighbor_count_mismatch():
    camera = _camera()
    camera.get_neighbor_pixels = lambda: []
    groups, members = _group_tables()

    with pytest.raises(ValueError, match="neighbor indices"):
        analyze_trigger_patch_configuration(camera, groups, members)


def test_analyze_trigger_patch_configuration_marks_single_member_group_distances_undefined():
    groups = Table(rows=[(1, MAJORITY)], names=[GROUP_ID, "kind"])
    members = Table(rows=[(1, 10)], names=[GROUP_ID, PIXEL_ID])

    pixels, group_summary = analyze_trigger_patch_configuration(_camera(), groups, members)

    assert pixels[PATCH_COUNT][0] == 1
    assert np.isnan(pixels[TYPICAL_DISTANCE][0])
    assert pixels[DISTANCE_STATUS][0] == "no other members in patch"
    assert np.isnan(group_summary[MEDIAN_PAIR_DISTANCE][0])
    assert group_summary[COMPONENTS][0] == 1


def test_format_pixel_ids_limits_long_lists():
    assert _format_pixel_ids([]) == "none"
    expected_ids = ", ".join(map(str, range(20))) + ", ..."
    assert _format_pixel_ids(list(range(21))) == expected_ids


def test_format_trigger_patch_summary_limits_group_details():
    groups = Table(
        rows=[(group_id, ANALOG_SUM) for group_id in range(1, 7)],
        names=[GROUP_ID, "kind"],
    )
    members = Table(
        rows=[(group_id, pixel_id) for group_id in range(1, 7) for pixel_id in (10, 14)],
        names=[GROUP_ID, PIXEL_ID],
    )
    pixels, group_summary = analyze_trigger_patch_configuration(_camera(), groups, members)

    summary = format_trigger_patch_summary(pixels, group_summary)

    assert "... and 1 more; see groups table" in summary
    assert "  pixels: 10, 14" in summary


def test_format_trigger_patch_summary_only_lists_outlier_spans():
    groups = Table(
        rows=[(group_id, ANALOG_SUM) for group_id in range(1, 8)],
        names=[GROUP_ID, "kind"],
    )
    members = Table(
        rows=[(group_id, pixel_id) for group_id in range(1, 7) for pixel_id in (10, 11)]
        + [(7, pixel_id) for pixel_id in (10, 14)],
        names=[GROUP_ID, PIXEL_ID],
    )
    pixels, group_summary = analyze_trigger_patch_configuration(_camera(), groups, members)

    summary = format_trigger_patch_summary(pixels, group_summary)

    assert "7 (analog_sum)" in summary
    assert "1 (analog_sum)" not in summary
    assert "max span 11.00 cm" in summary


def test_format_trigger_patch_summary_omits_common_connected_spans():
    groups = Table(
        rows=[(group_id, ANALOG_SUM) for group_id in range(1, 7)],
        names=[GROUP_ID, "kind"],
    )
    members = Table(
        rows=[(group_id, pixel_id) for group_id in range(1, 7) for pixel_id in (10, 11)],
        names=[GROUP_ID, PIXEL_ID],
    )
    pixels, group_summary = analyze_trigger_patch_configuration(_camera(), groups, members)

    summary = format_trigger_patch_summary(pixels, group_summary)

    assert "No outlying spans or disconnected groups." in summary
    assert "pixels: 10, 11" not in summary


def test_format_trigger_patch_summary_wraps_full_patch_pixel_list():
    pixel_ids = list(range(1, 26))
    camera = SimpleNamespace(
        pixels={
            PIXEL_IDS: pixel_ids,
            "x": [float(pixel_id) for pixel_id in pixel_ids],
            "y": [0.0] * len(pixel_ids),
        },
        get_neighbor_pixels=lambda: [[] for _index in range(len(pixel_ids))],
    )
    groups = Table(rows=[(1, ANALOG_SUM)], names=[GROUP_ID, "kind"])
    members = Table(rows=[(1, pixel_id) for pixel_id in pixel_ids], names=[GROUP_ID, PIXEL_ID])
    pixels, group_summary = analyze_trigger_patch_configuration(camera, groups, members)

    summary = format_trigger_patch_summary(pixels, group_summary)

    group_pixel_text = summary.split("  pixels: ", maxsplit=1)[1]
    assert "\n    " in group_pixel_text
    assert all(str(pixel_id) in group_pixel_text for pixel_id in pixel_ids)


def test_report_keeps_coverage_first_and_adds_distance_page(mocker, tmp_test_directory):
    groups, members = _group_tables()
    pixels, group_summary = analyze_trigger_patch_configuration(_camera(), groups, members)

    class FakePdf:
        """Capture page titles without writing a PDF during this unit test."""

        def __init__(self):
            self.page_titles = []
            self.axis_bounds = {CAMERA_AXIS_BOUNDS: []}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def add(self, figure):
            self.page_titles.append(figure._suptitle.get_text())
            self.axis_bounds[CAMERA_AXIS_BOUNDS].append(figure.axes[0].get_position().bounds)
            plt.close(figure)

    fake_pdf = FakePdf()
    mocker.patch.object(patch_plot, "pdf_report", return_value=fake_pdf)
    images = []

    def capture_image(_camera, image, **_kwargs):
        images.append(image)

    mocker.patch.object(patch_plot, "plot_pixel_layout_with_image", side_effect=capture_image)

    write_trigger_patch_report(_camera(), pixels, group_summary, tmp_test_directory / "report.pdf")

    assert fake_pdf.page_titles[0].endswith("patch configuration")
    assert "typical distance" in fake_pdf.page_titles[1]
    assert np.allclose(
        fake_pdf.axis_bounds[CAMERA_AXIS_BOUNDS][0],
        fake_pdf.axis_bounds[CAMERA_AXIS_BOUNDS][1],
    )
    assert images[0].tolist() == [1, 2, 2, 1, 0]
    assert np.isclose(images[1][0], 1.5)
    assert np.ma.is_masked(images[1][4])


def test_report_renders_when_no_pixel_has_a_defined_distance(tmp_test_directory):
    groups = Table(rows=[(1, MAJORITY)], names=[GROUP_ID, "kind"])
    members = Table(rows=[(1, 10)], names=[GROUP_ID, PIXEL_ID])
    pixels, group_summary = analyze_trigger_patch_configuration(_camera(), groups, members)
    pdf_path = Path(tmp_test_directory) / "singleton-patch.pdf"

    write_trigger_patch_report(_camera(), pixels, group_summary, pdf_path)

    assert pdf_path.is_file()
    assert pdf_path.stat().st_size > 0
    assert not list(Path(tmp_test_directory).glob("*.png"))
