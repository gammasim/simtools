"""Comprehensive unit tests for plot_camera module."""

from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.table import Table

from simtools.visualization import plot_camera as camera_plot
from simtools.visualization.camera_plot_utils import (
    setup_camera_axis_properties,
)
from simtools.visualization.plot_camera import (
    _color_normalization,
    _plot_axes_def,
    _plot_one_axis_def,
    parse_pixel_ids_to_print,
    plot_pixel_layout,
    plot_pixel_layout_with_image,
)

GROUP_ID = "group_id"
PIXEL_ID = "pixel_id"
PIXEL_IDS = "pix_id"
MESSAGE = "message"
LST_DESIGN = "LSTN-design"
TRIGGER_PIXEL = "trigger_pixel"
PIXEL = "pixel"


@pytest.fixture
def simple_camera():
    """Create a mock camera with basic properties."""
    camera = MagicMock()
    camera.telescope_name = "LST-01"
    camera.focal_length = 10.0
    camera.pixels = {
        "x": [0.0, 1.0, -1.0],
        "y": [0.0, 1.0, -1.0],
        PIXEL_IDS: [0, 1, 2],
        "pix_on": [True, True, False],
        "pixel_diameter": 0.5,
        "pixel_shape": 1,
        "rotate_angle": 0.0,
        "orientation": 0.0,
    }
    camera.calc_fov.return_value = (10.0, 5.0)
    camera.effective_focal_length = [2923.7, 0.0, 0.0, 0.0, 0.0]
    camera.get_edge_pixels.return_value = [1]
    return camera


@pytest.fixture
def camera_hexagon():
    """Camera with hexagonal pixels."""
    camera = MagicMock()
    camera.telescope_name = "MST-01"
    camera.focal_length = 16.0
    camera.pixels = {
        "x": [0.0, 1.0, -1.0, 0.5, -0.5],
        "y": [0.0, 1.0, -1.0, 0.5, -0.5],
        PIXEL_IDS: [0, 1, 2, 3, 4],
        "pix_on": [True, True, True, False, True],
        "pixel_diameter": 0.3,
        "pixel_shape": 1,
        "rotate_angle": 0.0,
        "orientation": 0.0,
    }
    camera.calc_fov.return_value = (8.0, 4.0)
    camera.effective_focal_length = [2923.7, 0.0, 0.0, 0.0, 0.0]
    camera.get_edge_pixels.return_value = [0, 3]
    return camera


@pytest.fixture
def camera_square():
    """Camera with square pixels."""
    camera = MagicMock()
    camera.telescope_name = "SST-01"
    camera.focal_length = 5.6
    camera.pixels = {
        "x": [0.0, 1.0, -1.0],
        "y": [0.0, 1.0, -1.0],
        PIXEL_IDS: [0, 1, 2],
        "pix_on": [True, True, False],
        "pixel_diameter": 0.5,
        "pixel_shape": 2,
        "rotate_angle": 0.0,
        "orientation": 0.0,
    }
    camera.calc_fov.return_value = (20.0, 10.0)
    camera.effective_focal_length = [2923.7, 0.0, 0.0, 0.0, 0.0]
    camera.get_edge_pixels.return_value = [0]
    return camera


def test_color_normalization_none_image():
    result = _color_normalization(None, "viridis")
    assert result == (None, None, None)


def test_plot_layout_with_no_image(simple_camera):
    fig = plot_pixel_layout_with_image(simple_camera, image=None)
    assert fig is not None


def test_color_normalization_linear():
    image = np.array([0.0, 0.5, 1.0])
    colors, _, _ = _color_normalization(image, "viridis", norm_type="lin")
    assert colors is not None
    assert len(colors) == 3


def test_color_normalization_log():
    image = np.array([1.0, 10.0, 100.0])
    colors, _, _ = _color_normalization(image, "viridis", norm_type="log")
    assert colors is not None


def test_setup_camera_axis_with_scale_factor(camera_hexagon):
    ax = MagicMock()
    setup_camera_axis_properties(ax, camera_hexagon, y_scale_factor=1.42)
    ax.axis.assert_called_once()


def test_setup_camera_axis_grid_no_alpha(camera_hexagon):
    ax = MagicMock()
    setup_camera_axis_properties(ax, camera_hexagon, grid=True)
    ax.grid.assert_called_with(True)


def test_setup_camera_axis_below(camera_hexagon):
    ax = MagicMock()
    setup_camera_axis_properties(ax, camera_hexagon, axis_below=True)
    ax.set_axisbelow.assert_called_with(True)


def test_plot_layout_with_image_values(simple_camera):
    image = np.array([0.0, 0.5, 1.0])
    fig = plot_pixel_layout_with_image(simple_camera, image=image)
    assert fig is not None


def test_plot_layout_with_image_no_bar(camera_square):
    image = np.array([0.0, 0.5, 1.0])
    fig = plot_pixel_layout_with_image(camera_square, image=image, add_color_bar=False)
    assert fig is not None


def test_plot_layout_with_image_axes(camera_square):
    _, ax = plt.subplots()
    image = np.array([0.0, 0.5, 1.0])
    fig = plot_pixel_layout_with_image(camera_square, image=image, ax=ax)
    assert fig is not None
    assert len(ax.collections) == 1
    assert not ax.patches
    plt.close(fig)


def test_plot_one_axis_no_invert():
    plt_mock = MagicMock()
    _plot_one_axis_def(plt_mock, "X", "Y", 0.7, 0.12, 0.0, "black", False)
    assert plt_mock.gca.return_value.annotate.call_count == 2


def test_plot_one_axis_with_invert():
    plt_mock = MagicMock()
    _plot_one_axis_def(plt_mock, "X", "Y", 0.8, 0.12, np.pi / 2, "blue", True)
    assert plt_mock.gca.return_value.annotate.call_count == 2


def _axis_direction(annotation_call):
    """Return the vector from the axis origin to its labelled arrow tip."""
    origin = np.asarray(annotation_call.kwargs["xy"])
    arrow_tip = np.asarray(annotation_call.kwargs["xytext"])
    return (arrow_tip - origin).tolist()


def test_plot_axes_dual_mirror_directions():
    camera = MagicMock()
    camera.telescope_name = "LST-01"
    plt_mock = MagicMock()

    with patch("simtools.visualization.plot_camera.is_two_mirror_telescope", return_value=True):
        _plot_axes_def(camera, plt_mock, 0.0)

    calls = plt_mock.gca.return_value.annotate.call_args_list
    assert _axis_direction(calls[2]) == pytest.approx([0.0, -0.1])  # x_cam: down
    assert _axis_direction(calls[3]) == pytest.approx([0.1, 0.0])  # y_cam: right


def test_plot_axes_single_mirror_directions():
    camera = MagicMock()
    camera.telescope_name = "MST-01"
    plt_mock = MagicMock()
    rotate_angle = np.deg2rad(30.0)

    with patch("simtools.visualization.plot_camera.is_two_mirror_telescope", return_value=False):
        _plot_axes_def(camera, plt_mock, rotate_angle)

    calls = plt_mock.gca.return_value.annotate.call_args_list
    assert _axis_direction(calls[0]) == pytest.approx([-0.05, -np.sqrt(3) * 0.05])
    assert _axis_direction(calls[1]) == pytest.approx([-np.sqrt(3) * 0.05, 0.05])
    assert _axis_direction(calls[2]) == pytest.approx([0.0, -0.1])  # x_cam: down
    assert _axis_direction(calls[3]) == pytest.approx([-0.1, 0.0])  # y_cam: left


def test_plot_axes_large_rotation():
    camera = MagicMock()
    camera.telescope_name = "LST-01"
    camera.pixels = {"rotate_angle": 2.0}
    plt_mock = MagicMock()

    with patch("simtools.visualization.plot_camera.is_two_mirror_telescope", return_value=False):
        _plot_axes_def(camera, plt_mock, np.deg2rad(120))

    assert plt_mock.gca.return_value.annotate.call_count >= 2


def test_plot_pixel_layout_does_not_mutate_camera_coordinates(simple_camera):
    simple_camera.telescope_name = "LSTN-01"
    original_y = list(simple_camera.pixels["y"])

    with (
        patch("simtools.visualization.plot_camera.is_two_mirror_telescope", return_value=False),
        patch(
            "simtools.visualization.plot_camera.create_pixel_patches_by_type",
            return_value=([], [], []),
        ) as mock_create_patches,
    ):
        figure = plot_pixel_layout(simple_camera, camera_in_sky_coor=False)

    plotted_camera = mock_create_patches.call_args.args[0]
    assert plotted_camera.pixels["y"] == [-value for value in original_y]
    assert simple_camera.pixels["y"] == original_y
    plt.close(figure)


def _mock_camera(n_pixels=1855):
    camera = MagicMock()
    camera.get_number_of_pixels.return_value = n_pixels
    camera.calc_fov.return_value = (4.5, 120.0)
    return camera


def test_parse_pixel_ids_to_print_integer():
    camera = _mock_camera()
    assert parse_pixel_ids_to_print(50, camera) == 50


def test_parse_pixel_ids_to_print_zero_returns_minus_one():
    camera = _mock_camera()
    assert parse_pixel_ids_to_print(0, camera) == -1


def test_parse_pixel_ids_to_print_all():
    camera = _mock_camera(n_pixels=1855)
    assert parse_pixel_ids_to_print("All", camera) == 1855


def test_parse_pixel_ids_to_print_invalid_raises():
    camera = _mock_camera()
    with pytest.raises(ValueError, match="must be integer or 'All'"):
        parse_pixel_ids_to_print("invalid", camera)


def _pixel_patches(telescope, pixel_id, camera_pixels):
    """Look up trigger patches for a pixel from a mock telescope's membership table."""
    members = telescope.get_parameter_table("camera_trigger_members")
    return camera_plot.trigger_patches_for_pixel(members, pixel_id, camera_pixels)


def _read_selected(model, patch_ids, camera_pixels):
    """Resolve selected trigger patches from a mock model's group and member tables."""
    definitions = model.get_parameter_table("camera_trigger_groups")
    members = model.get_parameter_table("camera_trigger_members")
    return camera_plot.read_selected_trigger_patches(definitions, members, patch_ids, camera_pixels)


@pytest.mark.parametrize(PIXEL, [0, 75])
def test_trigger_patches_for_pixel(pixel):
    telescope = MagicMock()
    telescope.get_parameter_table.return_value = Table(
        {GROUP_ID: [8, 3, 8, 9], PIXEL_ID: [pixel, pixel, pixel, 99]}
    )
    assert _pixel_patches(telescope, pixel, {pixel, 99}) == [3, 8]


@pytest.mark.parametrize((PIXEL, MESSAGE), [(999, "not in the camera"), (75, "no trigger patches")])
def test_trigger_patches_for_pixel_invalid(pixel, message):
    telescope = MagicMock()
    telescope.get_parameter_table.return_value = Table({GROUP_ID: [0], PIXEL_ID: [0]})
    with pytest.raises(ValueError, match=message):
        _pixel_patches(telescope, pixel, {0, 75})


@pytest.mark.parametrize("sky", [False, True])
def test_selected_patch_layout(simple_camera, sky):
    simple_camera.telescope_name = LST_DESIGN
    groups = {7: {0, 1}, 12: {1, 2}, 20: {2}}
    original_y = list(simple_camera.pixels["y"])
    fig = camera_plot.plot_selected_trigger_patches(simple_camera, groups, sky)
    overview, zoom = fig.axes
    assert len(overview.collections) == 4
    overlap_colours = overview.collections[1].get_facecolors()
    expected_colours = {tuple(camera_plot.mcolors.to_rgba(c)) for c in ("#0072B2", "#D55E00")}
    assert {tuple(colour) for colour in overlap_colours} == expected_colours
    assert {text.get_text() for text in zoom.texts} == {"0", "1", "2"}
    labels = [text.get_text() for text in overview.get_legend().texts]
    assert "Patches 7 + 12" in labels
    assert "Patches 12 + 20" in labels
    assert simple_camera.pixels["y"] == original_y
    assert zoom.texts[1].get_position()[1] == pytest.approx(1.0 if sky else -1.0)
    plt.close(fig)


@pytest.mark.parametrize("n_colours", [1, 2, 3])
def test_pixel_stripes(simple_camera, n_colours):
    colours = ["red", "blue", "green"][:n_colours]
    patches, facecolours = camera_plot._striped_pixel_shapes(simple_camera, [(0, 1, -2)], colours)
    assert len(patches) == len(facecolours)
    assert set(facecolours) == set(colours)
    full_shape = camera_plot.pixel_shape(simple_camera, 1, -2)
    bounds = full_shape.get_path().transformed(full_shape.get_transform()).get_extents()
    for shape in patches:
        vertices = shape.get_path().vertices
        assert vertices[:, 0].min() >= bounds.xmin - 1e-10
        assert vertices[:, 0].max() <= bounds.xmax + 1e-10


@pytest.mark.parametrize("ids", [[7], [7, 12], [7, 12, 20], [7, 12, 20, 21, 22, 23]])
def test_read_selected_trigger_patches(ids):
    model = MagicMock()
    model.get_parameter_table.side_effect = [
        Table({GROUP_ID: [7, 12, 20, 21, 22, 23]}),
        Table({GROUP_ID: [7, 7, 12, 20, 21, 22, 23], PIXEL_ID: [0, 1, 0, 0, 0, 0, 0]}),
    ]
    groups = _read_selected(model, ids, {0, 1, 2})
    assert list(groups) == ids
    assert groups[7] == {0, 1}


@pytest.mark.parametrize(
    ("ids", MESSAGE),
    [([], "at least one"), (list(range(7)), "at most 6"), ([7, 7], "distinct")],
)
def test_selected_trigger_patch_count_invalid(ids, message):
    model = MagicMock()
    model.get_parameter_table.side_effect = [
        Table({GROUP_ID: list(range(6))}),
        Table({GROUP_ID: list(range(6)), PIXEL_ID: [0] * 6}),
    ]
    with pytest.raises(ValueError, match=message):
        _read_selected(model, ids, {0})


def test_maximum_patch_membership_ignores_duplicate_entries():
    members = Table({GROUP_ID: [0, 0, 1, 2], PIXEL_ID: [9, 9, 9, 8]})
    assert camera_plot._maximum_patch_membership(members) == 2
    assert camera_plot._maximum_patch_membership(members[:0]) == 0


@pytest.mark.parametrize("count", [6, 7])
def test_many_patch_stripes(simple_camera, count):
    simple_camera.telescope_name = LST_DESIGN
    fig = camera_plot.plot_selected_trigger_patches(
        simple_camera, {i: {0} for i in range(count)}, False
    )
    assert len({tuple(c) for c in fig.axes[1].collections[0].get_facecolors()}) == count
    plt.close(fig)


@pytest.mark.parametrize(
    ("members", "ids", MESSAGE),
    [
        ([0], [99], "Unknown trigger patch"),
        ([], [7], "no members"),
        ([99], [7], "absent from the camera"),
    ],
)
def test_selected_trigger_patch_members_invalid(members, ids, message):
    model = MagicMock()
    model.get_parameter_table.side_effect = [
        Table({GROUP_ID: [7]}),
        Table({GROUP_ID: [7] * len(members), PIXEL_ID: members}),
    ]
    with pytest.raises(ValueError, match=message):
        _read_selected(model, ids, {0})
