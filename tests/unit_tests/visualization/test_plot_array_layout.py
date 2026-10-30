from pathlib import Path
from unittest.mock import Mock

import astropy.units as u
import matplotlib.figure as mpl_fig
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.table import QTable, Table
from matplotlib.collections import PatchCollection
from matplotlib.colors import to_rgba

from simtools.utils import geometry as transf
from simtools.visualization import plot_array_layout as pal
from simtools.visualization.plot_array_layout import (
    finalize_plot,
    generate_plot_combinations,
    get_patches,
    get_positions,
    get_sphere_radius,
    get_telescope_name,
    get_telescope_patch,
    plot_array_layout,
    update_legend,
)


@pytest.fixture
def telescopes():
    return QTable(
        {
            "telescope_name": ["LSTN-01", "MSTN-01", "LSTN-02"],
            "position_x": [0, 100, 200] * u.m,
            "position_y": [0, 100, 200] * u.m,
            "pos_x_rotated": [0, 100, 200] * u.m,
            "pos_y_rotated": [0, 100, 200] * u.m,
            "sphere_radius": [12, 8, 12] * u.m,
        }
    )


def _layout_table(names, positions, radii):
    return QTable(
        {
            "telescope_name": names,
            "position_x": positions * u.m,
            "position_y": positions * u.m,
            "sphere_radius": radii * u.m,
        }
    )


def test_finalize_plot_with_range():
    _, ax = plt.subplots()
    finalize_plot(
        ax, [mpatches.Circle((0, 0), 1)], "Easting [m]", "Northing [m]", (-10, 10), (-10, 10)
    )

    assert ax.get_xlabel() == "Easting [m]"
    assert ax.get_ylabel() == "Northing [m]"
    assert ax.get_xlim() == (-10, 10)
    assert ax.get_ylim() == (-10, 10)
    assert [coll for coll in ax.collections if isinstance(coll, PatchCollection)]
    for tick in ax.get_xticklabels():
        assert tick.get_fontsize() == 8


def test_finalize_plot_without_range():
    _, ax = plt.subplots()
    finalize_plot(ax, [mpatches.Circle((0, 0), 1)], "Easting [m]", "Northing [m]", None, None)

    assert ax.get_xlabel() == "Easting [m]"
    assert ax.get_ylabel() == "Northing [m]"
    assert ax.get_xlim() != (-10, 10) or ax.get_ylim() != (-10, 10)
    assert [coll for coll in ax.collections if isinstance(coll, PatchCollection)]


@pytest.mark.parametrize(
    ("grayed_out", "expected_labels"),
    [
        ([], ["LSTN (2)", "MSTN (1)"]),
        (["LSTN-01"], ["LSTN (1)", "MSTN (1)"]),
    ],
)
def test_update_legend_counts(telescopes, grayed_out, expected_labels):
    _, ax = plt.subplots()
    update_legend(ax, telescopes, grayed_out_elements=grayed_out)

    labels = [txt.get_text() for txt in ax.get_legend().get_texts()]
    for expected in expected_labels:
        prefix, count = expected.split(" ")
        assert any(prefix in label and count in label for label in labels)


@pytest.mark.parametrize(
    ("table_columns", "expected"),
    [
        ({"sphere_radius": [5.0] * u.m}, 5.0),
        ({}, 10.0),  # default radius when the column is missing
    ],
)
def test_get_sphere_radius(table_columns, expected):
    tbl = QTable({"telescope_name": ["TEL-01"], **table_columns})
    result = get_sphere_radius(tbl[0])
    assert np.isclose(result.value, expected)
    assert result.unit.is_equivalent(u.m)


class DummyTel:
    """Telescope row stand-in with an index attribute."""

    def __init__(self, data, index):
        self.data = data
        self.index = index
        self.colnames = list(data.keys())

    def __getitem__(self, key):
        return self.data[key]


def test_get_telescope_name():
    row = Table({"asset_code": ["ASSET"], "sequence_number": [101]})[0]
    dummy = DummyTel(
        {"asset_code": row["asset_code"], "sequence_number": row["sequence_number"]}, 0
    )
    assert get_telescope_name(dummy) == "ASSET-101"
    # Fallback to the table index when no name columns exist
    assert get_telescope_name(DummyTel({"other_column": "value"}, index=5)) == "tel_5"


def dummy_rotate(x, y, angle):
    return x * 2, y * 2


def test_get_positions_with_position_columns(monkeypatch):
    x = [1, -2, 3] * u.m
    y = [4, 5, -6] * u.m
    monkeypatch.setattr(transf, "rotate", dummy_rotate)
    rotated_x, rotated_y = get_positions(QTable({"position_x": x, "position_y": y}))
    np.testing.assert_allclose(rotated_x.value, (x * 2).value)
    np.testing.assert_allclose(rotated_y.value, (y * 2).value)


def test_get_positions_with_utm_columns(monkeypatch):
    x = [10, 20, 30] * u.m
    y = [40, 50, 60] * u.m

    def dummy_rotate_no_call(x_val, y_val, angle):
        return x_val * 3, y_val * 3

    monkeypatch.setattr(transf, "rotate", dummy_rotate_no_call)
    # Rotation angle is 0 for UTM coordinates, so values are returned unchanged
    rotated_x, rotated_y = get_positions(QTable({"utm_east": x, "utm_north": y}))
    np.testing.assert_allclose(rotated_x.value, x.value)
    np.testing.assert_allclose(rotated_y.value, y.value)


def test_get_positions_missing_columns():
    with pytest.raises(ValueError, match=r"Missing required position columns."):
        get_positions(QTable({"some_column": [1, 2, 3] * u.m}))


def test_get_patches_simplest(monkeypatch):
    telescopes = QTable({"position_x": [0, 0] * u.m, "position_y": [0, 0] * u.m})
    dummy_x = u.Quantity([1, 2], u.m)
    dummy_y = u.Quantity([3, 4], u.m)
    dummy_patches = ["dummy_patch1", "dummy_patch2"]

    monkeypatch.setattr(
        "simtools.visualization.plot_array_layout.get_positions",
        lambda _telescopes: (dummy_x, dummy_y),
    )
    monkeypatch.setattr(
        "simtools.visualization.plot_array_layout.create_patches",
        lambda *args, **kwargs: (dummy_patches, [2, 3] * u.m, [], []),
    )

    _, ax = plt.subplots()
    patches, returned_range, highlighted_patches, _, _ = get_patches(ax, telescopes, False, 50, 1.0)
    assert patches == dummy_patches
    assert returned_range == 50
    assert highlighted_patches == []
    assert "pos_x_rotated" in telescopes.colnames
    assert "pos_y_rotated" in telescopes.colnames

    # Without an explicit range, the range is derived from the patch extent
    _, returned_range, _, _, _ = get_patches(ax, telescopes, False, None, 1.0)
    assert returned_range == pytest.approx(7.7)


def test_get_patches_with_highlighted_elements(telescopes):
    _, ax = plt.subplots()
    patches, returned_range, highlighted_patches, _, _ = get_patches(
        ax,
        telescopes,
        False,
        None,
        1.0,
        grayed_out_elements=["LSTN-02"],
        highlighted_elements=["LSTN-01", "MSTN-01"],
    )

    assert len(patches) == 3
    assert len(highlighted_patches) == 2
    assert returned_range > 0
    for highlight_patch in highlighted_patches:
        assert isinstance(highlight_patch, mpatches.Circle)
        assert highlight_patch.get_fill() is False
        assert highlight_patch.get_facecolor()[:3] == to_rgba("red")[:3]


def test_get_patches_empty_with_axes_range():
    telescopes = _layout_table(["LSTN-01"], [100], [12])
    _, ax = plt.subplots()
    patches, axes_range, _, _, _ = get_patches(
        ax,
        telescopes,
        show_tel_label=False,
        axes_range=1000.0,
        marker_scaling=1.0,
        filter_x_lim=(500, 600),  # filters out the telescope at 100
        filter_y_lim=(500, 600),
    )
    assert len(patches) == 0
    assert np.isclose(axes_range, 1000.0)


@pytest.mark.parametrize(
    ("telescope_type", "expected_class", "check"),
    [
        ("SCTS", mpatches.Rectangle, "square"),  # square in legend handlers
        ("HESS", mpatches.RegularPolygon, "hexagon"),  # hexagon in legend handlers
    ],
)
def test_get_telescope_patch(telescope_type, expected_class, check):
    x, y, radius = 10 * u.m, 20 * u.m, 2 * u.m
    patch = get_telescope_patch(telescope_type, x, y, radius)
    assert isinstance(patch, expected_class)
    if check == "square":
        assert patch.get_xy() == ((x - radius / 2).value, (y - radius / 2).value)
        assert patch.get_width() == radius.to(u.m).value
        assert patch.get_height() == radius.to(u.m).value
        assert patch.get_fill() is False
        assert patch.get_edgecolor() == to_rgba("black")
    else:
        assert patch.numvertices == 6
        assert patch.get_fill() is True


def test_plot_array_layout_calls_adjust_text(monkeypatch, telescopes):
    fig = plot_array_layout(telescopes, show_tel_label=True)
    assert isinstance(fig, mpl_fig.Figure)
    assert fig.axes[0].texts


def test_plot_array_layout_with_highlighted_elements():
    telescopes = _layout_table(["LSTN-01", "MSTN-01", "LSTN-02"], [0, 100, 200], [12, 8, 12])

    fig = plot_array_layout(
        telescopes,
        highlighted_elements=["LSTN-01"],
        grayed_out_elements=["MSTN-01"],
        show_tel_label=False,
        axes_range=300,
    )

    assert isinstance(fig, mpl_fig.Figure)
    ax = fig.get_axes()[0]
    collections = [coll for coll in ax.collections if isinstance(coll, PatchCollection)]
    assert len(collections) >= 2  # normal patches and highlighted patches


@pytest.mark.parametrize(
    ("kwargs", "check"),
    [
        ({"bounds_mode": "symmetric", "padding": 0.2}, "symmetric"),
        ({"x_lim": (-500, 500)}, "x_lim"),
        ({"y_lim": (-300, 300)}, "y_lim"),
        ({"x_lim": (500, 600), "y_lim": (500, 600)}, "filtered_out"),
    ],
)
def test_plot_array_layout_bounds(kwargs, check):
    telescopes = _layout_table(["LSTN-01", "MSTN-01"], [0, 100], [12, 8])
    if check == "symmetric":
        telescopes["position_x"] = [100, 200] * u.m
        telescopes["position_y"] = [100, 200] * u.m

    fig = plot_array_layout(telescopes, **kwargs)
    ax = fig.get_axes()[0]
    assert isinstance(fig, mpl_fig.Figure)
    if check == "symmetric":
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        assert abs(xlim[0] + xlim[1]) < 1e-6
        assert abs(ylim[0] + ylim[1]) < 1e-6
    elif check == "x_lim":
        assert ax.get_xlim() == (-500, 500)
    elif check == "y_lim":
        assert ax.get_ylim() == (-300, 300)
    plt.close(fig)


def test_plot_array_layout_with_background_symmetric_mode():
    telescopes = _layout_table(["LSTN-01"], [50], [12])
    background = _layout_table(["MSTN-01"], [200], [8])

    fig = plot_array_layout(telescopes, background_telescopes=background, bounds_mode="symmetric")

    ax = fig.get_axes()[0]
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    # Background should expand the symmetric range
    assert abs(xlim[0] + xlim[1]) < 1e-6
    assert abs(ylim[0] + ylim[1]) < 1e-6
    plt.close(fig)


def _layout_args(**overrides):
    args = {
        "show_labels": False,
        "axes_range": 100,
        "marker_scaling": 1.0,
        "grayed_out_array_elements": [],
        "highlighted_array_elements": [],
        "legend_location": "best",
        "bounds": "exact",
        "padding": 0.1,
        "x_lim": None,
        "y_lim": None,
        "figure_name": None,
        "site": None,
        "coordinate_system": "utm",
        "model_version": "6.0.0",
    }
    args.update(overrides)
    return args


@pytest.mark.parametrize(
    ("figure_name", "site", "coordinate_system", "layout_name", "expected_filename"),
    [
        (None, None, "utm", "testlayout", "array_layout_testlayout_utm_6-0-0"),
        (None, "North", "ground", "alpha", "array_layout_alpha_North_ground_6-0-0"),
        ("custom_name", "North", "ground", "alpha", "custom_name"),
    ],
    ids=["layout_name", "default_filename_includes_model_version", "figure_name_override"],
)
def test_plot_array_layouts(
    monkeypatch,
    tmp_test_directory,
    figure_name,
    site,
    coordinate_system,
    layout_name,
    expected_filename,
):
    captured = {"name": None, "plotted": False}

    def dummy_plot_array_layout(**kwargs):
        captured["plotted"] = True
        return Mock()

    def dummy_save_figure(fig, path, dpi):
        assert fig is not None
        assert path.parent == Path(tmp_test_directory)
        captured["name"] = path.name

    monkeypatch.setattr(pal, "plot_array_layout", dummy_plot_array_layout)
    monkeypatch.setattr(pal.visualize, "save_figure", dummy_save_figure)

    layouts = [
        {
            "array_elements": _layout_table(["LSTN-01"], [0], [12]),
            "name": layout_name,
            "site": site,
        }
    ]
    args = _layout_args(figure_name=figure_name, site=site, coordinate_system=coordinate_system)
    pal.plot_array_layouts(args, Path(tmp_test_directory), layouts)

    assert captured["plotted"]
    assert captured["name"] == expected_filename


def test_generate_plot_combinations_default():
    args = {
        "model_version": "1.0.0",
        "site": "North",
        "all_model_versions": False,
        "all_sites": False,
    }
    assert list(generate_plot_combinations(args)) == [("1.0.0", "North")]


def test_generate_plot_combinations_all_sites_and_versions(monkeypatch):
    args = {
        "model_version": None,
        "site": None,
        "all_model_versions": True,
        "all_sites": True,
    }
    model_reader = Mock()
    model_reader.get_model_versions.return_value = ["6.0.0", "7.0.0"]
    monkeypatch.setattr(
        "simtools.visualization.plot_array_layout.names.site_names",
        Mock(return_value=["North", "South"]),
    )

    assert list(generate_plot_combinations(args, model_reader=model_reader)) == [
        ("6.0.0", "North"),
        ("6.0.0", "South"),
        ("7.0.0", "North"),
        ("7.0.0", "South"),
    ]
