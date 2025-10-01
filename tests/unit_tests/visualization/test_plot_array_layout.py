import builtins

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
from simtools.visualization.plot_array_layout import (
    create_patches,
    finalize_plot,
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


def test_finalize_plot_with_range():
    _, ax = plt.subplots()
    # Create a dummy patch for testing
    dummy_patch = mpatches.Circle((0, 0), 1)
    x_title = "Easting [m]"
    y_title = "Northing [m]"
    axes_range = 10

    finalize_plot(ax, [dummy_patch], x_title, y_title, axes_range)

    assert ax.get_xlabel() == x_title
    assert ax.get_ylabel() == y_title

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    assert xlim == (-axes_range, axes_range)
    assert ylim == (-axes_range, axes_range)

    # Verify that a PatchCollection was added
    collections = [coll for coll in ax.collections if isinstance(coll, PatchCollection)]
    assert collections, "Expected a PatchCollection to be added to the axes."

    xticklabels = ax.get_xticklabels()
    if xticklabels:
        for tick in xticklabels:
            assert tick.get_fontsize() == 8


def test_finalize_plot_without_range():
    _, ax = plt.subplots()
    dummy_patch = mpatches.Circle((0, 0), 1)
    x_title = "Easting [m]"
    y_title = "Northing [m]"

    finalize_plot(ax, [dummy_patch], x_title, y_title, None)
    assert ax.get_xlabel() == x_title
    assert ax.get_ylabel() == y_title

    # When axes_range is None, limits are not reset explicitly
    # Check that the updated limits differ from what would be set if a range were provided
    new_xlim = ax.get_xlim()
    new_ylim = ax.get_ylim()
    assert new_xlim != (-10, 10) or new_ylim != (-10, 10)

    collections = [coll for coll in ax.collections if isinstance(coll, PatchCollection)]
    assert collections, "Expected a PatchCollection to be added to the axes."


def test_update_legend_with_telescopes(telescopes):
    """Test update_legend with telescopes."""
    _, ax = plt.subplots()
    update_legend(ax, telescopes)

    legend = ax.get_legend()
    assert legend is not None

    labels = [txt.get_text() for txt in legend.get_texts()]

    # Should have counts for telescope types present in the data
    # LSTN appears twice, MSTN appears once
    assert any("LSTN" in label and "(2)" in label for label in labels)
    assert any("MSTN" in label and "(1)" in label for label in labels)


def test_update_legend_without_telescopes():
    """Test update_legend with empty telescope list."""

    empty_telescopes = QTable()

    _, ax = plt.subplots()
    update_legend(ax, empty_telescopes)

    # Legend should either not exist or have no labels
    legend = ax.get_legend()
    if legend:
        texts = [txt.get_text() for txt in legend.get_texts()]
        assert len(texts) == 0


def test_get_sphere_radius_with_column():
    # Create a table with 'sphere_radius' column
    tbl = QTable(
        {
            "telescope_name": ["TEL-01"],
            "sphere_radius": [5.0] * u.m,  # This creates a column with units
        }
    )
    row = tbl[0]

    result = get_sphere_radius(row)
    assert np.isclose(result.value, 5.0)
    assert result.unit.is_equivalent(u.m)


def test_get_sphere_radius_without_column():
    # Create a table without a 'sphere_radius' column
    tbl = QTable({"telescope_name": ["TEL-01"]})
    row = tbl[0]

    # Since the 'sphere_radius' column is missing, the function should return 1.0 m.
    result = get_sphere_radius(row)
    assert np.isclose(result.value, 10.0)
    assert result.unit.is_equivalent(u.m)


# Helper class to simulate a telescope row with an index attribute
class DummyTel:
    def __init__(self, data, index):
        self.data = data
        self.index = index
        self.colnames = list(data.keys())

    def __getitem__(self, key):
        return self.data[key]


def test_get_telescope_name_with_telescope_name():
    # When telescope_name column exists, it should return its value.
    tbl = Table({"telescope_name": ["Telescope-01"]})
    row = tbl[0]
    # astropy table row does not have an 'index' attribute, so we simulate it using DummyTel.
    dummy = DummyTel({"telescope_name": row["telescope_name"]}, index=0)
    result = get_telescope_name(dummy)
    assert result == "Telescope-01"


def test_get_telescope_name_with_asset_code_and_sequence_number():
    # When telescope_name column is missing but asset_code and sequence_number exist.
    tbl = Table({"asset_code": ["ASSET"], "sequence_number": [101]})
    row = tbl[0]
    dummy = DummyTel(
        {"asset_code": row["asset_code"], "sequence_number": row["sequence_number"]}, index=0
    )
    result = get_telescope_name(dummy)
    assert result == "ASSET-101"


def test_get_telescope_name_default_fallback():
    # When neither telescope_name nor asset_code/sequence_number exist,
    # fallback should use tel.index.
    data = {"other_column": "value"}
    dummy = DummyTel(data, index=5)
    result = get_telescope_name(dummy)
    assert result == "tel_5"


def dummy_rotate(x, y, angle):
    # Dummy rotate: simply multiply coordinates by 2 for testing purposes.
    return x * 2, y * 2


def test_get_positions_with_position_columns(monkeypatch):
    # Test branch when table has "position_x" and "position_y".
    x = [1, -2, 3] * u.m
    y = [4, 5, -6] * u.m
    tbl = QTable({"position_x": x, "position_y": y})
    # For "position_x" branch, locale_rotate_angle = 90 deg.
    monkeypatch.setattr(transf, "rotate", dummy_rotate)
    rotated_x, rotated_y = get_positions(tbl)
    # Expect dummy_rotate to be used: x*2, y*2.
    for orig, rot in zip(x, rotated_x):
        assert rot == orig * 2
    for orig, rot in zip(y, rotated_y):
        assert rot == orig * 2


def test_get_positions_with_utm_columns(monkeypatch):
    # Test branch when table has "utm_east" and "utm_north".
    x = [10, 20, 30] * u.m
    y = [40, 50, 60] * u.m
    tbl = QTable({"utm_east": x, "utm_north": y})

    def dummy_rotate_no_call(x_val, y_val, angle):
        return x_val * 3, y_val * 3

    monkeypatch.setattr(transf, "rotate", dummy_rotate_no_call)
    rotated_x, rotated_y = get_positions(tbl)
    # Since locale_rotate_angle is 0, transf.rotate should not be called.
    # Returned values should be unchanged.
    for orig, rot in zip(x, rotated_x):
        assert rot == orig
    for orig, rot in zip(y, rotated_y):
        assert rot == orig


def test_get_positions_missing_columns():
    # Test that a table missing required position columns raises ValueError.
    tbl = QTable({"some_column": [1, 2, 3] * u.m})
    with pytest.raises(ValueError, match=r"Missing required position columns."):
        get_positions(tbl)


def test_get_patches_simplest(monkeypatch):
    telescopes = QTable(
        {
            "position_x": [0, 0] * u.m,  # Two rows to match dummy_x length
            "position_y": [0, 0] * u.m,  # Two rows to match dummy_y length
        }
    )

    # Dummy rotated positions to be injected (length matches table)
    dummy_x = u.Quantity([1, 2], u.m)
    dummy_y = u.Quantity([3, 4], u.m)

    def dummy_get_positions(telescopes):
        return dummy_x, dummy_y

    dummy_patches = ["dummy_patch1", "dummy_patch2"]  # Two patches
    dummy_radii = [2, 3] * u.m  # Two radii

    def dummy_create_patches(telescopes, scale, show_label, ax):
        return dummy_patches, dummy_radii

    monkeypatch.setattr(
        "simtools.visualization.plot_array_layout.get_positions",
        dummy_get_positions,
    )
    monkeypatch.setattr(
        "simtools.visualization.plot_array_layout.create_patches", dummy_create_patches
    )

    _, ax = plt.subplots()

    provided_range = 50
    patches, returned_range = get_patches(ax, telescopes, False, provided_range, 1.0)

    # Verify that get_patches returns the dummy patches and the provided axes_range
    assert patches == dummy_patches
    assert returned_range == provided_range

    # Verify that the rotated position columns were added to the table
    assert "pos_x_rotated" in telescopes.colnames
    assert "pos_y_rotated" in telescopes.colnames

    patches, returned_range = get_patches(ax, telescopes, False, None, 1.0)
    assert returned_range == pytest.approx(7.7)


def test_get_telescope_patch_circle(monkeypatch):
    def dummy_get_type(name):
        return "MSTN"

    monkeypatch.setattr(
        "simtools.visualization.plot_array_layout.names.get_array_element_type_from_name",
        dummy_get_type,
    )

    x = 15 * u.m
    y = 25 * u.m
    radius = 3 * u.m

    patch = get_telescope_patch("dummy", x, y, radius)
    assert isinstance(patch, mpatches.Circle)

    expected_center = (x.to(u.m).value, y.to(u.m).value)
    assert patch.center == expected_center
    assert patch.radius == radius.to(u.m).value

    assert patch.get_fill() is True
    assert to_rgba("dodgerblue") == patch.get_facecolor()


def test_get_telescope_patch_rectangle(monkeypatch):
    def dummy_get_type(name):
        return "SCTS"

    monkeypatch.setattr(
        "simtools.visualization.plot_array_layout.names.get_array_element_type_from_name",
        dummy_get_type,
    )

    x = 10 * u.m
    y = 20 * u.m
    radius = 2 * u.m

    patch = get_telescope_patch("dummy", x, y, radius)
    assert isinstance(patch, mpatches.Rectangle)

    expected_xy = ((x - radius / 2).value, (y - radius / 2).value)
    assert patch.get_xy() == expected_xy
    assert patch.get_width() == radius.to(u.m).value
    assert patch.get_height() == radius.to(u.m).value

    assert patch.get_fill() is False
    assert patch.get_edgecolor() == to_rgba("black")


def test_plot_array_layout_calls_helpers(monkeypatch):
    calls = {"get_patches_count": 0, "update_legend_called": False, "finalize_plot_called": False}

    def dummy_get_patches(ax, telescopes, show_tel_label, axes_range, marker_scaling):
        calls["get_patches_count"] += 1
        # Create real patch objects instead of strings
        dummy_patch = mpatches.Circle((0, 0), 1)  # Simple circle patch
        dummy_bg_patch = mpatches.Circle((0, 0), 1)  # Another circle patch
        # For first call (primary telescopes)
        if calls["get_patches_count"] == 1:
            return ([dummy_patch], 50)
        # For second call (background telescopes)
        return ([dummy_bg_patch], 60)

    def dummy_update_legend(ax, telescopes):
        calls["update_legend_called"] = True

    def dummy_finalize_plot(ax, patches, x_title, y_title, axes_range):
        calls["finalize_plot_called"] = True

    monkeypatch.setattr("simtools.visualization.plot_array_layout.get_patches", dummy_get_patches)
    monkeypatch.setattr(
        "simtools.visualization.plot_array_layout.update_legend", dummy_update_legend
    )
    monkeypatch.setattr(
        "simtools.visualization.plot_array_layout.finalize_plot", dummy_finalize_plot
    )

    # Create minimal dummy tables for telescopes and background_telescopes
    dummy_telescopes = Table()
    dummy_background = Table()

    fig = plot_array_layout(
        dummy_telescopes,
        show_tel_label=True,
        axes_range=100,
        marker_scaling=2.0,
        background_telescopes=dummy_background,
    )

    assert calls["get_patches_count"] == 2
    assert calls["update_legend_called"]
    assert calls["finalize_plot_called"]
    assert isinstance(fig, mpl_fig.Figure)

    # Monkey patch max() to count its calls
    call_count = {"count": 0}
    original_max = builtins.max

    def dummy_max(*args, **kwargs):
        call_count["count"] += 1
        return original_max(*args, **kwargs)

    monkeypatch.setattr(builtins, "max", dummy_max)

    _ = plot_array_layout(
        dummy_telescopes,
        show_tel_label=True,
        axes_range=None,
        marker_scaling=2.0,
        background_telescopes=dummy_background,
    )

    assert call_count["count"] > 0


def test_create_patches(telescopes):
    _, ax = plt.subplots()
    patches, radii = create_patches(telescopes, 1.0, True, ax)

    assert len(patches)
    assert len(radii)
