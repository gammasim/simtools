import astropy.units as u
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from astropy.table import Table
from matplotlib.collections import PatchCollection

from simtools.visualization.plot_array_layout import (
    finalize_plot,
    get_sphere_radius,
    get_telescope_name,
    update_legend,
)


def test_finalize_plot_with_range():
    fig, ax = plt.subplots()
    # Create a dummy patch for testing
    dummy_patch = mpatches.Circle((0, 0), 1)
    x_title = "Easting [m]"
    y_title = "Northing [m]"
    axes_range = 10

    finalize_plot(ax, [dummy_patch], x_title, y_title, axes_range)

    # Verify axis labels
    assert ax.get_xlabel() == x_title
    assert ax.get_ylabel() == y_title

    # Verify axis limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    assert xlim == (-axes_range, axes_range)
    assert ylim == (-axes_range, axes_range)

    # Verify that a PatchCollection was added
    collections = [coll for coll in ax.collections if isinstance(coll, PatchCollection)]
    assert collections, "Expected a PatchCollection to be added to the axes."

    # Optionally verify tick label size if any tick labels exist
    xticklabels = ax.get_xticklabels()
    if xticklabels:
        for tick in xticklabels:
            assert tick.get_fontsize() == 8


def test_finalize_plot_without_range():
    fig, ax = plt.subplots()
    dummy_patch = mpatches.Circle((0, 0), 1)
    x_title = "Easting [m]"
    y_title = "Northing [m]"

    # Save default limits before calling finalize_plot

    finalize_plot(ax, [dummy_patch], x_title, y_title, None)

    # Verify axis labels
    assert ax.get_xlabel() == x_title
    assert ax.get_ylabel() == y_title

    # When axes_range is None, limits are not reset explicitly
    # Check that the updated limits differ from what would be set if a range were provided
    new_xlim = ax.get_xlim()
    new_ylim = ax.get_ylim()
    assert new_xlim != (-10, 10) or new_ylim != (-10, 10)

    # Verify that a PatchCollection was added
    collections = [coll for coll in ax.collections if isinstance(coll, PatchCollection)]
    assert collections, "Expected a PatchCollection to be added to the axes."


def test_update_legend_with_telescopes():
    """Test update_legend with telescopes."""
    # Create a proper astropy Table instead of custom objects
    telescopes = Table(
        {
            "telescope_name": ["LSTN-01", "MSTN-01", "LSTN-02"],
            "position_x": [0, 100, 200] * u.m,
            "position_y": [0, 100, 200] * u.m,
            "sphere_radius": [12, 8, 12] * u.m,
        }
    )

    fig, ax = plt.subplots()
    update_legend(ax, telescopes)

    # Check that legend was created
    legend = ax.get_legend()
    assert legend is not None

    # Get legend labels
    labels = [txt.get_text() for txt in legend.get_texts()]

    # Should have counts for telescope types present in the data
    # LSTN appears twice, MSTN appears once
    assert any("LSTN" in label and "(2)" in label for label in labels)
    assert any("MSTN" in label and "(1)" in label for label in labels)


def test_update_legend_without_telescopes():
    """Test update_legend with empty telescope list."""

    # Create empty table
    empty_telescopes = Table()

    fig, ax = plt.subplots()
    update_legend(ax, empty_telescopes)

    # Legend should either not exist or have no labels
    legend = ax.get_legend()
    if legend:
        texts = [txt.get_text() for txt in legend.get_texts()]
        assert len(texts) == 0


def test_get_sphere_radius_with_column():
    # Create a table with 'sphere_radius' column
    tbl = Table(
        {
            "telescope_name": ["TEL-01"],
            "sphere_radius": [5.0] * u.m,  # This creates a column with units
        }
    )
    row = tbl[0]

    # The function returns the raw numeric value, not a Quantity
    result = get_sphere_radius(row)
    assert result == 5.0  # Function returns numpy float64, not Quantity


def test_get_sphere_radius_without_column():
    # Create a table without a 'sphere_radius' column
    tbl = Table({"telescope_name": ["TEL-01"]})
    row = tbl[0]

    # Since the 'sphere_radius' column is missing, the function should return 1.0 m.
    result = get_sphere_radius(row)
    assert result == 1.0 * u.m


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
    # When neither telescope_name nor asset_code/sequence_number exist, fallback should use tel.index.
    data = {"other_column": "value"}
    dummy = DummyTel(data, index=5)
    result = get_telescope_name(dummy)
    assert result == "tel_5"
