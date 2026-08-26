from unittest.mock import patch

import matplotlib.pyplot as plt
from astropy.table import Column, Table, vstack

from simtools.visualization import plot_corsika_limits


def create_test_table(
    zenith,
    azimuth,
    nsb_level,
    array_name="test_array",
    lower_energy_limit=0.01,
    br_energy_min=None,
    br_core_scatter_max=None,
    br_viewcone_max=None,
):
    """Create a minimal CORSIKA limits table row for plotting tests."""
    columns = [
        Column(data=["gamma"], name="primary_particle"),
        Column(data=[array_name], name="array_name"),
        Column(data=[[1, 2, 3]], name="telescope_ids"),
        Column(data=[zenith], name="zenith"),
        Column(data=[azimuth], name="azimuth"),
        Column(data=[nsb_level], name="nsb_level"),
        Column(data=[lower_energy_limit], name="lower_energy_limit"),
        Column(data=[2000], name="upper_radius_limit"),
        Column(data=[10], name="viewcone_radius"),
    ]

    if br_energy_min is not None:
        columns.append(Column(data=[br_energy_min], name="br_energy_min"))
    if br_core_scatter_max is not None:
        columns.append(Column(data=[br_core_scatter_max], name="br_core_scatter_max"))
    if br_viewcone_max is not None:
        columns.append(Column(data=[br_viewcone_max], name="br_viewcone_max"))

    return Table(columns)


@patch("simtools.visualization.plot_corsika_limits.plt.savefig")
def test_plot_grid_coverage(mock_savefig, tmp_test_directory):
    table = vstack(
        [
            create_test_table(20, 0, "dark", "layout1"),
            create_test_table(40, 0, "dark", "layout1"),
        ]
    )
    grid_definition = {
        "zenith": [20, 40],
        "azimuth": [0],
        "nsb_level": ["dark"],
        "array_name": ["layout1"],
    }

    output_files = plot_corsika_limits.plot_grid_coverage(
        table, grid_definition, tmp_test_directory
    )
    assert len(output_files) == 1
    mock_savefig.assert_called_once()

    _, ax = plt.subplots()
    found_combinations_str = {("20", "0", "dark", "layout1"), ("40", "0", "dark", "layout1")}
    plot_corsika_limits._plot_single_grid_coverage(
        ax,
        [20, 40],
        [0],
        "dark",
        "layout1",
        found_combinations_str,
        "gamma",
    )
    assert "Primary Particle=gamma" in ax.get_title()
    plt.close(ax.figure)

    mock_savefig.reset_mock()
    output_files = plot_corsika_limits.plot_grid_coverage(table, None, tmp_test_directory)
    assert not output_files
    mock_savefig.assert_not_called()


@patch("simtools.visualization.plot_corsika_limits.plt.savefig")
@patch("simtools.visualization.plot_corsika_limits.plt.suptitle")
def test_plot_limits(mock_suptitle, mock_savefig, tmp_test_directory):
    table = vstack(
        [
            create_test_table(20, 0, "dark", "layout1"),
            create_test_table(40, 0, "dark", "layout1"),
            create_test_table(20, 0, "moon", "layout1"),
        ]
    )

    output_files = plot_corsika_limits.plot_limits(table, tmp_test_directory)
    assert len(output_files) == 1
    mock_savefig.assert_called_once()
    suptitle_text = mock_suptitle.call_args.args[0]
    assert "gamma" in suptitle_text


@patch("simtools.visualization.plot_corsika_limits.plt.savefig")
def test_plot_limits_combines_azimuths_with_filled_and_open_markers(
    mock_savefig, tmp_test_directory
):
    table = vstack(
        [
            create_test_table(20, 0, "dark", "layout1"),
            create_test_table(40, 0, "dark", "layout1"),
            create_test_table(20, 180, "dark", "layout1"),
            create_test_table(40, 180, "dark", "layout1"),
        ]
    )

    with patch("simtools.visualization.plot_corsika_limits.plt.close"):
        output_files = plot_corsika_limits.plot_limits(table, tmp_test_directory)

    assert len(output_files) == 1
    assert output_files[0].name == "limits_layout1_gamma.png"
    marker_face_colors = [line.get_markerfacecolor() for line in plt.gcf().axes[0].lines]
    assert any(color == "none" for color in marker_face_colors)
    assert any(color != "none" for color in marker_face_colors)
    assert [label.get_text() for label in plt.gcf().legends[0].get_texts()] == [
        "NSB=dark GHz",
        "Az=0 deg",
        "Az=180 deg",
    ]
    plt.close(plt.gcf())


@patch("simtools.visualization.plot_corsika_limits.plt.savefig")
def test_plot_limits_with_broad_range_overlay(mock_savefig, tmp_test_directory):
    table = vstack(
        [
            create_test_table(
                20,
                0,
                "dark",
                "layout1",
                lower_energy_limit=0.01,
                br_energy_min=0.008,
                br_core_scatter_max=2200,
                br_viewcone_max=11,
            ),
            create_test_table(
                40,
                0,
                "dark",
                "layout1",
                lower_energy_limit=0.02,
                br_energy_min=0.015,
                br_core_scatter_max=2300,
                br_viewcone_max=12,
            ),
        ]
    )

    with patch("simtools.visualization.plot_corsika_limits.plt.close"):
        output_files = plot_corsika_limits.plot_limits(table, tmp_test_directory)

    assert len(output_files) == 1
    assert mock_savefig.called
    fig = plt.gcf()
    axes = fig.axes[:3]
    assert axes[0].get_ylim()[0] <= 0.008
    assert axes[0].get_ylim()[1] >= 0.015
    assert any(
        line.get_linestyle() == "--" and line.get_color() == "gray"
        for axis in axes
        for line in axis.get_lines()
    )
    plt.close(fig)
