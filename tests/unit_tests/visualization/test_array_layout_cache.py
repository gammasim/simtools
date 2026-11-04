from astropy.table import Table

from simtools.visualization.array_layout_cache import array_layout_figure_name


def test_figure_name_tracks_rendered_positions():
    args = {
        "coordinate_system": "ground",
        "site": "North",
        "model_version": "8.0.0",
        "figure_format": ["png"],
    }
    layout = {
        "name": "4lst",
        "site": "North",
        "array_elements": Table({"name": ["LSTN-01"], "pos_x": [0.0], "pos_y": [12.0]}),
    }
    shifted = {
        **layout,
        "array_elements": Table({"name": ["LSTN-01"], "pos_x": [10.0], "pos_y": [12.0]}),
    }

    assert array_layout_figure_name(layout, args) != array_layout_figure_name(shifted, args)


def test_figure_name_reuses_aliases_and_tracks_visible_options():
    args = {
        "coordinate_system": "ground",
        "site": "North",
        "model_version": "8.0.0",
        "figure_format": ["png"],
        "show_labels": False,
        "legend_location": "best",
        "x_lim": None,
        "y_lim": None,
    }
    layout = {
        "name": "4lst",
        "site": "North",
        "array_elements": Table({"name": ["LSTN-01"], "pos_x": [0.0], "pos_y": [12.0]}),
    }
    alias = {**layout, "name": "CTAO-North-Alpha", "site": "South"}

    base_name = array_layout_figure_name(layout, args)

    assert array_layout_figure_name(alias, args) == base_name
    assert array_layout_figure_name(layout, {**args, "show_labels": True}) != base_name
    assert array_layout_figure_name(layout, {**args, "legend_location": "upper left"}) != base_name
    assert array_layout_figure_name(layout, {**args, "x_lim": (-20.0, 20.0)}) != base_name
    assert array_layout_figure_name(layout, {**args, "figure_png_palette": True}) != base_name
