"""Cache identities for rendered array layouts."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from simtools.visualization import visualize


def array_layout_figure_name(layout, args_dict, background_layout=None):
    """Return the content-derived filename stem for an array layout figure.

    Layout names and sites are intentionally omitted because they are not drawn in the figure.
    Rendered data, display options, renderer source, and output settings determine the identity.

    Parameters
    ----------
    layout : dict
        Layout data containing ``array_elements`` and optional ``site`` fields.
    args_dict : dict
        Plot options used to render the layout.
    background_layout : astropy.table.Table or None, optional
        Optional telescope data drawn behind the layout.

    Returns
    -------
    str
        Filename stem shared by figures with identical rendering inputs.
    """
    plot_config = {
        key: args_dict.get(key)
        for key in (
            "show_labels",
            "axes_range",
            "marker_scaling",
            "grayed_out_array_elements",
            "highlighted_array_elements",
            "legend_location",
            "bounds",
            "padding",
            "x_lim",
            "y_lim",
            "coordinate_system",
        )
    }
    output_settings = visualize.figure_output_settings().copy()
    output_settings.update(
        {
            key: args_dict[key]
            for key in ("figure_format", "figure_dpi", "figure_png_palette")
            if args_dict.get(key) is not None
        }
    )
    figure_context = SimpleNamespace(args=output_settings)
    cache_key = visualize.figure_cache_key(
        renderer="plot_array_layout.plot_array_layout",
        renderer_modules={
            "plot_array_layout": Path(__file__).with_name("plot_array_layout.py"),
            "array_layout_cache": __file__,
        },
        plot_config=plot_config,
        rendered_data={
            "layout": _array_layout_data(layout["array_elements"]),
            "background": (
                _array_layout_data(background_layout) if background_layout is not None else None
            ),
        },
        context=figure_context,
    )
    return f"array_layout_{cache_key}"


def _array_layout_data(table):
    """Convert a telescope table to canonical renderer input data."""
    return {
        name: {
            "unit": str(getattr(table[name], "unit", None) or ""),
            "values": np.asarray(table[name]).tolist(),
        }
        for name in table.colnames
    }
