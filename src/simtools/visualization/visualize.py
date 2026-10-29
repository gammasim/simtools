#!/usr/bin/python3
"""Module for visualization."""

import logging
import re
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path

import astropy.units as u
from astropy.table import QTable
from cycler import cycler
from PIL import Image

from simtools.settings import config
from simtools.utils import value_conversion
from simtools.visualization.matplotlib_backend import lazy_module
from simtools.visualization.matplotlib_backend import pyplot as plt

gridspec = lazy_module("matplotlib.gridspec")
pdf_backend = lazy_module("matplotlib.backends.backend_pdf")

_FIGURE_SUFFIXES = {".eps", ".jpeg", ".jpg", ".pdf", ".png", ".ps", ".svg"}
_PLOT_KWARGS = set(
    "color linestyle linewidth marker markersize markerfacecolor markeredgecolor "
    "markeredgewidth alpha label zorder dashes gapcolor solid_capstyle solid_joinstyle "
    "dash_capstyle dash_joinstyle antialiased".split()
)


def _palette(hex_codes):
    """Return a list of '#'-prefixed hex colours from a space-separated string."""
    return [f"#{code}" for code in hex_codes.split()]


_CLASSIC = "ba2c54 5B90DC FFAB44 0C9FB3 57271B 3B507D 794D88 FD6989 8A978E 3B507D D8153C cc9214"
_MODIFIED = "D6088F 424D9C 178084 AF99DA F58D46 634B5B 0C9FB3 7C438A 328cd6 8D0F25 8A978E ffcb3d"
_AUTUMN = "A9434D 4E615D 3C8DAB A4657A 424D9C DC575A 1D2D38 634B5B 56276D 577580 134663 196096"
_PURPLES = "a57bb7 343D80 EA60BF B7308E E099C3 7C438A AF99DA 4D428E 56276D CC4B93 DC4E76 5C4AE4"
_GREENS = "268F92 abc14d 8A978E 0C9FB3 BDA962 B0CB9E 769168 5E93A5 178084 B7BBAD 163317 76A63F"
COLORS = {
    "classic": _palette(_CLASSIC),
    "modified classic": _palette(_MODIFIED),
    "autumn": _palette(_AUTUMN),
    "purples": _palette(_PURPLES),
    "greens": _palette(_GREENS),
}
COLORS["default"] = COLORS["classic"]

MARKERS = ["o", "s", "v", "^", "*", "P", "d", "X", "p", "<", ">", "h"]
LINES = [
    (0, ()),  # solid
    (0, (1, 1)),  # densely dotted
    (0, (3, 1, 1, 1)),  # densely dashdotted
    (0, (5, 5)),  # dashed
    (0, (3, 1, 1, 1, 1, 1)),  # densely dashdotdotted
    (0, (5, 1)),  # densely dashed
    (0, (1, 5)),  # dotted
    (0, (3, 5, 1, 5)),  # dashdotted
    (0, (3, 5, 1, 5, 1, 5)),  # dashdotdotted
    (0, (5, 10)),  # loosely dashed
    (0, (1, 10)),  # loosely dotted
    (0, (3, 10, 1, 10)),  # loosely dashdotted
]


def _add_unit(title, array):
    """Return ``title`` with the unit of ``array`` added as ``[unit]`` if missing."""
    if title and "[" in title and "]" in title:
        return title

    if not isinstance(array, u.Quantity) or not str(array[0].unit):
        return title

    unit_str = f" [{array[0].unit}]"
    if re.search(r"\d", unit_str):
        unit_str = re.sub(r"(\d)", r"^\1", unit_str)
        unit_str = unit_str.replace("[", r"[$").replace("]", r"$]")
    return f"{title}{unit_str}"


def set_style(palette="default", big_plot=False):
    """
    Set the plotting style (colour palette, font and marker sizes) for following plots.

    Parameters
    ----------
    palette: str
        Colour palette: classic (default), modified classic, autumn, purples or greens.
    big_plot: bool
        If True, fonts and markers are made bigger.

    Raises
    ------
    KeyError
        if provided palette does not exist.
    """
    if palette not in COLORS:
        raise KeyError(f"palette must be one of {', '.join(COLORS)}")

    fontsize = 30 if big_plot else 17
    prop_cycle = cycler(color=COLORS[palette]) + cycler(linestyle=LINES) + cycler(marker=MARKERS)
    plt.rc("lines", linewidth=2, markersize=18 if big_plot else 6)
    plt.rc(
        "axes",
        prop_cycle=prop_cycle,
        titlesize=fontsize,
        labelsize=fontsize,
        labelpad=5,
        grid=True,
        axisbelow=True,
    )
    plt.rc("xtick", labelsize=fontsize)
    plt.rc("ytick", labelsize=fontsize)
    plt.rc("legend", loc="best", shadow=False, fontsize="medium")
    plt.rc("font", family="serif", size=fontsize)


def plot_1d(data, **kwargs):
    """
    Produce a high contrast one dimensional plot from multiple data sets.

    Parameters
    ----------
    data: numpy structured array or dict of structured arrays
        Columns: x, y and optionally errors. Dict keys are the legend labels; the first entry
        is the reference for ratio or difference plots.
    **kwargs:
        Plot options (palette, big_plot, title, no_legend, legend_fontsize, no_markers,
        empty_markers, plot_ratio, plot_difference, xscale, yscale, xlim, ylim, xtitle,
        ytitle, error_type, error_label) and any kwargs accepted by plt.plot.

    Returns
    -------
    pyplot.figure
        Figure in which the plot was produced.

    Raises
    ------
    ValueError
        if a ratio or difference is requested with one data set.
    """
    _apply_marker_options(kwargs)

    if not isinstance(data, dict):
        data = {"_default": data}
    plot_ratio = kwargs.get("plot_ratio", False)
    ratio = plot_ratio or kwargs.get("plot_difference", False)
    if ratio and len(data) < 2:
        raise ValueError("Asked to plot a ratio or difference with just one set of data")

    set_style(kwargs.get("palette", "default"), kwargs.get("big_plot", False))
    plot_args = {k: v for k, v in kwargs.items() if k in _PLOT_KWARGS}

    fig, gs = _create_figure(ratio)
    plt.subplot(gs[0])
    ax1 = plt.gca()

    for label, data_now in data.items():
        x_col, y_col, x_err_col, y_err_col = _get_data_columns(data_now)
        x_title_unit = _add_unit(kwargs.get("xtitle") or x_col, data_now[x_col])
        y_title_unit = _add_unit(kwargs.get("ytitle") or y_col, data_now[y_col])
        (line,) = plt.plot(data_now[x_col], data_now[y_col], label=label, **plot_args)
        _plot_error_plots(kwargs, data_now, x_col, y_col, x_err_col, y_err_col, line.get_color())

    _decorate_plot(data, kwargs, ratio, x_title_unit, y_title_unit)

    if ratio:
        plot_ratio_difference(ax1, data, plot_ratio, gs, plot_args)
    else:
        plt.tight_layout()
    return fig


def _apply_marker_options(kwargs):
    """Translate the ``no_markers`` and ``empty_markers`` options into plot keywords."""
    if kwargs.get("no_markers"):
        kwargs["marker"] = "None"
        kwargs["linewidth"] = 4
    if kwargs.get("empty_markers"):
        kwargs["markerfacecolor"] = "None"


def _create_figure(ratio):
    """Return the figure and grid spec, with a ratio panel below the main panel if requested."""
    if ratio:
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        fig = plt.figure(figsize=(8, 8))
        gs.update(hspace=0.02)
    else:
        gs = gridspec.GridSpec(1, 1)
        fig = plt.figure(figsize=(8, 6))
    return fig, gs


def _decorate_plot(data, kwargs, ratio, x_title_unit, y_title_unit):
    """Set scales, limits, labels, title and legend of the main plot panel."""
    plt.xscale(kwargs.get("xscale", "linear"))
    plt.yscale(kwargs.get("yscale", "linear"))
    plt.xlim(kwargs.get("xlim", (None, None)))
    plt.ylim(kwargs.get("ylim", (None, None)))
    plt.ylabel(y_title_unit)
    if not ratio:
        plt.xlabel(x_title_unit)
    if kwargs.get("title"):
        plt.title(kwargs["title"], y=1.02)
    if "_default" not in data and not kwargs.get("no_legend"):
        plt.legend(fontsize=kwargs.get("legend_fontsize", "xx-small"))


def _plot_error_plots(kwargs, data_now, x_col, y_col, x_err_col, y_err_col, color):
    """Plot error bands or error bars for one data set."""
    if kwargs.get("error_type") == "fill_between" and y_err_col:
        plt.fill_between(
            data_now[x_col],
            data_now[y_col] - data_now[y_err_col],
            data_now[y_col] + data_now[y_err_col],
            alpha=0.2,
            color=color,
        )
    if kwargs.get("error_type") == "errorbar" and (x_err_col or y_err_col):
        plt.errorbar(
            data_now[x_col],
            data_now[y_col],
            xerr=data_now[x_err_col] if x_err_col else None,
            yerr=data_now[y_err_col] if y_err_col else None,
            fmt=".",
            color=color,
            capsize=4,
            elinewidth=1.5,
            capthick=1.5,
            label=kwargs.get("error_label"),
        )


def _get_data_columns(data_now):
    """Return x, y, x-error and y-error column names (errors may be None)."""
    columns = data_now.dtype.names
    assert len(columns) >= 2, "Input array must have at least two columns with titles."
    x_err_col = columns[2] if len(columns) == 4 else None
    y_err_col = columns[-1] if len(columns) in (3, 4) else None
    return columns[0], columns[1], x_err_col, y_err_col


def plot_ratio_difference(ax1, data_dict, plot_ratio, gs, plot_args):
    """Plot the ratio or difference plot in the lower panel."""
    plt.subplot(gs[1], sharex=ax1)
    plt.plot([], [])  # Advance cycler for consistent colors/styles

    data_ref_name = next(iter(data_dict))
    for label, data_now in data_dict.items():
        if label == data_ref_name:
            continue
        x_title, y_title = data_now.dtype.names[0], data_now.dtype.names[1]
        x_title_unit = _add_unit(x_title, data_now[x_title])

        if plot_ratio:
            y_values = data_now[y_title] / data_dict[data_ref_name][y_title]
            y_title_ratio = f"Ratio to {data_ref_name}"
        else:
            y_values = data_now[y_title] - data_dict[data_ref_name][y_title]
            y_title_ratio = f"Difference to {data_ref_name}"
        plt.plot(data_now[x_title], y_values, **plot_args)

    plt.xlabel(x_title_unit)
    if len(y_title_ratio) > 20 and plot_ratio:
        y_title_ratio = "Ratio"
    plt.ylabel(y_title_ratio)
    ylim = plt.gca().get_ylim()
    nbins = min(int((ylim[1] - ylim[0]) / 0.05 + 1), 6)
    plt.locator_params(axis="y", nbins=nbins)


def plot_table(table, y_title, **kwargs):
    """
    Produce a high contrast one dimensional plot from an astropy.Table.

    The first column is the x-axis; each further column is one data set.

    Parameters
    ----------
    table: astropy.Table or astropy.QTable
        Table with at least two columns.
    y_title: str
        The y-axis title.
    **kwargs:
        Options as for :func:`plot_1d`.

    Returns
    -------
    pyplot.fig
        Figure instance.

    Raises
    ------
    ValueError
        if table has less than two columns.
    """
    if len(table.keys()) < 2:
        raise ValueError("Table has to have at least two columns")

    x_axis = table.keys()[0]
    data_dict = {
        column: QTable([table[x_axis], table[column]], names=[x_axis, y_title])
        for column in table.keys()[1:]
    }
    kwargs["xtitle"] = x_axis
    kwargs["ytitle"] = y_title
    return plot_1d(data_dict, **kwargs)


def plot_hist_2d(data, ax=None, **kwargs):
    """
    Produce a two dimensional histogram plot.

    Parameters
    ----------
    data: numpy structured array
        Columns are used as the x-axis and y-axis titles.
    ax: matplotlib.axes.Axes, optional
        Axes to draw on. If not provided, a new figure is created.
    **kwargs:
        title, cmap and any kwargs accepted by plt.hist2d.

    Returns
    -------
    pyplot.figure
        Figure in which the plot was produced.
    """
    cmap = kwargs.pop("cmap", plt.cm.gist_heat_r)
    title = kwargs.pop("title", "")

    created_figure = ax is None
    if created_figure:
        # Set default style since the usual options do not affect 2d plots (for now).
        set_style()
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(gridspec.GridSpec(1, 1)[0])
    else:
        fig = ax.figure

    assert len(data.dtype.names) == 2, "Input array must have two columns with titles."
    x_title, y_title = data.dtype.names[0], data.dtype.names[1]
    ax.hist2d(data[x_title], data[y_title], cmap=cmap, **kwargs)
    ax.set_xlabel(_add_unit(x_title, data[x_title]))
    ax.set_ylabel(_add_unit(y_title, data[y_title]))
    ax.set_aspect("equal", adjustable="box")
    if len(title) > 0:
        ax.set_title(title, y=1.02)
    if created_figure:
        fig.tight_layout()
    return fig


def plot_histogram(data, ax=None, **kwargs):
    """
    Plot a one-dimensional histogram on the provided or current axes.

    Parameters
    ----------
    data : array-like
        Values to plot.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If not provided, the current axes are used.
    **kwargs
        Keyword arguments passed to ``Axes.hist``.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the histogram.
    """
    if ax is None:
        ax = plt.gca()
    ax.hist(data, **kwargs)
    return ax.figure


def save_figure(
    fig, output_file, figure_format=None, log_title="", dpi="figure", close=False, pil_kwargs=None
):
    """
    Save figure to output file(s).

    A figure suffix in ``output_file`` is replaced by each format; other dots are kept.
    Formats default to the configured ``figure_format`` or PNG. PNG files are converted to a
    256-color palette if ``figure_png_palette`` is configured.

    Parameters
    ----------
    fig: plt.figure
        Figure to save.
    output_file: Path, str
        Output path, with or without a format suffix.
    figure_format: list
        List of formats to save the figure.
    log_title: str
        Title of the figure to be added to the log message.
    dpi : str or int, optional
        DPI for ``fig.savefig``; a configured ``figure_dpi`` overrides it for PNG.
    close : bool, optional
        Close the figure after saving. Defaults to False.
    pil_kwargs : dict, optional
        Keyword arguments passed to Pillow when saving PNG files.
    """
    configured_dpi = config.args.get("figure_dpi")
    png_palette = config.args.get("figure_png_palette", False)

    figure_format = figure_format or config.args.get("figure_format") or ["png"]
    base = Path(output_file)
    if base.suffix.lower() in _FIGURE_SUFFIXES:
        base = base.with_suffix("")

    for fmt in value_conversion.ensure_list(figure_format):
        _file = base.with_name(f"{base.name}.{fmt}")
        save_dpi = configured_dpi if fmt == "png" and configured_dpi is not None else dpi
        save_kwargs = {"format": fmt, "bbox_inches": "tight", "dpi": save_dpi}
        if fmt == "png" and pil_kwargs is not None:
            save_kwargs["pil_kwargs"] = pil_kwargs
        fig.savefig(_file, **save_kwargs)
        if fmt == "png" and png_palette:
            convert_png_to_palette(_file)
        logging.info(f"Saved plot {log_title} to {_file}")

    fig.clf()
    if close:
        plt.close(fig)


class _PdfReport:
    """Write PDF pages and close each figure after it is appended."""

    def __init__(self, path):
        path = Path(path).absolute()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._document = pdf_backend.PdfPages(path.as_posix())
        self.page_labels = []

    def add(self, fig):
        """Append a figure as a page and close it."""
        try:
            fig.tight_layout()
            self._document.savefig(fig)
            self.page_labels.append(fig.get_label())
        finally:
            plt.close(fig)

    def __len__(self):
        return len(self.page_labels)

    def close(self):
        """Close the PDF document."""
        self._document.close()


@contextmanager
def pdf_report(path):
    """
    Write figures to a multi-page PDF, closing each figure after it is added.

    Parameters
    ----------
    path : pathlib.Path or str
        Destination PDF path.

    Yields
    ------
    _PdfReport
        Report with an ``add(fig)`` method.
    """
    report = _PdfReport(path)
    try:
        yield report
    finally:
        report.close()


def convert_png_to_palette(png_file):
    """
    Convert a PNG file in place to a 256-color palette image if that makes it smaller.

    Parameters
    ----------
    png_file : Path or str
        PNG file to convert.
    """
    png_file = Path(png_file)
    with Image.open(png_file) as image:
        if image.mode == "RGBA":
            palette_image = image.quantize(colors=256, method=Image.Quantize.FASTOCTREE)
        else:
            palette_image = image.convert("RGB").quantize(colors=256)
    buffer = BytesIO()
    palette_image.save(buffer, format="PNG", optimize=True)
    if buffer.tell() < png_file.stat().st_size:
        png_file.write_bytes(buffer.getvalue())
