#!/usr/bin/python3
"""Plot incident angle histograms for focal, primary, and secondary mirrors.

Plots the primary-mirror hit radius if available.
"""

import logging
from functools import partial
from pathlib import Path

import astropy.units as u
import numpy as np

from simtools.visualization.matplotlib_backend import pyplot as plt
from simtools.visualization.visualize import save_figure

Y_AXIS_BIN_COUNT_LABEL = "Density"
FOCAL_COLUMN = "angle_incidence_focal"
STACKED_ZORDERS = (3, 1, 4)


def _tables(results_by_offset, columns):
    """Yield (off, table) in sorted offset order for non-empty tables with all columns."""
    for off in sorted(results_by_offset):
        tab = results_by_offset[off]
        if tab is not None and len(tab) > 0 and all(c in tab.colnames for c in columns):
            yield off, tab


def _finite(*arrays):
    """Return the arrays restricted to rows where all of them are finite."""
    mask = np.all([np.isfinite(arr) for arr in arrays], axis=0)
    return [arr[mask] for arr in arrays]


def _gather_arrays(results_by_offset, column, unit, log):
    """Return the column of each usable table converted to unit."""
    if column == FOCAL_COLUMN:
        for off, tab in results_by_offset.items():
            if tab is None or len(tab) == 0:
                log.warning(f"Empty results for off-axis={off}")
    return [tab[column].to(unit).value for _, tab in _tables(results_by_offset, [column])]


def _compute_bins(arrays, bin_width, log, name):
    """Return histogram bin edges covering all finite values, or None if there are none."""
    if not arrays:
        return None
    all_vals = np.concatenate(arrays)
    finite_vals = all_vals[np.isfinite(all_vals)]
    if finite_vals.size == 0:
        log.warning("No %s values to plot for this telescope type", name)
        return None
    vmin = float(np.floor(finite_vals.min() / bin_width) * bin_width)
    vmax = float(np.ceil(finite_vals.max() / bin_width) * bin_width)
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        log.warning("Invalid bin edges for %s: vmin=%s vmax=%s", name, vmin, vmax)
        return None
    if vmax <= vmin:
        vmax = vmin + bin_width
    return np.arange(vmin, vmax + bin_width * 0.5, bin_width)


def _save(fig, out_path):
    """Tight-layout, save at 300 dpi and close the figure."""
    plt.tight_layout()
    save_figure(fig, out_path.with_suffix(""), figure_format=[out_path.suffix[1:]], dpi=300)
    plt.close(fig)


def _plot_overlay(
    results_by_offset,
    column,
    unit,
    bins,
    title,
    xlabel,
    out_path,
    zorders=None,
    model_version=None,
):
    """Plot one stacked density histogram per off-axis angle, overlaid on one axis."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    z1, z2, z3 = zorders or (None, None, None)
    for off, tab in _tables(results_by_offset, [column]):
        (data,) = _finite(tab[column].to(unit).value)
        if data.size == 0:
            continue
        hist = partial(ax.hist, data, bins=bins, density=True, stacked=True)
        _, _, patches = hist(
            histtype="step", linewidth=0.5, label=f"off-axis {off:g} deg", zorder=z1
        )
        color = patches[0].get_edgecolor() if patches else None
        hist(
            histtype="stepfilled",
            alpha=0.15,
            color=color,
            edgecolor="none",
            label="_nolegend_",
            zorder=z2,
        )
        hist(histtype="step", linewidth=0.5, color=color, label="_nolegend_", zorder=z3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(Y_AXIS_BIN_COUNT_LABEL)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    if model_version:
        ax.text(
            0.03,
            0.97,
            f"Model version: {model_version}",
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            ha="left",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.5},
        )
    _save(fig, out_path)


def _plot_angles(
    results_by_offset, column, title, xlabel, out_path, bin_width_deg, log, **overlay_kwargs
):
    """Plot the incidence-angle overlay histogram for one column."""
    name = "focal-surface incidence angle" if column == FOCAL_COLUMN else column
    bins = _compute_bins(
        _gather_arrays(results_by_offset, column, u.deg, log), bin_width_deg, log, name
    )
    if bins is not None:
        _plot_overlay(
            results_by_offset, column, u.deg, bins, title, xlabel, out_path, **overlay_kwargs
        )


def _plot_radius_histograms(
    results_by_offset, radius_col, title, xlabel, out_path, bin_width_m, log
):
    """Plot the stacked density histogram of a hit radius per off-axis angle."""
    bins = _compute_bins(
        _gather_arrays(results_by_offset, radius_col, u.m, log),
        bin_width_m,
        log,
        f"{radius_col}_m",
    )
    if bins is not None:
        _plot_overlay(
            results_by_offset,
            radius_col,
            u.m,
            bins,
            title,
            xlabel,
            out_path,
            zorders=STACKED_ZORDERS,
        )


def _plot_radius_vs_angle(results_by_offset, radius_col, angle_col, title, out_path, log):
    """Scatter hit radius against incidence angle for all off-axis angles."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    any_points = False
    for off, tab in _tables(results_by_offset, [radius_col, angle_col]):
        r, a = _finite(tab[radius_col].to(u.m).value, tab[angle_col].to(u.deg).value)
        if r.size == 0:
            continue
        any_points = True
        ax.scatter(r, a, s=4, alpha=0.25, label=f"off-axis {off:g} deg")
    if not any_points:
        plt.close(fig)
        log.warning("No valid data to plot for %s", title)
        return
    ax.set_xlabel("Hit radius (m)")
    ax.set_ylabel("Angle of incidence (deg)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(markerscale=3)
    _save(fig, out_path)


def _plot_xy_heatmaps_per_offset(
    results_by_offset, x_col, y_col, title_prefix, file_stem, out_dir, label, bins=400
):
    """Plot one X-Y hit-position heatmap per off-axis angle."""
    for off, tab in _tables(results_by_offset, [x_col, y_col]):
        x, y = _finite(tab[x_col].to(u.m).value, tab[y_col].to(u.m).value)
        if x.size == 0:
            continue
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        h = ax.hist2d(x, y, bins=bins, cmap="viridis", norm=None)
        ax.set_xlabel("X hit (m)")
        ax.set_ylabel("Y hit (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{title_prefix} (off-axis {off:g} deg)")
        plt.colorbar(h[3], ax=ax).set_label("Counts per bin")
        _save(fig, out_dir / f"{file_stem}{off:g}_{label}.png")


def _plot_debug_plots(results_by_offset, out_dir, label, radius_bin_width_m, log):
    """Plot hit radius, radius-vs-angle and X-Y hit diagnostics for primary and secondary."""
    for mirror, mirror_number in (("primary", "M1"), ("secondary", "M2")):
        cap = mirror.capitalize()
        _plot_radius_histograms(
            results_by_offset,
            f"{mirror}_hit_radius",
            f"{cap} mirror hit radius vs off-axis angle",
            f"{cap}-hit radius on {mirror_number} (m)",
            out_dir / f"incident_radius_{mirror}_multi_{label}.png",
            radius_bin_width_m,
            log,
        )
        _plot_radius_vs_angle(
            results_by_offset,
            f"{mirror}_hit_radius",
            f"angle_incidence_{mirror}",
            f"{cap} mirror: hit radius vs incidence angle",
            out_dir / f"incident_{mirror}_radius_vs_angle_multi_{label}.png",
            log,
        )
        _plot_xy_heatmaps_per_offset(
            results_by_offset,
            f"{mirror}_hit_x",
            f"{mirror}_hit_y",
            f"{cap} mirror: X-Y hit distribution",
            f"incident_{mirror}_xy_heatmap_off",
            out_dir,
            label,
        )


def plot_incident_angles(
    results_by_offset,
    output_dir,
    label,
    bin_width_deg=0.1,
    radius_bin_width_m=0.01,
    debug_plots=False,
    logger=None,
    model_version=None,
):
    """Plot overlaid histograms of focal, primary, secondary angles, and primary hit radius.

    Parameters
    ----------
    results_by_offset : dict
        Mapping from off-axis angle to result tables containing angle and radius columns.
    output_dir : path-like
        Base output directory where the ``plots`` subdirectory will be created.
    label : str
        Label used to distinguish this set of plots in the output filenames.
    bin_width_deg : float, optional
        Bin width in degrees for the angle-of-incidence histograms.
    radius_bin_width_m : float, optional
        Bin width in meters for the primary mirror hit-radius histograms.
    debug_plots : bool, optional
        If True, generate additional diagnostic plots.
    logger : logging.Logger or None, optional
        Logger instance to use for messages. If None, a module-level logger is used.
    model_version : str or None, optional
        Semantic model version identifier to annotate the generated plots. If None,
        no model version text is added to the figures.
    """
    log = logger or logging.getLogger(__name__)
    if not results_by_offset:
        log.warning("No results provided for multi-offset plot")
        return

    out_dir = Path(output_dir) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    _plot_angles(
        results_by_offset,
        FOCAL_COLUMN,
        "Incident angle distribution vs off-axis angle",
        "Angle of incidence at focal surface (deg) w.r.t. optical axis",
        out_dir / f"incident_angles_multi_{label}.png",
        bin_width_deg,
        log,
        zorders=STACKED_ZORDERS,
        model_version=model_version,
    )
    for mirror in ("primary", "secondary"):
        _plot_angles(
            results_by_offset,
            f"angle_incidence_{mirror}",
            f"Incident angle on {mirror} mirror (w.r.t. normal) vs off-axis angle",
            "Angle of incidence (deg)",
            out_dir / f"incident_angles_{mirror}_multi_{label}.png",
            bin_width_deg,
            log,
            model_version=model_version,
        )

    if debug_plots:
        _plot_debug_plots(results_by_offset, out_dir, label, radius_bin_width_m, log)
