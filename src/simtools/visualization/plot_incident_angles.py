#!/usr/bin/python3
"""Plot incident angle histograms for focal, primary, and secondary mirrors.

Plots the primary-mirror hit radius if available.
"""

import logging
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

__all__ = ["plot_incident_angles"]

Y_AXIS_BIN_COUNT_LABEL = "Density"


def _gather_angle_arrays(results_by_offset, column, log):
    arrays = []
    for off, tab in results_by_offset.items():
        if tab is None or len(tab) == 0:
            if column == "angle_incidence_focal":
                log.warning(f"Empty results for off-axis={off}")
            continue
        if column not in tab.colnames:
            continue
        arrays.append(tab[column].to(u.deg).value)
    return arrays


def _gather_radius_arrays(results_by_offset, column, log):
    arrays = []
    for off, tab in results_by_offset.items():
        if tab is None or len(tab) == 0 or column not in tab.colnames:
            continue
        try:
            arrays.append(tab[column].to(u.m).value)
        except (AttributeError, ValueError, TypeError):
            log.warning("Skipping radius values for off-axis=%s due to unit/format issue", off)
    return arrays


def _plot_radius_vs_angle(
    results_by_offset,
    radius_col,
    angle_col,
    title,
    out_path,
    log,
):
    any_points = False
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for off in sorted(results_by_offset.keys()):
        tab = results_by_offset[off]
        if tab is None or len(tab) == 0:
            continue
        if radius_col not in tab.colnames or angle_col not in tab.colnames:
            continue
        r = tab[radius_col].to(u.m).value
        a = tab[angle_col].to(u.deg).value
        mask = np.isfinite(r) & np.isfinite(a)
        r, a = r[mask], a[mask]
        if r.size == 0 or a.size == 0:
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
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_xy_heatmap(
    results_by_offset,
    x_col,
    y_col,
    title,
    out_path,
    log,
    bins=400,
):
    any_points = False
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    h = None
    for _off, x, y in _iter_xy_valid_points(results_by_offset, x_col, y_col):
        any_points = True
        h = ax.hist2d(x, y, bins=bins, cmap="viridis", norm=None)
    if not any_points:
        plt.close(fig)
        log.warning("No valid data to plot for %s", title)
        return
    ax.set_xlabel("X hit (m)")
    ax.set_ylabel("Y hit (m)")
    ax.set_title(title)
    ax.grid(False)
    cb = plt.colorbar(h[3], ax=ax)
    cb.set_label("Counts per bin")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_xy_heatmaps_per_offset(
    results_by_offset,
    x_col,
    y_col,
    title_prefix,
    file_stem,
    out_dir,
    label,
    bins=400,
):
    for off, x, y in _iter_xy_valid_points(results_by_offset, x_col, y_col):
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        h = ax.hist2d(x, y, bins=bins, cmap="viridis", norm=None)
        ax.set_xlabel("X hit (m)")
        ax.set_ylabel("Y hit (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{title_prefix} (off-axis {off:g} deg)")
        cb = plt.colorbar(h[3], ax=ax)
        cb.set_label("Counts per bin")
        plt.tight_layout()
        out_path = out_dir / f"{file_stem}{off:g}_{label}.png"
        plt.savefig(out_path, dpi=300)
        plt.close(fig)


def _iter_xy_valid_points(results_by_offset, x_col, y_col):
    """Yield (off, x, y) arrays for valid entries with finite X/Y in meters.

    Filters out None/empty tables, missing columns, and non-finite rows.
    Offsets are iterated in sorted order.
    """
    for off in sorted(results_by_offset.keys()):
        tab = results_by_offset[off]
        if tab is None or len(tab) == 0:
            continue
        if x_col not in tab.colnames or y_col not in tab.colnames:
            continue
        x = tab[x_col].to(u.m).value
        y = tab[y_col].to(u.m).value
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if x.size == 0:
            continue
        yield off, x, y


def _compute_bins(all_vals, bin_width, log, context):
    finite_mask = np.isfinite(all_vals)
    if not np.any(finite_mask):
        if context == "focal":
            log.warning("No focal-surface incidence angle values to plot for this telescope type")
        else:
            log.warning("No %s values to plot for this telescope type", context)
        return None
    vals = all_vals[finite_mask]
    vmin = float(np.floor(vals.min() / bin_width) * bin_width)
    vmax = float(np.ceil(vals.max() / bin_width) * bin_width)
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        log.warning("Invalid bin edges for %s: vmin=%s vmax=%s", context, vmin, vmax)
        return None
    if vmax <= vmin:
        vmax = vmin + bin_width
    return np.arange(vmin, vmax + bin_width * 0.5, bin_width)


def _plot_radius_histograms(
    results_by_offset,
    radius_col,
    title,
    xlabel,
    out_path,
    bin_width_m,
    log,
):
    arrays = _gather_radius_arrays(results_by_offset, radius_col, log)
    if not arrays:
        return
    all_vals = np.concatenate(arrays)
    bins_m = _compute_bins(all_vals, bin_width=bin_width_m, log=log, context=f"{radius_col}_m")
    if bins_m is None:
        return
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for off in sorted(results_by_offset.keys()):
        tab = results_by_offset[off]
        if tab is None or len(tab) == 0 or radius_col not in tab.colnames:
            continue
        data = tab[radius_col].to(u.m).value
        data = data[np.isfinite(data)]
        if data.size == 0:
            continue
        _, _, patches = ax.hist(
            data,
            bins=bins_m,
            density=True,
            stacked=True,
            histtype="step",
            linewidth=0.5,
            label=f"off-axis {off:g} deg",
            zorder=3,
        )
        color = patches[0].get_edgecolor() if patches else None
        ax.hist(
            data,
            bins=bins_m,
            density=True,
            stacked=True,
            histtype="stepfilled",
            alpha=0.15,
            color=color,
            edgecolor="none",
            label="_nolegend_",
            zorder=1,
        )
        ax.hist(
            data,
            bins=bins_m,
            density=True,
            stacked=True,
            histtype="step",
            linewidth=0.5,
            color=color,
            label="_nolegend_",
            zorder=4,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(Y_AXIS_BIN_COUNT_LABEL)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_debug_plots(results_by_offset, out_dir, label, radius_bin_width_m, log):
    _plot_radius_histograms(
        results_by_offset,
        radius_col="primary_hit_radius",
        title="Primary mirror hit radius vs off-axis angle",
        xlabel="Primary-hit radius on M1 (m)",
        out_path=out_dir / f"incident_radius_primary_multi_{label}.png",
        bin_width_m=radius_bin_width_m,
        log=log,
    )
    _plot_radius_histograms(
        results_by_offset,
        radius_col="secondary_hit_radius",
        title="Secondary mirror hit radius vs off-axis angle",
        xlabel="Secondary-hit radius on M2 (m)",
        out_path=out_dir / f"incident_radius_secondary_multi_{label}.png",
        bin_width_m=radius_bin_width_m,
        log=log,
    )

    _plot_radius_vs_angle(
        results_by_offset,
        radius_col="primary_hit_radius",
        angle_col="angle_incidence_primary",
        title="Primary mirror: hit radius vs incidence angle",
        out_path=out_dir / f"incident_primary_radius_vs_angle_multi_{label}.png",
        log=log,
    )
    _plot_radius_vs_angle(
        results_by_offset,
        radius_col="secondary_hit_radius",
        angle_col="angle_incidence_secondary",
        title="Secondary mirror: hit radius vs incidence angle",
        out_path=out_dir / f"incident_secondary_radius_vs_angle_multi_{label}.png",
        log=log,
    )

    _plot_xy_heatmaps_per_offset(
        results_by_offset,
        x_col="primary_hit_x",
        y_col="primary_hit_y",
        title_prefix="Primary mirror: X-Y hit distribution",
        file_stem="incident_primary_xy_heatmap_off",
        out_dir=out_dir,
        label=label,
    )
    _plot_xy_heatmaps_per_offset(
        results_by_offset,
        x_col="secondary_hit_x",
        y_col="secondary_hit_y",
        title_prefix="Secondary mirror: X-Y hit distribution",
        file_stem="incident_secondary_xy_heatmap_off",
        out_dir=out_dir,
        label=label,
    )


def _plot_overlay_angles(results_by_offset, column, bins, ax, use_zorder):
    for off in sorted(results_by_offset.keys()):
        tab = results_by_offset[off]
        if tab is None or len(tab) == 0 or column not in tab.colnames:
            continue
        data = tab[column].to(u.deg).value
        data = data[np.isfinite(data)]
        if data.size == 0:
            continue
        z1, z2, z3 = (3, 1, 4) if use_zorder else (None, None, None)
        _, _, patches = ax.hist(
            data,
            bins=bins,
            density=True,
            stacked=True,
            histtype="step",
            linewidth=0.5,
            label=f"off-axis {off:g} deg",
            zorder=z1,
        )
        color = patches[0].get_edgecolor() if patches else None
        ax.hist(
            data,
            bins=bins,
            density=True,
            stacked=True,
            histtype="stepfilled",
            alpha=0.15,
            color=color,
            edgecolor="none",
            label="_nolegend_",
            zorder=z2,
        )
        ax.hist(
            data,
            bins=bins,
            density=True,
            stacked=True,
            histtype="step",
            linewidth=0.5,
            color=color,
            label="_nolegend_",
            zorder=z3,
        )


def _plot_component_angles(
    results_by_offset,
    column,
    title_suffix,
    out_path,
    bin_width_deg,
    log,
):
    arrays = _gather_angle_arrays(results_by_offset, column, log)
    if not arrays:
        return
    bins = _compute_bins(np.concatenate(arrays), bin_width_deg, log, context=column)
    if bins is None:
        return
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    _plot_overlay_angles(results_by_offset, column, bins, ax, use_zorder=False)
    ax.set_xlabel("Angle of incidence (deg)")
    ax.set_ylabel(Y_AXIS_BIN_COUNT_LABEL)
    ax.set_title(f"Incident angle {title_suffix} vs off-axis angle")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_incident_angles(
    results_by_offset,
    output_dir,
    label,
    bin_width_deg=0.1,
    radius_bin_width_m=0.01,
    debug_plots=False,
    logger=None,
):
    """Plot overlaid histograms of focal, primary, secondary angles, and primary hit radius."""
    log = logger or logging.getLogger(__name__)
    if not results_by_offset:
        log.warning("No results provided for multi-offset plot")
        return

    out_dir = Path(output_dir) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Focal-surface angles
    arrays = _gather_angle_arrays(results_by_offset, "angle_incidence_focal", log)
    if arrays:
        bins = _compute_bins(np.concatenate(arrays), bin_width_deg, log, context="focal")
        if bins is not None:
            fig, ax = plt.subplots(1, 1, figsize=(7, 5))
            _plot_overlay_angles(
                results_by_offset, "angle_incidence_focal", bins, ax, use_zorder=True
            )
            ax.set_xlabel("Angle of incidence at focal surface (deg) w.r.t. optical axis")
            ax.set_ylabel(Y_AXIS_BIN_COUNT_LABEL)
            ax.set_title("Incident angle distribution vs off-axis angle")
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f"incident_angles_multi_{label}.png", dpi=300)
            plt.close(fig)

    # Primary and secondary mirror angles
    _plot_component_angles(
        results_by_offset=results_by_offset,
        column="angle_incidence_primary",
        title_suffix="on primary mirror (w.r.t. normal)",
        out_path=out_dir / f"incident_angles_primary_multi_{label}.png",
        bin_width_deg=bin_width_deg,
        log=log,
    )
    _plot_component_angles(
        results_by_offset=results_by_offset,
        column="angle_incidence_secondary",
        title_suffix="on secondary mirror (w.r.t. normal)",
        out_path=out_dir / f"incident_angles_secondary_multi_{label}.png",
        bin_width_deg=bin_width_deg,
        log=log,
    )

    # Debug plots
    if debug_plots:
        _plot_debug_plots(results_by_offset, out_dir, label, radius_bin_width_m, log)
