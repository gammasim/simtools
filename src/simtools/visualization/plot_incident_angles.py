#!/usr/bin/python3
"""Plot incident angle histograms for focal plane, primary, and secondary mirrors."""

import logging
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import QTable

__all__ = ["plot_incident_angles"]


def _gather_arrays(results_by_offset: dict[float, QTable], column: str, log: logging.Logger):
    arrays: list[np.ndarray] = []
    for off, tab in results_by_offset.items():
        if tab is None or len(tab) == 0:
            if column == "angle_incidence_focal":
                log.warning(f"Empty results for off-axis={off}")
            continue
        if column not in tab.colnames:
            continue
        arrays.append(tab[column].to(u.deg).value)
    return arrays


def _compute_bins(all_vals: np.ndarray, bin_width_deg: float, log: logging.Logger, context: str):
    finite_mask = np.isfinite(all_vals)
    if not np.any(finite_mask):
        if context == "focal":
            log.warning("No finite focal-surface incidence angle values to plot")
        else:
            log.warning("No finite %s values to plot", context)
        return None
    vals = all_vals[finite_mask]
    vmin = float(np.floor(vals.min() / bin_width_deg) * bin_width_deg)
    vmax = float(np.ceil(vals.max() / bin_width_deg) * bin_width_deg)
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        if context == "focal":
            log.warning("Invalid focal-surface histogram edges: vmin=%s vmax=%s", vmin, vmax)
        else:
            log.warning(
                "Invalid bin edges computed for %s histogram: vmin=%s vmax=%s",
                context,
                vmin,
                vmax,
            )
        return None
    if vmax <= vmin:
        vmax = vmin + bin_width_deg
    return np.arange(vmin, vmax + bin_width_deg * 0.5, bin_width_deg)


def _plot_overlay(
    results_by_offset: dict[float, QTable], column: str, bins: np.ndarray, ax, use_zorder: bool
):
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
            histtype="step",
            linewidth=0.5,
            label=f"off-axis {off:g} deg",
            zorder=z1,
        )
        color = patches[0].get_edgecolor() if patches else None
        ax.hist(
            data,
            bins=bins,
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
            histtype="step",
            linewidth=0.5,
            color=color,
            label="_nolegend_",
            zorder=z3,
        )


def _plot_component(
    results_by_offset: dict[float, QTable],
    column: str,
    title_suffix: str,
    out_path: Path,
    bin_width_deg: float,
    log: logging.Logger,
):
    arrays = _gather_arrays(results_by_offset, column, log)
    if not arrays:
        log.warning("No finite %s values to plot", column)
        return
    bins = _compute_bins(np.concatenate(arrays), bin_width_deg, log, context=column)
    if bins is None:
        return
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    _plot_overlay(results_by_offset, column, bins, ax, use_zorder=False)
    ax.set_xlabel("Angle of incidence (deg)")
    ax.set_ylabel("Count / Bin")
    ax.set_title(f"Incident angle {title_suffix} vs off-axis angle")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_incident_angles(
    results_by_offset: dict[float, QTable],
    output_dir: Path,
    label: str,
    bin_width_deg: float = 0.1,
    logger: logging.Logger | None = None,
) -> None:
    """Plot overlaid histograms of focal-surface incidence angles for multiple offsets.

    Parameters
    ----------
    results_by_offset : dict(float -> QTable)
        Mapping from off-axis angle in degrees to result table containing column
        ``angle_incidence_focal`` with ``astropy.units``.
    output_dir : Path
        Directory to write PNG plots into (created if needed).
    label : str
        Label used to compose the output filename.
    bin_width_deg : float, optional
        Histogram bin width in degrees (default: 0.1 deg).
    logger : logging.Logger, optional
        Logger to emit warnings; if not provided, a module-level logger is used.
    """
    log = logger or logging.getLogger(__name__)
    if not results_by_offset:
        log.warning("No results provided for multi-offset plot")
        return

    arrays = _gather_arrays(results_by_offset, "angle_incidence_focal", log)
    if not arrays:
        log.warning("No non-empty results to plot")
        return
    bins = _compute_bins(np.concatenate(arrays), bin_width_deg, log, context="focal")
    if bins is None:
        return

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    _plot_overlay(results_by_offset, "angle_incidence_focal", bins, ax, use_zorder=True)

    ax.set_xlabel("Angle of incidence at focal surface (deg) w.r.t. optical axis")
    ax.set_ylabel("Count / Bin")
    ax.set_title("Incident angle distribution vs off-axis angle")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    out_dir = Path(output_dir) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"incident_angles_multi_{label}.png"
    plt.savefig(out_png, dpi=300)
    plt.close(fig)

    _plot_component(
        results_by_offset=results_by_offset,
        column="angle_incidence_primary",
        title_suffix="on primary mirror (w.r.t. normal)",
        out_path=out_dir / f"incident_angles_primary_multi_{label}.png",
        bin_width_deg=bin_width_deg,
        log=log,
    )
    _plot_component(
        results_by_offset=results_by_offset,
        column="angle_incidence_secondary",
        title_suffix="on secondary mirror (w.r.t. normal)",
        out_path=out_dir / f"incident_angles_secondary_multi_{label}.png",
        bin_width_deg=bin_width_deg,
        log=log,
    )
