"""Plot gain-resolved pulse-response measurements."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

GAINS = ("HG", "LG")
INTENSITY = "illumination"
PEAK = "peak_adc"
ADC_SUM = "adc_sum"
WIDTH = "width_ns"
RESPONSE = "response"
PIXEL = "pixel"
PIXEL_PREFIX = " pixel "


def plot_response(summary, waves, output_file):
    """Save peak, charge, width and mean-waveform panels in one PDF.

    Parameters
    ----------
    summary : astropy.table.Table
        Mean metrics with standard deviations, counts, and illumination units.
    waves : list of dict
        Time in ns and baseline-subtracted waveform for each response and gain.
    output_file : path-like
        Destination PDF file.
    """
    axis = summary.meta["illumination_unit"]
    threshold = summary.meta.get("threshold_adc")
    width_label = f"Width at {summary.meta['width_fraction']:.0%} peak [ns]"
    if threshold is not None:
        width_label = f"Width above {threshold:g} ADC [ns]"
    labels = ("Mean event peak [ADC above pedestal]", "Mean ADC sum [ADC samples]", width_label)
    with PdfPages(output_file) as pdf:
        for gain in GAINS:
            fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
            try:
                _plot_metrics(axes, summary[summary["gain"] == gain], labels, axis)
                _plot_waves(axes[1, 1], waves, gain, axis)
                fig.suptitle(f"{gain} pulse response (error bars: event standard deviation)")
                pdf.savefig(fig)
            finally:
                plt.close(fig)


def _plot_metrics(axes, selected, labels, axis):
    """Plot event means and dispersions for each response and selected pixel."""
    keys = [RESPONSE]
    if PIXEL in selected.colnames:
        keys.append(PIXEL)
    for group in selected.group_by(keys).groups:
        label = str(group[RESPONSE][0])
        if PIXEL in group.colnames:
            label += f"{PIXEL_PREFIX}{group[PIXEL][0]}"
        for ax, metric, ylabel in zip(axes.flat, (PEAK, ADC_SUM, WIDTH), labels):
            order = np.argsort(group[INTENSITY])
            ax.errorbar(
                group[INTENSITY][order],
                group[metric][order],
                yerr=group[f"{metric}_std"][order],
                marker="o",
                label=label,
            )
            ax.set(xscale="log", xlabel=f"Illumination [{axis}]", ylabel=ylabel)
            if metric == ADC_SUM:
                ax.set_yscale("log")
            ax.grid(alpha=0.3)
            ax.legend(fontsize="small")


def _plot_waves(ax, waves, gain, axis):
    """Overlay the mean waveforms, keeping their time origins."""
    for wave in waves:
        if wave["gain"] != gain:
            continue
        label = f"{wave[RESPONSE]} {wave[INTENSITY]:g} {axis}"
        if PIXEL in wave:
            label += f"{PIXEL_PREFIX}{wave[PIXEL]}"
        ax.plot(wave["time"], wave["waveform"], label=label)
    ax.set(xlabel="Time [ns]", ylabel="Mean waveform [ADC above pedestal]")
    ax.legend(fontsize="x-small")
