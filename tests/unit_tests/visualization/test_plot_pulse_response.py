"""Check the response plot renders both raw gain panels."""

from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.table import Table

from simtools.visualization.plot_pulse_response import _plot_metrics, _plot_waves, plot_response

ILLUMINATION = "illumination"
RESPONSE = "response"
PIXEL = "pixel"
FLASHER = "flasher"
_CAPTURED_FIGURES = []


def _capture_figure(figure):
    """Keep the figure for checking its labels after PDF export."""
    _CAPTURED_FIGURES.append(figure)


@pytest.mark.parametrize(("sums", "scale"), [([1, 2], "log"), ([-1, 2], "linear")])
def test_adc_sum_scale(sums, scale):
    table = Table(
        {
            ILLUMINATION: [1, 2],
            RESPONSE: [FLASHER, FLASHER],
            "peak_adc": [1, 2],
            "adc_sum": sums,
            "width_ns": [1, 1],
            "peak_adc_std": [0, 0],
            "adc_sum_std": [0, 0],
            "width_ns_std": [0, 0],
        }
    )
    fig, axes = plt.subplots(2, 2)
    try:
        _plot_metrics(axes, table, ("peak", "sum", "width"), "photons")
        assert axes[0, 1].get_yscale() == scale
    finally:
        plt.close(fig)


def test_plot_waves_selects_gain():
    waves = [
        {
            "gain": gain,
            RESPONSE: FLASHER,
            ILLUMINATION: 1,
            PIXEL: 0,
            "time": np.arange(3),
            "waveform": [0, 1, 0],
        }
        for gain in ("HG", "LG")
    ]
    figure, axis = plt.subplots()
    try:
        _plot_waves(axis, waves, "HG", "photons")
        assert len(axis.lines) == 1
        np.testing.assert_array_equal(axis.lines[0].get_xdata(), [0, 1, 2])
        np.testing.assert_array_equal(axis.lines[0].get_ydata(), [0, 1, 0])
        assert axis.lines[0].get_label() == "flasher 1 photons pixel 0"
        assert axis.get_xlabel() == "Time [ns]"
    finally:
        plt.close(figure)


def test_plot_response_default_width_label():
    table = Table(
        {
            ILLUMINATION: [1],
            "gain": ["HG"],
            RESPONSE: [FLASHER],
            "peak_adc": [1],
            "adc_sum": [2],
            "width_ns": [1],
            "peak_adc_std": [0],
            "adc_sum_std": [0],
            "width_ns_std": [0],
        },
        meta={"illumination_unit": "photons", "width_fraction": 0.95},
    )
    waves = [{"gain": "HG", RESPONSE: FLASHER, ILLUMINATION: 1, "time": [0], "waveform": [0]}]
    output = BytesIO()
    try:
        with plt.rc_context({"figure.hooks": [f"{__name__}:_capture_figure"]}):
            plot_response(table, waves, output)
        assert len(_CAPTURED_FIGURES) == 1
        assert _CAPTURED_FIGURES[0].axes[2].get_ylabel() == "Width at 95% peak [ns]"
    finally:
        _CAPTURED_FIGURES.clear()
    assert output.getvalue().startswith(b"%PDF-")
