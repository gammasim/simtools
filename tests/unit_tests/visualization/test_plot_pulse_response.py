"""Check the response plot renders both raw gain panels."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.table import Table

from simtools.visualization.plot_pulse_response import _plot_metrics, plot_response

ILLUMINATION = "illumination"
RESPONSE = "response"
PIXEL = "pixel"
FLASHER = "flasher"


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


def test_plot_response(tmp_test_directory):
    table = Table(
        {
            ILLUMINATION: [1, 2, 1, 2],
            "gain": ["HG", "HG", "LG", "LG"],
            RESPONSE: [FLASHER] * 4,
            PIXEL: [0] * 4,
            "peak_adc": [1, 2, 1, 2],
            "adc_sum": [2, 4, 2, 4],
            "width_ns": [1] * 4,
            "peak_adc_std": [0] * 4,
            "adc_sum_std": [0] * 4,
            "width_ns_std": [0] * 4,
        },
        meta={"illumination_unit": "photons", "width_fraction": 0.95},
    )
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
    output = Path(tmp_test_directory) / "response.pdf"
    plot_response(table, waves, output)
    assert output.is_file()
    table.meta["threshold_adc"] = 0.5
    plot_response(table, waves, output)
