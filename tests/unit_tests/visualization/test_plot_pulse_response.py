"""Check the response plot renders both raw gain panels."""

from pathlib import Path

import numpy as np
from astropy.table import Table

from simtools.visualization.plot_pulse_response import plot_response

ILLUMINATION = "illumination"
RESPONSE = "response"
PIXEL = "pixel"
FLASHER = "flasher"


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
