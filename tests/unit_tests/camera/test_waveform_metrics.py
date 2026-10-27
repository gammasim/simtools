"""Tests of gain-independent waveform measurements."""

import numpy as np
import pytest

from simtools.camera.waveform_metrics import measure_waveform

TRUNCATED = "truncated"
ADC_SUM = "adc_sum"
INTEGRAL = "integral_adc_ns"
WIDTH = "width_ns"


def test_triangle_and_sampling():
    result = measure_waveform([0, 2, 4, 2, 0], 0.5, width_fraction=0.5)
    assert result["peak_adc"] == pytest.approx(4)
    assert result[ADC_SUM] == pytest.approx(8)
    assert result[INTEGRAL] == pytest.approx(4)
    assert result[WIDTH] == pytest.approx(1)
    assert not result[TRUNCATED]


def test_plateau_and_absolute_threshold():
    result = measure_waveform([0, 4, 4, 4, 0], 1, threshold_adc=2)
    assert result[WIDTH] == pytest.approx(3)
    assert result["peak_time_ns"] == pytest.approx(1)
    assert measure_waveform([4, 4, 0], 1)[TRUNCATED]


def test_no_crossing_and_negative_tail():
    result = measure_waveform([0, 2, 0, -1], 1, threshold_adc=3)
    assert result[WIDTH] == pytest.approx(0)
    assert result[ADC_SUM] == pytest.approx(1)
    assert np.isfinite(result[INTEGRAL])


@pytest.mark.parametrize(("wave", "dt"), [([], 1), ([0], 1), ([0, np.nan], 1), ([0, 1], 0)])
def test_invalid_waveform(wave, dt):
    with pytest.raises(ValueError, match=r"finite|samples|interval"):
        measure_waveform(wave, dt)


@pytest.mark.parametrize(("fraction", "threshold"), [(0, None), (1, None), (0.5, -1)])
def test_invalid_threshold(fraction, threshold):
    with pytest.raises(ValueError, match=r"threshold|fraction"):
        measure_waveform([0, 1, 0], 1, fraction, threshold)


def test_signed_sum_is_not_clamped():
    result = measure_waveform([0, 2, 0, -5], 1)
    assert result[ADC_SUM] == pytest.approx(-3)
