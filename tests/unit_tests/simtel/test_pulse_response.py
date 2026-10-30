"""Tests for pulse-response tables and flasher scans."""

from unittest.mock import MagicMock

import numpy as np
import pytest
from astropy.table import Table

from simtools.simtel import pulse_response

FREQUENCY = "fadc_mhz"
FADC_AMPLITUDE = "fadc_amplitude"
TELESCOPE = "LSTN-01"
RESPONSE = "response"
AMPLITUDE = "amplitude"
INTENSITY = "intensity"
SCAN_FILE = "scan.simtel"
PEAK = "peak_adc"
ADC_SUM = "adc_sum"


@pytest.fixture
def pulse_model():
    model = MagicMock()
    parameters = {
        FREQUENCY: 1000,
        FADC_AMPLITUDE: 2,
        "fadc_lg_amplitude": 1,
        "fadc_sensitivity": 1,
        "fadc_lg_sensitivity": -1,
        "fadc_pedestal": 10,
        "fadc_lg_pedestal": -1,
        "fadc_max_signal": 100,
        "transit_time_jitter": 0,
    }
    model.get_parameter_value.side_effect = parameters.__getitem__
    rows = []
    for gain in pulse_response.GAINS:
        rows.extend(
            {
                "template_kind": "reference",
                "gain": gain,
                INTENSITY: 1.0,
                "time": time,
                AMPLITUDE: amplitude,
            }
            for time, amplitude in zip(
                [0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 1.0, 0.0, 0.0], strict=True
            )
        )
        for intensity, amplitudes in (
            (10.0, [0.0, 0.0, 10.0, 0.0, 0.0]),
            (20.0, [0.0, 5.0, 15.0, 5.0, 0.0]),
        ):
            rows.extend(
                {
                    "template_kind": "response",
                    "gain": gain,
                    INTENSITY: intensity,
                    "time": time,
                    AMPLITUDE: amplitude,
                }
                for time, amplitude in zip([0.0, 1.0, 2.0, 3.0, 4.0], amplitudes, strict=True)
            )
    unified = Table(rows=rows)
    model.get_parameter_table.return_value = unified
    return model


def test_template_scaling_interpolation_and_holding(pulse_model):
    table, waves = pulse_response.template_responses(pulse_model, [5, 10, 15, 20, 30])
    selected = table[(table["gain"] == "HG") & (table[RESPONSE] == "nonlinear")]
    assert selected[PEAK].tolist() == pytest.approx([10, 20, 25, 30, 30])
    assert selected[ADC_SUM].tolist() == pytest.approx([10, 20, 35, 50, 50])
    assert selected["above_table"].tolist() == [False, False, False, False, True]
    assert len(waves) == 20


def test_template_response_scan_requires_unified_parameter(pulse_model):
    pulse_model.get_parameter_table.return_value = Table({"time": [0, 1], AMPLITUDE: [0, 1]})
    with pytest.raises(ValueError, match="requires unified"):
        pulse_response.template_responses(pulse_model, [1])


def test_template_clipping(pulse_model):
    table, _ = pulse_response.template_responses(pulse_model, [100])
    linear = table[(table["gain"] == "HG") & (table[RESPONSE] == "linear_clipped")]
    assert linear[PEAK][0] == pytest.approx(90)


def test_template_responses_reads_unified_reference_and_gain_templates(pulse_model):
    rows = []
    for gain in ("HG", "LG"):
        rows.extend(
            {
                "template_kind": "reference",
                "gain": gain,
                INTENSITY: 1.0,
                "time": time,
                AMPLITUDE: amplitude,
            }
            for time, amplitude in zip([0, 1, 2, 3, 4], [0, 0, 1, 0, 0], strict=True)
        )
        for intensity, pulse in ((10, [0, 0, 10, 0, 0]), (20, [0, 0, 20, 0, 0])):
            rows.extend(
                {
                    "template_kind": "response",
                    "gain": gain,
                    INTENSITY: intensity,
                    "time": time,
                    AMPLITUDE: amplitude,
                }
                for time, amplitude in zip([0, 1, 2, 3, 4], pulse, strict=True)
            )
    unified = Table(rows=rows)
    pulse_model.get_parameter_table.side_effect = lambda _name: unified
    table, _ = pulse_response.template_responses(pulse_model, [10])
    assert table[table["response"] == "nonlinear"][PEAK].tolist() == pytest.approx([20, 10])
    assert pulse_model.get_parameter_table.call_args_list[0].args == ("fadc_pulse_shape",)


def test_invalid_intensities(pulse_model):
    with pytest.raises(ValueError, match="illumination"):
        pulse_response.template_responses(pulse_model, [-1])


def test_scan_reads_both_gains_and_pedestals(mocker):
    event = {
        "event_id": 1,
        "telescope_events": {1: {"adc_samples": np.array([[[10, 12, 10]], [[20, 21, 20]]])}},
        "camera_monitorings": {1: {"pedestal": [[20], [40]], "n_ped_slices": 2}},
    }
    mocker.patch.object(
        pulse_response,
        "get_sim_telarray_telescope_id_to_telescope_name_mapping",
        return_value={1: TELESCOPE},
    )
    reader = mocker.patch.object(pulse_response, "SimTelFile")
    reader.return_value.__enter__.return_value = MagicMock(
        telescope_descriptions={1: {"pixel_settings": {"time_slice": 1}}},
        __iter__=lambda _: iter([event]),
    )
    table, waves = pulse_response.flasher_responses([SCAN_FILE], [100], [1], TELESCOPE, [0])
    assert table[PEAK].tolist() == pytest.approx([2, 1])
    assert table[ADC_SUM].tolist() == pytest.approx([2, 1])
    assert len(waves) == 2
    with pytest.raises(ValueError, match="count"):
        pulse_response.flasher_responses([SCAN_FILE], [100], [2], TELESCOPE, [0])


def test_scan_rejects_ambiguous_steps():
    with pytest.raises(ValueError, match=r"one file|steps"):
        pulse_response.flasher_responses(["a", "b"], [1, 2, 3], [1], TELESCOPE, [0])


def test_summary_and_outputs(pulse_model, tmp_test_directory, mocker):
    table, waves = pulse_response.template_responses(pulse_model, [5, 15])
    plot = mocker.patch.object(pulse_response, "plot_response")
    path = pulse_response.write_response(table, waves, tmp_test_directory, RESPONSE)
    loaded = Table.read(path)
    summary = Table.read(tmp_test_directory / "response_summary.ecsv")
    assert len(loaded) == len(summary) == 8
    assert summary["event_count"].tolist() == [1] * 8
    assert np.allclose(summary["peak_adc_std"], 0)
    plot.assert_called_once()


@pytest.mark.parametrize("value", [0, -1, float("nan")])
def test_invalid_gain_parameters(pulse_model, value):
    original = pulse_model.get_parameter_value.side_effect
    pulse_model.get_parameter_value.side_effect = lambda name: (
        value if name == FADC_AMPLITUDE else original(name)
    )
    with pytest.raises(ValueError, match=r"positive|finite"):
        pulse_response.template_responses(pulse_model, [5])


def test_invalid_frequency(pulse_model):
    original = pulse_model.get_parameter_value.side_effect
    pulse_model.get_parameter_value.side_effect = lambda name: (
        0 if name == FREQUENCY else original(name)
    )
    with pytest.raises(ValueError, match="frequency"):
        pulse_response.template_responses(pulse_model, [5])


@pytest.mark.parametrize(
    "table",
    [
        Table({INTENSITY: [1, 1], "time": [0, 1], AMPLITUDE: [0, 1]}),
        Table({INTENSITY: [1, 1, 2, 2], "time": [1, 0, 1, 0], AMPLITUDE: [0, 1, 0, 2]}),
    ],
)
def test_invalid_nonlinear_table(table):
    with pytest.raises(ValueError, match=r"levels|increasing"):
        pulse_response._interpolate_response(table, np.arange(3), np.ones(3), 2)


def test_missing_or_explicit_pedestals():
    assert np.allclose(pulse_response._pedestals({}, 1, (2, 3), 10), 10)
    with pytest.raises(ValueError, match="monitoring pedestals"):
        pulse_response._pedestals({}, 1, (2, 3), None)
    with pytest.raises(ValueError, match="finite"):
        pulse_response._pedestals({}, 1, (2, 3), np.nan)


@pytest.mark.parametrize(
    ("counts", "pixels"), [([0], [0]), ([1, 2], [0]), ([1], []), ([1], [0, 0])]
)
def test_invalid_scan_configuration(counts, pixels):
    with pytest.raises(ValueError, match=r"counts|pixel"):
        pulse_response.flasher_responses(["a"], [100], counts, TELESCOPE, pixels)
