"""Tests for pulse-response tables and flasher scans."""

from unittest.mock import MagicMock

import matplotlib.pyplot as plt
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
_CAPTURED_FIGURES = []


def _capture_figure(figure):
    """Keep figures for checking their labels after PDF export."""
    _CAPTURED_FIGURES.append(figure)


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
        "fadc_lg_max_signal": -1,
        "transit_time_jitter": 0,
        "fadc_pulse_shape": "pulse.ecsv",
    }
    model.get_parameter_value.side_effect = parameters.__getitem__
    rows = []
    for gain in pulse_response.GAINS:
        rows.extend(
            {
                "template_kind": "single_pe",
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
                    "template_kind": "intensity_dependent",
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


def test_unsorted_reference_samples(pulse_model):
    expected, _ = pulse_response.template_responses(pulse_model, [5])
    table = pulse_model.get_parameter_table.return_value
    pulse_model.get_parameter_table.return_value = table[::-1]
    actual, _ = pulse_response.template_responses(pulse_model, [5])
    assert actual[PEAK].tolist() == pytest.approx(expected[PEAK].tolist())
    assert actual[ADC_SUM].tolist() == pytest.approx(expected[ADC_SUM].tolist())


def test_first_response_level_uses_table(pulse_model):
    table = pulse_model.get_parameter_table.return_value
    selected = (table["template_kind"] == "intensity_dependent") & (table[INTENSITY] == 10)
    table[AMPLITUDE][selected] *= 0.8
    result, _ = pulse_response.template_responses(pulse_model, [9, 10, 10.001])
    hg = result[(result["gain"] == "HG") & (result[RESPONSE] == "nonlinear")]
    assert hg[PEAK].tolist() == pytest.approx([18, 16, 16.0014])


@pytest.mark.parametrize("low_gain", [True, False])
def test_template_response_scan_with_linear_table(pulse_model, low_gain):
    data = {"time": [0, 1, 2, 3, 4], AMPLITUDE: [0, 0, 1, 0, 0]}
    if low_gain:
        data["amplitude_low_gain"] = [0, 1, 1, 0, 0]
    pulse_model.get_parameter_table.return_value = Table(data)
    table, waves = pulse_response.template_responses(pulse_model, [5, 10])
    assert set(table[RESPONSE]) == {"linear_clipped"}
    assert len(table) == len(waves) == 4
    assert table[PEAK].tolist() == pytest.approx([10, 20, 5, 10])
    assert table[ADC_SUM][-1] == pytest.approx(20 if low_gain else 10)


@pytest.mark.parametrize("low_gain", [True, False])
def test_template_response_scan_with_dictionary_pulse(pulse_model, low_gain):
    columns = ["time", AMPLITUDE]
    rows = [[0, 0], [1, 0], [2, 1], [3, 0], [4, 0]]
    if low_gain:
        columns.append("amplitude (low gain)")
        for row, amplitude in zip(rows, [0, 1, 1, 0, 0], strict=True):
            row.append(amplitude)
    value = {
        "columns": columns,
        "column_units": ["ns"] + ["dimensionless"] * (len(columns) - 1),
        "rows": rows,
    }
    original = pulse_model.get_parameter_value.side_effect
    pulse_model.get_parameter_value.side_effect = lambda name: (
        value if name == "fadc_pulse_shape" else original(name)
    )
    table, _ = pulse_response.template_responses(pulse_model, [10])
    lg = table[table["gain"] == "LG"]
    assert lg[ADC_SUM][0] == pytest.approx(20 if low_gain else 10)


def test_template_clipping(pulse_model):
    table, _ = pulse_response.template_responses(pulse_model, [100])
    linear = table[(table["gain"] == "HG") & (table[RESPONSE] == "linear_clipped")]
    assert linear[PEAK][0] == pytest.approx(90)


@pytest.mark.parametrize(("maximum", "expected"), [(40, 30), (-1, 90)])
def test_low_gain_clipping_uses_its_own_maximum(pulse_model, maximum, expected):
    original = pulse_model.get_parameter_value.side_effect
    pulse_model.get_parameter_value.side_effect = lambda name: (
        maximum if name == "fadc_lg_max_signal" else original(name)
    )
    table, _ = pulse_response.template_responses(pulse_model, [100])
    linear = table[(table["gain"] == "LG") & (table[RESPONSE] == "linear_clipped")]
    assert linear[PEAK][0] == pytest.approx(expected)


def test_template_responses_reads_unified_reference_and_gain_templates(pulse_model):
    rows = []
    for gain in ("HG", "LG"):
        rows.extend(
            {
                "template_kind": "single_pe",
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
                    "template_kind": "intensity_dependent",
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


def test_scan_reads_both_gains_and_pedestals():
    event = {
        "event_id": 1,
        "telescope_events": {1: {"adc_samples": np.array([[[10, 12, 10]], [[20, 21, 20]]])}},
        "camera_monitorings": {1: {"pedestal": [[20], [40]], "n_ped_slices": 2}},
    }

    class FakeStream:
        telescope_descriptions = {1: {"pixel_settings": {"time_slice": 1}}}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def __iter__(self):
            return iter([event])

    parsed = list(
        pulse_response._read_scan_file(
            SCAN_FILE,
            TELESCOPE,
            [0],
            np.asarray([100]),
            np.asarray([0]),
            {pulse_response.WIDTH_FRACTION: 0.95, pulse_response.THRESHOLD_ADC: None},
            None,
            stream_factory=lambda *_args, **_kwargs: FakeStream(),
            mapping_reader=lambda _file: {1: TELESCOPE},
        )
    )
    rows = [item[0] for item in parsed]
    assert [row[PEAK] for row in rows] == pytest.approx([2, 1])
    assert [row[ADC_SUM] for row in rows] == pytest.approx([2, 1])
    assert len(parsed) == 2
    with pytest.raises(ValueError, match="count"):
        list(
            pulse_response._read_scan_file(
                SCAN_FILE,
                TELESCOPE,
                [0],
                np.asarray([100]),
                np.asarray([0, 0]),
                {pulse_response.WIDTH_FRACTION: 0.95, pulse_response.THRESHOLD_ADC: None},
                None,
                stream_factory=lambda *_args, **_kwargs: FakeStream(),
                mapping_reader=lambda _file: {1: TELESCOPE},
            )
        )

    partial = list(
        pulse_response._read_scan_file(
            SCAN_FILE,
            TELESCOPE,
            [0],
            np.asarray([100]),
            np.asarray([0, 0]),
            {pulse_response.WIDTH_FRACTION: 0.95, pulse_response.THRESHOLD_ADC: None},
            None,
            stream_factory=lambda *_args, **_kwargs: FakeStream(),
            mapping_reader=lambda _file: {1: TELESCOPE},
            allow_incomplete=True,
        )
    )
    assert len(partial) == 2


def test_scan_rejects_ambiguous_steps():
    with pytest.raises(ValueError, match=r"one file|steps"):
        pulse_response.flasher_responses(["a", "b"], [1, 2, 3], [1], TELESCOPE, [0])


def test_summary_and_outputs(pulse_model, tmp_test_directory):
    table, waves = pulse_response.template_responses(pulse_model, [5, 15], threshold_adc=0.5)
    try:
        with plt.rc_context({"figure.hooks": [f"{__name__}:_capture_figure"]}):
            path = pulse_response.write_response(table, waves, tmp_test_directory, RESPONSE)
        assert len(_CAPTURED_FIGURES) == 2
        assert all(
            figure.axes[2].get_ylabel() == "Width above 0.5 ADC [ns]"
            for figure in _CAPTURED_FIGURES
        )
    finally:
        _CAPTURED_FIGURES.clear()
    loaded = Table.read(path)
    summary = Table.read(tmp_test_directory / "response_summary.ecsv")
    assert len(loaded) == len(summary) == 8
    assert path.with_suffix(".pdf").read_bytes().startswith(b"%PDF-")
    assert summary["event_count"].tolist() == [1] * 8
    assert np.allclose(summary["peak_adc_std"], 0)


def test_summary_keeps_repeated_illumination_steps_separate():
    table = Table(
        {
            "illumination": [100, 100],
            "gain": ["HG", "HG"],
            RESPONSE: ["flasher", "flasher"],
            "step": [0, 1],
            "pixel": [0, 0],
            "truncated": [False, False],
            PEAK: [10, 20],
            ADC_SUM: [10, 20],
            "integral_adc_ns": [10, 20],
            "width_ns": [1, 2],
        }
    )

    summary = pulse_response.summarize_response(table)

    assert summary["step"].tolist() == [0, 1]
    assert summary[PEAK].tolist() == [10, 20]
    assert summary["event_count"].tolist() == [1, 1]


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


def test_nonlinear_response_sorts_samples_before_interpolation():
    table = Table(
        {
            INTENSITY: [1, 1, 2, 2],
            "time": [1, 0, 1, 0],
            AMPLITUDE: [2, 0, 4, 0],
        }
    )

    response = pulse_response._interpolate_response(table, np.asarray([0, 0.5, 1]), np.ones(3), 2)

    assert response.tolist() == pytest.approx([0, 2, 4])


@pytest.mark.parametrize(
    "table",
    [
        Table({INTENSITY: [1, 1], "time": [0, 1], AMPLITUDE: [0, 1]}),
        Table({INTENSITY: [1, 1, 2, 2], "time": [0, 0, 0, 1], AMPLITUDE: [0, 1, 0, 2]}),
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


def test_individual_scan_file_allows_zero_triggers():
    class EmptyStream:
        telescope_descriptions = {1: {"pixel_settings": {"time_slice": 1}}}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def __iter__(self):
            return iter(())

    rows = list(
        pulse_response._read_scan_file(
            SCAN_FILE,
            TELESCOPE,
            [0],
            np.asarray([100]),
            np.asarray([0] * 20),
            {pulse_response.WIDTH_FRACTION: 0.95, pulse_response.THRESHOLD_ADC: None},
            None,
            stream_factory=lambda *_args, **_kwargs: EmptyStream(),
            mapping_reader=lambda _file: {1: TELESCOPE},
            allow_incomplete=True,
        )
    )
    assert rows == []


def test_zero_trigger_level_is_reported_without_signal():
    empty = pulse_response._metrics_table([], "photons", 0.95, None)
    empty.meta["scan_levels"] = [100.0]
    empty.meta["scan_expected_events"] = [20]
    empty.meta["scan_pixels"] = [0]
    summary = pulse_response.summarize_response(empty)
    assert len(summary) == 1
    assert summary["illumination"][0] == 100
    assert summary["event_count"][0] == 0
    assert summary["expected_event_count"][0] == 20
    assert np.isnan(summary["peak_adc"][0])


def test_write_response_with_no_triggers_creates_report(tmp_test_directory):
    empty = pulse_response._metrics_table([], "photons", 0.95, None)
    empty.meta["scan_levels"] = [100.0]
    empty.meta["scan_expected_events"] = [20]
    empty.meta["scan_pixels"] = [0]
    path = pulse_response.write_response(empty, [], tmp_test_directory, "flasher_response")
    assert path.is_file()
    assert path.with_name("flasher_response_summary.ecsv").is_file()
    assert path.with_name("flasher_response.pdf").stat().st_size > 0


def test_empty_combined_scan_has_no_ambiguous_event_assignments():
    class EmptyStream:
        telescope_descriptions = {1: {"pixel_settings": {"time_slice": 1}}}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def __iter__(self):
            return iter(())

    rows = list(
        pulse_response._read_scan_file(
            SCAN_FILE,
            TELESCOPE,
            [0],
            np.asarray([100, 300]),
            np.asarray([0, 0, 1, 1]),
            {pulse_response.WIDTH_FRACTION: 0.95, pulse_response.THRESHOLD_ADC: None},
            None,
            stream_factory=lambda *_args, **_kwargs: EmptyStream(),
            mapping_reader=lambda _file: {1: TELESCOPE},
        )
    )
    assert rows == []


def test_summary_distinguishes_zero_and_partial_recorded_levels():
    row = {
        "file": "partial.simtel",
        "event_id": 1,
        "step": 1,
        "pixel": 0,
        "gain": "HG",
        "response": "flasher",
        "illumination": 300.0,
        **pulse_response.measure_waveform(np.array([0.0, 1.0, 0.0]), 1.0),
    }
    table = pulse_response._metrics_table([row], "photons", 0.95, None)
    table.meta["scan_levels"] = [100.0, 300.0]
    table.meta["scan_expected_events"] = [20, 20]
    table.meta["scan_pixels"] = [0]
    summary = pulse_response.summarize_response(table)
    assert summary["event_count"].tolist() == [0, 1]
    assert summary["expected_event_count"].tolist() == [20, 20]
    assert np.isnan(summary["peak_adc"][0])
    assert summary["peak_adc"][1] == pytest.approx(1.0)
