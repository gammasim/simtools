import numpy as np
import pytest

from simtools.sim_events import production_comparison
from simtools.sim_events.production_comparison import ProductionDescriptor


def _event(triggered_pixels=3):
    adc_samples = np.full((2, 2, 20), 20.0)
    adc_samples[0, 0, 4] = 30.0
    return {
        "adc_samples": adc_samples,
        "pixel_lists": {0: {"pixels": triggered_pixels}},
    }


def test_collect_signal_metrics_for_all_layout_telescopes(mocker):
    mocker.patch(
        "simtools.sim_events.production_comparison"
        ".get_sim_telarray_telescope_id_to_telescope_name_mapping",
        return_value={1: "LSTN-01", 2: "MSTN-01"},
    )
    mocker.patch(
        "simtools.sim_events.production_comparison.read_events",
        side_effect=lambda _file, _telescope, **_kwargs: ([1], {}, [_event()]),
    )

    metrics = production_comparison.collect_signal_metrics(
        [ProductionDescriptor("baseline", ["baseline-1.simtel", "baseline-2.simtel"])],
        ["CTAO-North-Alpha"],
    )

    assert set(metrics) == {"LSTN-01", "MSTN-01"}
    assert metrics["LSTN-01"][0].label == "baseline"
    assert metrics["LSTN-01"][0].signals.tolist() == [10.0, 0.0, 10.0, 0.0]
    assert metrics["LSTN-01"][0].triggered_pixels.tolist() == [3, 3]
    assert metrics["LSTN-01"][0].peak_timing.size == 4


def test_collect_signal_metrics_rejects_missing_layout_telescope(mocker):
    mocker.patch(
        "simtools.sim_events.production_comparison"
        ".get_sim_telarray_telescope_id_to_telescope_name_mapping",
        side_effect=[{1: "LSTN-01"}, {1: "LSTN-01", 2: "MSTN-01"}],
    )

    with pytest.raises(ValueError, match="telescope set"):
        production_comparison.collect_signal_metrics(
            [
                ProductionDescriptor("baseline", ["baseline.simtel"]),
                ProductionDescriptor("candidate", ["candidate.simtel"]),
            ],
            "CTAO-North-Alpha",
        )


def test_collect_signal_metrics_requires_one_layout():
    with pytest.raises(ValueError, match="exactly one array_layout_name"):
        production_comparison.collect_signal_metrics([], [])
