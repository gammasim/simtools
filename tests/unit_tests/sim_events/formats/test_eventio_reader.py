"""Test eventio-specific interpretation and unit conversions."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from eventio import iact
from eventio.simtel import ArrayEvent, MCEvent, TrackingPosition, TriggerInformation

from simtools.sim_events.formats.eventio_reader import (
    EventioReader,
    get_corsika_run_and_event_headers,
)

one_two_three = "LSTN-01,LSTN-02,MSTN-01"


@pytest.fixture
def eventio_reader():
    """Reader state for synthetic native records."""
    return EventioReader("synthetic.simtel")


def test_process_array_event(eventio_reader):
    mock_array_event = MagicMock(spec=ArrayEvent)
    mock_array_event.event_id = 42

    mock_trigger = MagicMock(spec=TriggerInformation)
    mock_trigger.parse.return_value = {"triggered_telescopes": [1, 2, 3]}
    mock_tracking = MagicMock(spec=TrackingPosition)
    mock_tracking.parse.return_value = {"altitude_raw": 0.5, "azimuth_raw": 1.2}

    mock_array_event.__iter__.return_value = [mock_trigger, mock_tracking]

    eventio_reader.shower_data.append({"shower_id": 1, "event_id": 42, "file_id": 0})

    with patch.object(
        eventio_reader, "_map_telescope_names", return_value=one_two_three.split(",")
    ):
        eventio_reader._process_array_event(mock_array_event, 0)

    assert len(eventio_reader.trigger_data) == 1
    trigger_event = eventio_reader.trigger_data[0]
    assert trigger_event["shower_id"] == 1
    assert trigger_event["event_id"] == 42
    assert trigger_event["telescope_list"] == one_two_three


def test_process_array_event_empty(eventio_reader):
    mock_array_event = MagicMock(spec=ArrayEvent)
    mock_array_event.__iter__.return_value = []

    # Initial length of trigger data
    initial_len = len(eventio_reader.trigger_data)
    eventio_reader._process_array_event(mock_array_event, 0)

    # Verify no data was added
    assert len(eventio_reader.trigger_data) == initial_len


def test_get_nsb_level_from_file_name(eventio_reader):
    assert eventio_reader._get_nsb_level_from_file_name("dark_file.simtel.zst") == pytest.approx(
        0.24
    )

    assert eventio_reader._get_nsb_level_from_file_name(
        "half_nsb_file.simtel.zst"
    ) == pytest.approx(0.835)

    assert eventio_reader._get_nsb_level_from_file_name(
        "gamma_full_moon_file.simtel.zst"
    ) == pytest.approx(1.2)

    assert eventio_reader._get_nsb_level_from_file_name("DARK_FILE.simtel.zst") == pytest.approx(
        0.24
    )

    assert eventio_reader._get_nsb_level_from_file_name(
        "gamma_run_moon+magic.simtel.zst"
    ) == pytest.approx(0.835)


def test_get_nsb_level_from_file_name_unknown(eventio_reader):
    with pytest.raises(ValueError, match="Cannot determine NSB level"):
        eventio_reader._get_nsb_level_from_file_name("file.simtel.zst")


def test_get_nsb_level_from_file_name_invalid_input(eventio_reader):
    with pytest.raises(AttributeError, match=r"Invalid file name."):
        eventio_reader._get_nsb_level_from_file_name(None)


def test_get_nsb_level_from_metadata_invalid_value_falls_back(mocker, eventio_reader, caplog):
    mocker.patch(
        "simtools.sim_events.formats.eventio_reader.read_sim_telarray_metadata",
        return_value=({"nsb_integrated_flux": "not-a-number"}, {}),
    )
    fallback = mocker.patch.object(
        eventio_reader,
        "_get_nsb_level_from_file_name",
        return_value=0.24,
    )

    with caplog.at_level("WARNING"):
        nsb = eventio_reader.get_nsb_level_from_sim_telarray_metadata("dummy_dark.simtel.zst")

    fallback.assert_called_once_with("dummy_dark.simtel.zst")
    assert "Invalid nsb_integrated_flux value 'not-a-number'" in caplog.text
    assert nsb == pytest.approx(0.24)


def test_process_mc_event(eventio_reader):
    eventio_reader.n_use = 2
    eventio_reader.shower_data = [
        {"shower_id": 1, "event_id": None},
        {"shower_id": 1, "event_id": None},
    ]

    mock_event = MagicMock(spec=MCEvent)
    mock_event.parse.return_value = {
        "event_id": 1001,
        "shower_num": 1,
        "xcore": 100.0,
        "ycore": 200.0,
        "aweight": 1.5,
    }

    eventio_reader._process_mc_event(mock_event)

    updated_event = eventio_reader.shower_data[1]  # event_id is 10001
    assert updated_event["event_id"] == 1001
    assert updated_event["x_core"] == pytest.approx(100.0)
    assert updated_event["y_core"] == pytest.approx(200.0)
    assert updated_event["area_weight"] == pytest.approx(1.5)


def test_process_mc_event_inconsistent_shower(eventio_reader):
    eventio_reader.n_use = 2
    eventio_reader.shower_data = [
        {"shower_id": 1, "event_id": None},
        {"shower_id": 1, "event_id": None},
    ]

    # Create mock MC event with mismatched shower number
    mock_event = MagicMock(spec=MCEvent)
    mock_event.parse.return_value = {
        "event_id": 1001,
        "shower_num": 2,  # Different from shower_id in data
        "xcore": 100.0,
        "ycore": 200.0,
        "aweight": 1.5,
    }

    with pytest.raises(IndexError, match="Inconsistent shower and MC event data for shower id 2"):
        eventio_reader._process_mc_event(mock_event)

    mock_event.parse.return_value = {
        "event_id": 109999,
        "shower_num": 2,  # Different from shower_id in data
        "xcore": 100.0,
        "ycore": 200.0,
        "aweight": 1.5,
    }

    with pytest.raises(IndexError, match="Inconsistent shower and MC event data for shower id 2"):
        eventio_reader._process_mc_event(mock_event)


def test_process_mc_shower_from_iact_simple(eventio_reader):
    mock_eventio_object = MagicMock()
    mock_eventio_object.parse.return_value = {
        "n_reuse": 2,
        "event_number": 7,
        "total_energy": 42.0,
        "reuse_x": [100.0, 200.0],
        "reuse_y": [300.0, 400.0],
        "azimuth": 0.1,
        "angle_array_x_magnetic_north": 0.05,
        "zenith": 0.2,
    }

    eventio_reader._process_mc_shower_from_iact(mock_eventio_object, 1)

    assert len(eventio_reader.shower_data) == 2
    assert eventio_reader.shower_data[0]["shower_id"] == 7
    assert eventio_reader.shower_data[0]["event_id"] == 700
    assert eventio_reader.shower_data[1]["event_id"] == 701
    assert eventio_reader.shower_data[0]["simulated_energy"] == pytest.approx(0.042)
    assert eventio_reader.shower_data[0]["x_core"] == pytest.approx(1.0)
    assert eventio_reader.shower_data[1]["x_core"] == pytest.approx(2.0)
    assert eventio_reader.shower_data[0]["y_core"] == pytest.approx(3.0)
    assert eventio_reader.shower_data[1]["y_core"] == pytest.approx(4.0)
    assert eventio_reader.shower_data[0]["file_id"] == 1
    assert eventio_reader.shower_data[1]["file_id"] == 1
    assert eventio_reader.shower_data[0]["area_weight"] == pytest.approx(1.0)
    assert eventio_reader.shower_data[1]["area_weight"] == pytest.approx(1.0)


def test_process_file_info_else(monkeypatch, tmp_test_directory):
    file_path = Path(tmp_test_directory) / "test.iact"
    file_path.touch()

    fake_run_header = {"x_scatter": 10000.0}
    fake_event_header = {
        "particle_id": 3,
        "energy_spectrum_slope": -2.3,
        "energy_min": 0.5,
        "energy_max": 5.0,
        "zenith": 0.5,
        "azimuth": 1.0,
        "angle_array_x_magnetic_north": 0.1,
        "viewcone_inner_angle": 0.1,
        "viewcone_outer_angle": 0.2,
    }

    monkeypatch.setattr(
        "simtools.sim_events.formats.eventio_reader.get_corsika_run_and_event_headers",
        lambda f: (fake_run_header, fake_event_header),
    )

    writer = EventioReader(str(file_path))
    writer._process_file_info(1, str(file_path))

    assert len(writer.file_info) == 1
    info = writer.file_info[0]
    assert info["file_name"] == str(file_path)
    assert info["file_id"] == 1
    assert info["particle_id"] == 3
    assert info["spectral_index"] == pytest.approx(-2.3)
    assert info["energy_min"] == pytest.approx(0.0005)
    assert info["energy_max"] == pytest.approx(0.005)
    assert info["viewcone_min"] == pytest.approx(0.1)
    assert info["viewcone_max"] == pytest.approx(0.2)
    assert info["core_scatter_min"] == pytest.approx(0.0)
    assert info["core_scatter_max"] == pytest.approx(100.0)
    assert info["zenith"] == pytest.approx(28.64788975654116)
    assert info["azimuth"] == pytest.approx(180.0 - np.rad2deg(0.9))
    assert info["nsb_level"] == pytest.approx(0.0)


def test_read_simulation_parameters(mocker):
    run = {
        "n_showers": 7,
        "n_observation_levels": 1,
        "observation_height": [220000],
    }
    event = {
        "theta_min": 20,
        "theta_max": 40,
        "phi_min": 60,
        "phi_max": 60,
        "angle_array_x_magnetic_north": np.deg2rad(-4),
        "particle_id": 14,
        "n_reuse": 3,
        "viewcone_outer_angle": 5,
    }
    reader = EventioReader("showers.iact")
    mocker.patch.object(reader, "read_run_headers", return_value=(run, event))
    parameters = reader.read_simulation_parameters()
    assert parameters["zenith_angle"] == pytest.approx(30)
    assert parameters["azimuth_angle"] == pytest.approx(116)
    assert parameters["zenith_min"] == pytest.approx(20)
    assert parameters["observation_levels"] == pytest.approx([2200])
    assert parameters["shower_events"] == 7
    assert parameters["mc_events"] == 21
    assert parameters["primary_particle"].name == "proton"


@pytest.mark.parametrize("headers", [(None, None), ({}, None)])
def test_missing_simulation_headers(headers, mocker):
    reader = EventioReader("invalid.iact")
    mocker.patch.object(reader, "read_run_headers", return_value=headers)
    with pytest.raises(ValueError, match="Missing IACT run or event header"):
        reader.read_simulation_parameters()


def test_reader_flushes_complete_reused_showers(eventio_objects, mocker):
    reader = EventioReader("simulation.simtel")
    mocker.patch(
        "simtools.sim_events.formats.eventio_reader.read_sim_telarray_metadata",
        return_value=({"nsb_integrated_flux": 0.24}, {}),
    )
    mocker.patch(
        "simtools.sim_events.formats.eventio_reader.get_sim_telarray_telescope_id_to_telescope_name_mapping",
        return_value={1: "LSTN-01"},
    )
    objects = [eventio_objects["run_header"]()]
    for shower_id in (1, 2):
        objects.extend(
            [
                eventio_objects["shower"](shower_id),
                eventio_objects["event"](shower_id, shower_id * 100),
                eventio_objects["event"](shower_id, shower_id * 100 + 1),
            ]
        )
    file = mocker.patch("simtools.sim_events.formats.eventio_reader.EventIOFile")
    file.return_value.__enter__.return_value = objects
    rows = list(reader.iter_records(file_id=4))
    showers = [row for name, row in rows if name == "SHOWERS"]
    assert [row["event_id"] for row in showers] == [100, 101, 200, 201]
    assert all(row["x_core"] == pytest.approx(0.1) for row in showers)
    assert all(row["file_id"] == 4 for row in showers)
    assert reader.shower_data == []
    assert reader.read_metadata()["run_number"] == 123


def test_iact_numpy_headers_are_mappings(mocker):
    run = np.array((7.0,), dtype=[("run_number", "f4")])[()]
    event = np.array((7.0, 100.0), dtype=[("run_number", "f4"), ("energy_min", "f4")])[()]
    run_object = mocker.Mock(spec=iact.RunHeader)
    run_object.parse.return_value = run
    event_object = mocker.Mock(spec=iact.EventHeader)
    event_object.parse.return_value = event
    file = mocker.patch("simtools.sim_events.formats.eventio_reader.EventIOFile")
    file.return_value.__enter__.return_value = [run_object, event_object]
    actual_run, actual_event = get_corsika_run_and_event_headers("showers.iact")
    assert actual_run == {"run_number": 7.0}
    assert actual_event == {"run_number": 7.0, "energy_min": 100.0}


def test_iact_provenance_retains_software_and_run_header(mocker):
    reader = EventioReader("showers.iact")
    mocker.patch.object(reader, "read_run_headers", return_value=({"run_number": 7}, {}))
    reader._record_input_metadata(0, "showers.iact", None)
    metadata = reader.read_metadata()
    assert metadata["simulation_software"] == "corsika"
    assert metadata["run_number"] == 7
    assert metadata["run_header"] == {"run_number": 7}
