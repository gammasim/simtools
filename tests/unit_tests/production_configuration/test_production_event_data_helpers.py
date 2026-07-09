"""Tests for production event-data helpers."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from simtools.production_configuration import production_event_data_helpers as helpers


def test_normalize_event_data_file_accepts_str_and_list():
    assert helpers.normalize_event_data_file("pattern*.hdf5") == ["pattern*.hdf5"]
    assert helpers.normalize_event_data_file(["a*.hdf5", "b*.hdf5"]) == ["a*.hdf5", "b*.hdf5"]

    with pytest.raises(TypeError, match="event_data_file must be str or list"):
        helpers.normalize_event_data_file(("a*.hdf5",))


def test_build_production_subdirectories_creates_unique_names(tmp_test_directory, mocker):
    output_dir = tmp_test_directory / "plots"
    mocker.patch(
        "simtools.production_configuration.production_event_data_helpers.get_uuid",
        side_effect=["uuid-1", "uuid-2"],
    )

    subdirs = helpers.build_production_subdirectories(
        ["a/output.hdf5", "b/output.hdf5", "c/output.hdf5"], output_dir
    )

    assert [Path(path).name for path in subdirs.values()] == [
        "production_output",
        "production_output_uuid-1",
        "production_output_uuid-2",
    ]
    assert all(Path(path).is_dir() for path in subdirs.values())
    mocker.patch(
        "simtools.production_configuration.production_event_data_helpers.get_uuid",
        return_value="uuid-1",
    )
    assert helpers.get_production_directory_name("a/output.hdf5", {"production_output"}) == (
        "production_output_uuid-1"
    )


def test_resolve_telescope_configs_prefers_layout_and_falls_back_to_array_elements(mocker):
    mock_resolve = mocker.patch(
        "simtools.production_configuration.production_event_data_helpers.resolve_array_layout_name",
        return_value="alpha",
    )
    mock_get = mocker.patch(
        "simtools.production_configuration.production_event_data_helpers."
        "get_array_elements_from_db_for_layouts",
        return_value={"alpha": ["LSTN-01"]},
    )

    assert helpers.resolve_telescope_configs(
        {"array_layout_name": "alpha", "model_version": "1.0.0", "site": "North"}
    ) == {"alpha": ["LSTN-01"]}
    mock_resolve.assert_called_once_with("alpha", "1.0.0")
    mock_get.assert_called_once_with(["alpha"], "North", "1.0.0")
    assert helpers.resolve_telescope_configs({"array_element_list": ["LSTN-01"]}) == {
        "array_element_list": ["LSTN-01"]
    }

    with pytest.raises(ValueError, match="No telescope configuration provided"):
        helpers.resolve_telescope_configs({})


def test_normalize_telescope_configs_normalizes_dict_and_list_inputs(mocker):
    mocker.patch(
        "simtools.production_configuration.production_event_data_helpers."
        "normalize_array_element_identifier_container",
        side_effect=lambda value: [str(item).upper() for item in value],
    )

    from_dict = helpers.normalize_telescope_configs({"alpha": ["lstn-01"]})
    from_list = helpers.normalize_telescope_configs(
        [{"array_name": "beta", "telescope_ids": ["mstn-01"]}]
    )

    assert from_dict == [{"array_name": "alpha", "telescope_ids": ["LSTN-01"]}]
    assert from_list == [{"array_name": "beta", "telescope_ids": ["MSTN-01"]}]


def test_accumulate_histograms_by_telescope_config_accumulates_and_finalizes(mocker):
    event_source = mocker.Mock()
    reader = mocker.Mock()
    reader.filter_by_telescopes.return_value = ("filtered_data", "filtered_shower")
    event_source.iter_event_data.return_value = [
        ("reader", ("file_info", "shower", "triggered_shower", "triggered_data"))
    ]
    accumulator = mocker.Mock()
    accumulator_all = mocker.Mock()
    mocker.patch(
        "simtools.production_configuration.production_event_data_helpers.EventDataHistograms",
        return_value=event_source,
    )
    mocker.patch(
        "simtools.production_configuration.production_event_data_helpers."
        "EventDataHistograms.create_accumulator",
        side_effect=[accumulator, accumulator_all],
    )
    event_source.iter_event_data.return_value = [
        (reader, ("file_info", "shower", "triggered_shower", "triggered_data"))
    ]

    result = helpers.accumulate_histograms_by_telescope_config(
        "prod*.hdf5",
        [
            {"array_name": "alpha", "telescope_ids": ["LSTN-01"]},
            {"array_name": "all", "telescope_ids": []},
        ],
        energy_bins_per_decade=4,
        fill_efficiency_histogram=True,
    )

    reader.filter_by_telescopes.assert_called_once_with(
        "triggered_data",
        "triggered_shower",
        telescope_list=["LSTN-01"],
    )
    accumulator.accumulate.assert_called_once_with(
        "file_info", "shower", "filtered_shower", "filtered_data"
    )
    accumulator_all.accumulate.assert_called_once_with(
        "file_info", "shower", "triggered_shower", "triggered_data"
    )
    accumulator.finalize.assert_called_once_with(fill_efficiency_histogram=True)
    accumulator_all.finalize.assert_called_once_with(fill_efficiency_histogram=True)
    assert result == [
        ({"array_name": "alpha", "telescope_ids": ["LSTN-01"]}, accumulator),
        ({"array_name": "all", "telescope_ids": []}, accumulator_all),
    ]


def test_accumulate_histograms_by_telescope_config_collects_trigger_topology(mocker):
    event_source = mocker.Mock()
    reader = mocker.Mock()
    event_source.iter_event_data.return_value = [(reader, ("file_info", "shower", None, None))]
    accumulator = mocker.Mock()
    topology = {"marker": 1}
    mocker.patch(
        "simtools.production_configuration.production_event_data_helpers.EventDataHistograms",
        return_value=event_source,
    )
    mocker.patch(
        "simtools.production_configuration.production_event_data_helpers."
        "EventDataHistograms.create_accumulator",
        return_value=accumulator,
    )
    mock_init = mocker.patch(
        "simtools.production_configuration.production_event_data_helpers."
        "_initialize_trigger_topology_accumulator",
        return_value=topology,
    )
    mock_accumulate = mocker.patch(
        "simtools.production_configuration.production_event_data_helpers."
        "_accumulate_trigger_topology"
    )

    result = helpers.accumulate_histograms_by_telescope_config(
        "prod*.hdf5",
        [{"array_name": "alpha", "telescope_ids": []}],
        energy_bins_per_decade=4,
        collect_trigger_topology=True,
    )

    mock_init.assert_called_once_with()
    mock_accumulate.assert_called_once_with(None, None, topology, allowed_telescopes=[])
    assert result == [({"array_name": "alpha", "telescope_ids": []}, accumulator, topology)]


def test_trigger_topology_helpers_track_counts_and_subsets(mocker):
    mocker.patch(
        "simtools.production_configuration.production_event_data_helpers."
        "names.get_array_element_type_from_name",
        side_effect=lambda telescope: telescope[:3],
    )
    accumulator = helpers._initialize_trigger_topology_accumulator()
    shower_data = SimpleNamespace(
        simulated_energy=[0.2, 0.5],
        core_distance_shower=[10.0, 20.0],
        angular_distance=[0.1, 0.2],
    )
    triggered_data = SimpleNamespace(telescope_list=[["LST-1"], ["LST-1", "MST-1"]])

    helpers._accumulate_trigger_topology(shower_data, triggered_data, accumulator)

    assert accumulator["trigger_multiplicity"] == {1: 1, 2: 1}
    assert accumulator["trigger_combinations"] == {"LST-1": 1, "LST-1,MST-1": 1}
    assert accumulator["telescope_participation"] == {"LST-1": 2, "MST-1": 1}
    assert accumulator["subset_multiplicity"]["single_telescope"] == {1: 1}
    assert accumulator["subset_multiplicity"]["mixed_type"] == {2: 1}
    assert accumulator["subset_values"]["energy"]["mixed_type"] == pytest.approx([0.5])
    assert helpers._subset_counts_for_trigger(("LST-1", "LST-2")) == {"LST": 2}
    helpers._accumulate_trigger_topology(None, triggered_data, accumulator)


def test_normalize_trigger_telescopes_filters_unknown_and_non_selected(mocker):
    mocker.patch(
        "simtools.production_configuration.production_event_data_helpers."
        "names.get_array_element_type_from_name",
        side_effect=lambda telescope: {"LST-1": "LST", "MST-1": "MST"}[telescope],
    )

    normalized = helpers._normalize_trigger_telescopes(
        ["Unknown_30", "MST-1", "LST-1"],
        {"LST-1", "MST-1"},
    )

    assert normalized == ("LST-1", "MST-1")


def test_accumulate_trigger_topology_ignores_unknown_and_non_selected_telescopes(mocker):
    mocker.patch(
        "simtools.production_configuration.production_event_data_helpers."
        "names.get_array_element_type_from_name",
        side_effect=lambda telescope: {"LST-1": "LST", "MST-1": "MST"}[telescope],
    )
    accumulator = helpers._initialize_trigger_topology_accumulator()
    shower_data = SimpleNamespace(
        simulated_energy=[0.2, 0.5],
        core_distance_shower=[10.0, 20.0],
        angular_distance=[0.1, 0.2],
    )
    triggered_data = SimpleNamespace(
        telescope_list=[["LST-1", "Unknown_30"], ["LST-1", "MST-1", "Unknown_30"]]
    )

    helpers._accumulate_trigger_topology(
        shower_data,
        triggered_data,
        accumulator,
        allowed_telescopes=["LST-1", "MST-1"],
    )

    assert accumulator["trigger_multiplicity"] == {1: 1, 2: 1}
    assert accumulator["trigger_combinations"] == {"LST-1": 1, "LST-1,MST-1": 1}
