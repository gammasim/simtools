from pathlib import Path

import astropy.units as u
import h5py
import numpy as np
import pytest
from astropy.table import Table

from simtools.io import table_handler
from simtools.sim_events import output_validator
from simtools.sim_events.output_validator import (
    validate_event_numbers,
    validate_reduced_event_data_file,
    validate_sim_events,
    validate_trigger_histogram_file,
)


def _reduced_event_tables(trigger_shower_id=1, duplicate_shower=False):
    """Return minimal linked reduced-event tables for structural validation tests."""
    shower_ids = [1, 1] if duplicate_shower else [1]
    return {
        "SHOWERS": Table(
            {
                "file_id": np.array([1] * len(shower_ids), dtype=np.uint32),
                "event_id": [2] * len(shower_ids),
                "shower_id": shower_ids,
            }
        ),
        "TRIGGERS": Table(
            {
                "file_id": np.array([1], dtype=np.uint32),
                "event_id": [2],
                "shower_id": [trigger_shower_id],
            }
        ),
        "FILE_INFO": Table({"file_id": np.array([1], dtype=np.uint32)}),
    }


def _mock_reduced_event_dependencies(mocker, tables):
    """Mock file access and table-schema details for reduced-event validator tests."""
    mocker.patch(
        "simtools.sim_events.output_validator.table_handler.read_table_file_type",
        return_value="HDF5",
    )
    mocker.patch(
        "simtools.sim_events.output_validator.table_handler.read_table_list",
        return_value={
            name: [name]
            for name in ("SHOWERS", "TRIGGERS", "FILE_INFO", "METADATA", "SIMULATION_METADATA")
        },
    )
    mocker.patch(
        "simtools.sim_events.output_validator.table_handler.read_tables", return_value=tables
    )
    mocker.patch(
        "simtools.sim_events.output_validator.table_handler.read_metadata_document", return_value={}
    )
    mocker.patch("simtools.sim_events.output_validator.schema.validate_dict_using_schema")
    mocker.patch("simtools.sim_events.output_validator.validate_simulation_metadata")
    return mocker.patch("simtools.sim_events.output_validator.validate_data.DataValidator")


def test_validate_reduced_event_data_file_validates_all_tables(mocker):
    mock_validator = _mock_reduced_event_dependencies(mocker, _reduced_event_tables())

    assert validate_reduced_event_data_file("events.hdf5") is True
    assert mock_validator.call_count == 3


def test_validate_reduced_event_data_file_rejects_unmatched_trigger(mocker):
    _mock_reduced_event_dependencies(mocker, _reduced_event_tables(trigger_shower_id=3))

    with pytest.raises(ValueError, match="triggers without matching showers"):
        validate_reduced_event_data_file("events.hdf5")


def test_validate_reduced_event_data_file_rejects_missing_entries(mocker):
    _mock_reduced_event_dependencies(mocker, _reduced_event_tables())
    mocker.patch(
        "simtools.sim_events.output_validator.table_handler.read_table_list",
        return_value={"SHOWERS": ["SHOWERS"], "TRIGGERS": [], "FILE_INFO": ["FILE_INFO"]},
    )

    with pytest.raises(ValueError, match="missing required entries: TRIGGERS"):
        validate_reduced_event_data_file("events.hdf5")


def test_validate_reduced_event_data_file_rejects_non_hdf5_input(mocker):
    mocker.patch(
        "simtools.sim_events.output_validator.table_handler.read_table_file_type",
        return_value="FITS",
    )

    with pytest.raises(ValueError, match="must be an HDF5 file"):
        validate_reduced_event_data_file("events.fits")


def test_validate_reduced_event_data_file_rejects_unknown_file_id(mocker):
    tables = _reduced_event_tables()
    tables["TRIGGERS"]["file_id"][0] = 3
    _mock_reduced_event_dependencies(mocker, tables)

    with pytest.raises(ValueError, match=r"references unknown file_id values: \[3\]"):
        validate_reduced_event_data_file("events.hdf5")


def test_validate_reduced_event_data_file_rejects_duplicate_shower_key(mocker):
    _mock_reduced_event_dependencies(mocker, _reduced_event_tables(duplicate_shower=True))

    with pytest.raises(ValueError, match="duplicate shower composite keys"):
        validate_reduced_event_data_file("events.hdf5")


def _write_trigger_histogram_file(output_file, topology_reference_id="reference_0"):
    """Write a minimal, structurally valid trigger-histogram file."""
    metadata = Table(
        {
            "reference_id": ["reference_0"],
            "production_index": np.array([0], dtype=np.int64),
            "site": ["North"],
            "array_name": ["alpha"],
            "telescope_ids": ["LSTN-01"],
            "primary_particle": ["gamma"],
            "zenith": u.Quantity([20.0], u.deg),
            "azimuth": u.Quantity([0.0], u.deg),
            "nsb_level": np.array([0.0], dtype=np.float64),
            "spectral_index": np.array([-2.0], dtype=np.float64),
            "energy_min": u.Quantity([0.01], u.TeV),
            "energy_max": u.Quantity([100.0], u.TeV),
            "viewcone_min": u.Quantity([0.0], u.deg),
            "viewcone_max": u.Quantity([5.0], u.deg),
            "core_scatter_min": u.Quantity([0.0], u.m),
            "core_scatter_max": u.Quantity([500.0], u.m),
            "scatter_area": u.Quantity([1.0], u.cm**2),
            "solid_angle": u.Quantity([1.0], u.sr),
            "angular_distance_min": u.Quantity([0.0], u.deg),
            "angular_distance_max": u.Quantity([5.0], u.deg),
            "energy_bins_per_decade": np.array([5], dtype=np.int64),
            "angular_distance_bin_width": u.Quantity([1.0], u.deg),
            "angular_distance_bin_count": np.array([5], dtype=np.int64),
            "core_distance_bin_count": np.array([1], dtype=np.int64),
            "total_simulated_events": np.array([1], dtype=np.int64),
            "total_triggered_events": np.array([1], dtype=np.int64),
        }
    )
    metadata.meta["EXTNAME"] = "TRIGGER_REFERENCE_METADATA"
    bins = Table(
        {
            "reference_id": ["reference_0"],
            "production_index": np.array([0], dtype=np.int64),
            "array_name": ["alpha"],
            "angular_distance_bin_index": np.array([0], dtype=np.int64),
            "energy_bin_index": np.array([0], dtype=np.int64),
            "core_distance_bin_index": np.array([0], dtype=np.int64),
            "angular_distance_low": u.Quantity([0.0], u.deg),
            "angular_distance_high": u.Quantity([1.0], u.deg),
            "energy_low": u.Quantity([0.01], u.TeV),
            "energy_high": u.Quantity([0.1], u.TeV),
            "core_distance_low": u.Quantity([0.0], u.m),
            "core_distance_high": u.Quantity([500.0], u.m),
            "simulated_count": np.array([1], dtype=np.int64),
            "triggered_count": np.array([1], dtype=np.int64),
            "trigger_efficiency": np.array([1.0], dtype=np.float64),
        }
    )
    bins.meta["EXTNAME"] = "TRIGGER_REFERENCE_BINS"
    topology = Table(
        {
            "reference_id": [topology_reference_id],
            "count_type": ["trigger_multiplicity"],
            "subset": [""],
            "key": ["1"],
            "count": np.array([1], dtype=np.int64),
        }
    )
    topology.meta["EXTNAME"] = "TRIGGER_TOPOLOGY_COUNTS"
    subset = Table(
        {
            "reference_id": ["reference_0"],
            "subset": ["all"],
            "quantity": ["energy"],
            "bin_index": np.array([0], dtype=np.int64),
            "bin_low": np.array([0.01], dtype=np.float64),
            "bin_high": np.array([0.1], dtype=np.float64),
            "count": np.array([1], dtype=np.int64),
        }
    )
    subset.meta["EXTNAME"] = "TRIGGER_SUBSET_HISTOGRAMS"
    table_handler.write_tables(
        [metadata, bins, topology, subset],
        output_file,
        file_type="HDF5",
        metadata_documents={"METADATA": {"cta": {}}},
    )
    import h5py

    with h5py.File(output_file, "a") as hdf5_file:
        histogram_group = hdf5_file.create_group("TRIGGER_HISTOGRAM_DENSE/reference_0/energy")
        histogram_group.create_dataset("values", data=[1])
        histogram_group.create_dataset("edges_0", data=[0.1, 1.0])


def test_validate_trigger_histogram_file_validates_references_and_payload(
    tmp_test_directory, mocker
):
    output_file = Path(tmp_test_directory) / "trigger_histograms.hdf5"
    _write_trigger_histogram_file(output_file)
    mocker.patch("simtools.sim_events.output_validator.schema.validate_dict_using_schema")

    assert validate_trigger_histogram_file(output_file) is True


def test_validate_trigger_histogram_file_rejects_unknown_table_reference(
    tmp_test_directory, mocker
):
    output_file = Path(tmp_test_directory) / "trigger_histograms.hdf5"
    _write_trigger_histogram_file(output_file, topology_reference_id="missing")
    mocker.patch("simtools.sim_events.output_validator.schema.validate_dict_using_schema")

    with pytest.raises(
        ValueError,
        match=r"TRIGGER_TOPOLOGY_COUNTS.*unknown reference IDs: \['missing'\]",
    ):
        validate_trigger_histogram_file(output_file)


def test_validate_trigger_histogram_file_rejects_non_hdf5_input(mocker):
    mocker.patch(
        "simtools.sim_events.output_validator.table_handler.read_table_file_type",
        return_value="FITS",
    )

    with pytest.raises(ValueError, match="must be an HDF5 file"):
        validate_trigger_histogram_file("trigger_histograms.fits")


def test_validate_trigger_histogram_file_rejects_missing_entries(mocker):
    mocker.patch(
        "simtools.sim_events.output_validator.table_handler.read_table_file_type",
        return_value="HDF5",
    )
    mocker.patch(
        "simtools.sim_events.output_validator.table_handler.read_table_list",
        return_value={
            "TRIGGER_REFERENCE_METADATA": [],
            "TRIGGER_REFERENCE_BINS": ["TRIGGER_REFERENCE_BINS"],
            "TRIGGER_TOPOLOGY_COUNTS": ["TRIGGER_TOPOLOGY_COUNTS"],
            "TRIGGER_SUBSET_HISTOGRAMS": ["TRIGGER_SUBSET_HISTOGRAMS"],
        },
    )

    with pytest.raises(ValueError, match="missing required entries: TRIGGER_REFERENCE_METADATA"):
        validate_trigger_histogram_file("trigger_histograms.hdf5")


@pytest.mark.parametrize(
    ("metadata_ids", "bin_ids", "message"),
    [
        ([], ["reference_0"], "no reference metadata rows"),
        (["reference_0", "reference_0"], ["reference_0"], "duplicate reference IDs"),
        (["reference_0"], [], "references without histogram bins"),
    ],
)
def test_validate_trigger_histogram_table_references_rejects_invalid_ids(
    metadata_ids, bin_ids, message
):
    tables = {
        "TRIGGER_REFERENCE_METADATA": Table({"reference_id": metadata_ids}),
        "TRIGGER_REFERENCE_BINS": Table({"reference_id": bin_ids}),
        "TRIGGER_TOPOLOGY_COUNTS": Table({"reference_id": metadata_ids[:1]}),
        "TRIGGER_SUBSET_HISTOGRAMS": Table({"reference_id": metadata_ids[:1]}),
    }

    with pytest.raises(ValueError, match=message):
        output_validator._validate_trigger_histogram_table_references(tables, "output.hdf5")


def test_dense_histogram_group_validation_rejects_missing_group(tmp_test_directory):
    output_file = Path(tmp_test_directory) / "missing_group.hdf5"
    with h5py.File(output_file, "w") as hdf5_file:
        with pytest.raises(ValueError, match="no dense payload group"):
            output_validator._get_dense_histogram_group(hdf5_file, output_file)


def test_dense_histogram_group_validation_rejects_non_group(tmp_test_directory):
    output_file = Path(tmp_test_directory) / "invalid_group.hdf5"
    with h5py.File(output_file, "w") as hdf5_file:
        hdf5_file.create_dataset("TRIGGER_HISTOGRAM_DENSE", data=[1])
        with pytest.raises(ValueError, match="dense payload is not a group"):
            output_validator._get_dense_histogram_group(hdf5_file, output_file)


def test_dense_histogram_validation_rejects_reference_mismatch(tmp_test_directory):
    output_file = Path(tmp_test_directory) / "reference_mismatch.hdf5"
    with h5py.File(output_file, "w") as hdf5_file:
        dense_group = hdf5_file.create_group("dense")
        dense_group.create_group("other")
        with pytest.raises(ValueError, match="do not match metadata"):
            output_validator._validate_dense_reference_ids(
                dense_group, {"reference_0"}, output_file
            )


def test_dense_histogram_validation_rejects_empty_reference_payload(tmp_test_directory):
    output_file = Path(tmp_test_directory) / "empty_reference.hdf5"
    with pytest.raises(ValueError, match="has no payload"):
        output_validator._validate_dense_reference_payload("reference_0", None, output_file)


@pytest.mark.parametrize("payload_kind", ["not_group", "no_values", "invalid_values"])
def test_dense_histogram_validation_rejects_invalid_payload(tmp_test_directory, payload_kind):
    output_file = Path(tmp_test_directory) / f"{payload_kind}.hdf5"
    with h5py.File(output_file, "w") as hdf5_file:
        if payload_kind == "not_group":
            payload = hdf5_file.create_dataset("payload", data=[1])
            expected_message = "is not a group"
        else:
            payload = hdf5_file.create_group("payload")
            if payload_kind == "invalid_values":
                payload.create_dataset("values", data=1)
                expected_message = "invalid values data"
            else:
                expected_message = "has no values dataset"
        with pytest.raises(ValueError, match=expected_message):
            output_validator._validate_dense_histogram_payload("histogram", payload, output_file)


def test_validate_trigger_histogram_file_rejects_missing_table_column(tmp_test_directory, mocker):
    output_file = Path(tmp_test_directory) / "trigger_histograms.hdf5"
    table_names = (
        "TRIGGER_REFERENCE_METADATA",
        "TRIGGER_REFERENCE_BINS",
        "TRIGGER_TOPOLOGY_COUNTS",
        "TRIGGER_SUBSET_HISTOGRAMS",
    )
    tables = []
    for table_name in table_names:
        table = Table({"reference_id": ["reference_0"]})
        table.meta["EXTNAME"] = table_name
        tables.append(table)
    table_handler.write_tables(
        tables,
        output_file,
        file_type="HDF5",
        metadata_documents={"METADATA": {"cta": {}}},
    )
    mocker.patch("simtools.sim_events.output_validator.schema.validate_dict_using_schema")

    with pytest.raises(KeyError, match="Missing required column production_index"):
        validate_trigger_histogram_file(output_file)


@pytest.mark.parametrize("malformation", ["name", "length", "group", "nonnumeric"])
def test_validate_trigger_histogram_file_rejects_malformed_edges(
    tmp_test_directory, mocker, malformation
):
    output_file = Path(tmp_test_directory) / f"trigger_histograms_{malformation}.hdf5"
    _write_trigger_histogram_file(output_file)
    mocker.patch("simtools.sim_events.output_validator.schema.validate_dict_using_schema")

    import h5py

    with h5py.File(output_file, "a") as hdf5_file:
        histogram_group = hdf5_file["TRIGGER_HISTOGRAM_DENSE/reference_0/energy"]
        del histogram_group["edges_0"]
        if malformation == "name":
            histogram_group.create_dataset("edges_bad", data=[0.1, 1.0])
        elif malformation == "length":
            histogram_group.create_dataset("edges_0", data=[0.1])
        elif malformation == "nonnumeric":
            histogram_group.create_dataset("edges_0", data=np.array(["a", "b"], dtype="S1"))
        else:
            histogram_group.create_group("edges_0")

    with pytest.raises(ValueError, match=r"bin-edge|one-dimensional|expected|numeric"):
        validate_trigger_histogram_file(output_file)


def test_validate_sim_events_calls_validate_event_numbers(tmp_path, monkeypatch):
    test_files = [tmp_path / f"test_{i}.hdf5" for i in range(2)]
    for f in test_files:
        f.touch()

    mock_tables = {"SHOWERS": [1, 2, 3, 4]}
    monkeypatch.setattr(
        "simtools.sim_events.output_validator.table_handler.read_tables",
        lambda *args, **kwargs: mock_tables,
    )

    assert validate_sim_events(test_files, 4) is None


def test_validate_sim_events_rejects_fits_files(tmp_path):
    test_file = tmp_path / "test_events.fits"
    test_file.touch()

    with pytest.raises(ValueError, match="Only HDF5 files"):
        validate_sim_events(str(test_file), 3)


def test_validate_event_numbers_single_file_mismatch(tmp_path, monkeypatch):
    test_file = tmp_path / "test_events.hdf5"
    test_file.touch()

    mock_tables = {"SHOWERS": [1, 2, 3]}
    monkeypatch.setattr(
        "simtools.sim_events.output_validator.table_handler.read_tables",
        lambda *args, **kwargs: mock_tables,
    )

    with pytest.raises(ValueError, match="Inconsistent event counts found in reduced event lists"):
        validate_event_numbers(str(test_file), 5)


def test_validate_event_numbers_missing_showers_table(tmp_path, monkeypatch):
    test_file = tmp_path / "test_events.hdf5"
    test_file.touch()

    mock_tables = {"OTHER_TABLE": [1, 2, 3]}
    monkeypatch.setattr(
        "simtools.sim_events.output_validator.table_handler.read_tables",
        lambda *args, **kwargs: mock_tables,
    )

    with pytest.raises(ValueError, match="SHOWERS table not found in reduced event list"):
        validate_event_numbers(str(test_file), 3)
