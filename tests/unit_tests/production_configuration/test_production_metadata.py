"""Tests for production metadata backfill and validation workflows."""

import struct
from pathlib import Path
from types import SimpleNamespace

import astropy.units as u
import h5py
import pytest
from astropy.tests.helper import assert_quantity_allclose

from simtools.io.ascii_handler import write_data_to_file
from simtools.production_configuration import jobs_io, production_metadata
from simtools.production_configuration.production_file_selection import (
    ProductionManifest,
    check_manifest,
)
from simtools.production_configuration.production_metadata import (
    REQUIRED_SIMULATION_JOB_METADATA_ARGUMENTS,
    _primary_name,
    _resolved_model_parameter_overrides,
    _showers_per_run,
    build_production_job_manifest,
    build_simulation_job_metadata,
    get_sim_telarray_event_counts,
)


def _job_grid_row():
    """Return one complete authoritative job-grid row."""
    return {
        "run_number": 1,
        "primary": "gamma",
        "azimuth_angle": 180 * u.deg,
        "zenith_angle": 20 * u.deg,
        "energy_min": 0.03 * u.TeV,
        "energy_max": 300 * u.TeV,
        "cores_per_shower": 10,
        "core_scatter_max": 500 * u.m,
        "view_cone_min": 0 * u.deg,
        "view_cone_max": 10 * u.deg,
        "showers_per_run": 100,
        "model_version": "7.0.0",
        "array_layout_name": "CTAO-North-Alpha",
        "corsika_le_interaction": "urqmd",
        "corsika_he_interaction": "qgs3",
    }


def test_check_existing_manifests_reports_missing_job_manifest(tmp_test_directory):
    production_path = Path(tmp_test_directory)
    with pytest.raises(FileNotFoundError, match="No production job directories"):
        production_metadata._check_existing_manifests(production_path)

    (production_path / "job-000001").mkdir()

    with pytest.raises(FileNotFoundError, match="Missing production metadata manifest"):
        production_metadata._check_existing_manifests(production_path)


def test_write_production_metadata_dispatches_check_and_write_modes(mocker, tmp_test_directory):
    production_path = Path(tmp_test_directory)
    mock_check = mocker.patch(
        "simtools.production_configuration.production_metadata._check_existing_manifests"
    )
    mock_write = mocker.patch(
        "simtools.production_configuration.production_metadata._write_manifests"
    )

    production_metadata.write_production_metadata(
        {"production_path": str(production_path), "check": True}
    )
    production_metadata.write_production_metadata(
        {"production_path": str(production_path), "job_grid_file": "job-grid.ecsv"}
    )

    mock_check.assert_called_once_with(production_path)
    mock_write.assert_called_once_with(production_path, "job-grid.ecsv", mocker.ANY, None)


def test_check_existing_manifests_validates_each_manifest(mocker, tmp_test_directory):
    production_path = Path(tmp_test_directory)
    job_directory = production_path / "job-000001"
    job_directory.mkdir()
    manifest_path = job_directory / production_metadata.SIMULATE_PROD_JOB_METADATA
    manifest_path.touch()
    mock_check = mocker.patch(
        "simtools.production_configuration.production_metadata.check_manifest"
    )

    production_metadata._check_existing_manifests(production_path)

    mock_check.assert_called_once_with(manifest_path)


def test_write_manifests_does_not_mark_incomplete_job_complete(
    mocker, tmp_test_directory, run_context
):
    production_path = Path(tmp_test_directory)
    (production_path / "job-000001").mkdir()
    mocker.patch(
        "simtools.production_configuration.production_metadata.read_job_grid",
        return_value=(
            [_job_grid_row()],
            {"site": "North", "simulation_software": ["corsika", "sim_telarray"]},
        ),
    )
    write_file = mocker.patch(
        "simtools.production_configuration.production_metadata.write_data_to_file"
    )

    with pytest.raises(ValueError, match="Incomplete production job"):
        production_metadata._write_manifests(
            production_path,
            "job-grid.ecsv",
            {"overwrite": False},
            run_context,
        )

    write_file.assert_not_called()


def test_write_manifests_rejects_missing_job_directory(mocker, tmp_test_directory, run_context):
    mocker.patch(
        "simtools.production_configuration.production_metadata.read_job_grid",
        return_value=([_job_grid_row()], {}),
    )

    with pytest.raises(FileNotFoundError, match="Production job directory not found"):
        production_metadata._write_manifests(
            Path(tmp_test_directory), "job-grid.ecsv", {}, run_context
        )


def test_write_manifests_supports_zero_based_job_directories(
    mocker, tmp_test_directory, run_context
):
    production_path = Path(tmp_test_directory)
    job_directory = production_path / "job-000000"
    job_directory.mkdir()
    (job_directory / "gamma_run000001.simtel.zst").touch()
    mocker.patch(
        "simtools.production_configuration.production_metadata.read_job_grid",
        return_value=(
            [_job_grid_row()],
            {"site": "North", "simulation_software": ["sim_telarray"]},
        ),
    )
    write_file = mocker.patch(
        "simtools.production_configuration.production_metadata.write_data_to_file"
    )

    production_metadata._write_manifests(
        production_path, "job-grid.ecsv", {"overwrite": False}, run_context
    )

    assert write_file.call_args.args[1] == job_directory / "simulate_prod_job_metadata.yml"


def test_write_manifests_requires_overwrite_for_existing_manifest(
    mocker, tmp_test_directory, run_context
):
    production_path = Path(tmp_test_directory)
    job_directory = production_path / "job-000001"
    job_directory.mkdir()
    (job_directory / production_metadata.SIMULATE_PROD_JOB_METADATA).touch()
    mocker.patch(
        "simtools.production_configuration.production_metadata.read_job_grid",
        return_value=([_job_grid_row()], {}),
    )

    with pytest.raises(FileExistsError, match="Use --overwrite"):
        production_metadata._write_manifests(production_path, "job-grid.ecsv", {}, run_context)


def test_write_manifests_omits_unknown_catalog_and_atmosphere_values(
    mocker, tmp_test_directory, run_context
):
    production_path = Path(tmp_test_directory)
    job_directory = production_path / "job-000001"
    job_directory.mkdir()
    (job_directory / "gamma_run000001.simtel.zst").touch()
    mocker.patch(
        "simtools.production_configuration.production_metadata.read_job_grid",
        return_value=(
            [_job_grid_row()],
            {"site": "North", "simulation_software": ["corsika", "sim_telarray"]},
        ),
    )
    write_file = mocker.patch(
        "simtools.production_configuration.production_metadata.write_data_to_file"
    )

    production_metadata._write_manifests(
        production_path,
        "job-grid.ecsv",
        {"overwrite": False},
        run_context,
    )

    manifest = write_file.call_args.args[0]
    assert "sct" not in manifest["catalog_metadata"]
    assert manifest["configuration"]["atmosphere"] == {}


def test_write_manifests_requires_primary_and_existing_production_path(
    tmp_test_directory, run_context
):
    job_grid = Path(tmp_test_directory) / "job-grid.ecsv"
    jobs_io.serialize_job_grid([{**_job_grid_row(), "nsb_rate": 1.0, "primary": ""}], job_grid)
    (Path(tmp_test_directory) / "job-000001").mkdir()

    with pytest.raises(KeyError, match="primary"):
        production_metadata._write_manifests(Path(tmp_test_directory), job_grid, {}, run_context)
    with pytest.raises(FileNotFoundError, match="Production path not found"):
        production_metadata._check_existing_manifests(Path(tmp_test_directory) / "missing")


def _args(**updates):
    args = {
        "array_layout_name": "CTAO-South-Alpha",
        "site": "South",
        "primary": "Gamma",
        "azimuth_angle": 190 * u.deg,
        "zenith_angle": 20 * u.deg,
        "view_cone": (0 * u.deg, 1.5 * u.deg),
        "model_version": "7.0.0",
    }
    args.update(updates)
    return args


def _simulator(*array_elements, run_number=12):
    return SimpleNamespace(
        array_models=[SimpleNamespace(array_elements=dict.fromkeys(array_elements))],
        run_number=run_number,
    )


def test_required_simulation_job_metadata_arguments_cover_manifest_inputs():
    assert REQUIRED_SIMULATION_JOB_METADATA_ARGUMENTS == (
        "primary",
        "azimuth_angle",
        "zenith_angle",
        "energy_range",
        "core_scatter",
        "view_cone",
        "showers_per_run",
        "model_version",
        "array_layout_name",
        "site",
        "simulation_software",
    )


def test_build_simulation_job_metadata_uses_catalog_conventions():
    metadata = build_simulation_job_metadata(
        _args(dec=-45 * u.deg, ha=123 * u.deg),
        _simulator("MSTS-01", "SCTS-01"),
    )

    assert metadata == {
        "array_layout": "CTAO-South-Alpha",
        "site": "Paranal",
        "particle": "gamma",
        "phiP": 350.0,
        "thetaP": 20.0,
        "sct": "True",
        "view_cone_min": 0.0,
        "view_cone_max": 1.5,
        "runNumber": 12,
        "model_version": "7.0.0",
        "dec": -45.0,
        "ha": 123.0,
    }


def test_sim_telarray_only_metadata_uses_resolved_simulation_parameters():
    simulator = _simulator("MSTS-01")
    simulator.corsika_configurations = [None]
    simulator.simulation_parameters = [
        SimpleNamespace(primary_particle=SimpleNamespace(name="gamma"), shower_events=10)
    ]

    assert _primary_name({"primary": None}, simulator) == "gamma"
    assert _showers_per_run({"showers_per_run": None}, simulator) == 10


def test_build_simulation_job_metadata_rounds_angles_to_two_decimal_places():
    metadata = build_simulation_job_metadata(
        _args(zenith_angle=20.123 * u.deg, dec=-45.987 * u.deg, ha=10.0056 * u.deg),
        _simulator("MSTS-01"),
    )

    assert metadata["thetaP"] == pytest.approx(20.12)
    assert metadata["dec"] == pytest.approx(-45.99)
    assert metadata["ha"] == pytest.approx(10.01)


def test_build_simulation_job_metadata_omits_missing_coordinates_and_sets_sct_false():
    metadata = build_simulation_job_metadata(
        _args(site="North", azimuth_angle=180 * u.deg),
        _simulator("LSTN-01", run_number=5),
    )

    assert metadata["site"] == "LaPalma"
    assert metadata["phiP"] == pytest.approx(0.0)
    assert metadata["sct"] == "False"
    assert metadata["runNumber"] == 5
    assert "dec" not in metadata
    assert "ha" not in metadata


def test_build_production_job_manifest_contains_selection_fields(tmp_test_directory):
    tmp_test_directory = Path(tmp_test_directory)
    output_directory = tmp_test_directory / "job-000001"
    output_directory.mkdir()
    simtel_file = output_directory / "gamma_run000012.simtel.zst"
    event_data_file = output_directory / "gamma_run000012.reduced_event_data.hdf5"
    simtel_file.touch()
    event_data_file.touch()
    simulator = _simulator("MSTS-01", run_number=12)
    simulator.get_files = lambda file_type: {
        "sim_telarray_output": [tmp_test_directory / simtel_file.name],
        "sim_telarray_event_data": [tmp_test_directory / event_data_file.name],
    }.get(file_type, [])

    manifest = build_production_job_manifest(
        _args(
            production_id="production-42",
            energy_range=(0.03 * u.TeV, 300 * u.TeV),
            core_scatter=(10, 500 * u.m),
            showers_per_run=100,
            simulation_software=["corsika", "sim_telarray"],
            corsika_he_interaction="qgs3",
            corsika_le_interaction="urqmd",
        ),
        simulator,
        output_directory,
    )

    assert manifest["schema_version"] == "2.0.0"
    assert manifest["product_type"] == "simulate_prod_job"
    assert manifest["production_id"] == "production-42"
    assert manifest["catalog_metadata"]["runNumber"] == 12
    assert manifest["configuration"]["run_number"] == 12
    assert manifest["configuration"]["zenith_angle"] == 20 * u.deg
    assert manifest["configuration"]["cores_per_shower"] == 10
    assert manifest["files"] == {
        "reduced_event_data": ["gamma_run000012.reduced_event_data.hdf5"],
        "sim_telarray": ["gamma_run000012.simtel.zst"],
    }


def test_filter_ctao_metadata_keeps_registry_scalar_values():
    metadata = {"dataid": "id-1", "simzen": 20.0, "unknown": "value", "nested": {"x": 1}}

    assert production_metadata._filter_ctao_metadata(metadata, {"dataid", "simzen", "nested"}) == {
        "dataid": "id-1",
        "simzen": 20.0,
    }


def test_manifest_records_ctao_metadata_and_skips_missing_global_metadata(
    tmp_test_directory, caplog
):
    output_directory = Path(tmp_test_directory)
    valid = output_directory / "valid.simtel"
    incomplete = output_directory / "incomplete.simtel"
    payload = b"\x02\x06dataid\x04id-1\x06simzen\x0220"
    valid.write_bytes(struct.pack("<IIiI", 0xD41F8A37, 75, -1, len(payload)) + payload)
    incomplete.write_bytes(struct.pack("<IIiI", 0xD41F8A37, 75, 1, len(payload)) + payload)

    manifest = build_production_job_manifest(
        _args(
            energy_range=(0.03 * u.TeV, 300 * u.TeV),
            core_scatter=(10, 500 * u.m),
            showers_per_run=100,
            simulation_software=["corsika", "sim_telarray"],
        ),
        _simulator("MSTS-01", run_number=12),
        output_directory,
    )

    assert manifest["ctao_metadata"] == {valid.name: {"dataid": "id-1", "simzen": "20"}}
    assert "Unable to read CTAO metadata" in caplog.text
    assert incomplete.name in caplog.text


def test_build_production_job_manifest_uses_resolved_angles_from_input_file(tmp_test_directory):
    output_directory = Path(tmp_test_directory) / "job-000001"
    output_directory.mkdir()
    simulator = _simulator("MSTS-01", run_number=12)
    simulator.corsika_configurations = [None]
    simulator.simulation_parameters = [
        SimpleNamespace(
            primary_particle=SimpleNamespace(name="gamma"),
            shower_events=10,
            azimuth_angle=180.0,
            zenith_angle=40.0,
        )
    ]

    manifest = build_production_job_manifest(
        _args(
            azimuth_angle=0 * u.deg,
            zenith_angle=0 * u.deg,
            energy_range=(100 * u.GeV, 100 * u.GeV),
            core_scatter=(1, 840 * u.m),
            showers_per_run=None,
            simulation_software=["sim_telarray"],
            corsika_file="showers.dat",
        ),
        simulator,
        output_directory,
    )

    assert_quantity_allclose(manifest["configuration"]["azimuth_angle"], 180 * u.deg)
    assert_quantity_allclose(manifest["configuration"]["zenith_angle"], 40 * u.deg)


def test_build_production_job_manifest_preserves_generated_shower_angle_precision(
    tmp_test_directory,
):
    simulator = _simulator("MSTS-01")
    simulator.simulation_parameters = [SimpleNamespace(azimuth_angle=190.12, zenith_angle=20.12)]
    manifest = build_production_job_manifest(
        _args(
            azimuth_angle=190.123 * u.deg,
            zenith_angle=20.123 * u.deg,
            energy_range=(0.03 * u.TeV, 300 * u.TeV),
            core_scatter=(10, 500 * u.m),
            showers_per_run=100,
            simulation_software=["corsika", "sim_telarray"],
        ),
        simulator,
        Path(tmp_test_directory),
    )

    assert_quantity_allclose(manifest["configuration"]["azimuth_angle"], 190.123 * u.deg)
    assert_quantity_allclose(manifest["configuration"]["zenith_angle"], 20.123 * u.deg)


def test_build_production_job_manifest_records_reduced_event_counts(tmp_test_directory):
    output_directory = Path(tmp_test_directory) / "job-000012"
    output_directory.mkdir()
    (output_directory / "gamma_run000012.simtel.zst").touch()
    reduced_file = output_directory / "gamma_run000012.reduced_event_data.hdf5"
    with h5py.File(reduced_file, "w") as data_file:
        data_file.create_dataset("SHOWERS", shape=(12,), dtype="i8")
        data_file.create_dataset("TRIGGERS", shape=(5,), dtype="i8")

    manifest = build_production_job_manifest(
        _args(
            energy_range=(0.03 * u.TeV, 300 * u.TeV),
            core_scatter=(10, 500 * u.m),
            showers_per_run=100,
            simulation_software=["corsika", "sim_telarray"],
        ),
        _simulator("MSTS-01", run_number=12),
        output_directory,
    )

    assert manifest["statistics"] == {
        "outputs": {
            "gamma_run000012.simtel.zst": {
                "simulated_events": 12,
                "triggered_events": 5,
            }
        }
    }
    manifest_path = output_directory / "simulate_prod_job_metadata.yml"
    write_data_to_file(manifest, manifest_path)
    assert check_manifest(ProductionManifest(path=manifest_path, data=manifest))["valid"]
    legacy_manifest = manifest | {
        "schema_version": "1.0.0",
        "configuration": manifest["configuration"]
        | {"simulation_software": "corsika_sim_telarray"},
        "statistics": {"simulated_events": 12, "triggered_events": 5},
    }
    assert check_manifest(ProductionManifest(path=manifest_path, data=legacy_manifest))["valid"]


def test_get_sim_telarray_event_counts_falls_back_to_log(tmp_test_directory):
    output_directory = Path(tmp_test_directory)
    log_file = output_directory / "run000001.simtel.log"
    simtel_file = output_directory / "run000001.simtel.zst"
    log_file.write_text(
        "Sim_telarray finished at time after 1/2/3/4 tel., 100/17 events\n",
        encoding="utf-8",
    )
    simtel_file.touch()

    assert get_sim_telarray_event_counts(
        output_directory,
        {"sim_telarray": [simtel_file.name], "sim_telarray_log": [log_file.name]},
    ) == {"outputs": {simtel_file.name: {"simulated_events": 100, "triggered_events": 17}}}


def test_matching_output_files_uses_registered_product_suffixes():
    inventory = {
        "reduced_event_data": ["run.reduced_event_data.hdf5.zst"],
        "sim_telarray_log": ["run.simtel.log.zst", "run.simtel.err.zst"],
    }

    assert production_metadata._matching_output_files("run.simtel.gz", inventory) == {
        "reduced_event_data": ["run.reduced_event_data.hdf5.zst"],
        "sim_telarray_log": ["run.simtel.log.zst"],
    }


def test_get_sim_telarray_event_counts_keeps_outputs_separate(tmp_test_directory):
    output_directory = Path(tmp_test_directory)
    first_output = output_directory / "gamma_run000001.simtel.zst"
    second_output = output_directory / "proton_run000002.simtel.zst"
    first_output.touch()
    second_output.touch()
    first_event_data = output_directory / "gamma_run000001.reduced_event_data.hdf5"
    with h5py.File(first_event_data, "w") as data_file:
        data_file.create_dataset("SHOWERS", shape=(12,), dtype="i8")
        data_file.create_dataset("TRIGGERS", shape=(5,), dtype="i8")
    second_log = output_directory / "proton_run000002.simtel.log"
    second_log.write_text(
        "Sim_telarray finished at time after 1/2/3/4 tel., 80/9 events\n",
        encoding="utf-8",
    )

    assert get_sim_telarray_event_counts(
        output_directory,
        {
            "sim_telarray": [first_output.name, second_output.name],
            "reduced_event_data": [first_event_data.name],
            "sim_telarray_log": [second_log.name],
        },
    ) == {
        "outputs": {
            first_output.name: {"simulated_events": 12, "triggered_events": 5},
            second_output.name: {"simulated_events": 80, "triggered_events": 9},
        }
    }


def test_get_sim_telarray_event_counts_matches_exact_output_in_model_directories(
    tmp_test_directory,
):
    output_directory = Path(tmp_test_directory)
    inventory = {"sim_telarray": [], "reduced_event_data": [], "sim_telarray_log": []}
    for model_version, simulated, triggered in (("6.0.2", 12, 5), ("6.1.1", 30, 11)):
        model_directory = output_directory / model_version / "sim_telarray" / "run000001"
        model_directory.mkdir(parents=True)
        simtel_name = "gamma_run000001.simtel.zst"
        reduced_name = "gamma_run000001.reduced_event_data.hdf5"
        (model_directory / simtel_name).touch()
        with h5py.File(model_directory / reduced_name, "w") as data_file:
            data_file.create_dataset("SHOWERS", shape=(simulated,), dtype="i8")
            data_file.create_dataset("TRIGGERS", shape=(triggered,), dtype="i8")
        inventory["sim_telarray"].append(
            (Path(model_version) / "sim_telarray" / "run000001" / simtel_name).as_posix()
        )
        inventory["reduced_event_data"].append(
            (Path(model_version) / "sim_telarray" / "run000001" / reduced_name).as_posix()
        )

    assert get_sim_telarray_event_counts(output_directory, inventory) == {
        "outputs": {
            "6.0.2/sim_telarray/run000001/gamma_run000001.simtel.zst": {
                "simulated_events": 12,
                "triggered_events": 5,
            },
            "6.1.1/sim_telarray/run000001/gamma_run000001.simtel.zst": {
                "simulated_events": 30,
                "triggered_events": 11,
            },
        }
    }


def test_get_sim_telarray_event_counts_matches_same_run_outputs_by_full_basename(
    tmp_test_directory,
):
    output_directory = Path(tmp_test_directory)
    inventory = {"sim_telarray": [], "reduced_event_data": [], "sim_telarray_log": []}
    expected = {}
    for model_version, triggered in (("6.0.2", 16), ("6.1.1", 9)):
        basename = f"gamma_run000001_test_{model_version}_multiple_model_versions"
        simtel_name = f"{basename}.simtel.zst"
        reduced_name = f"{basename}.reduced_event_data.hdf5"
        (output_directory / simtel_name).touch()
        with h5py.File(output_directory / reduced_name, "w") as data_file:
            data_file.create_dataset("SHOWERS", shape=(25,), dtype="i8")
            data_file.create_dataset("TRIGGERS", shape=(triggered,), dtype="i8")
        inventory["sim_telarray"].append(simtel_name)
        inventory["reduced_event_data"].append(reduced_name)
        expected[simtel_name] = {"simulated_events": 25, "triggered_events": triggered}

    assert get_sim_telarray_event_counts(output_directory, inventory) == {"outputs": expected}


def test_get_sim_telarray_event_counts_rejects_paths_outside_job_directory(tmp_test_directory):
    root = Path(tmp_test_directory)
    job = root / "job"
    job.mkdir()
    with h5py.File(root / "run000001.reduced_event_data.hdf5", "w") as data_file:
        data_file.create_dataset("SHOWERS", shape=(25,), dtype="i8")
        data_file.create_dataset("TRIGGERS", shape=(9,), dtype="i8")
    (root / "run000001.simtel.log").write_text(
        "Sim_telarray finished after 1/2/3/4 tel., 100/17 events\n", encoding="utf-8"
    )
    inventory = {
        "sim_telarray": ["../run000001.simtel.zst"],
        "reduced_event_data": ["../run000001.reduced_event_data.hdf5"],
        "sim_telarray_log": ["../run000001.simtel.log"],
    }

    assert get_sim_telarray_event_counts(
        root, {key: [Path(value[0]).name] for key, value in inventory.items()}
    ) == {"outputs": {"run000001.simtel.zst": {"simulated_events": 25, "triggered_events": 9}}}
    assert get_sim_telarray_event_counts(job, inventory) is None
    log_only = {key: value for key, value in inventory.items() if key != "reduced_event_data"}
    assert get_sim_telarray_event_counts(
        root, {key: [Path(value[0]).name] for key, value in log_only.items()}
    ) == {"outputs": {"run000001.simtel.zst": {"simulated_events": 100, "triggered_events": 17}}}
    assert get_sim_telarray_event_counts(job, log_only) is None


def test_build_production_job_manifest_discovers_all_packaged_outputs(tmp_test_directory):
    output_directory = Path(tmp_test_directory) / "job-000012"
    output_directory.mkdir()
    for name in (
        "gamma_run000012.simtel.zst",
        "gamma_run000012.corsika.zst",
        "gamma_run000012.simtel.log.gz",
    ):
        (output_directory / name).touch()

    manifest = build_production_job_manifest(
        _args(
            energy_range=(0.03 * u.TeV, 300 * u.TeV),
            core_scatter=(10, 500 * u.m),
            showers_per_run=100,
            simulation_software=["corsika", "sim_telarray"],
        ),
        _simulator("MSTS-01", run_number=12),
        output_directory,
    )

    assert manifest["files"] == {
        "corsika": ["gamma_run000012.corsika.zst"],
        "sim_telarray": ["gamma_run000012.simtel.zst"],
        "sim_telarray_log": ["gamma_run000012.simtel.log.gz"],
    }


def test_build_production_job_manifest_reads_primary_from_corsika_input(
    tmp_test_directory,
):
    output_directory = Path(tmp_test_directory) / "job-000012"
    output_directory.mkdir()
    (output_directory / "gamma_run000012.simtel.zst").touch()
    simulator = _simulator("MSTS-01", run_number=12)
    simulator.corsika_configurations = SimpleNamespace(
        primary_particle=SimpleNamespace(name="gamma"),
        use_curved_atmosphere=False,
        shower_events=10,
    )

    manifest = build_production_job_manifest(
        _args(
            energy_range=(0.03 * u.TeV, 300 * u.TeV),
            core_scatter=(10, 500 * u.m),
            showers_per_run=None,
            primary=None,
            simulation_software=["sim_telarray"],
        ),
        simulator,
        output_directory,
    )

    assert manifest["catalog_metadata"]["particle"] == "gamma"
    assert manifest["configuration"]["primary"] == "gamma"
    assert manifest["configuration"]["showers_per_run"] == 10


def test_build_production_job_manifest_preserves_multiple_model_versions(tmp_test_directory):
    output_directory = Path(tmp_test_directory) / "job-000012"
    output_directory.mkdir()
    (output_directory / "gamma_run000012.simtel.zst").touch()
    manifest = build_production_job_manifest(
        _args(
            energy_range=(0.03 * u.TeV, 300 * u.TeV),
            core_scatter=(10, 500 * u.m),
            showers_per_run=100,
            model_version=["6.0.2", "7.0.0"],
            simulation_software=["sim_telarray"],
        ),
        _simulator("MSTS-01", run_number=12),
        output_directory,
    )

    assert manifest["configuration"]["model_version"] == ["6.0.2", "7.0.0"]
    assert manifest["catalog_metadata"]["model_version"] == ["6.0.2", "7.0.0"]


def test_build_production_job_manifest_preserves_truthful_backfill_metadata(tmp_test_directory):
    output_directory = Path(tmp_test_directory) / "job-000012"
    output_directory.mkdir()
    simulator = _simulator("MSTS-01", run_number=12)
    catalog_metadata = {"runNumber": 12, "particle": "gamma"}
    atmosphere = {"curved_atmosphere_min_zenith_angle": 70 * u.deg}

    manifest = build_production_job_manifest(
        _args(
            energy_range=(0.03 * u.TeV, 300 * u.TeV),
            core_scatter=(10, 500 * u.m),
            showers_per_run=100,
            simulation_software=["corsika", "sim_telarray"],
        ),
        simulator,
        output_directory,
        file_inventory={"sim_telarray": ["gamma_run000012.simtel.zst"]},
        catalog_metadata=catalog_metadata,
        atmosphere_configuration=atmosphere,
    )

    assert manifest["catalog_metadata"] == catalog_metadata
    assert manifest["configuration"]["atmosphere"] == atmosphere


def test_build_production_job_manifest_records_resolved_overrides_and_atmosphere(
    tmp_test_directory,
):
    output_directory = Path(tmp_test_directory) / "job-000012"
    output_directory.mkdir()
    simtel_file = output_directory / "gamma_run000012.simtel.zst"
    simtel_file.touch()
    site_model = SimpleNamespace(
        parameters={
            "atmospheric_profile": {"value": "prod-atmosphere"},
            "reference_point_altitude": {"value": 2150 * u.m},
        }
    )
    array_model = SimpleNamespace(
        array_elements={},
        model_version="7.0.0",
        overwrite_model_parameter_dict={"LSTN-design": {"mirror_area": {"value": 400}}},
        site_model=site_model,
    )
    simulator = SimpleNamespace(
        array_models=[array_model],
        corsika_configurations=SimpleNamespace(use_curved_atmosphere=True),
        run_number=12,
        get_files=lambda file_type: [simtel_file] if file_type == "sim_telarray_output" else [],
    )

    manifest = build_production_job_manifest(
        _args(
            energy_range=(0.03 * u.TeV, 300 * u.TeV),
            core_scatter=(10, 500 * u.m),
            showers_per_run=100,
            simulation_software=["corsika", "sim_telarray"],
            curved_atmosphere_min_zenith_angle=70 * u.deg,
            overwrite_model_parameters="/submission/path/overrides.yml",
        ),
        simulator,
        output_directory,
    )

    configuration = manifest["configuration"]
    assert configuration["model_parameter_overrides"] == {
        "LSTN-design": {"mirror_area": {"value": 400}}
    }
    assert "overwrite_model_parameters" not in configuration
    assert configuration["atmosphere"]["use_curved_atmosphere"] is True
    assert configuration["atmosphere"]["site_parameters"]["atmospheric_profile"] == (
        "prod-atmosphere"
    )


def test_telescope_only_manifest_uses_resolved_atmosphere_parameter(tmp_test_directory):
    simulator = SimpleNamespace(
        array_models=[],
        corsika_configurations=[None],
        simulation_parameters=[SimpleNamespace(use_curved_atmosphere=True)],
        run_number=12,
        get_files=lambda file_type: [],
    )

    manifest = build_production_job_manifest(
        _args(
            simulation_software=["sim_telarray"],
            energy_range=(0.03 * u.TeV, 300 * u.TeV),
            core_scatter=(10, 500 * u.m),
            showers_per_run=100,
        ),
        simulator,
        tmp_test_directory,
    )

    assert manifest["configuration"]["atmosphere"]["use_curved_atmosphere"] is True


def test_resolved_model_parameter_overrides_keeps_values_by_model_version():
    models = [
        SimpleNamespace(
            model_version="7.0.0",
            overwrite_model_parameter_dict={"first": 1},
        ),
        SimpleNamespace(
            model_version="7.1.0",
            overwrite_model_parameter_dict={"second": 2},
        ),
    ]

    assert _resolved_model_parameter_overrides(SimpleNamespace(array_models=models)) == {
        "7.0.0": {"first": 1},
        "7.1.0": {"second": 2},
    }


def test_build_production_job_manifest_keeps_nested_output_paths(tmp_test_directory):
    output_directory = Path(tmp_test_directory) / "job-000012"
    simtel_directory = output_directory / "sim_telarray" / "run000012"
    simtel_directory.mkdir(parents=True)
    simtel_file = simtel_directory / "gamma_run000012.simtel.zst"
    simtel_file.touch()
    simulator = _simulator("MSTS-01", run_number=12)
    simulator.get_files = lambda file_type: (
        [simtel_file] if file_type == "sim_telarray_output" else []
    )

    manifest = build_production_job_manifest(
        _args(
            energy_range=(0.03 * u.TeV, 300 * u.TeV),
            core_scatter=(10, 500 * u.m),
            showers_per_run=100,
            simulation_software=["sim_telarray"],
        ),
        simulator,
        output_directory,
    )

    assert manifest["files"] == {
        "sim_telarray": ["sim_telarray/run000012/gamma_run000012.simtel.zst"]
    }


def test_build_simulation_job_metadata_rounds_view_cone_to_two_decimal_places():
    metadata = build_simulation_job_metadata(
        _args(view_cone=(0.12345 * u.deg, 5.6789 * u.deg)),
        _simulator("MSTS-01"),
    )

    assert metadata["view_cone_min"] == pytest.approx(0.12)
    assert metadata["view_cone_max"] == pytest.approx(5.68)


@pytest.mark.parametrize(
    ("geographic_az", "expected_phip"),
    [
        (0.0, 180.0),
        (90.0, 90.0),
        (180.0, 0.0),
        (270.0, 270.0),
        (190.0, 350.0),
        (360.0, 180.0),
        (45.7, 134.3),
        (135.3, 44.7),
        (225.5, 314.5),
        (315.8, 224.2),
    ],
)
def test_phip_is_corsika_azimuth_conversion(geographic_az, expected_phip):
    """phiP is the CORSIKA-space azimuth derived from the geographic azimuth."""
    metadata = build_simulation_job_metadata(
        _args(azimuth_angle=geographic_az * u.deg),
        _simulator("MSTS-01"),
    )
    assert metadata["phiP"] == pytest.approx(expected_phip, abs=1e-3)
