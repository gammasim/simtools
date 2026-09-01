"""Tests for simulation job metadata generation."""

from pathlib import Path
from types import SimpleNamespace

import astropy.units as u
import pytest

from simtools.production_configuration.job_metadata import (
    REQUIRED_SIMULATION_JOB_METADATA_ARGUMENTS,
    _add_optional_configuration_value,
    _ensure_list,
    _resolved_model_parameter_overrides,
    build_production_job_manifest,
    build_simulation_job_metadata,
)


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
            energy_range=(0.03 * u.TeV, 300 * u.TeV),
            core_scatter=(10, 500 * u.m),
            showers_per_run=100,
            simulation_software="corsika_sim_telarray",
            corsika_he_interaction="qgs3",
            corsika_le_interaction="urqmd",
        ),
        simulator,
        output_directory,
    )

    assert manifest["schema_version"] == "1.0.0"
    assert manifest["product_type"] == "simulate_prod_job"
    assert manifest["catalog_metadata"]["runNumber"] == 12
    assert manifest["configuration"]["run_number"] == 12
    assert manifest["configuration"]["zenith_angle"] == 20 * u.deg
    assert manifest["configuration"]["cores_per_shower"] == 10
    assert manifest["files"] == {
        "reduced_event_data": ["gamma_run000012.reduced_event_data.hdf5"],
        "sim_telarray": ["gamma_run000012.simtel.zst"],
    }


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
            simulation_software="corsika_sim_telarray",
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
            simulation_software="corsika_sim_telarray",
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


def test_optional_configuration_values_and_file_lists():
    configuration = {}
    _add_optional_configuration_value(configuration, "present", 1)
    _add_optional_configuration_value(configuration, "missing", None)

    assert configuration == {"present": 1}
    assert _ensure_list(None) == []
    assert _ensure_list((1, 2)) == [1, 2]
    assert _ensure_list("one") == ["one"]


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
            simulation_software="sim_telarray",
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
