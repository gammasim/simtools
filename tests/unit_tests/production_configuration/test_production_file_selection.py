"""Tests for production metadata file discovery, filtering, and grouping."""

from pathlib import Path

import astropy.units as u
import pytest

from simtools.io.ascii_handler import collect_data_from_file, write_data_to_file
from simtools.production_configuration.production_file_selection import (
    SIMULATE_PROD_JOB_METADATA,
    ProductionFileGroup,
    ProductionManifest,
    _compare_simtel_metadata,
    _embedded_metadata_matches,
    _get_dotted_value,
    _missing_run_numbers,
    _parse_selection,
    _production_file_type,
    _quantity_matches,
    _resolve_relative_manifest_path,
    _validate_file_type,
    _validate_filename_run_number,
    _validate_manifest_structure,
    check_manifest,
    discover_product_manifests,
    filter_manifests,
    find_manifests,
    group_selected_files,
    inventory_production_files,
    load_manifest,
    normalize_for_comparison,
    select_file_groups,
    selection_summary,
    stable_configuration_hash,
    validate_required_production_outputs,
    write_selection_file,
)


def _write_manifest(base, job_name, run_number, he_interaction="qgs3", zenith=None):
    base = Path(base)
    job_directory = base / job_name
    job_directory.mkdir()
    data_file = job_directory / f"gamma_run{run_number:06d}.reduced_event_data.hdf5"
    simtel_file = job_directory / f"gamma_run{run_number:06d}.simtel.zst"
    data_file.touch()
    simtel_file.touch()
    manifest = {
        "schema_name": "simulate_prod_job_metadata",
        "schema_version": "1.0.0",
        "product_type": "simulate_prod_job",
        "job_id": job_name,
        "status": "complete",
        "catalog_metadata": {"runNumber": run_number},
        "configuration": {
            "run_number": run_number,
            "primary": "gamma",
            "site": "North",
            "array_layout_name": "CTAO-North-Alpha",
            "model_version": "7.0.0",
            "simulation_software": "corsika_sim_telarray",
            "corsika_he_interaction": he_interaction,
            "azimuth_angle": {"value": 180.0, "unit": "deg"},
            "zenith_angle": zenith or {"value": 20.0, "unit": "deg"},
            "energy_min": {"value": 0.03, "unit": "TeV"},
            "energy_max": {"value": 300.0, "unit": "TeV"},
            "view_cone_min": {"value": 0.0, "unit": "deg"},
            "view_cone_max": {"value": 10.0, "unit": "deg"},
            "cores_per_shower": 10,
            "core_scatter_max": {"value": 500.0, "unit": "m"},
            "showers_per_run": 100,
            "model_parameter_overrides": {},
            "atmosphere": {},
        },
        "files": {
            "reduced_event_data": [data_file.name],
            "sim_telarray": [simtel_file.name],
        },
    }
    manifest_path = job_directory / SIMULATE_PROD_JOB_METADATA
    write_data_to_file(manifest, manifest_path)
    return manifest_path


def test_select_file_groups_filters_quantities_and_orders_runs(tmp_test_directory):
    _write_manifest(tmp_test_directory, "job-000002", 2, he_interaction="qgs3")
    _write_manifest(tmp_test_directory, "job-000001", 1, he_interaction="qgs3")
    _write_manifest(tmp_test_directory, "job-000003", 3, he_interaction="epos")

    result = select_file_groups(
        tmp_test_directory,
        selections=[
            "configuration.corsika_he_interaction=qgs3",
            "configuration.zenith_angle=20 deg",
        ],
        file_type="reduced_event_data",
    )

    assert result["metadata_files_read"] == 3
    assert result["matching_jobs"] == 2
    assert result["configuration_groups"] == 1
    assert result["groups"][0].run_numbers == [1, 2]
    assert [path.name for path in result["groups"][0].file_paths] == [
        "gamma_run000001.reduced_event_data.hdf5",
        "gamma_run000002.reduced_event_data.hdf5",
    ]


def test_select_file_groups_separates_different_configurations(tmp_test_directory):
    _write_manifest(tmp_test_directory, "job-000001", 1, he_interaction="qgs3")
    _write_manifest(
        tmp_test_directory,
        "job-000002",
        2,
        he_interaction="qgs3",
        zenith={"value": 0.3490658503988659, "unit": "rad"},
    )

    result = select_file_groups(tmp_test_directory)

    assert result["configuration_groups"] == 1
    assert result["groups"][0].run_numbers == [1, 2]


def test_select_file_groups_reports_missing_runs(tmp_test_directory):
    _write_manifest(tmp_test_directory, "job-000001", 1)
    _write_manifest(tmp_test_directory, "job-000003", 3)

    result = select_file_groups(tmp_test_directory)

    assert result["groups"][0].missing_run_numbers == [2]


def test_select_file_groups_keeps_multiple_files_from_one_run(tmp_test_directory):
    manifest_path = _write_manifest(tmp_test_directory, "job-000001", 1)
    second_file = manifest_path.parent / "gamma_run000001.part0002.reduced_event_data.hdf5"
    second_file.touch()
    manifest = collect_data_from_file(manifest_path)
    manifest["files"]["reduced_event_data"].append(second_file.name)
    write_data_to_file(manifest, manifest_path)

    result = select_file_groups(tmp_test_directory)

    assert result["groups"][0].run_numbers == [1]
    assert [path.name for path in result["groups"][0].file_paths] == [
        "gamma_run000001.reduced_event_data.hdf5",
        "gamma_run000001.part0002.reduced_event_data.hdf5",
    ]


def test_select_file_groups_can_require_complete_runs(tmp_test_directory):
    _write_manifest(tmp_test_directory, "job-000001", 1)
    _write_manifest(tmp_test_directory, "job-000003", 3)

    with pytest.raises(ValueError, match=r"Missing run numbers.*2"):
        select_file_groups(tmp_test_directory, require_complete_runs=True)


def test_select_file_groups_accepts_complete_runs(tmp_test_directory):
    _write_manifest(tmp_test_directory, "job-000001", 1)
    _write_manifest(tmp_test_directory, "job-000002", 2)

    result = select_file_groups(tmp_test_directory, require_complete_runs=True)

    assert result["groups"][0].missing_run_numbers == []


def test_select_file_groups_rejects_duplicate_runs(tmp_test_directory):
    _write_manifest(tmp_test_directory, "job-000001", 1)
    _write_manifest(tmp_test_directory, "job-000002", 1)

    with pytest.raises(ValueError, match="Duplicate run numbers"):
        select_file_groups(tmp_test_directory)


def test_check_manifest_rejects_paths_outside_job_directory(tmp_test_directory):
    manifest_path = _write_manifest(tmp_test_directory, "job-000001", 1)
    manifest = {
        "schema_name": "simulate_prod_job_metadata",
        "schema_version": "1.0.0",
        "product_type": "simulate_prod_job",
        "job_id": "job-000001",
        "status": "complete",
        "catalog_metadata": {"runNumber": 1},
        "configuration": {
            "run_number": 1,
            "primary": "gamma",
            "site": "North",
            "array_layout_name": "CTAO-North-Alpha",
            "model_version": "7.0.0",
            "simulation_software": "corsika_sim_telarray",
            "azimuth_angle": {"value": 180.0, "unit": "deg"},
            "zenith_angle": {"value": 20.0, "unit": "deg"},
            "energy_min": {"value": 0.03, "unit": "TeV"},
            "energy_max": {"value": 300.0, "unit": "TeV"},
            "view_cone_min": {"value": 0.0, "unit": "deg"},
            "view_cone_max": {"value": 10.0, "unit": "deg"},
            "cores_per_shower": 10,
            "core_scatter_max": {"value": 500.0, "unit": "m"},
            "showers_per_run": 100,
            "model_parameter_overrides": {},
            "atmosphere": {},
        },
        "files": {"sim_telarray": ["../gamma_run000001.simtel.zst"]},
    }
    write_data_to_file(manifest, manifest_path)

    with pytest.raises(ValueError, match="escapes job directory"):
        check_manifest(manifest_path)


def test_check_manifest_rejects_missing_files(tmp_test_directory):
    manifest_path = _write_manifest(tmp_test_directory, "job-000001", 1)
    (manifest_path.parent / "gamma_run000001.reduced_event_data.hdf5").unlink()

    with pytest.raises(FileNotFoundError, match="references missing file"):
        check_manifest(manifest_path)


def test_check_manifest_rejects_unlisted_production_files(tmp_test_directory):
    manifest_path = _write_manifest(tmp_test_directory, "job-000001", 1)
    (manifest_path.parent / "gamma_run000001.simtel.log.gz").touch()

    with pytest.raises(ValueError, match="Unexpected production files"):
        check_manifest(manifest_path)


def test_check_manifest_rejects_duplicate_output_entries(tmp_test_directory):
    simtel_file = Path(tmp_test_directory) / "gamma_run000001.simtel.zst"
    simtel_file.touch()
    manifest = ProductionManifest(
        path=Path(tmp_test_directory) / "manifest.yml",
        data={
            "schema_version": "1.0.0",
            "product_type": "custom_product",
            "status": "complete",
            "configuration": {},
            "files": {"sim_telarray": [simtel_file.name, simtel_file.name]},
        },
    )

    with pytest.raises(ValueError, match="Duplicate output file listed"):
        check_manifest(manifest)


def test_check_manifest_accepts_non_simulation_product_manifest(tmp_test_directory):
    manifest = ProductionManifest(
        path=Path(tmp_test_directory) / "selection.yml",
        data={
            "schema_version": "1.0.0",
            "product_type": "production_file_selection",
            "status": "complete",
            "configuration": {},
            "files": {},
        },
    )

    assert check_manifest(manifest) == {"valid": True, "unverifiable_fields": []}


def test_inventory_production_files_discovers_nested_simulation_outputs(tmp_test_directory):
    job_directory = Path(tmp_test_directory)
    simtel_file = job_directory / "sim_telarray" / "run000001" / "gamma_run000001.simtel.zst"
    log_file = job_directory / "sim_telarray" / "run000001" / "gamma_run000001.simtel.log.gz"
    reduced_file = (
        job_directory / "sim_telarray" / "run000001" / "gamma_run000001.reduced_event_data.hdf5"
    )
    simtel_file.parent.mkdir(parents=True)
    simtel_file.touch()
    log_file.touch()
    reduced_file.touch()
    (job_directory / "trigger.trigger_histograms.hdf5").touch()
    (job_directory / "unrecognized.txt").touch()

    assert inventory_production_files(job_directory) == {
        "reduced_event_data": ["sim_telarray/run000001/gamma_run000001.reduced_event_data.hdf5"],
        "sim_telarray": ["sim_telarray/run000001/gamma_run000001.simtel.zst"],
        "sim_telarray_log": ["sim_telarray/run000001/gamma_run000001.simtel.log.gz"],
    }


def test_embedded_metadata_matches_in_memory_and_serialized_quantities():
    assert _embedded_metadata_matches(0 * u.deg, "0.0")
    assert _embedded_metadata_matches({"value": 20.0, "unit": "deg"}, "20")
    assert not _embedded_metadata_matches(20 * u.deg, "21")
    assert _embedded_metadata_matches("Gamma", "gamma")


def test_check_manifest_compares_embedded_simtel_metadata(tmp_test_directory, mocker):
    manifest_path = _write_manifest(tmp_test_directory, "job-000001", 1)
    mocker.patch(
        "simtools.production_configuration.production_file_selection.read_sim_telarray_metadata",
        return_value=({"primary": "gamma", "zenith": "20 deg"}, None),
    )

    result = check_manifest(manifest_path)

    assert "primary" not in result["unverifiable_fields"]
    assert "zenith_angle" not in result["unverifiable_fields"]


def test_compare_simtel_metadata_skips_configuration_fields_without_metadata(mocker):
    manifest = ProductionManifest(
        path=Path("manifest.yml"),
        data={"configuration": {"primary": "gamma"}},
    )
    mocker.patch(
        "simtools.production_configuration.production_file_selection.read_sim_telarray_metadata",
        return_value=({"primary": "gamma"}, None),
    )

    assert _compare_simtel_metadata(Path("file.simtel"), manifest, ["primary"]) == []


def test_check_manifest_rejects_mismatched_embedded_simtel_metadata(tmp_test_directory, mocker):
    manifest_path = _write_manifest(tmp_test_directory, "job-000001", 1)
    mocker.patch(
        "simtools.production_configuration.production_file_selection.read_sim_telarray_metadata",
        return_value=({"primary": "proton"}, None),
    )

    with pytest.raises(ValueError, match=r"configuration\.primary"):
        check_manifest(manifest_path)


def test_validate_required_production_outputs_rejects_incomplete_job(tmp_test_directory):
    with pytest.raises(ValueError, match="no 'sim_telarray' output"):
        validate_required_production_outputs(
            {"reduced_event_data": ["gamma.reduced_event_data.hdf5"]},
            "corsika_sim_telarray",
            tmp_test_directory,
        )


def test_validate_required_production_outputs_accepts_required_output(tmp_test_directory):
    validate_required_production_outputs(
        {"corsika": ["shower.corsika.zst"]}, "corsika", tmp_test_directory
    )


def test_stable_configuration_hash_is_deterministic_and_short():
    configuration = {"zenith_angle": {"value": 20.0, "unit": "deg"}}

    first_hash = stable_configuration_hash(configuration)
    second_hash = stable_configuration_hash({"zenith_angle": {"unit": "deg", "value": 20.0}})

    assert first_hash == second_hash
    assert len(first_hash) == 8
    assert stable_configuration_hash(configuration, length=12) != first_hash


def test_find_manifests_rejects_missing_production_path(tmp_test_directory):
    with pytest.raises(FileNotFoundError, match="Production path not found"):
        find_manifests(Path(tmp_test_directory) / "missing")


def test_discover_product_manifests_filters_invalid_and_other_products(tmp_test_directory, mocker):
    paths = [Path(tmp_test_directory) / name for name in ("keep.yml", "skip.yml", "invalid.yml")]
    for path in paths:
        path.touch()
    data = {
        "keep.yml": {
            "schema_version": "1.0.0",
            "product_type": "custom_product",
            "status": "complete",
            "configuration": {},
            "files": {},
        },
        "skip.yml": {"product_type": "other_product"},
        "invalid.yml": ["not a mapping"],
    }
    mocker.patch(
        "simtools.production_configuration.production_file_selection.ascii_handler.collect_data_from_file",
        side_effect=lambda path: data[Path(path).name],
    )

    result = discover_product_manifests(tmp_test_directory, "custom_product")

    assert [manifest.path.name for manifest in result] == ["keep.yml"]


def test_discover_product_manifests_rejects_missing_production_path(tmp_test_directory):
    with pytest.raises(FileNotFoundError, match="Production path not found"):
        discover_product_manifests(Path(tmp_test_directory) / "missing", "custom_product")


def test_group_selected_files_rejects_missing_file_type(tmp_test_directory):
    manifest_path = _write_manifest(tmp_test_directory, "job-000001", 1)
    manifest = load_manifest(manifest_path)

    with pytest.raises(ValueError, match="lists no files"):
        group_selected_files([manifest], file_type="corsika")


def test_write_selection_file_and_summary(tmp_test_directory):
    group = ProductionFileGroup(
        configuration={"primary": "gamma"},
        run_numbers=[1, 3],
        file_paths=[Path("job-000001/file.simtel.zst")],
        missing_run_numbers=[2],
    )
    result = {
        "metadata_files_read": 3,
        "matching_jobs": 2,
        "configuration_groups": 1,
        "groups": [group],
    }
    output_file = Path(tmp_test_directory) / "selection.yml"

    write_selection_file(result, output_file)

    assert "Missing runs: 2" in selection_summary(result)
    assert collect_data_from_file(output_file)["groups"][0]["files"] == [
        "job-000001/file.simtel.zst"
    ]


def test_normalize_for_comparison_handles_quantities_and_sequences():
    assert normalize_for_comparison(1 * u.m) == (1.0, "m")
    assert normalize_for_comparison([1 * u.m, (2 * u.s,)]) == ((1.0, "m"), ((2.0, "s"),))


def test_filter_manifests_supports_unqualified_configuration_keys():
    manifest = ProductionManifest(
        path=Path("manifest.yml"),
        data={"configuration": {"primary": "gamma"}},
    )

    assert filter_manifests([manifest], ["primary=gamma"]) == [manifest]
    assert _get_dotted_value({"configuration": {}}, "configuration.primary") is None
    assert not filter_manifests([manifest], ["missing=value"])


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (None, "expected a mapping"),
        ({}, "missing 'schema_version'"),
        (
            {
                "schema_version": "9.9.9",
                "product_type": "custom",
                "status": "complete",
                "configuration": {},
                "files": {},
            },
            "Unsupported production metadata schema",
        ),
        (
            {
                "schema_version": "1.0.0",
                "product_type": "custom",
                "status": "running",
                "configuration": {},
                "files": {},
            },
            "not complete",
        ),
        (
            {
                "schema_version": "1.0.0",
                "product_type": "custom",
                "status": "complete",
                "configuration": [],
                "files": {},
            },
            "configuration is not a mapping",
        ),
        (
            {
                "schema_version": "1.0.0",
                "product_type": "simulate_prod_job",
                "status": "complete",
                "configuration": {},
                "files": {},
            },
            "configuration.run_number",
        ),
        (
            {
                "schema_version": "1.0.0",
                "product_type": "custom",
                "status": "complete",
                "configuration": {},
                "files": [],
            },
            "files is not a mapping",
        ),
    ],
)
def test_validate_manifest_structure_rejects_malformed_data(data, message, tmp_test_directory):
    with pytest.raises(ValueError, match=message):
        _validate_manifest_structure(data, Path(tmp_test_directory) / "manifest.yml")


@pytest.mark.parametrize("selection", ["missing_separator", "=value"])
def test_parse_selection_rejects_invalid_syntax(selection):
    with pytest.raises(ValueError, match="Selection"):
        _parse_selection(selection)


def test_quantity_selection_rejects_invalid_expected_quantity():
    assert not _quantity_matches({"value": 1, "unit": "m"}, "not-a-quantity")


def test_manifest_path_and_file_suffix_validation(tmp_test_directory):
    with pytest.raises(ValueError, match="must be relative"):
        _resolve_relative_manifest_path(tmp_test_directory, "/absolute/file.simtel.zst")
    with pytest.raises(ValueError, match="suffix is inconsistent"):
        _validate_file_type(Path(tmp_test_directory) / "file.txt", "sim_telarray", "manifest.yml")


@pytest.mark.parametrize(
    ("file_type", "suffix"),
    [
        ("reduced_event_data", ".reduced_event_data.hdf5"),
        ("sim_telarray", ".simtel"),
        ("sim_telarray_log", ".simtel.log"),
        ("sim_telarray_histogram", ".hdata"),
        ("corsika", ".corsika"),
        ("corsika_log", ".corsika.log"),
        ("trigger_histograms", ".trigger_histograms.hdf5"),
    ],
)
@pytest.mark.parametrize("compression", ["", ".gz", ".zst"])
def test_all_production_file_types_accept_compression_suffixes(
    file_type, suffix, compression, tmp_test_directory
):
    _validate_file_type(
        Path(tmp_test_directory) / f"run000001{suffix}{compression}",
        file_type,
        "manifest.yml",
    )


def test_filename_run_number_validation_allows_unencoded_names_and_rejects_mismatch():
    no_run_manifest = ProductionManifest(Path("manifest.yml"), {"configuration": {}})
    _validate_filename_run_number(Path("output.simtel"), no_run_manifest)
    run_manifest = ProductionManifest(Path("manifest.yml"), {"configuration": {"run_number": 2}})

    with pytest.raises(ValueError, match="Run number mismatch"):
        _validate_filename_run_number(Path("gamma_run000001.simtel"), run_manifest)


def test_production_file_type_and_quantity_helpers():
    assert _production_file_type(Path("file.simtel.zst")) == "sim_telarray"
    assert _production_file_type(Path("file.txt")) is None
    assert _quantity_matches({"value": 100, "unit": "GeV"}, "0.1 TeV")


def test_empty_selection_summary_has_no_missing_runs():
    result = {"metadata_files_read": 0, "matching_jobs": 0, "configuration_groups": 0, "groups": []}
    assert selection_summary(result).endswith("Missing runs: none")
    assert _missing_run_numbers([]) == []
