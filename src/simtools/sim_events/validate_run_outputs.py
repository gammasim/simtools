"""Validation of run outputs: sim_telarray, CORSIKA, and reduced event list files."""

import logging
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from eventio.simtel.simtelfile import SimTelFile

from simtools.constants import METADATA_JSON_SCHEMA, SCHEMA_PATH
from simtools.data_model import schema, validate_data
from simtools.io import table_handler
from simtools.job_execution.log_inspector import check_plain_logs
from simtools.model.mirrors import uses_segmented_dual_mirror_geometry
from simtools.model_repository.asset_names import get_simtel_table_file_name
from simtools.production_configuration.trigger_histograms import (
    TRIGGER_HISTOGRAM_BINS_TABLE,
    TRIGGER_HISTOGRAM_DENSE_GROUP,
    TRIGGER_HISTOGRAM_METADATA_TABLE,
    TRIGGER_SUBSET_HISTOGRAMS_TABLE,
    TRIGGER_TOPOLOGY_COUNTS_TABLE,
)
from simtools.sim_events import file_info
from simtools.sim_events.file_info import get_corsika_run_number
from simtools.sim_events.metadata import validate_simulation_metadata
from simtools.simtel import sim_telarray_metadata
from simtools.simtel.simtel_io_metadata import (
    get_sim_telarray_telescope_id,
    read_sim_telarray_metadata,
)
from simtools.utils import random, value_conversion

_logger = logging.getLogger(__name__)

_REDUCED_EVENT_TABLE_SCHEMAS = {
    "SHOWERS": SCHEMA_PATH / "reduced_event_showers.schema.yml",
    "TRIGGERS": SCHEMA_PATH / "reduced_event_triggers.schema.yml",
    "FILE_INFO": SCHEMA_PATH / "reduced_event_file_info.schema.yml",
}
_TRIGGER_HISTOGRAM_TABLES = (
    TRIGGER_HISTOGRAM_METADATA_TABLE,
    TRIGGER_HISTOGRAM_BINS_TABLE,
    TRIGGER_TOPOLOGY_COUNTS_TABLE,
    TRIGGER_SUBSET_HISTOGRAMS_TABLE,
)
_TRIGGER_HISTOGRAM_TABLE_SCHEMAS = {
    TRIGGER_HISTOGRAM_METADATA_TABLE: SCHEMA_PATH
    / "trigger_histogram_reference_metadata.schema.yml",
    TRIGGER_HISTOGRAM_BINS_TABLE: SCHEMA_PATH / "trigger_histogram_reference_bins.schema.yml",
    TRIGGER_TOPOLOGY_COUNTS_TABLE: SCHEMA_PATH / "trigger_histogram_topology_counts.schema.yml",
    TRIGGER_SUBSET_HISTOGRAMS_TABLE: SCHEMA_PATH / "trigger_histogram_subset_histograms.schema.yml",
}


def _raise_inconsistent(title, errors):
    """Raise a ValueError listing inconsistent entries under a title."""
    raise ValueError(f"{title}\n" + "\n".join(f" - {error}" for error in errors))


# --- sim_telarray -----------------------------------------------------------------------


def validate_sim_telarray(
    data_files,
    log_files,
    array_models=None,
    expected_mc_events=None,
    expected_shower_events=None,
    curved_atmo=False,
    allow_for_changes=None,
):
    """
    Validate sim_telarray output files and metadata (raises ValueError if inconsistent).

    Parameters
    ----------
    data_files, log_files: list of Path
        sim_telarray output and log files.
    array_models: list of ArrayModel
        Array models to compare with.
    expected_mc_events, expected_shower_events: int
        Expected numbers of MC and shower events.
    curved_atmo: bool
        CORSIKA executable compiled with curved atmosphere option.
    allow_for_changes: list
        Model parameters changed with '-C' (metadata may differ for these).
    """
    data_files = value_conversion.ensure_list(data_files)
    log_files = value_conversion.ensure_list(log_files)

    if array_models:
        validate_sim_telarray_metadata(data_files, array_models, allow_for_changes)

    validate_sim_telarray_logs(log_files, expected_mc_events, expected_shower_events, curved_atmo)
    if expected_mc_events is not None and expected_shower_events is not None:
        validate_sim_telarray_event_numbers(data_files, expected_mc_events, expected_shower_events)


def validate_sim_telarray_metadata(files, array_models, allow_for_changes=None):
    """Validate metadata in the sim_telarray output files."""
    for model in array_models:
        output_file = next((f for f in files if model.model_version in str(f)), None)
        if output_file:
            _logger.info(f"Validating metadata for {output_file}")
            assert_sim_telarray_metadata(output_file, model, allow_for_changes)
            _logger.info(f"Metadata for sim_telarray file {output_file} is valid.")
        else:
            _logger.warning(
                f"No sim_telarray file found for model version {model.model_version}: {files}"
            )


def validate_sim_telarray_logs(
    log_files, expected_mc_events=None, expected_shower_events=None, curved_atmo=False
):
    """Validate sim_telarray log files."""
    patterns = ["Finished.", "Sim_telarray finished at"]
    forbidden = []
    if expected_mc_events is not None and expected_shower_events is not None:
        patterns.append(
            f"Run(s) completed as expected after {expected_mc_events} events "
            f"({expected_shower_events} showers)."
        )
    (patterns if curved_atmo else forbidden).append("CORSIKA was compiled with CURVED option.")

    if not check_plain_logs(log_files, {"pattern": patterns, "forbidden_pattern": forbidden}):
        raise ValueError(f"Sim_telarray log files validation failed for {log_files}")
    _logger.info("Sim_telarray log files validation passed: %s", log_files)


def assert_sim_telarray_metadata(file, array_model, allow_for_changes=None):
    """
    Assert consistency of sim_telarray metadata with given array model.

    Parameters
    ----------
    file: Path
        sim_telarray file.
    array_model: ArrayModel
        Array model to compare with.
    allow_for_changes: list
        Model parameters changed with '-C' (metadata may differ for these).
    """
    global_meta, telescope_meta = read_sim_telarray_metadata(file)
    sim_telarray_metadata.validate_metadata_values(global_meta)
    for metadata in telescope_meta.values():
        sim_telarray_metadata.validate_metadata_values(metadata)
    _logger.info(f"Found metadata in sim_telarray file for {len(telescope_meta)} telescopes")
    site_mismatch = _assert_model_parameters(global_meta, array_model.site_model, allow_for_changes)
    seed_mismatch = _assert_sim_telarray_seed(global_meta, array_model.sim_telarray_seed, file)
    if seed_mismatch:
        site_mismatch.append(seed_mismatch)

    if len(telescope_meta) != len(array_model.telescope_models):
        raise ValueError(
            f"Number of telescopes in sim_telarray file ({len(telescope_meta)}) does not match "
            f"number of telescopes in array model ({len(array_model.telescope_models)})"
        )
    for telescope_name in array_model.telescope_models.keys():
        if not get_sim_telarray_telescope_id(telescope_name, file):
            raise ValueError(f"Telescope {telescope_name} not found in sim_telarray file metadata")

    # all mismatches are reported, not only those found first
    mismatches = [
        _assert_model_parameters(telescope_meta[i], model, allow_for_changes)
        for i, model in enumerate(array_model.telescope_models.values(), start=1)
    ]
    mismatches.append(site_mismatch)
    if any(mismatches):
        raise ValueError(
            "Telescope or site model parameters do not match sim_telarray metadata: "
            f"{[m for m in mismatches if m]}"
        )


def _extract_parameter_value(metadata, sim_telarray_name, parameter_type):
    """Extract a parameter value from sim_telarray metadata based on its type."""
    value = metadata[sim_telarray_name]
    if parameter_type in ("string", "file", "dict", "boolean"):
        return int(value) if isinstance(value, str) and value.isnumeric() else value

    if isinstance(value, str):
        value = value.removeprefix("all:").strip()
        column = value.split(",") if "," in value else value.split()
    else:
        column = [value]
    column = [None if item is None or str(item).lower() == "none" else item for item in column]
    if parameter_type == "bool":
        column = [bool(int(item)) for item in column]
    if len(column) == 1:
        return None if column[0] is None else np.array(column, dtype=np.dtype(parameter_type))[0]
    return np.array(column, dtype=np.dtype(parameter_type))


def _assert_model_parameters(metadata, model, allow_for_changes=None):
    """
    Return the list of model parameters whose values do not match the sim_telarray metadata.

    Parameters
    ----------
    metadata: dict
        Metadata dictionary.
    model: SiteModel, TelescopeModel
        Model to compare with.
    allow_for_changes: list
        Model parameters changed with the command line ('-C'); metadata may differ for these.
    """
    invalid_parameter_list = []
    for param in model.parameters:
        # sim_telarray names: "reference_point_latitude" -> "latitude", "altitude" -> ...
        sim_telarray_name = param.replace("reference_point_", "")
        if sim_telarray_name == "altitude":
            sim_telarray_name = "corsika_observation_level"
        if sim_telarray_name == "array_triggers" or sim_telarray_name not in metadata:
            continue

        parameter_type = model.parameters[param]["type"]
        model_value = model.parameters[param]["value"]
        if param == "mirror_list" and uses_segmented_dual_mirror_geometry(model.parameters):
            model_value = "none"
        elif parameter_type == "file":
            model_value = _resolve_file_parameter_value(model_value, param, model)
        value = _extract_parameter_value(metadata, sim_telarray_name, parameter_type)

        if is_equal(value, model_value, parameter_type):
            continue
        if allow_for_changes and param in allow_for_changes:
            _logger.warning(
                f"Parameter {param} mismatch between sim_telarray file: {value}, "
                f"and model: {model_value}, but allowed for changes."
            )
            continue
        invalid_parameter_list.append(
            f"Parameter {param} mismatch between sim_telarray file: {value}, "
            f"and model: {model_value}"
        )
    return invalid_parameter_list


def _resolve_file_parameter_value(model_value, parameter_name, model):
    """Resolve the generated sim_telarray filename for an ECSV file parameter."""
    if not isinstance(model_value, str) or not model_value.lower().endswith(".ecsv"):
        return model_value

    shared_name = get_simtel_table_file_name(
        getattr(model, "parameters", {}).get(parameter_name, {})
    )
    if shared_name is not None:
        return shared_name

    model_name = getattr(model, "name", None)
    config_file_path = getattr(model, "config_file_path", None)
    if isinstance(model_name, str) and isinstance(config_file_path, (str, Path)):
        return f"{parameter_name}-{Path(config_file_path).stem}.dat"
    return model_value


def _assert_sim_telarray_seed(metadata, sim_telarray_seed, file=None):
    """
    Assert that sim_telarray seed matches the values in the sim_telarray metadata.

    Seeds are regenerated with ``random.seeds`` and compared with the metadata.

    Returns an error message if the seeds do not match, None otherwise.

    Parameters
    ----------
    metadata: dict
        Metadata dictionary.
    sim_telarray_seed: SimtelSeeds
        sim_telarray seed.
    file : Path
        sim_telarray file.
    """
    if sim_telarray_seed is None:
        return None
    if "instrument_seed" not in metadata or "instrument_instances" not in metadata:
        return None

    if str(metadata.get("instrument_seed")) != str(sim_telarray_seed.instrument_seed):
        return (
            "Parameter instrument_seed mismatch between sim_telarray file: "
            f"{metadata['instrument_seed']}, and model: {sim_telarray_seed.instrument_seed}"
        )
    _logger.info(
        f"sim_telarray_seed in sim_telarray file: {metadata['instrument_seed']}, "
        f"and model: {sim_telarray_seed.instrument_seed}"
    )
    if file:
        run_number_modified = get_corsika_run_number(file) - 1
        test_seeds = random.seeds(
            n_seeds=int(metadata["instrument_instances"]),
            max_seed=np.iinfo(np.int32).max,
            fixed_seed=int(metadata["instrument_seed"]),
        )
        # no +1 as in sim_telarray (as we count from 0)
        seed_used = run_number_modified % int(metadata["instrument_instances"])
        if str(metadata.get("rng_select_seed")) != str(test_seeds[seed_used]):
            return (
                "Parameter rng_select_seed mismatch between sim_telarray file: "
                f"{metadata['rng_select_seed']}, and model: {test_seeds[seed_used]}"
            )
    return None


def is_equal(value1, value2, value_type):
    """
    Check if two values are equal based on their type.

    Parameters
    ----------
    value1, value2: any
        Values to compare.
    value_type: str
        Type of the values ('string', 'dict', etc.).

    Returns
    -------
    bool
        True if the values are equal, False otherwise.
    """
    value1 = value1[0] if isinstance(value1, tuple) else value1
    value2 = value2[0] if isinstance(value2, tuple) else value2
    if (value1 is None or value2 is None) and value1 in ("none", None) and value2 in ("none", None):
        return True
    if value_type in ("string", "file"):
        return str(value1).strip() == str(value2).strip()
    if value_type == "dict":
        return value1 == value2
    if value_type == "boolean":
        return bool(value1) == bool(value2)
    return _is_numeric_equal(value1, value2)


def _is_numeric_equal(value1, value2):
    """Compare numeric values, arrays and lists of repeated numbers with a relative tolerance."""
    numbers = (float, int, np.integer, np.floating)
    if isinstance(value1, np.ndarray | list) and isinstance(value2, np.ndarray | list):
        return bool(np.allclose(np.array(value1), np.array(value2), rtol=1e-10))
    if isinstance(value1, list) and isinstance(value2, numbers):
        if all(x == value1[0] for x in value1):
            return bool(np.isclose(float(value1[0]), float(value2), rtol=1e-10))
    if isinstance(value2, list) and isinstance(value1, numbers):
        if all(x == value2[0] for x in value2):
            return bool(np.isclose(float(value1), float(value2[0]), rtol=1e-10))
    return bool(np.isclose(float(value1), float(value2), rtol=1e-10))


def validate_sim_telarray_event_numbers(data_files, expected_mc_events, expected_shower_events):
    """
    Verify the number of simulated MC and shower events in sim_telarray output files.

    Parameters
    ----------
    data_files: list of Path
        sim_telarray output files.
    expected_mc_events, expected_shower_events: int
        Expected numbers of simulated MC and shower events.
    """
    event_errors = []
    for file in value_conversion.ensure_list(data_files):
        shower_events, mc_events = file_info.get_simulated_events(file)
        if (shower_events, mc_events) != (expected_shower_events, expected_mc_events):
            event_errors.append(
                f"Event mismatch: shower/MC events in {file}: {shower_events}/{mc_events}"
                f" (expected: {expected_shower_events}/{expected_mc_events})"
            )
        else:
            _logger.info(
                "Consistent number of events in sim_telarray output: "
                f"shower events: {shower_events} (expected {expected_shower_events}), "
                f"MC events: {mc_events} (expected {expected_mc_events})"
                f" (file: {file})"
            )

    if event_errors:
        _logger.error("Inconsistent event counts found:")
        for error in event_errors:
            _logger.error(f" - {error}")
        _raise_inconsistent("Inconsistent event counts found:", event_errors)


def assert_n_showers_and_energy_range(file):
    """
    Assert the number of showers (1% tolerance) and the energy range of a sim_telarray file.

    Parameters
    ----------
    file: Path
        sim_telarray file.
    """
    with SimTelFile(file, skip_non_triggered=False) as f:
        simulation_config = f.mc_run_headers[0]
        try:
            simulated_energies = [event["mc_shower"]["energy"] for event in f]
        except KeyError as exc:
            raise KeyError(
                f"Expected 'mc_shower' information in sim_telarray file {file} for checking "
                "number of showers and energy range, but it was not found."
            ) from exc

    # Inconsistent filling of run header depending on how simulations are run
    # The relative tolerance is set to 1% because ~0.5% shower simulations do not
    # succeed, without resulting in an error. This tolerance therefore is not an issue.
    n_showers = simulation_config["n_showers"]
    consistent_n_showers = np.isclose(len(simulated_energies), n_showers, rtol=1e-2)
    consistent_n_showers_unique = np.isclose(
        len(np.unique(simulated_energies)), n_showers, rtol=1e-2
    )
    if not consistent_n_showers and not consistent_n_showers_unique:
        raise ValueError(
            f"Number of showers in sim_telarray file {file} does not match the configuration. "
            f"Simulated showers: {len(simulated_energies)}, "
            f"Unique simulated showers: {len(np.unique(simulated_energies))}, "
            f"Configuration: {n_showers}"
        )

    e_min, e_max = simulation_config["E_range"]
    if not all(e_min <= energy <= e_max for energy in simulated_energies):
        raise ValueError(
            f"Energy range in sim_telarray file {file} does not match "
            f"the configuration. Simulated energies: {simulated_energies}, "
            f"configuration: {simulation_config}"
        )
    return True


def assert_expected_sim_telarray_output_and_event_type(
    file, expected_sim_telarray_output, event_type="shower"
):
    """
    Validate expected event values and type during one file traversal.

    Parameters
    ----------
    file: Path
        Sim_telarray output file.
    expected_sim_telarray_output: dict or None
        Expected event values and optional event type.
    event_type: str
        Expected event type (default: "shower").

    Returns
    -------
    bool
        True when all requested event checks pass.
    """
    expected = expected_sim_telarray_output or {}
    event_types = set()
    item_to_check = _read_sim_telarray_items(file, expected, event_types)
    output_valid = True
    if expected_sim_telarray_output is not None:
        _logger.debug(
            "Extracted event numbers from sim_telarray file: "
            f"telescope events: {item_to_check['n_telescope_events']}, "
            f"calibration events: {item_to_check['n_calibration_events']}"
        )
        for key, value in expected_sim_telarray_output.items():
            if key == "event_type":
                continue
            if len(item_to_check[key]) == 0:
                _logger.error(f"No data found for {key}")
                output_valid = False
                break
            if not value[0] < np.mean(item_to_check[key]) < value[1]:
                _logger.error(
                    f"Mean of {key} is not in the expected range, got {np.mean(item_to_check[key])}"
                )
                output_valid = False
                break
    event_type_valid = _assert_event_type_in_set(event_types, event_type, file)
    return output_valid and event_type_valid


def _read_sim_telarray_items(file, expected, event_types):
    """Read the items needed for checking against expected output in one traversal."""
    item_to_check = defaultdict(list)
    item_to_check["n_telescope_events"] = 0
    item_to_check["n_calibration_events"] = 0
    with SimTelFile(file) as f:
        for event in f:
            if "trigger_time" in expected:
                item_to_check["trigger_time"].extend(event["trigger_information"]["trigger_times"])
            event_types.add(event["type"])
            if event["type"] == "calibration":
                item_to_check["n_calibration_events"] += 1
            elif "pe_sum" in expected or "photons" in expected:
                sums = event["photoelectron_sums"]
                if "pe_sum" in expected:
                    item_to_check["pe_sum"].extend(sums["n_pe"][sums["n_pe"] > 0])
                if "photons" in expected:
                    item_to_check["photons"].extend(sums["photons_atm_qe"][sums["photons"] > 0])
            if len(event.get("telescope_events", [])) > 0:
                item_to_check["n_telescope_events"] += 1
    return item_to_check


def assert_expected_sim_telarray_metadata(file, expected_sim_telarray_metadata):
    """
    Assert that expected metadata is present in the sim_telarray file.

    Parameters
    ----------
    file: Path
        sim_telarray file.
    expected_sim_telarray_metadata: dict
        Expected metadata values.
    """
    if expected_sim_telarray_metadata is None:
        return True
    global_meta, telescope_meta = read_sim_telarray_metadata(file)

    for key, value in expected_sim_telarray_metadata.items():
        if key not in global_meta and key not in telescope_meta:
            _logger.error(f"Metadata key {key} not found in sim_telarray file {file}")
            return False
        if key in global_meta and global_meta[key] != value:
            _logger.error(
                f"Metadata key {key} has value {global_meta[key]} instead of expected {value}"
            )
            return False
        _logger.debug(f"Metadata key {key} matches expected value {value}")
    return True


def assert_events_of_type(file, event_type="shower"):
    """
    Assert that events of the expected type are present in the sim_telarray file.

    Parameters
    ----------
    file: Path
        sim_telarray file.
    event_type: str
        Expected event type (e.g., "shower", "flasher").
    """
    expected_event_type = _simtel_event_type(event_type)
    with SimTelFile(file) as f:
        for event in f:
            if event["type"] == expected_event_type:
                return True
    return _assert_event_type_in_set(set(), event_type, file)


def _simtel_event_type(event_type):
    """Return the eventio type corresponding to a sim_telarray event type."""
    return "calibration" if event_type in ("pedestal", "direct_injection") else "data"


def _assert_event_type_in_set(event_types, event_type, file):
    """Check whether an expected sim_telarray event type was observed."""
    if _simtel_event_type(event_type) in event_types:
        return True
    _logger.error(f"No events of type {event_type} found in sim_telarray file {file}")
    return False


# --- CORSIKA ----------------------------------------------------------------------------


def validate_corsika_output(data_files, log_files, expected_shower_events=None, curved_atmo=False):
    """
    Validate CORSIKA output files.

    Parameters
    ----------
    data_files, log_files: list of Path
        CORSIKA data and log files.
    expected_shower_events: int, optional
        Expected number of shower events.
    curved_atmo: bool, optional
        Whether CORSIKA was run with the curved atmosphere option.
    """
    validate_corsika_event_numbers(data_files, expected_shower_events)
    validate_corsika_logs(
        log_files, expected_shower_events=expected_shower_events, curved_atmo=curved_atmo
    )


def validate_corsika_event_numbers(data_files, expected_shower_events=None, tolerance=1.0e-3):
    """
    Validate the number of simulated events in CORSIKA output files.

    A small relative mismatch (tolerance) is only logged as a warning.

    Parameters
    ----------
    data_files: list of Path
        CORSIKA data files.
    expected_shower_events: int
        Expected number of simulated shower events.
    tolerance: float
        Relative tolerance for a small mismatch.
    """
    if expected_shower_events is None:
        _logger.info("Skipping CORSIKA event-count validation: no expected count was supplied.")
        return

    event_errors = []
    for file in value_conversion.ensure_list(data_files):
        shower_events, _ = file_info.get_simulated_events(file)
        if shower_events == expected_shower_events:
            _logger.info(
                f"Consistent number of events in: {file}: shower events: {shower_events}"
                f" (expected: {expected_shower_events})"
            )
        elif (
            abs(shower_events - expected_shower_events) / max(shower_events, expected_shower_events)
            <= tolerance
        ):
            _logger.warning(
                f"Small mismatch in number of events in: {file}: "
                f"shower events: {shower_events} (expected: {expected_shower_events})"
            )
        else:
            event_errors.append(
                f"Number of simulated shower events ({shower_events}) does not match "
                f"the expected number ({expected_shower_events}) in CORSIKA {file}."
            )

    if event_errors:
        _raise_inconsistent("Inconsistent event counts found in CORSIKA output:", event_errors)


def validate_corsika_logs(log_files, expected_shower_events=None, curved_atmo=False):
    """
    Validate CORSIKA log files (raises ValueError if the expected patterns are missing).

    Parameters
    ----------
    log_files: list of Path
        CORSIKA log files.
    expected_shower_events: int, optional
        Expected number of shower events.
    curved_atmo: bool, optional
        Whether CORSIKA was run with the curved atmosphere option.
    """
    patterns = ["========== END OF RUN ======="]
    forbidden = []
    if expected_shower_events is not None:
        patterns.append(f"NUMBER OF GENERATED EVENTS =          {expected_shower_events}")
    if curved_atmo:
        patterns.append("CURVED VERSION WITH SLIDING PLANAR ATMOSPHERE")
    else:
        forbidden.append("CORSIKA was compiled without CURVED option.")

    if not check_plain_logs(log_files, {"pattern": patterns, "forbidden_pattern": forbidden}):
        raise ValueError(
            f"Log files {log_files} do not contain expected patterns "
            "indicating successful completion."
        )


# --- reduced event lists and trigger histograms -----------------------------------------


def validate_sim_events(data_files, expected_mc_events):
    """
    Validate that the number of simulated events in reduced event lists matches the expectation.

    Parameters
    ----------
    data_files: str, Path, list
        Reduced event list files (HDF5).
    expected_mc_events: int
        Expected number of simulated MC events.
    """
    event_errors = []
    for data_file in value_conversion.ensure_list(data_files):
        if table_handler.read_table_file_type([data_file]) != "HDF5":
            raise ValueError(
                f"Unsupported reduced event data format for '{data_file}'. "
                "Only HDF5 files with suffix '.hdf5' or '.h5' are supported."
            )
        tables = table_handler.read_tables(data_file, ["SHOWERS"], file_type="HDF5")
        try:
            mc_events = len(tables["SHOWERS"])
        except KeyError as exc:
            raise ValueError(f"SHOWERS table not found in reduced event list {data_file}.") from exc

        if mc_events != expected_mc_events:
            event_errors.append(
                f"Number of simulated MC events ({mc_events}) does not match "
                f"the expected number ({expected_mc_events}) in reduced event list {data_file}."
            )
        else:
            _logger.info(
                f"Consistent number of events in reduced event list: {data_file}: MC events:"
                f" {mc_events} (expected: {expected_mc_events})"
            )

    if event_errors:
        _logger.error("Inconsistent event counts found in reduced event lists:")
        for error in event_errors:
            _logger.error(f" - {error}")
        _raise_inconsistent("Inconsistent event counts found in reduced event lists:", event_errors)


def validate_reduced_event_data_file(data_file):
    """
    Validate structure, tables, metadata, and references of one reduced-event file.

    Parameters
    ----------
    data_file: str or pathlib.Path
        HDF5 reduced-event file to validate.

    Returns
    -------
    bool
        True when the file passes all checks.
    """
    if table_handler.read_table_file_type([data_file]) != "HDF5":
        raise ValueError(f"Reduced event data file '{data_file}' must be an HDF5 file.")

    required_entries = [*_REDUCED_EVENT_TABLE_SCHEMAS, "METADATA", "SIMULATION_METADATA"]
    _check_entries_present(
        data_file, table_handler.read_table_list(data_file, required_entries), "Reduced event data"
    )

    tables = table_handler.read_tables(
        data_file, list(_REDUCED_EVENT_TABLE_SCHEMAS), file_type="HDF5"
    )
    for table_name, schema_file in _REDUCED_EVENT_TABLE_SCHEMAS.items():
        validate_data.DataValidator(
            schema_file=schema_file,
            data_table=tables[table_name],
        ).validate_and_transform()

    standard_metadata = table_handler.read_metadata_document(data_file, "METADATA")
    schema.validate_dict_using_schema(standard_metadata, schema_file=METADATA_JSON_SCHEMA)
    validate_simulation_metadata(
        table_handler.read_metadata_document(data_file, "SIMULATION_METADATA")
    )
    _validate_reduced_event_table_references(tables, data_file)
    return True


def _check_entries_present(data_file, available_entries, label):
    """Raise ValueError if any required entry is missing from a file."""
    missing_entries = [name for name, entries in available_entries.items() if not entries]
    if missing_entries:
        raise ValueError(
            f"{label} file '{data_file}' is missing required entries: {', '.join(missing_entries)}."
        )


def validate_trigger_histogram_file(data_file):
    """
    Validate structure, tables, metadata, and references of one trigger-histogram file.

    Parameters
    ----------
    data_file: str or pathlib.Path
        HDF5 trigger-histogram file to validate.

    Returns
    -------
    bool
        True when the file passes all checks.
    """
    if table_handler.read_table_file_type([data_file]) != "HDF5":
        raise ValueError(f"Trigger histogram file '{data_file}' must be an HDF5 file.")

    _check_entries_present(
        data_file,
        table_handler.read_table_list(data_file, list(_TRIGGER_HISTOGRAM_TABLES)),
        "Trigger histogram",
    )
    standard_metadata = table_handler.read_metadata_document(data_file, "METADATA")
    schema.validate_dict_using_schema(standard_metadata, schema_file=METADATA_JSON_SCHEMA)
    tables = table_handler.read_tables(data_file, list(_TRIGGER_HISTOGRAM_TABLES), file_type="HDF5")
    for table_name, schema_file in _TRIGGER_HISTOGRAM_TABLE_SCHEMAS.items():
        validate_data.DataValidator(
            schema_file=schema_file,
            data_table=tables[table_name],
        ).validate_and_transform()
    reference_ids = _validate_trigger_histogram_table_references(tables, data_file)
    _validate_trigger_histogram_dense_payload(data_file, reference_ids)
    return True


def validate_trigger_histograms_file(data_file):
    """Validate a trigger-histogram product using its registered product name."""
    return validate_trigger_histogram_file(data_file)


def _validate_trigger_histogram_table_references(tables, data_file):
    """Validate reference IDs used by trigger-histogram tables."""
    reference_ids = {str(row["reference_id"]) for row in tables[TRIGGER_HISTOGRAM_METADATA_TABLE]}
    if not reference_ids:
        raise ValueError(f"Trigger histogram file '{data_file}' has no reference metadata rows.")
    if len(reference_ids) != len(tables[TRIGGER_HISTOGRAM_METADATA_TABLE]):
        raise ValueError(f"Trigger histogram file '{data_file}' has duplicate reference IDs.")

    for table_name in _TRIGGER_HISTOGRAM_TABLES[1:]:
        table_reference_ids = {str(row["reference_id"]) for row in tables[table_name]}
        unknown_ids = table_reference_ids.difference(reference_ids)
        if unknown_ids:
            raise ValueError(
                f"Trigger histogram file '{data_file}' table '{table_name}' references unknown "
                f"reference IDs: {sorted(unknown_ids)}."
            )

    bin_reference_ids = {str(row["reference_id"]) for row in tables[TRIGGER_HISTOGRAM_BINS_TABLE]}
    missing_bin_ids = reference_ids.difference(bin_reference_ids)
    if missing_bin_ids:
        raise ValueError(
            f"Trigger histogram file '{data_file}' has references without histogram bins: "
            f"{sorted(missing_bin_ids)}."
        )
    return reference_ids


def _validate_trigger_histogram_dense_payload(data_file, reference_ids):
    """Validate dense histogram groups and their links to reference metadata."""
    prefix = f"Trigger histogram file '{data_file}'"
    with h5py.File(data_file, "r") as hdf5_file:
        if TRIGGER_HISTOGRAM_DENSE_GROUP not in hdf5_file:
            raise ValueError(f"{prefix} has no dense payload group.")
        dense_group = hdf5_file[TRIGGER_HISTOGRAM_DENSE_GROUP]
        if not isinstance(dense_group, h5py.Group):
            raise ValueError(f"{prefix} dense payload is not a group.")
        dense_reference_ids = set(dense_group.keys())
        if dense_reference_ids != reference_ids:
            raise ValueError(
                f"{prefix} dense reference IDs do not match metadata: "
                f"metadata={sorted(reference_ids)}, dense={sorted(dense_reference_ids)}."
            )
        for reference_id, reference_group in dense_group.items():
            if not isinstance(reference_group, h5py.Group) or not reference_group:
                raise ValueError(f"{prefix} reference '{reference_id}' has no payload.")
            for histogram_name, histogram_group in reference_group.items():
                _validate_dense_histogram_payload(histogram_name, histogram_group, data_file)


def _validate_dense_histogram_payload(histogram_name, histogram_group, data_file):
    """Validate one dense histogram's values and bin-edge datasets."""
    prefix = f"Trigger histogram file '{data_file}' payload '{histogram_name}'"
    if not isinstance(histogram_group, h5py.Group):
        raise ValueError(f"{prefix} is not a group.")
    values = histogram_group.get("values")
    if not isinstance(values, h5py.Dataset):
        raise ValueError(f"{prefix} has no values dataset.")
    if values.ndim == 0 or not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"{prefix} has invalid values data.")

    edge_names = [name for name in histogram_group if name != "values"]
    expected_edge_names = [f"edges_{axis_index}" for axis_index in range(values.ndim)]
    if set(edge_names) != set(expected_edge_names):
        raise ValueError(
            f"{prefix} has invalid bin-edge datasets: expected {expected_edge_names}, "
            f"found {sorted(edge_names)}."
        )
    for axis_index, edge_name in enumerate(expected_edge_names):
        edges = histogram_group[edge_name]
        if not isinstance(edges, h5py.Dataset) or edges.ndim != 1:
            raise ValueError(
                f"{prefix} edge dataset '{edge_name}' is not a one-dimensional dataset."
            )
        if not np.issubdtype(edges.dtype, np.number):
            raise ValueError(f"{prefix} edge dataset '{edge_name}' is not numeric.")
        expected_length = values.shape[axis_index] + 1
        if len(edges) != expected_length:
            raise ValueError(
                f"{prefix} edge dataset '{edge_name}' has {len(edges)} values, "
                f"expected {expected_length}."
            )


def _validate_reduced_event_table_references(tables, data_file):
    """Validate file and event references between reduced-event tables."""
    file_ids = set(np.asarray(tables["FILE_INFO"]["file_id"], dtype=int))
    for table_name in ("SHOWERS", "TRIGGERS"):
        unknown_file_ids = set(np.asarray(tables[table_name]["file_id"], dtype=int)).difference(
            file_ids
        )
        if unknown_file_ids:
            raise ValueError(
                f"Reduced event data file '{data_file}' table '{table_name}' references unknown "
                f"file_id values: {sorted(map(int, unknown_file_ids))}."
            )

    def _keys(table):
        return {
            tuple(int(row[name]) for name in ("file_id", "event_id", "shower_id")) for row in table
        }

    shower_ids = _keys(tables["SHOWERS"])
    if len(shower_ids) != len(tables["SHOWERS"]):
        raise ValueError(
            f"Reduced event data file '{data_file}' contains duplicate shower composite keys."
        )
    unknown_showers = _keys(tables["TRIGGERS"]).difference(shower_ids)
    if unknown_showers:
        raise ValueError(
            f"Reduced event data file '{data_file}' contains triggers without matching showers: "
            f"{sorted(unknown_showers)}."
        )
