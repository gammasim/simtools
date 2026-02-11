"""Validation of sim_telarray data and metadata."""

import logging
from collections import defaultdict

import numpy as np
from eventio.simtel.simtelfile import SimTelFile

from simtools.sim_events import file_info
from simtools.sim_events.file_info import get_corsika_run_number
from simtools.simtel.simtel_config_reader import SimtelConfigReader
from simtools.simtel.simtel_io_metadata import (
    get_sim_telarray_telescope_id,
    read_sim_telarray_metadata,
)
from simtools.testing.log_inspector import check_plain_logs
from simtools.utils import general, random

_logger = logging.getLogger(__name__)


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
    Validate sim_telarray output files and metadata.

    Parameters
    ----------
    data_files: list of Path
        List of sim_telarray output files.
    log_files: list of Path
        List of sim_telarray log files.
    array_models: list of ArrayModel
        List of array models to compare with.
    expected_mc_events: int
        Expected number of MC events in the sim_telarray output files.
    expected_shower_events: int
        Expected number of shower events in the sim_telarray output files.
    curved_atmo: bool
        CORSIKA executable compiled with curved atmosphere option.
    allow_for_changes: list
        List of model parameters that are changed from command line ('-C')
        Metadata checks allows these values to be different than expected from model.

    Raises
    ------
    ValueError
        If the sim_telarray output files or metadata are not consistent with the array models.
    """
    data_files = general.ensure_iterable(data_files)
    log_files = general.ensure_iterable(log_files)

    if array_models:
        validate_metadata(data_files, array_models, allow_for_changes)

    validate_log_files(log_files, expected_mc_events, expected_shower_events, curved_atmo)
    if expected_mc_events and expected_shower_events:
        validate_event_numbers(data_files, expected_mc_events, expected_shower_events)


def validate_metadata(files, array_models, allow_for_changes=None):
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


def validate_log_files(
    log_files, expected_mc_events=None, expected_shower_events=None, curved_atmo=False
):
    """Validate sim_telarray log files."""
    event_string = (
        (
            f"Run(s) completed as expected after {expected_mc_events} events "
            f"({expected_shower_events} showers)."
        )
        if expected_mc_events and expected_shower_events
        else ""
    )
    curved_good = None
    curved_bad = None
    if curved_atmo:
        curved_good = "CORSIKA was compiled with CURVED option."
    else:
        curved_bad = "CORSIKA was compiled with CURVED option."
    if not check_plain_logs(
        log_files,
        {
            "pattern": [
                "Finished.",
                "Sim_telarray finished at",
                event_string,
                curved_good,
            ],
            "forbidden_pattern": [
                curved_bad,
            ],
        },
    ):
        raise ValueError(f"Sim_telarray log files validation failed for {log_files}")
    _logger.info(f"Sim_telarray log files validation passed: {log_files}")


def assert_sim_telarray_metadata(file, array_model, allow_for_changes=None):
    """
    Assert consistency of sim_telarray metadata with given array model.

    Parameters
    ----------
    file: Path
        Path to the sim_telarray file.
    array_model: ArrayModel
        Array model to compare with.
    allow_for_changes: list
        List of model parameters that are changed from command line ('-C')
        Metadata checks allows these values to be different than expected from model.
    """
    global_meta, telescope_meta = read_sim_telarray_metadata(file)
    _logger.info(f"Found metadata in sim_telarray file for {len(telescope_meta)} telescopes")
    site_parameter_mismatch = _assert_model_parameters(
        global_meta, array_model.site_model, allow_for_changes
    )
    sim_telarray_seed_mismatch = _assert_sim_telarray_seed(
        global_meta, array_model.sim_telarray_seed, file
    )
    if sim_telarray_seed_mismatch:
        site_parameter_mismatch.append(sim_telarray_seed_mismatch)

    if len(telescope_meta) != len(array_model.telescope_models):
        raise ValueError(
            f"Number of telescopes in sim_telarray file ({len(telescope_meta)}) does not match "
            f"number of telescopes in array model ({len(array_model.telescope_models)})"
        )

    for telescope_name in array_model.telescope_models.keys():
        if not get_sim_telarray_telescope_id(telescope_name, file):
            raise ValueError(f"Telescope {telescope_name} not found in sim_telarray file metadata")

    telescope_parameter_mismatch = [
        _assert_model_parameters(telescope_meta[i], model, allow_for_changes)
        for i, model in enumerate(array_model.telescope_models.values(), start=1)
    ]

    # ensure printout of all mismatches, not only those found first
    telescope_parameter_mismatch.append(site_parameter_mismatch)
    if any(len(m) > 0 for m in telescope_parameter_mismatch):
        mismatches = [m for m in telescope_parameter_mismatch if len(m) > 0]
        raise ValueError(
            f"Telescope or site model parameters do not match sim_telarray metadata: {mismatches}"
        )


def _assert_model_parameters(metadata, model, allow_for_changes=None):
    """
    Assert that model parameter values matches the values in the sim_telarray metadata.

    Parameters
    ----------
    metadata: dict
        Metadata dictionary.
    model: SiteModel, TelescopeModel
        Model to compare with.
    allow_for_changes: list
        List of model parameters that are changed from command line ('-C')
        Metadata checks allows these values to be different than expected from model.

    Returns
    -------
    invalid_parameter_list: list
        List of parameters that do not match.

    """
    config_reader = SimtelConfigReader()

    invalid_parameter_list = []

    for param in model.parameters:
        sim_telarray_name = _sim_telarray_name_from_parameter_name(param)
        if sim_telarray_name in metadata.keys():
            parameter_type = model.parameters[param]["type"]
            if parameter_type not in ("string", "dict", "boolean"):
                value, _ = config_reader.extract_value_from_sim_telarray_column(
                    [metadata[sim_telarray_name]], parameter_type
                )
            else:
                value = metadata[sim_telarray_name]
                value = (int)(value) if value.isnumeric() else value

            if not is_equal(value, model.parameters[param]["value"], parameter_type):
                if allow_for_changes and param in allow_for_changes:
                    _logger.warning(
                        f"Parameter {param} mismatch between sim_telarray file: {value}, "
                        f"and model: {model.parameters[param]['value']}, but allowed for changes."
                    )
                else:
                    invalid_parameter_list.append(
                        f"Parameter {param} mismatch between sim_telarray file: {value}, "
                        f"and model: {model.parameters[param]['value']}"
                    )

    return invalid_parameter_list


def _assert_sim_telarray_seed(metadata, sim_telarray_seed, file=None):
    """
    Assert that sim_telarray seed matches the values in the sim_telarray metadata.

    Regenerate seeds using the sim_telarray_random_seeds function and compare with the metadata.

    Parameters
    ----------
    metadata: dict
        Metadata dictionary.
    sim_telarray_seed: SimtelSeeds
        sim_telarray seed.
    file : Path
        Path to the sim_telarray file.

    Returns
    -------
    invalid_parameter_list: list
        Error message if sim_telarray seeds do not match.

    """
    if sim_telarray_seed is None:
        return None

    if "instrument_seed" in metadata.keys() and "instrument_instances" in metadata.keys():
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


def _sim_telarray_name_from_parameter_name(parameter_name):
    """Return sim_telarray parameter name. Some specific fine tuning."""
    # parameters like "reference_point_latitude"
    sim_telarray_name = parameter_name.replace("reference_point_", "")

    if sim_telarray_name == "altitude":
        return "corsika_observation_level"
    if sim_telarray_name == "array_triggers":
        return None

    return sim_telarray_name


def is_equal(value1, value2, value_type):
    """
    Check if two values are equal based on their type.

    The complexity of this function reflects the complexity of the sim_telarray
    metadata output.

    Parameters
    ----------
    value1: any
        First value to compare.
    value2: any
        Second value to compare.
    value_type: str
        Type of the values ('string', 'dict', etc.).

    Returns
    -------
    bool
        True if the values are equal, False otherwise.
    """
    value1 = value1[0] if isinstance(value1, tuple) else value1
    value2 = value2[0] if isinstance(value2, tuple) else value2
    if value1 is None or value2 is None:
        if value1 in ("none", None) and value2 in ("none", None):
            return True
    if value_type == "string":
        return str(value1).strip() == str(value2).strip()
    if value_type == "dict":
        return value1 == value2
    if value_type == "boolean":
        return bool(value1) == bool(value2)
    return _is_equal_floats_or_ints(value1, value2)


def _is_equal_floats_or_ints(value1, value2):
    """Check if floats and ints are equal."""
    if isinstance(value1, np.ndarray | list) and isinstance(value2, np.ndarray | list):
        return bool(np.allclose(np.array(value1), np.array(value2), rtol=1e-10))
    if isinstance(value1, list) and isinstance(value2, float | int | np.integer | np.floating):
        if all(x == value1[0] for x in value1):
            return bool(np.isclose(float(value1[0]), float(value2), rtol=1e-10))
    if isinstance(value2, list) and isinstance(value1, float | int | np.integer | np.floating):
        if all(x == value2[0] for x in value2):
            return bool(np.isclose(float(value1), float(value2[0]), rtol=1e-10))
    return bool(np.isclose(float(value1), float(value2), rtol=1e-10))


def validate_event_numbers(data_files, expected_mc_events, expected_shower_events):
    """
    Verify the number of simulated events. Loops over all events in sim_telarray output files.

    Parameters
    ----------
    data_files: list of Path
        List of sim_telarray output files.
    expected_mc_events: int
        Expected number of simulated MC events.
    expected_shower_events: int
        Expected number of simulated shower events.

    Raises
    ------
    ValueError
        If the number of simulated events does not match the expected number.
    """
    event_errors = []
    for file in general.ensure_iterable(data_files):
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
        error_message = "Inconsistent event counts found:\n" + "\n".join(
            f" - {error}" for error in event_errors
        )
        raise ValueError(error_message)


def assert_n_showers_and_energy_range(file):
    """
    Assert the number of showers and the energy range.

    The number of showers should be consistent with the required one (up to 1% tolerance)
    and the energies simulated are required to be within the configured ones.

    Parameters
    ----------
    file: Path
        Path to the sim_telarray file.
    """
    simulated_energies = []
    simulation_config = {}
    with SimTelFile(file, skip_non_triggered=False) as f:
        simulation_config = f.mc_run_headers[0]
        try:
            simulated_energies.extend(event["mc_shower"]["energy"] for event in f)
        except KeyError as exc:
            raise KeyError(
                f"Expected 'mc_shower' information in sim_telarray file {file} for checking "
                "number of showers and energy range, but it was not found."
            ) from exc

    # Inconsistent filling of run header depending on how simulations are run
    # The relative tolerance is set to 1% because ~0.5% shower simulations do not
    # succeed, without resulting in an error. This tolerance therefore is not an issue.
    consistent_n_showers = np.isclose(
        len(simulated_energies), simulation_config["n_showers"], rtol=1e-2
    )
    consistent_n_showers_unique = np.isclose(
        len(np.unique(simulated_energies)), simulation_config["n_showers"], rtol=1e-2
    )
    if not consistent_n_showers and not consistent_n_showers_unique:
        raise ValueError(
            f"Number of showers in sim_telarray file {file} does not match the configuration. "
            f"Simulated showers: {len(simulated_energies)}, "
            f"Unique simulated showers: {len(np.unique(simulated_energies))}, "
            f"Configuration: {simulation_config['n_showers']}"
        )

    consistent_energy_range = all(
        simulation_config["E_range"][0] <= energy <= simulation_config["E_range"][1]
        for energy in simulated_energies
    )

    if not consistent_energy_range:
        raise ValueError(
            f"Energy range in sim_telarray file {file} does not match "
            f"the configuration. Simulated energies: {simulated_energies}, "
            f"configuration: {simulation_config}"
        )

    return True


def assert_expected_sim_telarray_output(file, expected_sim_telarray_output):
    """
    Assert that the expected output is present in the sim_telarray file.

    Parameters
    ----------
    file: Path
        Path to the sim_telarray file.
    expected_sim_telarray_output: dict
        Expected output values.

    """
    if expected_sim_telarray_output is None:
        return True

    item_to_check = _item_to_check_from_sim_telarray(file, expected_sim_telarray_output)
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
            return False

        if not value[0] < np.mean(item_to_check[key]) < value[1]:
            _logger.error(
                f"Mean of {key} is not in the expected range, got {np.mean(item_to_check[key])}"
            )
            return False

    return True


def _item_to_check_from_sim_telarray(file, expected_sim_telarray_output):
    """Read the relevant items from the sim_telarray file for checking against expected output."""
    item_to_check = defaultdict(list)
    for key in ("n_telescope_events", "n_calibration_events"):
        item_to_check[key] = 0
    with SimTelFile(file) as f:
        for event in f:
            if "pe_sum" in expected_sim_telarray_output:
                item_to_check["pe_sum"].extend(
                    event["photoelectron_sums"]["n_pe"][event["photoelectron_sums"]["n_pe"] > 0]
                )
            if "trigger_time" in expected_sim_telarray_output:
                item_to_check["trigger_time"].extend(event["trigger_information"]["trigger_times"])
            if "photons" in expected_sim_telarray_output:
                item_to_check["photons"].extend(
                    event["photoelectron_sums"]["photons_atm_qe"][
                        event["photoelectron_sums"]["photons"] > 0
                    ]
                )
            if "telescope_events" in event and len(event["telescope_events"]) > 0:
                item_to_check["n_telescope_events"] += 1
            if "type" in event and event["type"] == "calibration":
                item_to_check["n_calibration_events"] += 1

    return item_to_check


def assert_expected_sim_telarray_metadata(file, expected_sim_telarray_metadata):
    """
    Assert that expected metadata is present in the sim_telarray file.

    Parameters
    ----------
    file: Path
        Path to the sim_telarray file.
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
        Path to the sim_telarray file.
    event_type: str
        Expected event type (e.g., "shower", "flasher", etc.).

    """
    expected_event_type = "data"
    if event_type in ("pedestal", "direct_injection"):
        expected_event_type = "calibration"
    with SimTelFile(file) as f:
        for event in f:
            if event["type"] == expected_event_type:
                return True

    _logger.error(f"No events of type {event_type} found in sim_telarray file {file}")
    return False
