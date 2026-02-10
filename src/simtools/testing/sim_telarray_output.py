"""Validate sim_telarray output (event data and metadata)."""

import logging
from collections import defaultdict

import numpy as np
from eventio.simtel.simtelfile import SimTelFile

from simtools.simtel.simtel_io_metadata import read_sim_telarray_metadata

_logger = logging.getLogger(__name__)


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
        if key in ("require_telescope_events", "require_calibration_events"):
            test_events = key.replace("require_", "")
            if value and item_to_check.get(f"n_{test_events}", 0) == 0:
                _logger.error(f"Expected {test_events} but found none")
                return False
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


def assert_n_showers_and_energy_range(file, calibration_file=False):
    """
    Assert the number of showers and the energy range.

    The number of showers should be consistent with the required one (up to 1% tolerance)
    and the energies simulated are required to be within the configured ones.

    Parameters
    ----------
    file: Path
        Path to the sim_telarray file.
    calibration_file: bool
        Whether the file is a calibration file.
    """
    simulated_energies = []
    simulation_config = {}
    with SimTelFile(file, skip_non_triggered=False) as f:
        simulation_config = f.mc_run_headers[0]
        try:
            simulated_energies.extend(event["mc_shower"]["energy"] for event in f)
        except KeyError as exc:
            if calibration_file:
                _logger.debug("Skip testing calibration file for showers and energy range")
                return True
            raise KeyError(
                f"Expected 'mc_shower' information in sim_telarray file {file} for checking "
                "number of showers and energy range, but it was not found."
            ) from exc

    # The relative tolerance is set to 1% because ~0.5% shower simulations do not
    # succeed, without resulting in an error. This tolerance therefore is not an issue.
    consistent_n_showers = np.isclose(
        len(np.unique(simulated_energies)), simulation_config["n_showers"], rtol=1e-2
    )
    if not consistent_n_showers:
        _logger.error(
            f"Number of showers in sim_telarray file {file} does not match the configuration. "
            f"Simulated showers: {len(np.unique(simulated_energies))}, "
            f"configuration: {simulation_config['n_showers']}"
        )

    consistent_energy_range = all(
        simulation_config["E_range"][0] <= energy <= simulation_config["E_range"][1]
        for energy in simulated_energies
    )

    if not consistent_energy_range:
        _logger.error(
            f"Energy range in sim_telarray file {file} does not match "
            f"the configuration. Simulated energies: {simulated_energies}, "
            f"configuration: {simulation_config}"
        )

    return consistent_n_showers and consistent_energy_range
