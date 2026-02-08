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
    item_to_check = defaultdict(list)
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

    for key, value in expected_sim_telarray_output.items():
        if len(item_to_check[key]) == 0:
            _logger.error(f"No data found for {key}")
            return False

        if not value[0] < np.mean(item_to_check[key]) < value[1]:
            _logger.error(
                f"Mean of {key} is not in the expected range, got {np.mean(item_to_check[key])}"
            )
            return False

    return True


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
        simulated_energies.extend(event["mc_shower"]["energy"] for event in f)

    # The relative tolerance is set to 1% because ~0.5% shower simulations do not
    # succeed, without resulting in an error. This tolerance therefore is not an issue.
    consistent_n_showers = np.isclose(
        len(np.unique(simulated_energies)), simulation_config["n_showers"], rtol=1e-2
    )
    consistent_energy_range = all(
        simulation_config["E_range"][0] <= energy <= simulation_config["E_range"][1]
        for energy in simulated_energies
    )

    return consistent_n_showers and consistent_energy_range
