"""Functions asserting certain conditions are met (used e.g., in integration tests)."""

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

_logger = logging.getLogger(__name__)


def assert_file_type(file_type, file_name):
    """
    Assert that the file is of the given type.

    Parameters
    ----------
    file_type: str
        File type (json, yaml).
    file_name: str
        File name.

    """
    if file_type == "json":
        try:
            with open(file_name, encoding="utf-8") as file:
                json.load(file)
            return True
        except (json.JSONDecodeError, FileNotFoundError):
            return False
    if file_type in ("yaml", "yml"):
        if Path(file_name).suffix[1:] not in ("yaml", "yml"):
            return False
        try:
            with open(file_name, encoding="utf-8") as file:
                yaml.safe_load(file)
            return True
        except (yaml.YAMLError, FileNotFoundError):
            return False

    # no dedicated tests for other file types, checking suffix only
    _logger.info(f"File type test is checking suffix only for {file_name} (suffix: {file_type}))")
    return Path(file_name).suffix[1:] == file_type


def check_expected_output(file, expected_output):
    """
    Check if the expected output is present in the sim_telarray file.

    Parameters
    ----------
    file: Path
        Path to the sim_telarray file.
    expected_output: dict
        Expected output values.

    Raises
    ------
    ValueError
        If the file is not a zstd compressed file.
    """
    if not file.suffix == ".zst":
        raise ValueError(
            f"Expected output file {file} is not a zstd compressed file "
            f"(i.e., a sim_telarray file)."
        )

    from eventio.simtel.simtelfile import SimTelFile  # pylint: disable=import-outside-toplevel

    def check_n_showers_and_energy_range(file):
        simulated_energies = []
        simulation_config = {}
        with SimTelFile(file) as f:
            simulation_config = f.mc_run_headers[0]
            for event in f.iter_mc_events():
                simulated_energies.append(event["mc_shower"]["energy"])

        # The relative tolerance is set to 1% because ~0.5% shower simulations do not
        # succeed, without resulting in an error. This tolerance therefore is not an issue.
        assert np.isclose(
            len(np.unique(simulated_energies)), simulation_config["n_showers"], rtol=1e-2
        )
        assert all(
            simulation_config["E_range"][0] <= energy <= simulation_config["E_range"][1]
            for energy in simulated_energies
        )

    def check_telescope_info(file, expected_output):
        item_to_check = defaultdict(list)
        with SimTelFile(file) as f:
            for event in f:
                if "pe_sum" in expected_output:
                    item_to_check["pe_sum"].extend(
                        event["photoelectron_sums"]["n_pe"][event["photoelectron_sums"]["n_pe"] > 0]
                    )
                if "trigger_time" in expected_output:
                    item_to_check["trigger_time"].extend(
                        event["trigger_information"]["trigger_times"]
                    )
                if "photons" in expected_output:
                    item_to_check["photons"].extend(event["photoelectron_sums"]["photons_atm_qe"])

        for key, value in expected_output.items():
            assert len(item_to_check[key]) > 0, f"No data found for {key}"
            assert (
                value[0] < np.mean(item_to_check[key]) < value[1]
            ), f"Mean of {key} is not in the expected range, got {np.mean(item_to_check[key])}"

    check_n_showers_and_energy_range(file=file)
    check_telescope_info(file=file, expected_output=expected_output)
