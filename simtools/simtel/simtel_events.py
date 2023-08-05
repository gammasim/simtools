import logging
import math
from copy import copy

import astropy.units as u
import numpy as np
from eventio.simtel import SimTelFile

__all__ = ["InconsistentInputFile", "SimtelEvents"]


class InconsistentInputFile(Exception):
    """Exception for inconsistent input file."""


class SimtelEvents:
    """
    This class handle sim_telarray events. sim_telarray files are read with eventio package.

    Parameters
    ----------
    input_files: list
        List of sim_telarray output files (str of Path).
    """

    def __init__(self, input_files=None):
        """
        Initialize SimtelEvents.
        """
        self._logger = logging.getLogger(__name__)
        self.load_input_files(input_files)
        if self.number_of_files > 0:
            self.load_header_and_summary()

    def load_input_files(self, files=None):
        """
        Store list of input files into input_files attribute.

        Parameters
        ----------
        files: list
            List of sim_telarray files (str or Path).
        """
        if not hasattr(self, "input_files"):
            self.input_files = []

        if files is None:
            msg = "No input file was given"
            self._logger.debug(msg)
            return

        if not isinstance(files, list):
            files = [files]

        for file in files:
            self.input_files.append(file)
        return

    @property
    def number_of_files(self):
        """Return number of files loaded.

        Returns
        -------
        int
            Number of files loaded.
        """
        return len(self.input_files) if hasattr(self, "input_files") else 0

    def load_header_and_summary(self):
        """
        Read MC header from sim_telarray files and store it into _mc_header. Also fills \
        summary_events with energy and core radius of triggered events.
        """

        self._number_of_files = len(self.input_files)
        keys_to_grab = [
            "obsheight",
            "n_showers",
            "n_use",
            "core_range",
            "diffuse",
            "viewcone",
            "E_range",
            "spectral_index",
            "B_total",
        ]
        self._mc_header = {}

        def _are_headers_consistent(header0, header1):
            comparison = {}
            for k in keys_to_grab:
                value = header0[k] == header1[k]
                comparison[k] = value if isinstance(value, bool) else all(value)

            return all(comparison)

        is_first_file = True
        number_of_triggered_events = 0
        summary_energy, summary_rcore = [], []
        for file in self.input_files:
            with SimTelFile(file) as f:
                for event in f:
                    en = event["mc_shower"]["energy"]
                    rc = math.sqrt(
                        math.pow(event["mc_event"]["xcore"], 2)
                        + math.pow(event["mc_event"]["ycore"], 2)
                    )

                    summary_energy.append(en)
                    summary_rcore.append(rc)
                    number_of_triggered_events += 1

                if is_first_file:
                    # First file - grabbing parameters
                    self._mc_header.update({k: copy(f.mc_run_headers[0][k]) for k in keys_to_grab})
                else:
                    # Remaining files - Checking whether the parameters are consistent
                    if not _are_headers_consistent(self._mc_header, f.mc_run_headers[0]):
                        msg = "MC header pamameters from different files are inconsistent"
                        self._logger.error(msg)
                        raise InconsistentInputFile(msg)

                is_first_file = False

        self.summary_events = {
            "energy": np.array(summary_energy),
            "r_core": np.array(summary_rcore),
        }

        # Calculating number of events
        self._mc_header["n_events"] = (
            self._mc_header["n_use"] * self._mc_header["n_showers"] * self._number_of_files
        )
        self._mc_header["n_triggered"] = number_of_triggered_events

    @u.quantity_input(core_max=u.m)
    def count_triggered_events(self, energy_range=None, core_max=None):
        """
        Count number of triggered events within a certain energy range and core radius.

        Parameters
        ----------
        energy_range: Tuple with len 2
            Max and min energy of energy range, e.g. energy_range=(100 * u.GeV, 10 * u.TeV).
        core_max: astropy.Quantity distance
            Maximum core radius for selecting showers, e.g. core_max=1000 * u.m.

        Returns
        -------
        int
            Number of triggered events.
        """
        energy_range = self._validate_energy_range(energy_range)
        core_max = self._validate_core_max(core_max)

        is_in_energy_range = list(
            map(
                lambda e: energy_range[0] < e < energy_range[1],
                self.summary_events["energy"],
            )
        )
        is_in_core_range = list(map(lambda r: r < core_max, self.summary_events["r_core"]))
        return np.sum(np.array(is_in_energy_range) * np.array(is_in_core_range))

    @u.quantity_input(core_max=u.m)
    def select_events(self, energy_range=None, core_max=None):
        """
        Select sim_telarray events within a certain energy range and core radius.

        Parameters
        ----------
        energy_range: Tuple len 2
            Max and min energy of energy range, e.g. energy_range=(100 * u.GeV, 10 * u.TeV).
        core_max: astropy.Quantity distance
            Maximum core radius for selecting showers, e.g. core_max=1000 * u.m.

        Returns
        -------
        list
            List of events.
        """
        energy_range = self._validate_energy_range(energy_range)
        core_max = self._validate_core_max(core_max)

        selected_events = []
        for file in self.input_files:
            with SimTelFile(file) as f:
                for event in f:
                    energy = event["mc_shower"]["energy"]
                    if energy < energy_range[0] or energy > energy_range[1]:
                        continue

                    x_core = event["mc_event"]["xcore"]
                    y_core = event["mc_event"]["ycore"]
                    r_core = math.sqrt(math.pow(x_core, 2) + math.pow(y_core, 2))
                    if r_core > core_max:
                        continue

                    selected_events.append(event)
        return selected_events

    @u.quantity_input(core_max=u.m)
    def count_simulated_events(self, energy_range=None, core_max=None):
        """
        Count (or calculate) number of simulated events within a certain energy range and \
        core radius, based on the simulated power law.
        This calculation assumes the simulated spectrum is given by a single power law.

        Parameters
        ----------
        energy_range: Tuple len 2
            Max and min energy of energy range, e.g. energy_range=(100 * u.GeV, 10 * u.TeV).
        core_max: astropy.Quantity distance
            Maximum core radius for selecting showers, e.g. core_max=1000 * u.m.

        Returns
        -------
        int
            Number of simulated events.
        """
        energy_range = self._validate_energy_range(energy_range)
        core_max = self._validate_core_max(core_max)

        # energy factor
        def integral(erange):
            power = self._mc_header["spectral_index"] + 1
            return math.pow(erange[0], power) - math.pow(erange[1], power)

        energy_factor = integral(energy_range) / integral(self._mc_header["E_range"])

        # core factor
        core_factor = math.pow(core_max, 2) / math.pow(self._mc_header["core_range"][1], 2)

        return self._mc_header["n_events"] * energy_factor * core_factor

    def _validate_energy_range(self, energy_range):
        """
        Returns the default energy range from mc_header in case energy_range=None.
        Checks units, convert it to TeV and return it in the right format, otherwise.
        """
        if energy_range is None:
            return self._mc_header["E_range"]

        if not isinstance(energy_range[0], u.Quantity) or not isinstance(
            energy_range[1], u.Quantity
        ):
            msg = "energy_range must be given as u.Quantity in units of energy"
            self._logger.error(msg)
            raise TypeError(msg)

        try:
            return (energy_range[0].to(u.TeV).value, energy_range[1].to(u.TeV).value)
        except u.core.UnitConversionError as e:
            msg = "energy_range must be in units of energy"
            self._logger.error(msg)
            raise TypeError(msg) from e

    def _validate_core_max(self, core_max):
        """
        Returns the default core_max from mc_header in case core_max=None.
        Checks units, convert it to m and return it in the right format, otherwise.
        """
        return self._mc_header["core_range"][1] if core_max is None else core_max.to(u.m).value
