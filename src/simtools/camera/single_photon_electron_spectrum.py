"""Single photon electron spectrum analysis."""

import logging
import subprocess
from pathlib import Path

from simtools.io_operations import io_handler


class SinglePhotonElectronSpectrum:
    """
    Single photon electron spectrum analysis.

    Parameters
    ----------
    label: str
        Application label.
    args_dict: dict
        Dictionary with input arguments.
    """

    def __init__(self, label, args_dict):
        """Initialize SinglePhotonElectronSpectrum class."""
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Initialize SinglePhotonElectronSpectrum class.")

        self.label = label
        self.args_dict = args_dict
        self.io_handler = io_handler.IOHandler()
        self.data = ""  # Single photon electron spectrum data (as string)

    def derive_spectrum(self):
        """Derive single photon electron spectrum."""
        if self.args_dict.get("use_norm_spe"):
            return self._derive_spectrum_norm_spe()

        raise NotImplementedError(
            "Derivation of single photon electron spectrum using a simtool is not implemented."
        )

    def write_spectrum(self):
        """
        Write single photon electron spectrum plus metadata to disk.

        Includes writing in simtel and simtools (ecsv) formats.

        """
        simtel_file = self.io_handler.get_output_directory() / Path(
            self.args_dict["output_file"]
        ).with_suffix(".dat")
        self._logger.debug(f"norm_spe output file: {simtel_file}")
        with open(simtel_file, "w", encoding="utf-8") as simtel:
            simtel.write(self.data)

    def _derive_spectrum_norm_spe(self):
        """
        Derive single photon electron spectrum using sim_telarray tool 'norm_spe'.

        Returns
        -------
        None
        """
        command = [
            f"{self.args_dict['simtel_path']}/sim_telarray/bin/norm_spe",
            "-r",
            f"{self.args_dict['step_size']},{self.args_dict['max_amplitude']}",
            f"{self.args_dict['input_spectrum']}",
        ]
        if self.args_dict["afterpulse_spectrum"] is not None:
            command.insert(1, "-a")
            command.insert(2, f"{self.args_dict['afterpulse_spectrum']}")

        self._logger.debug(f"Running norm_spe command: {' '.join(command)}")
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as exc:
            raise exc

        self.data = result.stdout
        return result.returncode
