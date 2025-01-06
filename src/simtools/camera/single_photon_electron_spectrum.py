"""Single photon electron spectral analysis."""

import logging
import re
import subprocess
from io import BytesIO
from pathlib import Path

from astropy.table import Table

import simtools.data_model.model_data_writer as writer
from simtools.data_model.metadata_collector import MetadataCollector
from simtools.io_operations import io_handler


class SinglePhotonElectronSpectrum:
    """
    Single photon electron spectral analysis.

    Parameters
    ----------
    args_dict: dict
        Dictionary with input arguments.
    """

    def __init__(self, args_dict):
        """Initialize SinglePhotonElectronSpectrum class."""
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Initialize SinglePhotonElectronSpectrum class.")

        self.args_dict = args_dict
        # default output is of ecsv format
        self.args_dict["output_file"] = str(
            Path(self.args_dict["output_file"]).with_suffix(".ecsv")
        )
        self.io_handler = io_handler.IOHandler()
        self.data = ""  # Single photon electron spectrum data (as string)
        self.metadata = MetadataCollector(args_dict=self.args_dict)

    def derive_single_pe_spectrum(self):
        """Derive single photon electron spectrum."""
        if self.args_dict.get("use_norm_spe"):
            return self._derive_spectrum_norm_spe()

        raise NotImplementedError(
            "Derivation of single photon electron spectrum using a simtool is not implemented."
        )

    def write_single_pe_spectrum(self):
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

        cleaned_data = re.sub(r"%%%.+", "", self.data)  # remove norm_spe row metadata
        table = Table.read(
            BytesIO(cleaned_data.encode("utf-8")),
            format="ascii.no_header",
            comment="#",
            delimiter="\t",
        )
        table.rename_columns(
            ["col1", "col2", "col3"],
            ["amplitude", "frequency (prompt)", "frequency (prompt+afterpulsing)"],
        )

        writer.ModelDataWriter.dump(
            args_dict=self.args_dict,
            metadata=self.metadata.top_level_meta,
            product_data=table,
            validate_schema_file=None,  # TODO missing output schema
        )

    def _derive_spectrum_norm_spe(self):
        """
        Derive single photon electron spectrum using sim_telarray tool 'norm_spe'.

        Returns
        -------
        int
            Return code of the executed command

        Raises
        ------
        subprocess.CalledProcessError
            If the command execution fails.
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
            self._logger.error(f"Error running norm_spe: {exc}")
            raise exc

        self.data = result.stdout
        return result.returncode
