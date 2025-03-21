"""Single photon electron spectral analysis."""

import logging
import re
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path

from astropy.table import Table

import simtools.data_model.model_data_writer as writer
from simtools.constants import MODEL_PARAMETER_SCHEMA_URL, SCHEMA_PATH
from simtools.data_model import validate_data
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

    prompt_column = "frequency (prompt)"
    prompt_plus_afterpulse_column = "frequency (prompt+afterpulsing)"
    afterpulse_column = "frequency (afterpulsing)"

    input_schema = SCHEMA_PATH / "input" / "single_pe_spectrum.schema.yml"

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
        self.args_dict["metadata_product_data_name"] = "single_pe_spectrum"
        self.args_dict["metadata_product_data_url"] = (
            MODEL_PARAMETER_SCHEMA_URL + "/pm_photoelectron_spectrum.schema.yml"
        )
        self.metadata = MetadataCollector(args_dict=self.args_dict)

    def derive_single_pe_spectrum(self):
        """Derive single photon electron spectrum."""
        if self.args_dict.get("use_norm_spe"):
            return self._derive_spectrum_norm_spe()

        raise NotImplementedError(
            "Derivation of single photon electron spectrum using a simtool is not yet implemented."
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
            ["amplitude", self.prompt_column, self.prompt_plus_afterpulse_column],
        )

        writer.ModelDataWriter.dump(
            args_dict=self.args_dict,
            metadata=self.metadata,
            product_data=table,
            validate_schema_file=None,
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
        tmp_input_file = self._get_input_data(
            input_file=self.args_dict["input_spectrum"],
            frequency_column=self.prompt_column,
        )
        tmp_ap_file = self._get_input_data(
            input_file=self.args_dict.get("afterpulse_spectrum"),
            frequency_column=self.afterpulse_column,
        )

        command = [
            f"{self.args_dict['simtel_path']}/sim_telarray/bin/norm_spe",
            "-r",
            f"{self.args_dict['step_size']},{self.args_dict['max_amplitude']}",
        ]
        if tmp_ap_file:
            command.extend(["-a", f"{tmp_ap_file.name}"])
            command.extend(["-s", f"{self.args_dict['scale_afterpulse_spectrum']}"])
            command.extend(["-t", f"{self.args_dict['threshold_afterpulse_spectrum']}"])
        command.append(tmp_input_file.name)

        self._logger.info(f"Running norm_spe command: {' '.join(command)}")
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as exc:
            self._logger.error(f"Error running norm_spe: {exc}")
            self._logger.error(f"stderr: {exc.stderr}")
            raise exc
        finally:
            for tmp_file in [tmp_input_file, tmp_ap_file]:
                try:
                    Path(tmp_file.name).unlink()
                except (AttributeError, FileNotFoundError):
                    pass

        self.data = result.stdout
        return result.returncode

    def _get_input_data(self, input_file, frequency_column):
        """
        Return input data in the format required by the norm_spe tool as temporary file.

        The norm_spe tool requires the data to be space separated values of the amplitude spectrum,
        with two columns: amplitude and frequency.
        Input is validated using the single_pe_spectrum schema (legacy input is not validated).

        Parameters
        ----------
        input_file : str
            Input file with amplitude spectrum.
        frequency_column : str
            Column name of the frequency data.
        """
        if not input_file:
            return None
        input_file = Path(input_file)

        input_data = ""
        if input_file.suffix == ".ecsv":
            data_validator = validate_data.DataValidator(
                schema_file=self.input_schema, data_file=input_file
            )
            table = data_validator.validate_and_transform()
            input_data = "\n".join(f"{row['amplitude']} {row[frequency_column]}" for row in table)
        else:  # legacy format
            with open(input_file, encoding="utf-8") as f:
                input_data = (
                    f.read().replace(",", " ")
                    if frequency_column == self.prompt_column
                    else f.read()
                )

        with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8") as tmpfile:
            tmpfile.write(input_data)
        return tmpfile
