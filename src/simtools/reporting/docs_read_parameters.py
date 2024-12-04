#!/usr/bin/python3

r"""Class to read and manage relevant model parameters for a given telescope model."""

import logging
import os
import subprocess
from pathlib import Path

from simtools.io_operations import io_handler
from simtools.utils import names


class ReadParameters:
    """Read and manage model parameter data."""

    def __init__(self, telescope_model, output_path):
        """Initialise class with a telescope model."""
        self._logger = logging.getLogger(__name__)
        self.telescope_model = telescope_model
        self.output_path = output_path

    def _convert_to_md(self, input_file):
        """
        Convert a '.dat' or '.ecsv' file to a Markdown file, preserving formatting.

        Parameters
        ----------
        input_file: Path, str
           Simulation data file (in '.dat' or '.ecsv' format).

        Returns
        -------
        - Path to the created Markdown file.
        """
        input_file = Path(input_file)
        output_data_path = Path(self.output_path, "data_files")
        output_data_path.mkdir(parents=True, exist_ok=True)
        output_file = output_data_path / Path(input_file.stem + ".md")

        with (
            input_file.open("r", encoding="utf-8") as infile,
            output_file.open("w", encoding="utf-8") as outfile,
        ):

            for line in infile:
                line = line.strip()
                if line.startswith("#"):
                    comment = line.lstrip("#")
                    outfile.write(f"{comment}\n\n")

                else:
                    row = [col.strip() for col in line.split()]
                    if not row:
                        outfile.write("\n\n")
                        continue
                    outfile.write("| " + " | ".join(row) + " |\n\n")

        subprocess.run(["rm", input_file], check=True)
        return output_file

    def get_all_parameter_descriptions(self):
        """
        Get descriptions for all model parameters.

        Returns
        -------
            tuple: A tuple containing two dictionaries:
                - parameter_description: Maps parameter names to their descriptions.
                - short_description: Maps parameter names to their short descriptions.
                - inst_class: Maps parameter names to their respective class.
        """
        parameter_description = {}
        short_description = {}
        inst_class = {}

        for instrument_class in names.instrument_classes("telescope"):
            all_parameters = names.load_model_parameters(instrument_class)

            for parameter, details in all_parameters.items():
                parameter_description[parameter] = details.get("description", "To be added.")
                short_description[parameter] = details.get("short_description", "To be added.")
                inst_class[parameter] = instrument_class

        return parameter_description, short_description, inst_class

    def get_telescope_parameter_data(self):
        """
        Get model parameter data for a given telescope.

        Returns
        -------
            list: A list of lists containing parameter names, values with units,
                  descriptions, and short descriptions.
        """
        parameter_descriptions = self.get_all_parameter_descriptions()
        parameter_names = parameter_descriptions[0].keys()

        data = []
        output_path = io_handler.IOHandler().get_output_directory()

        for parameter in parameter_names:

            if self.telescope_model.has_parameter(parameter):
                value = self.telescope_model.get_parameter_value_with_unit(parameter)
                if isinstance(value, list):
                    value = ", ".join(str(q) for q in value)

                elif isinstance(value, str) and value.endswith((".dat", ".ecsv")):
                    try:
                        subprocess.run(
                            [
                                "python3",
                                "-m",
                                "simtools.applications.db_get_file_from_db",
                                "--file_name",
                                value,
                            ],
                            capture_output=True,
                            text=True,
                            check=True,
                        )

                        input_filename = os.path.join(output_path, os.path.basename(value))
                        output_filename = self._convert_to_md(input_filename)
                        value = f"[{os.path.basename(value)}]({output_filename.as_posix()})"

                    except FileNotFoundError:
                        value = f"File not found: {value}"

            else:
                continue

            # description = parameter_descriptions[0].get(parameter)
            short_description = parameter_descriptions[1].get(parameter)
            inst_class = parameter_descriptions[2].get(parameter)
            data.append([inst_class, parameter, value, short_description])
        return data
