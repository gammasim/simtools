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

    def __init__(self, telescope_model, output_folder):
        """Initialise class with a telescope model."""
        self._logger = logging.getLogger(__name__)
        self.telescope_model = telescope_model
        self.output_folder = output_folder

    def _get_parameter_classes(self):
        """Load parameter classifications by category."""
        return {
            "Structure": set(names.load_model_parameters(class_key_list="Structure")),
            "Camera": set(names.load_model_parameters(class_key_list="Camera")),
            "Telescope": set(names.load_model_parameters(class_key_list="Telescope")),
        }

    def _convert_to_md(self, input_file):
        """
        Convert a '.dat' or '.ecsv' file to a Markdown file, preserving formatting.

        Parameters
        ----------
        - input_file: the '.dat' or '.ecsv' filename.

        Returns
        -------
        - Path to the created Markdown file.
        """
        input_path = Path(input_file)
        subprocess.run(["mkdir", "-p", f"{self.output_folder}/data_files"], check=True)
        output_path = Path(f"{self.output_folder}/data_files/" + input_path.stem + ".md")

        with (
            input_path.open("r", encoding="utf-8") as infile,
            output_path.open("w", encoding="utf-8") as outfile,
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

        subprocess.run(["rm", input_path], check=True)
        return output_path

    def get_all_parameter_descriptions(self):
        """
        Get descriptions of all available parameters.

        Returns
        -------
            tuple: A tuple containing two dictionaries:
                - parameter_description: Maps parameter names to their descriptions.
                - short_description: Maps parameter names to their short descriptions.
                - inst_class: Maps parameter names to their respective class.
        """
        all_parameters = names.telescope_parameters()
        parameter_classes = self._get_parameter_classes()

        parameter_description = {}
        short_description = {}
        inst_class = {}

        for parameter, details in all_parameters.items():
            parameter_description[parameter] = details.get("description", "To be added.")
            short_description[parameter] = details.get("short_description", "To be added.")
            inst_class[parameter] = next(
                (cls for cls, params in parameter_classes.items() if parameter in params),
                None,
            )

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
        output_folder = io_handler.IOHandler().get_output_directory()

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

                        input_filename = os.path.join(output_folder, os.path.basename(value))
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
