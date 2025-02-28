#!/usr/bin/python3

r"""Class to read and manage relevant model parameters for a given telescope model."""

import logging
import textwrap
from collections import defaultdict
from itertools import groupby
from pathlib import Path

import numpy as np

from simtools.io_operations import io_handler
from simtools.utils import names

logger = logging.getLogger()


class ReadParameters:
    """Read and manage model parameter data."""

    def __init__(self, db_config, telescope_model, output_path):
        """Initialise class with a telescope model."""
        self._logger = logging.getLogger(__name__)
        self.db_config = db_config
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
        output_data_path = Path(self.output_path / "_data_files")
        output_data_path.mkdir(parents=True, exist_ok=True)
        output_file_name = Path(input_file.stem + ".md")
        output_file = output_data_path / output_file_name

        try:
            with (
                input_file.open("r", encoding="utf-8") as infile,
                output_file.open("w", encoding="utf-8") as outfile,
            ):

                outfile.write(f"# {input_file.stem}")
                outfile.write("\n")
                outfile.write("```")
                outfile.write("\n")
                file_contents = infile.read()
                outfile.write(file_contents)
                outfile.write("\n")
                outfile.write("```")

        except FileNotFoundError as exc:
            logger.exception(f"Data file not found: {input_file}.")
            raise FileNotFoundError(f"Data file not found: {input_file}.") from exc

        return f"_data_files/{output_file_name}"

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
        parameter_description, short_description, inst_class = {}, {}, {}

        for instrument_class in names.db_collection_to_instrument_class_key("telescopes"):
            for parameter, details in names.model_parameters(instrument_class).items():
                parameter_description[parameter] = details.get("description")
                short_description[parameter] = details.get("short_description")
                inst_class[parameter] = instrument_class

        return parameter_description, short_description, inst_class

    def get_array_element_parameter_data(self, telescope_model, collection="telescopes"):
        """
        Get model parameter data for a given array element.

        Currently only configures for telescope.

        Parameters
        ----------
        telescope_model : TelescopeModel
            The telescope model instance.

        Returns
        -------
        list: A list of lists containing parameter names, values with units,
                descriptions, and short descriptions.
        """
        all_params = telescope_model.db.get_model_parameters(
            site=telescope_model.site,
            array_element_name=telescope_model.name,
            collection=collection,
        )

        telescope_model.export_model_files()
        parameter_descriptions = self.get_all_parameter_descriptions()
        data = []

        for parameter in all_params:
            if all_params[parameter]["instrument"] != telescope_model.name:
                continue
            parameter_version = telescope_model.get_parameter_version(parameter)
            value = telescope_model.get_parameter_value_with_unit(parameter)
            if telescope_model.get_parameter_file_flag(parameter) and value:
                input_file_name = telescope_model.config_file_directory / Path(value)
                output_file_name = self._convert_to_md(input_file_name)
                value = f"[{Path(value).name}]({output_file_name})"
            elif isinstance(value, list):
                value = ", ".join(str(q) for q in value)
            else:
                value = str(value)

            description = parameter_descriptions[0].get(parameter)
            short_description = parameter_descriptions[1].get(parameter, description)
            inst_class = parameter_descriptions[2].get(parameter)
            data.append(
                [
                    inst_class,
                    parameter,
                    parameter_version,
                    value,
                    description,
                    short_description,
                ]
            )

        return data

    def _compare_parameter_across_versions(self, parameter_name):
        """
        Compare a parameter's value across different model versions.

        Parameters
        ----------
        parameter_name : str
            The name of the parameter to compare.

        Returns
        -------
        list
            A list of dictionaries containing model version, parameter value, and description.
        """
        all_versions = self.telescope_model.db.get_model_versions()
        all_versions.reverse()  # latest first
        grouped_data = defaultdict(list)

        for version in all_versions:
            all_params = self.telescope_model.db.get_model_parameters(
                site=self.telescope_model.site,
                array_element_name=self.telescope_model.name,
                collection="telescopes",
                model_version=version,
            )

            try:
                parameter_data = all_params[parameter_name]
            except KeyError:
                continue

            if parameter_data["instrument"] != self.telescope_model.name:
                return None

            try:
                unit = parameter_data["unit"] if parameter_data["unit"] else ""
                value_data = parameter_data["value"]

                if isinstance(value_data, str | int | float):
                    value = f"{value_data} {unit}"
                elif len(value_data) > 5 and np.allclose(value_data, value_data[0]):
                    value = f"all: {value_data[0]} {unit}"
                else:
                    value = ", ".join([f"{v:.3f} {u}" for v, u in zip(value_data, unit)])

                parameter_version = parameter_data["parameter_version"]
                model_version = version
                grouped_data[(value, parameter_version)].append(model_version)

                model_versions = ", ".join(grouped_data[(value, parameter_version)])
            except TypeError:
                continue

        return [
            {
                "value": value.strip(),
                "parameter_version": parameter_version,
                "model_version": model_versions if len(model_version) > 1 else model_version[0],
            }
            for (value, parameter_version), model_version in grouped_data.items()
        ]

    def produce_array_element_report(self):
        """
        Produce a markdown report of all model parameters per array element.

        Output
        ----------
        One markdown report of a given array element listing parameter values,
        versions, and descriptions.
        """
        output_filename = Path(self.output_path / (self.telescope_model.name + ".md"))
        output_filename.parent.mkdir(parents=True, exist_ok=True)
        data = self.get_array_element_parameter_data(self.telescope_model)
        # Sort data by class to prepare for grouping
        if not isinstance(data, str):
            data.sort(key=lambda x: (x[0], x[1]), reverse=True)

        with output_filename.open("w", encoding="utf-8") as file:
            # Group by class and write sections
            file.write(f"# {self.telescope_model.name}\n")

            if self.telescope_model.name != self.telescope_model.design_model:
                file.write(
                    "The design model can be found here: "
                    f"[{self.telescope_model.design_model}]"
                    f"({self.telescope_model.design_model}.md).\n"
                )
                file.write("\n\n")

            for class_name, group in groupby(data, key=lambda x: x[0]):
                group = sorted(group, key=lambda x: x[1])
                file.write(f"## {class_name}\n\n")

                # Write table header and separator row
                file.write(
                    "| Parameter Name      |  Parameter Version     "
                    "| Values      | Short Description           |\n"
                    "|---------------------|------------------------"
                    "|-------------|-----------------------------|\n"
                )

                # Write table rows
                column_widths = [20, 20, 20, 70]
                for (
                    _,
                    parameter_name,
                    parameter_version,
                    value,
                    description,
                    short_description,
                ) in group:
                    text = short_description if short_description else description
                    wrapped_text = textwrap.fill(str(text), column_widths[3]).split("\n")
                    wrapped_text = " ".join(wrapped_text)
                    file.write(
                        f"| {parameter_name:{column_widths[0]}} |"
                        f" {parameter_version:{column_widths[1]}} |"
                        f" {value:{column_widths[2]}} |"
                        f" {wrapped_text} |\n"
                    )
                file.write("\n\n")

    def produce_model_parameter_reports(self):
        """
        Produce a markdown report per parameter for a given array element.

        Output
        ----------
        One markdown report per model parameter of a given array element comparing
        values across model versions.
        """
        logger.info(
            f"Comparing parameters across model versions for Telescope: {self.telescope_model.name}"
            f" and Site: {self.telescope_model.site}."
        )
        io_handler_instance = io_handler.IOHandler()
        output_path = io_handler_instance.get_output_directory(
            label="reports", sub_dir=f"parameters/{self.telescope_model.name}"
        )

        all_params = self.telescope_model.db.get_model_parameters(
            site=self.telescope_model.site,
            array_element_name=self.telescope_model.name,
            collection="telescopes",
        )

        for parameter in all_params:
            comparison_data = []
            if all_params[parameter]["instrument"] == self.telescope_model.name:
                comparison_data = self._compare_parameter_across_versions(parameter)
            if comparison_data:
                output_filename = output_path / f"{parameter}.md"
                description = self.get_all_parameter_descriptions()[0].get(parameter)
                with output_filename.open("w", encoding="utf-8") as file:
                    # Write header
                    file.write(
                        f"# {parameter}\n\n"
                        f"**Telescope**: {self.telescope_model.name}\n\n"
                        f"**Description**: {description}\n\n"
                        "\n"
                    )

                    # Write table header
                    file.write(
                        "| Parameter Version      | Model Version(s)      "
                        "| Value                |\n"
                        "|------------------------|--------------------"
                        "|----------------------|\n"
                    )

                    # Write table rows
                    for item in comparison_data:
                        file.write(
                            f"| {item['parameter_version']} |"
                            f" {item['model_version']} |"
                            f"{item['value'].replace('](', '](../')} |\n"
                        )

                    file.write("\n")
                    if isinstance(comparison_data[0]["value"], str) and comparison_data[0][
                        "value"
                    ].endswith(".md)"):
                        file.write(
                            f"![Parameter plot.](_images/"
                            f"{self.telescope_model.name}_{parameter}.png)"
                        )
