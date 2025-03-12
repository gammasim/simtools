#!/usr/bin/python3

r"""Class to read and manage relevant model parameters for a given telescope model."""

import logging
import textwrap
from collections import defaultdict
from itertools import groupby
from pathlib import Path

import numpy as np

from simtools.db import db_handler
from simtools.io_operations import io_handler
from simtools.model.telescope_model import TelescopeModel
from simtools.utils import names

logger = logging.getLogger()


class ReadParameters:
    """Read and manage model parameter data for report generation."""

    def __init__(self, db_config, args, output_path):
        """Initialise class."""
        self._logger = logging.getLogger(__name__)
        self.db = db_handler.DatabaseHandler(mongo_db_config=db_config)
        self.db_config = db_config
        self.array_element = args.get("telescope")
        self.site = args.get("site")
        self.model_version = args.get("model_version", None)
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
        Get model parameter data and descriptions for a given array element.

        Currently only configured for telescope.

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
            model_version=telescope_model.model_version,
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

    def _compare_parameter_across_versions(self, all_param_data, all_parameter_names):
        """
        Compare a parameter's value across different model versions.

        Parameters
        ----------
        all_param_data : dict
            The dictionary containing parameter data for all versions.

        all_parameter_names : list
            The list of parameter names to compare across versions.

        Returns
        -------
        list
            A list of dictionaries containing model version, parameter value, description.
        """
        all_versions = self.db.get_model_versions()
        all_versions.reverse()  # latest first
        grouped_data = defaultdict(list)

        def format_value(value_data, unit, file_flag):
            """Format parameter value based on type and parameter name."""
            if file_flag:
                input_file_name = f"{self.output_path}/model/{value_data}"
                output_file_name = self._convert_to_md(input_file_name)
                return f"[{Path(value_data).name}]({output_file_name})"
            if isinstance(value_data, (str | int | float)):
                return f"{value_data} {unit}"
            if len(value_data) > 5 and np.allclose(value_data, value_data[0]):
                return f"all: {value_data[0]} {unit}"
            return (
                ", ".join(f"{v:.2f} {u}" for v, u in zip(value_data, unit))
                if isinstance(unit, list)
                else ", ".join(f"{v} {unit}" for v in value_data)
            )

        # Iterate over each model version
        for version in all_versions:
            Path(f"{self.output_path}/model").mkdir(parents=True, exist_ok=True)

            # Export model files (assuming '6.0.0' is the version you want)
            self.db.export_model_files(
                parameters=all_param_data.get(version), dest=f"{self.output_path}/model"
            )

            parameter_dict = all_param_data.get(version, {})

            # Iterate over each parameter name
            for parameter_name in all_parameter_names:
                # Skip if parameter_name is not present
                if parameter_name not in parameter_dict:
                    continue

                parameter_data = parameter_dict.get(parameter_name)

                # Skip if instrument doesn't match
                if parameter_data.get("instrument") != self.array_element:
                    continue

                unit = parameter_data.get("unit") or ""
                value_data = parameter_data.get("value")

                if not value_data:
                    continue

                file_flag = parameter_data.get("file", False)
                value = format_value(value_data, unit, file_flag)
                parameter_version = parameter_data.get("parameter_version")
                model_version = version

                # Group the data by parameter version and store model versions as a list
                grouped_data[parameter_name].append(
                    {
                        "value": value.strip(),
                        "parameter_version": parameter_version,
                        "model_version": model_version,
                        "file_flag": file_flag,
                    }
                )

            result = {}
            for parameter_name, items in grouped_data.items():
                # Group model versions by parameter version and track the correct values
                version_grouped = defaultdict(
                    lambda: {"model_versions": [], "value": None, "file_flag": None}
                )

                for item in items:
                    param_version = item["parameter_version"]
                    version_grouped[param_version]["model_versions"].append(item["model_version"])

                    # Ensure the correct value is taken for this specific parameter version
                    if version_grouped[param_version]["value"] is None:
                        version_grouped[param_version]["value"] = item["value"]
                        version_grouped[param_version]["file_flag"] = item["file_flag"]

                result[parameter_name] = [
                    {
                        "value": data["value"],
                        "parameter_version": param_version,
                        "file_flag": data["file_flag"],
                        "model_version": ", ".join(data["model_versions"]),
                    }
                    for param_version, data in version_grouped.items()
                ]

        return result

    def produce_array_element_report(self):
        """
        Produce a markdown report of all model parameters per array element.

        Outputs one markdown report of a given array element listing parameter values,
        versions, and descriptions.
        """
        telescope_model = TelescopeModel(
            site=self.site,
            telescope_name=self.array_element,
            model_version=self.model_version,
            label="reports",
            mongo_db_config=self.db_config,
        )

        output_filename = Path(self.output_path / (telescope_model.name + ".md"))
        output_filename.parent.mkdir(parents=True, exist_ok=True)
        data = self.get_array_element_parameter_data(telescope_model)
        # Sort data by class to prepare for grouping
        if not isinstance(data, str):
            data.sort(key=lambda x: (x[0], x[1]), reverse=True)

        with output_filename.open("w", encoding="utf-8") as file:
            # Group by class and write sections
            file.write(f"# {telescope_model.name}\n")

            if telescope_model.name != telescope_model.design_model:
                file.write(
                    "The design model can be found here: "
                    f"[{telescope_model.design_model}]"
                    f"({telescope_model.design_model}.md).\n"
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

        Outputs one markdown report per model parameter of a given array element comparing
        values across model versions.
        """
        logger.info(
            f"Comparing parameters across model versions for Telescope: {self.array_element}"
            f" and Site: {self.site}."
        )
        io_handler_instance = io_handler.IOHandler()
        output_path = io_handler_instance.get_output_directory(
            label="reports", sub_dir=f"parameters/{self.array_element}"
        )

        all_parameter_names = names.model_parameters(None).keys()
        all_parameter_data = self.db.get_model_parameters_for_all_model_versions(
            site=self.site, array_element_name=self.array_element, collection="telescopes"
        )

        comparison_data = self._compare_parameter_across_versions(
            all_parameter_data, all_parameter_names
        )

        for parameter in all_parameter_names:
            parameter_data = comparison_data.get(parameter)
            if not parameter_data:
                continue

            output_filename = output_path / f"{parameter}.md"
            description = self.get_all_parameter_descriptions()[0].get(parameter)
            with output_filename.open("w", encoding="utf-8") as file:
                # Write header
                file.write(
                    f"# {parameter}\n\n"
                    f"**Telescope**: {self.array_element}\n\n"
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
                for item in comparison_data.get(parameter):
                    file.write(
                        f"| {item['parameter_version']} |"
                        f" {item['model_version']} |"
                        f"{item['value'].replace('](', '](../')} |\n"
                    )

                file.write("\n")
                if comparison_data.get(parameter)[0]["file_flag"]:
                    file.write(f"![Parameter plot.](_images/{self.array_element}_{parameter}.png)")
