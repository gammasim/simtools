#!/usr/bin/python3

r"""Class to read and manage relevant model parameters for a given telescope model."""

import logging
import re
import textwrap
from collections import defaultdict
from itertools import groupby
from pathlib import Path

import numpy as np

from simtools.db import db_handler
from simtools.model.telescope_model import TelescopeModel
from simtools.utils import names

logger = logging.getLogger()
IMAGE_PATH = "../../_images"


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
        self.observatory = args.get("observatory")

    def _convert_to_md(self, parameter, input_file):
        """Convert a file to a Markdown file, preserving formatting."""
        input_file = Path(input_file)
        output_data_path = Path(self.output_path / "_data_files")
        output_data_path.mkdir(parents=True, exist_ok=True)
        output_file_name = Path(input_file.stem + ".md")
        output_file = output_data_path / output_file_name

        try:
            # First try with utf-8
            try:
                with input_file.open("r", encoding="utf-8") as infile:
                    file_contents = infile.read()
            except UnicodeDecodeError:
                # If utf-8 fails, try with latin-1 (which can read any byte sequence)
                with input_file.open("r", encoding="latin-1") as infile:
                    file_contents = infile.read()

            if self.model_version is not None:
                with output_file.open("w", encoding="utf-8") as outfile:
                    outfile.write(f"# {input_file.stem}\n")
                    outfile.write(
                        "The full file can be found in the Simulation Model repository [here]"
                        "(https://gitlab.cta-observatory.org/cta-science/simulations/"
                        "simulation-model/simulation-models/-/blob/main/simulation-models/"
                        f"model_parameters/Files/{input_file.stem}.dat).\n\n"
                    )
                    outfile.write(
                        f"![Parameter plot.](../{IMAGE_PATH}/{self.array_element}_"
                        f"{parameter}_{self.model_version.split('.')[0]}.png)\n"
                    )
                    outfile.write("\n\n")
                    outfile.write("The first 30 lines of the file are:\n")
                    outfile.write("```\n")
                    first_30_lines = "\n".join(file_contents.splitlines()[:30])
                    outfile.write(first_30_lines)
                    outfile.write("\n```")

        except FileNotFoundError as exc:
            logger.exception(f"Data file not found: {input_file}.")
            raise FileNotFoundError(f"Data file not found: {input_file}.") from exc

        return f"_data_files/{output_file_name}"

    def _format_parameter_value(self, parameter, value_data, unit, file_flag):
        """Format parameter value based on type."""
        if file_flag:
            input_file_name = f"{self.output_path}/model/{value_data}"
            output_file_name = self._convert_to_md(parameter, input_file_name)
            return f"[{Path(value_data).name}]({output_file_name})".strip()
        if isinstance(value_data, (str | int | float)):
            return f"{value_data} {unit}".strip()
        if len(value_data) > 5 and np.allclose(value_data, value_data[0]):
            return f"all: {value_data[0]} {unit}".strip()
        return (
            ", ".join(f"{v} {u}" for v, u in zip(value_data, unit))
            if isinstance(unit, list)
            else ", ".join(f"{v} {unit}" for v in value_data)
        ).strip()

    def _wrap_at_underscores(self, text, max_width):
        """Wrap text at underscores to fit within a specified width."""
        parts = text.split("_")
        lines = []
        current = []

        for part in parts:
            # Predict the new length if we add this part
            next_line = "_".join([*current, part])
            if len(next_line) <= max_width:
                current.append(part)
            else:
                lines.append("_".join(current))
                current = [part]

        if current:
            lines.append("_".join(current))

        return " ".join(lines)

    def _group_model_versions_by_parameter_version(self, grouped_data):
        """Group model versions by parameter version and track the parameter values."""
        result = {}

        for parameter_name, items in grouped_data.items():
            version_grouped = defaultdict(
                lambda: {"model_versions": [], "value": None, "file_flag": None}
            )

            for item in items:
                param_version = item["parameter_version"]
                version_grouped[param_version]["model_versions"].append(item["model_version"])

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

        # Iterate over each model version
        for version in all_versions:
            parameter_dict = all_param_data.get(version, {})
            if not parameter_dict:
                continue

            Path(f"{self.output_path}/model").mkdir(parents=True, exist_ok=True)
            self.db.export_model_files(
                parameters=all_param_data.get(version), dest=f"{self.output_path}/model"
            )

            for parameter_name in filter(parameter_dict.__contains__, all_parameter_names):
                parameter_data = parameter_dict.get(parameter_name)

                # Skip if instrument doesn't match
                if parameter_data.get("instrument") != self.array_element:
                    continue

                unit = parameter_data.get("unit") or " "
                value_data = parameter_data.get("value")

                if value_data is None:
                    continue

                file_flag = parameter_data.get("file", False)
                value = self._format_parameter_value(parameter_name, value_data, unit, file_flag)
                parameter_version = parameter_data.get("parameter_version")
                model_version = version

                # Group the data by parameter version and store model versions as a list
                grouped_data[parameter_name].append(
                    {
                        "value": value,
                        "parameter_version": parameter_version,
                        "model_version": model_version,
                        "file_flag": file_flag,
                    }
                )

        return self._group_model_versions_by_parameter_version(grouped_data)

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
        all_parameter_data = self.db.get_model_parameters(
            site=telescope_model.site,
            array_element_name=telescope_model.name,
            collection=collection,
            model_version=telescope_model.model_version,
        )

        Path(f"{self.output_path}/model").mkdir(parents=True, exist_ok=True)
        self.db.export_model_files(parameters=all_parameter_data, dest=f"{self.output_path}/model")
        parameter_descriptions = self.get_all_parameter_descriptions()
        data = []

        for parameter in filter(all_parameter_data.__contains__, names.model_parameters().keys()):
            parameter_data = all_parameter_data.get(parameter)
            parameter_version = telescope_model.get_parameter_version(parameter)
            unit = parameter_data.get("unit") or " "
            value_data = parameter_data.get("value")

            if value_data is None:
                continue

            file_flag = parameter_data.get("file", False)
            value = self._format_parameter_value(parameter, value_data, unit, file_flag)

            description = parameter_descriptions[0].get(parameter)
            short_description = parameter_descriptions[1].get(parameter)
            if short_description is None:
                short_description = description
            inst_class = parameter_descriptions[2].get(parameter)

            matching_instrument = parameter_data["instrument"] == telescope_model.name
            if not names.is_design_type(telescope_model.name) and matching_instrument:
                parameter = f"***{parameter}***"
                parameter_version = f"***{parameter_version}***"
                if not re.match(r"^\[.*\]\(.*\)$", value.strip()):
                    value = f"***{value}***"
                description = f"***{description}***"
                short_description = f"***{short_description}***"

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

    def produce_array_element_report(self):
        """
        Produce a markdown report of all model parameters per array element.

        Outputs one markdown report of a given array element listing parameter values,
        versions, and descriptions.
        """
        if self.observatory:
            self.produce_observatory_report()
            return

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
                    f"({telescope_model.design_model}.md).\n\n"
                )
                file.write(
                    "Parameters shown in ***bold and italics*** are specific to each telescope.\n"
                    "Parameters without emphasis are inherited from the design model.\n"
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
                column_widths = [10, 10, 20, 60]
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
                    parameter_name = self._wrap_at_underscores(parameter_name, column_widths[0])

                    file.write(
                        f"| {parameter_name:{column_widths[0]}} |"
                        f" {parameter_version:{column_widths[1]}} |"
                        f" {value:{column_widths[2]}} |"
                        f" {wrapped_text:{column_widths[3]}} |\n"
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
        output_path = self.output_path / f"{self.array_element}"
        Path(output_path).mkdir(parents=True, exist_ok=True)

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
                    file.write(
                        f"![Parameter plot.]({IMAGE_PATH}/{self.array_element}_{parameter}.png)"
                    )

    def _write_array_layouts_section(self, file, layouts):
        """Write the array layouts section of the report."""
        file.write("\n## Array Layouts\n\n")
        for layout in layouts:
            layout_name = layout["name"]
            elements = layout["elements"]
            file.write(f"### {layout_name}\n\n")
            file.write("| Element |\n|---------|\n")
            for element in sorted(elements):
                file.write(f"| [{element}]({element}.md) |\n")
            file.write("\n")
            image_path = (
                f"{IMAGE_PATH}/OBS-{self.site}_{layout_name}_{self.model_version.split('.')[0]}.png"
            )
            file.write(f"![{layout_name} Layout]({image_path})\n\n")
            file.write("\n")

    def _write_array_triggers_section(self, file, trigger_configs):
        """Write the array triggers section of the report."""
        file.write("\n## Array Trigger Configurations\n\n")
        file.write(
            "| Trigger Name | Multiplicity | Width | Hard Stereo | Min Separation |\n"
            "|--------------|--------------|--------|-------------|----------------|\n"
        )
        for config in trigger_configs:
            name = config["name"]
            mult = f"{config['multiplicity']['value']} {config['multiplicity']['unit'] or ''}"
            width = f"{config['width']['value']} {config['width']['unit'] or ''}"
            stereo = "Yes" if config["hard_stereo"]["value"] else "No"
            min_sep = (
                f"{config['min_separation']['value']} {config['min_separation']['unit'] or '-'}"
            )
            file.write(
                f"| {name} | {mult.strip()} | {width.strip()} | {stereo} | {min_sep.strip()} |\n"
            )
        file.write("\n")

    def _write_parameters_table(self, file, all_parameter_data):
        """Write the main parameters table of the report."""
        file.write(
            "| Parameter | Value | Parameter Version |\n"
            "|-----------|--------|-------------------|\n"
        )
        for param_name, param_data in sorted(all_parameter_data.items()):
            value = param_data.get("value")
            unit = param_data.get("unit") or " "
            file_flag = param_data.get("file", False)
            parameter_version = param_data.get("parameter_version")

            if value is None:
                continue

            if param_name == "array_layouts":
                file.write(
                    "| array_layouts | [View Array Layouts](#array-layouts)"
                    f" | {parameter_version} |\n"
                )
            elif param_name == "array_triggers":
                file.write(
                    "| array_triggers | [View Trigger Configurations]"
                    f"(#array-trigger-configurations) | {parameter_version} |\n"
                )
            else:
                formatted_value = self._format_parameter_value(param_name, value, unit, file_flag)
                file.write(f"| {param_name} | {formatted_value} | {parameter_version} |\n")
        file.write("\n")

    def produce_observatory_report(self):
        """Produce a markdown report of all observatory parameters for a given site."""
        output_filename = Path(self.output_path / f"OBS-{self.site}.md")
        output_filename.parent.mkdir(parents=True, exist_ok=True)

        all_parameter_data = self.db.get_model_parameters(
            site=self.site,
            array_element_name="OBS-" + self.site,
            collection="sites",
            model_version=self.model_version,
        )

        if not all_parameter_data:
            logger.warning(f"No observatory parameters found for site {self.site}")
            return

        Path(f"{self.output_path}/model").mkdir(parents=True, exist_ok=True)
        self.db.export_model_files(parameters=all_parameter_data, dest=f"{self.output_path}/model")

        with output_filename.open("w", encoding="utf-8") as file:
            file.write(f"# Observatory Parameters - {self.site} Site\n\n")
            self._write_parameters_table(file, all_parameter_data)

            if "array_layouts" in all_parameter_data:
                self._write_array_layouts_section(
                    file, all_parameter_data["array_layouts"]["value"]
                )

            if "array_triggers" in all_parameter_data:
                self._write_array_triggers_section(
                    file, all_parameter_data["array_triggers"]["value"]
                )
