#!/usr/bin/python3

r"""Class to read and manage relevant model parameters for a given telescope model."""

import logging
import re
import textwrap
from collections import defaultdict
from itertools import groupby
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from simtools.db import db_handler
from simtools.io_operations import io_handler
from simtools.model.telescope_model import TelescopeModel
from simtools.utils import general as gen
from simtools.utils import names
from simtools.visualization import plot_pixels, plot_tables

logger = logging.getLogger()


class ReadParameters:
    """Read and manage model parameter data for report generation."""

    def __init__(self, db_config, args, output_path):
        """Initialise class."""
        self._logger = logging.getLogger(__name__)
        self.db = db_handler.DatabaseHandler(mongo_db_config=db_config)
        self.db_config = db_config
        self.array_element = args.get("telescope", None)
        self.site = args.get("site", None)
        self.model_version = args.get("model_version", None)
        self.output_path = output_path
        self.observatory = args.get("observatory")
        self.software = args.get("simulation_software", None)

    @property
    def model_version(self):
        """Model version."""
        return self._model_version

    @model_version.setter
    def model_version(self, model_version):
        """
        Set model version.

        Parameters
        ----------
        _model_version: str or list
            Model version (e.g., "6.0.0").
            If a list is passed, it must contain exactly one element,
            and only that element will be used.

        Raises
        ------
        ValueError
            If more than one model version is passed.
        """
        if isinstance(model_version, list):
            raise ValueError(
                f"Only one model version can be passed to {self.__class__.__name__}, not a list."
            )
        self._model_version = model_version

    def _generate_plots(self, parameter, parameter_version, input_file, outpath, design_type):
        """Generate plots based on the parameter type."""
        plot_names = []

        if parameter == "camera_config_file":
            plot_names = self._plot_camera_config(parameter, parameter_version, input_file, outpath)
        elif parameter_version:
            plot_names = self._plot_parameter_tables(
                parameter, parameter_version, outpath, design_type
            )

        return plot_names

    def _plot_camera_config(self, parameter, parameter_version, input_file, outpath):
        """Generate plots for camera configuration files."""
        if not parameter_version:
            return []

        plot_names = []
        plot_name = input_file.stem.replace(".", "-")
        plot_path = Path(f"{outpath}/{plot_name}").with_suffix(".png")

        if not plot_path.exists():
            plot_config = {
                "file_name": input_file.name,
                "telescope": self.array_element,
                "parameter_version": parameter_version,
                "site": self.site,
                "model_version": self.model_version,
                "parameter": parameter,
            }

            plot_pixels.plot(
                config=plot_config,
                output_file=Path(f"{outpath}/{plot_name}"),
                db_config=self.db_config,
            )
            plot_names.append(plot_name)
        else:
            logger.info(
                "Camera configuration file plot already exists: %s",
                plot_path,
            )
            plot_names.append(plot_name)

        return plot_names

    def _plot_parameter_tables(self, parameter, parameter_version, outpath, design_type):
        """Generate plots for parameter tables."""
        telescope_design = self.db.get_design_model(
            self.model_version, self.array_element, collection="telescopes"
        )

        if not self.array_element:
            tel = None
        elif not design_type:
            tel = telescope_design
        else:
            tel = self.array_element

        config_data = plot_tables.generate_plot_configurations(
            parameter=parameter,
            parameter_version=parameter_version,
            site=self.site,
            telescope=tel,
            output_path=outpath,
            plot_type="all",
            db_config=self.db_config,
        )

        if not config_data:
            return []

        plot_configs, output_files = config_data
        plot_names = [i.stem for i in output_files]

        for plot_config, output_file in zip(plot_configs, output_files):
            image_output_file = outpath / output_file.name
            if not image_output_file.with_suffix(".png").exists():
                plot_tables.plot(
                    config=plot_config,
                    output_file=image_output_file,
                    db_config=self.db_config,
                )
                plt.close("all")

        return plot_names

    def _convert_to_md(self, parameter, parameter_version, input_file, design_type=False):
        """Convert a file to a Markdown file, preserving formatting."""
        input_file = Path(input_file)

        if not input_file.exists():
            raise FileNotFoundError(f"Data file not found: {input_file}")

        # Store the markdown output file path early and don't modify it
        output_data_path = Path(self.output_path / "_data_files")
        output_data_path.mkdir(parents=True, exist_ok=True)
        output_file_name = Path(input_file.stem + ".md")
        relative_path = f"_data_files/{output_file_name}"
        markdown_output_file = output_data_path / output_file_name

        if not markdown_output_file.exists():
            outpath = Path(io_handler.IOHandler().get_output_directory().parent / "_images")
            outpath.mkdir(parents=True, exist_ok=True)

            plot_names = self._generate_plots(
                parameter, parameter_version, input_file, outpath, design_type
            )
            # Write markdown file using the stored path
            file_contents = gen.read_file_encoded_in_utf_or_latin(input_file)

            with markdown_output_file.open("w", encoding="utf-8") as outfile:
                outfile.write(f"# {input_file.stem}\n")

                for plot_name in plot_names:
                    outfile.write(f"![Parameter plot.]({outpath}/{plot_name}.png)\n\n")

                outfile.write(
                    "\n\nThe full file can be found in the Simulation Model repository [here]"
                    "(https://gitlab.cta-observatory.org/cta-science/simulations/"
                    "simulation-model/simulation-models/-/blob/main/simulation-models/"
                    f"model_parameters/Files/{input_file.name}).\n\n"
                )
                outfile.write("\n\n")
                outfile.write("The first 30 lines of the file are:\n")
                outfile.write("```\n")
                first_30_lines = "".join(file_contents[:30])
                outfile.write(first_30_lines)
                outfile.write("\n```")

        return relative_path

    def _format_parameter_value(
        self, parameter, value_data, unit, file_flag, parameter_version=None, design_type=False
    ):
        """Format parameter value based on type."""
        if file_flag:
            input_file_name = f"{self.output_path}/model/{value_data}"
            if parameter_version is None:
                return (
                    f"[{Path(value_data).name}](https://gitlab.cta-observatory.org/"
                    "cta-science/simulations/simulation-model/simulation-models/-/blob/main/"
                    f"simulation-models/model_parameters/Files/{value_data})"
                ).strip()
            output_file_name = self._convert_to_md(
                parameter, parameter_version, input_file_name, design_type
            )
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
                parameter_version = parameter_data.get("parameter_version")
                value = self._format_parameter_value(
                    parameter_name, value_data, unit, file_flag, parameter_version=None
                )
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

    def get_all_parameter_descriptions(self, collection="telescopes"):
        """Get descriptions for all model parameters.

        Returns
        -------
        dict
            Nested dictionaries with first key as the parameter name and
            the following dictionary as the value:
            - key: description, value: description of the parameter.
            - key: short_description, value: short description of the parameter.
            - key: inst_class, value: class, for eg. Structure, Camera, etc.
        """
        parameter_dict = {}

        for instrument_class in names.db_collection_to_instrument_class_key(collection):
            for parameter, details in names.model_parameters(instrument_class).items():
                parameter_dict[parameter] = {
                    "description": details.get("description"),
                    "short_description": details.get("short_description"),
                    "inst_class": instrument_class,
                }

        return parameter_dict

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

            design_type = names.is_design_type(telescope_model.name)
            file_flag = parameter_data.get("file", False)
            value = self._format_parameter_value(
                parameter, value_data, unit, file_flag, parameter_version, design_type
            )

            description = parameter_descriptions.get(parameter).get("description")
            short_description = (
                parameter_descriptions.get(parameter).get("short_description") or description
            )
            inst_class = parameter_descriptions.get(parameter).get("inst_class")

            matching_instrument = parameter_data["instrument"] == telescope_model.name
            if not design_type and matching_instrument:
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

    def _write_to_file(self, data, file):
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
        ) in data:
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

    def get_simulation_configuration_data(self):
        """Get data and descriptions for simulation configuration parameters."""

        def get_param_data(telescope, site):
            """Retrieve and format parameter data for one telescope-site combo."""
            param_dict = self.db.get_simulation_configuration_parameters(
                simulation_software=self.software,
                site=site,
                array_element_name=telescope,
                model_version=self.model_version,
            )

            parameter_descriptions = self.get_all_parameter_descriptions(
                collection=f"configuration_{self.software}"
            )

            model_output_path = Path(f"{self.output_path}/model")
            model_output_path.mkdir(parents=True, exist_ok=True)
            self.db.export_model_files(parameters=param_dict, dest=str(model_output_path))

            data = []
            for parameter, parameter_data in param_dict.items():
                description = parameter_descriptions.get(parameter).get("description")
                short_description = parameter_descriptions.get(parameter).get(
                    "short_description", description
                )
                value_data = parameter_data.get("value")

                if value_data is None:
                    continue

                unit = parameter_data.get("unit") or " "
                file_flag = parameter_data.get("file", False)
                parameter_version = parameter_data.get("parameter_version")
                value = self._format_parameter_value(
                    parameter, value_data, unit, file_flag, parameter_version
                )

                data.append(
                    [
                        telescope,
                        parameter,
                        parameter_version,
                        value,
                        description,
                        short_description,
                    ]
                )
            return data

        if self.software == "corsika":
            return get_param_data(self.array_element, self.site)

        results = []
        telescopes = self.db.get_array_elements(self.model_version)
        for telescope in telescopes:
            valid_site = names.get_site_from_array_element_name(telescope)
            if not isinstance(valid_site, list):
                results.extend(get_param_data(telescope, valid_site))
            else:
                for site in valid_site:
                    results.extend(get_param_data(telescope, site))
        return results

    def produce_simulation_configuration_report(self):
        """Write simulation configuration report."""
        output_filename = Path(self.output_path / (f"configuration_{self.software}.md"))
        output_filename.parent.mkdir(parents=True, exist_ok=True)
        data = self.get_simulation_configuration_data()

        with output_filename.open("w", encoding="utf-8") as file:
            file.write(f"# configuration_{self.software}\n")
            file.write("\n\n")
            if self.software == "sim_telarray":
                data.sort(key=lambda x: (x[0], x[1]))
                for telescope, group in groupby(data, key=lambda x: x[0]):
                    file.write(f"## [{telescope}]({telescope}.md)\n")
                    file.write("\n\n")
                    self._write_to_file(group, file)
            else:
                self._write_to_file(data, file)

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
                self._write_to_file(group, file)

    def produce_model_parameter_reports(self, collection="telescopes"):
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
            site=self.site, array_element_name=self.array_element, collection=collection
        )

        comparison_data = self._compare_parameter_across_versions(
            all_parameter_data, all_parameter_names
        )

        for parameter in all_parameter_names:
            parameter_data = comparison_data.get(parameter)
            if not parameter_data:
                continue

            output_filename = output_path / f"{parameter}.md"

            parameter_descriptions = self.get_all_parameter_descriptions(collection=collection).get(
                parameter
            ) or self.get_all_parameter_descriptions(collection="telescopes").get(parameter)

            description = parameter_descriptions.get("description")
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
                        f"{item['value']} |\n"
                    )

                file.write("\n")
                if comparison_data.get(parameter)[0]["file_flag"]:
                    file.write(f"![Parameter plot.](/_images/{self.array_element}_{parameter}.png)")

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
            version = self.model_version.replace(".", "-")
            filename = f"OBS-{self.site}_{layout_name}_{version}.png"
            image_path = f"/_images/{filename}"
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
                formatted_value = self._format_parameter_value(
                    param_name, value, unit, file_flag, parameter_version
                )
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

    def get_calibration_data(self, all_parameter_data, array_element):
        """Get calibration data and descriptions for a given array element."""
        calibration_descriptions = self.get_all_parameter_descriptions(
            collection="calibration_devices"
        )
        # get descriptions of array element positions from the telescope collection
        telescope_descriptions = self.get_all_parameter_descriptions(collection="telescopes")
        data = []
        class_grouped_data = {}

        for parameter in all_parameter_data.keys():
            parameter_descriptions = calibration_descriptions.get(
                parameter
            ) or telescope_descriptions.get(parameter)

            parameter_data = all_parameter_data.get(parameter)
            parameter_version = parameter_data.get("parameter_version")
            unit = parameter_data.get("unit") or " "
            value_data = parameter_data.get("value")

            if value_data is None:
                continue

            file_flag = parameter_data.get("file", False)
            value = self._format_parameter_value(
                parameter, value_data, unit, file_flag, parameter_version
            )

            description = parameter_descriptions.get("description")
            short_description = parameter_descriptions.get("short_description") or description

            inst_class = parameter_descriptions.get("inst_class")

            matching_instrument = parameter_data["instrument"] == array_element
            if not names.is_design_type(array_element) and matching_instrument:
                parameter = f"***{parameter}***"
                parameter_version = f"***{parameter_version}***"
                if not re.match(r"^\[.*\]\(.*\)$", value.strip()):
                    value = f"***{value}***"
                description = f"***{description}***"
                short_description = f"***{short_description}***"

            # Group by class name
            if inst_class not in class_grouped_data:
                class_grouped_data[inst_class] = []

            class_grouped_data[inst_class].append(
                [
                    inst_class,
                    parameter,
                    parameter_version,
                    value,
                    description,
                    short_description,
                ]
            )

        data = []
        for class_name in sorted(class_grouped_data.keys(), reverse=True):
            sorted_class_data = sorted(class_grouped_data[class_name], key=lambda x: x[1])
            data.extend(sorted_class_data)

        return data

    def produce_calibration_reports(self):
        """Write calibration reports."""
        calibration_array_elements = self.db.get_array_elements(
            self.model_version, collection="calibration_devices"
        )
        array_elements = calibration_array_elements.copy()
        for element in calibration_array_elements:
            design_model = self.db.get_design_model(
                self.model_version, element, "calibration_devices"
            )
            if design_model and design_model not in array_elements:
                array_elements.append(design_model)

        for calibration_device in array_elements:
            all_parameter_data = self.db.get_model_parameters(
                site=names.get_site_from_array_element_name(calibration_device),
                array_element_name=calibration_device,
                collection="calibration_devices",
                model_version=self.model_version,
            )

            output_filename = Path(self.output_path / (f"{calibration_device}.md"))
            output_filename.parent.mkdir(parents=True, exist_ok=True)
            data = self.get_calibration_data(all_parameter_data, calibration_device)

            design_model = self.db.get_design_model(
                self.model_version, calibration_device, "calibration_devices"
            )

            with output_filename.open("w", encoding="utf-8") as file:
                file.write(f"# {calibration_device}\n")
                file.write("\n\n")

                if not names.is_design_type(calibration_device):
                    file.write(
                        "The design model can be found here: "
                        f"[{design_model}]"
                        f"({design_model}.md).\n\n"
                    )
                    file.write(
                        "Parameters shown in ***bold and italics*** are specific"
                        " to each array element.\n"
                        "Parameters without emphasis are inherited from the design model.\n"
                    )
                    file.write("\n\n")

                for class_name, group in groupby(data, key=lambda x: x[0]):
                    group = sorted(group, key=lambda x: x[1])
                    file.write(f"## {class_name}\n\n")
                    self._write_to_file(group, file)

        new_output_path = Path(self.output_path).parent.parent / "parameters"
        new_output_path.mkdir(parents=True, exist_ok=True)
        self.output_path = new_output_path
        for calibration_device in array_elements:
            self.site = names.get_site_from_array_element_name(calibration_device)
            self.array_element = calibration_device
            self.produce_model_parameter_reports(collection="calibration_devices")
