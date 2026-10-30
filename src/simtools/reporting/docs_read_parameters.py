#!/usr/bin/python3

r"""Class to read and manage relevant model parameters for a given telescope model."""

import logging
import re
import textwrap
from collections import defaultdict
from itertools import groupby
from pathlib import Path

import numpy as np

from simtools.context import RunContext
from simtools.data_model import parameter_catalog as catalog
from simtools.model.telescope_model import TelescopeModel
from simtools.reporting.docs_parameter_delta import ParameterDeltaMixin
from simtools.reporting.docs_parameter_plots import ParameterPlotMixin, scope_kwargs, version_key
from simtools.reporting.model_figures import (
    write_flasher_pulse_shape_plots,
    write_telescope_structure_figure,
)
from simtools.utils import names
from simtools.version import base_version_for_patch_delta, sort_versions

logger = logging.getLogger()


def _display_name(param):
    """Replace underscores by blanks, keeping markdown emphasis (***param***) intact."""
    if not isinstance(param, str):
        return param
    if param.startswith("***") and param.endswith("***"):
        return f"***{param.strip('*').replace('_', ' ')}***"
    return param.replace("_", " ")


class ReadParameters(ParameterDeltaMixin, ParameterPlotMixin):
    """Read and manage model parameter data for report generation."""

    def __init__(self, args, output_path, model_reader=None, context=None):
        """Initialise class."""
        self._logger = logging.getLogger(__name__)
        self.context = context or RunContext.from_environment()
        self.model_reader = model_reader or self.context.model_reader
        self.array_element = args.get("telescope", None)
        self.site = args.get("site", None)
        self.model_version = args.get("model_version", None)
        self.output_path = output_path
        self.observatory = args.get("observatory")
        self.software = args.get("simulation_software", None)
        self._telescope_identifiers = {}

    @property
    def model_version(self):
        """Model version."""
        return self._model_version

    @model_version.setter
    def model_version(self, model_version):
        """Set the model version; a list is rejected."""
        if isinstance(model_version, list):
            raise ValueError(
                f"Only one model version can be passed to {self.__class__.__name__}, not a list."
            )
        self._model_version = model_version

    def _format_file_parameter_value(
        self, parameter, value_data, parameter_version, telescope, site, model_version
    ):
        """Return a Markdown link to a parameter file (converted to Markdown if versioned)."""
        if parameter_version is None:
            repository_url = self._get_parameter_repository_url(
                parameter, value_data, telescope, site
            )
            return f"[{Path(value_data).name}]({repository_url})".strip()
        output_file_name = self._convert_to_md(
            parameter,
            parameter_version,
            f"{self.output_path}/model/{value_data}",
            **scope_kwargs(telescope, site),
            model_version=model_version,
        )
        return f"[{Path(value_data).name}]({output_file_name})".strip()

    def _format_parameter_value(
        self,
        parameter,
        value_data,
        unit,
        file_flag,
        parameter_version=None,
        telescope=None,
        site=None,
        model_version=None,
    ):
        """Format parameter value based on type."""
        if file_flag:
            return self._format_file_parameter_value(
                parameter, value_data, parameter_version, telescope, site, model_version
            )

        if isinstance(value_data, (str, int, float, dict)):
            return f"{value_data} {unit}".strip()

        if isinstance(value_data, (list, tuple)) and all(isinstance(v, dict) for v in value_data):
            title = parameter.replace("_", " ").title()
            return f"[View {title}](#{parameter.replace('_', '-')})"

        if len(value_data) > 5 and np.allclose(value_data, value_data[0]):
            return f"all: {value_data[0]} {unit}".strip()

        if isinstance(unit, list):
            return ", ".join(f"{v} {u}" for v, u in zip(value_data, unit)).strip()
        return ", ".join(f"{v} {unit}" for v in value_data).strip()

    def _group_model_versions_by_parameter_version(self, grouped_data):
        """Group model versions by parameter version and track the parameter values."""
        result = {}
        for parameter_name, items in grouped_data.items():
            groups = {}
            for item in items:
                group = groups.setdefault(
                    item["parameter_version"], {"versions": [], "entry": None}
                )
                group["versions"].append(item["model_version"])
                if group["entry"] is None or group["entry"]["value"] is None:
                    group["entry"] = item
            result[parameter_name] = [
                {
                    "value": group["entry"]["value"],
                    "parameter_version": param_version,
                    "file_flag": group["entry"]["file_flag"],
                    "dict_table": group["entry"].get("dict_table"),
                    "dict_unit": group["entry"].get("dict_unit"),
                    "model_version": ", ".join(sort_versions(group["versions"])),
                }
                for param_version, group in groups.items()
            ]
        return result

    def _is_list_of_dicts(self, value_data):
        """Return True if value_data is a non-empty list/tuple of dictionaries."""
        return bool(
            isinstance(value_data, (list, tuple))
            and value_data
            and all(isinstance(entry, dict) for entry in value_data)
        )

    def _export_model_files(self, parameters):
        """Export the model files of the given parameters to the output model directory."""
        Path(f"{self.output_path}/model").mkdir(parents=True, exist_ok=True)
        self.model_reader.export_model_files(
            parameters=parameters, dest=f"{self.output_path}/model"
        )

    def _build_version_parameter_item(self, version, parameter_name, parameter_data):
        """Build a grouped-data item for a given model version and parameter."""
        value_data = parameter_data.get("value")
        if parameter_data.get("instrument") != self.array_element or value_data is None:
            return None

        unit = parameter_data.get("unit") or " "
        file_flag = parameter_data.get("file", False)
        is_table = self._is_list_of_dicts(value_data)
        return {
            "value": self._format_parameter_value(parameter_name, value_data, unit, file_flag),
            "parameter_version": parameter_data.get("parameter_version"),
            "model_version": version,
            "file_flag": file_flag,
            "dict_table": value_data if is_table else None,
            "dict_unit": unit if is_table else None,
        }

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
        all_versions = self.model_reader.get_model_versions()
        all_versions.reverse()  # latest first
        grouped_data = defaultdict(list)

        for version in all_versions:
            parameter_dict = all_param_data.get(version, {})
            if not parameter_dict:
                continue
            self._export_model_files(all_param_data.get(version))
            for parameter_name in filter(parameter_dict.__contains__, all_parameter_names):
                item = self._build_version_parameter_item(
                    version, parameter_name, parameter_dict.get(parameter_name)
                )
                if item is not None:
                    grouped_data[parameter_name].append(item)

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
        return {
            parameter: {
                "description": details.get("description"),
                "short_description": details.get("short_description"),
                "inst_class": instrument_class,
            }
            for instrument_class in catalog.collection_to_instrument_class_key(collection)
            for parameter, details in catalog.model_parameters(instrument_class).items()
        }

    def _emphasize_specific(
        self,
        parameter,
        parameter_version,
        value,
        description,
        short_description,
        instrument,
        element,
    ):
        """Return the row fields in bold and italics if the parameter is specific to the element."""
        if names.is_design_type(element) or instrument != element:
            return parameter, parameter_version, value, description, short_description
        return (
            f"***{parameter}***",
            f"***{parameter_version}***",
            value if self.is_markdown_link(value) else f"***{value}***",
            f"***{description}***",
            f"***{short_description}***",
        )

    def get_array_element_parameter_data(self, telescope_model, collection="telescopes"):
        """
        Get model parameter data and descriptions for a given array element.

        Parameters
        ----------
        telescope_model : TelescopeModel
            The telescope model instance.

        Returns
        -------
        tuple: rows of [class, parameter, version, value, description, short description]
            and the dict tables (parameter, value, unit).
        """
        all_parameter_data = self.model_reader.get_model_parameters(
            site=telescope_model.site,
            array_element_name=telescope_model.name,
            collection=collection,
            model_version=telescope_model.model_version,
        )
        self._export_model_files(all_parameter_data)
        descriptions = self.get_all_parameter_descriptions()
        data = []
        dict_tables = []

        for parameter in filter(all_parameter_data.__contains__, catalog.model_parameters().keys()):
            parameter_data = all_parameter_data.get(parameter)
            parameter_version = telescope_model.get_parameter_version(parameter)
            unit = parameter_data.get("unit") or " "
            value_data = parameter_data.get("value")
            if value_data is None:
                continue
            if self._is_list_of_dicts(value_data):
                dict_tables.append((parameter, value_data, unit))
            value = self._format_parameter_value(
                parameter,
                value_data,
                unit,
                parameter_data.get("file", False),
                parameter_version,
                model_version=telescope_model.model_version,
            )
            info = descriptions.get(parameter)
            description = info.get("description")
            data.append(
                [
                    info.get("inst_class"),
                    *self._emphasize_specific(
                        parameter,
                        parameter_version,
                        value,
                        description,
                        info.get("short_description") or description,
                        parameter_data["instrument"],
                        telescope_model.name,
                    ),
                ]
            )

        return data, dict_tables

    def _write_to_file(self, data, file):
        """Write the parameter table (header, separator and rows)."""
        file.write(
            "| Parameter Name      |  Parameter Version     "
            "| Values      | Short Description           |\n"
            "|---------------------|------------------------"
            "|-------------|-----------------------------|\n"
        )
        for _, parameter_name, parameter_version, value, description, short_description in data:
            text = short_description if short_description else description
            wrapped_text = textwrap.fill(str(text), 60).replace("\n", " ")
            file.write(
                f"| {parameter_name:10} | {parameter_version:10} |"
                f" {value:20} | {wrapped_text:60} |\n"
            )
        file.write("\n\n")

    def get_simulation_configuration_data(self):
        """Get data and descriptions for simulation configuration parameters."""
        if self.software == "corsika":
            return self._get_simulation_configuration_data_for_telescope_site(
                self.array_element, self.site
            )

        results = []
        dict_tables = []
        for telescope in self.model_reader.get_array_elements(self.model_version):
            valid_site = names.get_site_from_array_element_name(telescope)
            for site in valid_site if isinstance(valid_site, list) else [valid_site]:
                data, tables = self._get_simulation_configuration_data_for_telescope_site(
                    telescope, site
                )
                results.extend(data)
                dict_tables.extend(tables)
        return results, dict_tables

    def _get_simulation_configuration_data_for_telescope_site(self, telescope, site):
        """Retrieve and format simulation-configuration parameter data for one telescope-site."""
        param_dict = self.model_reader.get_simulation_configuration_parameters(
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
        self.model_reader.export_model_files(parameters=param_dict, dest=str(model_output_path))

        data = []
        dict_tables = []
        for parameter, parameter_data in param_dict.items():
            value_data = parameter_data.get("value")
            if value_data is None:
                continue
            unit = parameter_data.get("unit") or " "
            info = parameter_descriptions.get(parameter)
            description = info.get("description")
            parameter_version = parameter_data.get("parameter_version")
            value = self._format_parameter_value(
                parameter,
                value_data,
                unit,
                parameter_data.get("file", False),
                parameter_version,
                telescope=telescope,
                site=site,
            )
            if self._is_list_of_dicts(value_data):
                dict_tables.append((telescope, parameter, value_data, unit))
            short_description = info.get("short_description", description)
            data.append(
                [telescope, parameter, parameter_version, value, description, short_description]
            )
        return data, dict_tables

    def produce_simulation_configuration_report(self):
        """Write simulation configuration report."""
        output_filename = Path(self.output_path / (f"configuration_{self.software}.md"))
        output_filename.parent.mkdir(parents=True, exist_ok=True)
        data, dict_tables = self.get_simulation_configuration_data()

        with output_filename.open("w", encoding="utf-8") as file:
            file.write(f"# configuration_{self.software}\n\n\n")
            if self.software == "sim_telarray":
                data.sort(key=lambda x: (x[0], x[1]))
                for telescope, group in groupby(data, key=lambda x: x[0]):
                    file.write(f"## [{telescope}]({telescope}.md)\n\n\n")
                    self._write_to_file(group, file)
                    for table_telescope, parameter, value_data, unit in dict_tables:
                        if table_telescope == telescope:
                            self._write_dict_table(parameter, file, value_data, unit)
            else:
                self._write_to_file(data, file)
                for _, parameter, value_data, unit in dict_tables:
                    self._write_dict_table(parameter, file, value_data, unit)

    def produce_array_element_report(self):
        """Produce a markdown report of all model parameters per array element."""
        if self.observatory:
            self.produce_observatory_report()
            return

        base_version = base_version_for_patch_delta(self.model_version)
        if base_version and self._produce_array_element_delta_report(base_version):
            return

        telescope_model = TelescopeModel(
            site=self.site,
            telescope_name=self.array_element,
            model_version=self.model_version,
            label="reports",
            ignore_software_version=True,
            model_reader=self.model_reader,
            model_directory=Path(self.output_path) / "model",
            context=self.context,
        )

        output_filename = Path(self.output_path / (telescope_model.name + ".md"))
        output_filename.parent.mkdir(parents=True, exist_ok=True)
        data, dict_tables = self.get_array_element_parameter_data(telescope_model)
        is_design_model = telescope_model.name == telescope_model.design_model
        if not is_design_model:
            data, dict_tables = self._select_specific_parameters(data, dict_tables)
        data.sort(key=lambda x: (x[0], x[1]), reverse=True)

        with output_filename.open("w", encoding="utf-8") as file:
            file.write(f"# {telescope_model.name}\n")
            write_telescope_structure_figure(
                file, telescope_model, self._logger, context=self.context
            )
            if not is_design_model:
                file.write(
                    f"Parameters specific to {telescope_model.name}"
                    " (shown in ***bold and italics***).\n"
                    "All other parameters are inherited from the design model "
                    f"[{telescope_model.design_model}]({telescope_model.design_model}.md).\n"
                    "\n\n"
                )
            for class_name, group in groupby(data, key=lambda x: x[0]):
                file.write(f"## {class_name}\n\n")
                self._write_to_file(sorted(group, key=lambda x: x[1]), file)
            for parameter, value_data, unit in dict_tables:
                self._write_dict_table(parameter, file, value_data, unit)

    @staticmethod
    def _select_specific_parameters(data, dict_tables):
        """Select the parameters specific to an array element (emphasised in bold italics)."""
        specific_data = [row for row in data if row[1].startswith("***")]
        specific_names = {row[1].strip("*") for row in specific_data}
        return specific_data, [e for e in dict_tables if e[0] in specific_names]

    def produce_model_parameter_reports(self, collection="telescopes"):
        """
        Produce a markdown report of all model parameters for a given array element.

        Outputs one markdown report per array element with one section per model parameter,
        comparing parameter values across model versions.
        """
        logger.info(
            f"Comparing parameters across model versions for Telescope: {self.array_element}"
            f" and Site: {self.site}."
        )
        all_parameter_names = catalog.model_parameters(None).keys()
        all_parameter_data = self.model_reader.get_model_parameters_for_all_model_versions(
            site=self.site, array_element_name=self.array_element, collection=collection
        )
        comparison_data = self._compare_parameter_across_versions(
            all_parameter_data, all_parameter_names
        )
        parameters = [p for p in all_parameter_names if comparison_data.get(p)]
        if not parameters:
            return

        Path(self.output_path).mkdir(parents=True, exist_ok=True)
        output_filename = Path(self.output_path) / f"{self.array_element}.md"
        with output_filename.open("w", encoding="utf-8") as file:
            file.write(
                f"# {self.array_element}\n\n"
                "Model parameters and their values for all model versions.\n\n"
            )
            for parameter in parameters:
                self._write_parameter_section(file, parameter, comparison_data, collection)

    def _write_parameter_section(self, file, parameter, comparison_data, collection):
        """Write the section of a single parameter comparing values across model versions."""
        info = self.get_all_parameter_descriptions(collection=collection).get(
            parameter
        ) or self.get_all_parameter_descriptions(collection="telescopes").get(parameter)
        file.write(f"## {parameter}\n\n**Description**: {info.get('description')}\n\n\n")

        file.write(
            "| Parameter Version      | Model Version(s)      "
            "| Value                |\n"
            "|------------------------|--------------------"
            "|----------------------|\n"
        )
        for item in comparison_data.get(parameter):
            file.write(
                f"| {item['parameter_version']} | {item['model_version']} | {item['value']} |\n"
            )
        file.write("\n")

        dict_items = [i for i in comparison_data.get(parameter) if i.get("dict_table") is not None]
        if dict_items:
            latest = max(dict_items, key=lambda x: version_key(x["parameter_version"]))
            self._write_dict_table(
                parameter,
                file,
                latest["dict_table"],
                latest.get("dict_unit") or " ",
                heading_level=3,
            )

        if comparison_data.get(parameter)[0]["file_flag"]:
            self._write_file_flag_section(file, parameter, comparison_data, collection)
        elif parameter == "flasher_pulse_shape":
            write_flasher_pulse_shape_plots(
                file,
                comparison_data[parameter],
                collection,
                self.model_reader,
                self.site,
                self.array_element,
                self.context,
            )
        file.write("\n\n")

    def _write_dict_table(self, parameter, file, value_data, unit, heading_level=2):
        """Write a markdown table for parameters that are lists of dictionaries."""
        if not value_data:
            return

        file.write(f"\n{'#' * heading_level} {parameter.replace('_', ' ').title()}\n\n")
        headers = value_data[0].keys()
        file.write("| " + " | ".join(headers) + " |\n")
        file.write("|" + " --- |" * len(headers) + "\n")
        for entry in value_data:
            cells = []
            for h in headers:
                cell_value = entry.get(h, "")
                u = unit if h == "value" and cell_value is not None else ""
                cells.append(f"{cell_value} {u}".strip())
            file.write("| " + " | ".join(cells) + " |\n")

    def _write_array_triggers_section(self, file, trigger_configs):
        """Write the array triggers section of the report."""
        file.write(
            "\n## Array Trigger Configurations\n\n"
            "| Trigger Name | Multiplicity | Width | Hard Stereo | Min Separation |\n"
            "|--------------|--------------|--------|-------------|----------------|\n"
        )
        for config in trigger_configs:
            mult = f"{config['multiplicity']['value']} {config['multiplicity']['unit'] or ''}"
            width = f"{config['width']['value']} {config['width']['unit'] or ''}"
            stereo = "Yes" if config["hard_stereo"]["value"] else "No"
            min_sep = (
                f"{config['min_separation']['value']} {config['min_separation']['unit'] or '-'}"
            )
            file.write(
                f"| {config['name']} | {mult.strip()} | {width.strip()} | {stereo} "
                f"| {min_sep.strip()} |\n"
            )
        file.write("\n")

    def _write_parameters_table(self, file, all_parameter_data):
        """Write the main parameters table of the report."""
        file.write(
            "| Parameter | Value | Parameter Version |\n"
            "|-----------|--------|-------------------|\n"
        )
        dict_tables = [
            (name, data.get("value"), data.get("unit") or " ")
            for name, data in sorted(all_parameter_data.items())
            if name not in ("array_layouts", "array_triggers")
            and self._is_list_of_dicts(data.get("value"))
        ]

        for param_name, param_data in sorted(all_parameter_data.items()):
            value = param_data.get("value")
            if value is None:
                continue
            unit = param_data.get("unit") or " "
            parameter_version = param_data.get("parameter_version")
            if param_name == "array_layouts":
                file.write(
                    f"| array_layouts | [View Array Layouts](#array-layouts) | "
                    f"{parameter_version} |\n"
                )
            elif param_name == "array_triggers":
                file.write(
                    "| array_triggers | [View Trigger Configurations]"
                    f"(#array-trigger-configurations) | {parameter_version} |\n"
                )
            else:
                formatted_value = self._format_parameter_value(
                    param_name,
                    value,
                    unit,
                    param_data.get("file", False),
                    parameter_version,
                    model_version=self.model_version,
                )
                file.write(f"| {param_name} | {formatted_value} | {parameter_version} |\n")

        file.write("\n")
        for param_name, value, unit in dict_tables:
            self._write_dict_table(param_name, file, value, unit)

    def produce_observatory_report(self):
        """Produce a markdown report of all observatory parameters for a given site."""
        base_version = base_version_for_patch_delta(self.model_version)
        if base_version and self._produce_observatory_delta_report(base_version):
            return

        output_filename = Path(self.output_path / f"OBS-{self.site}.md")
        output_filename.parent.mkdir(parents=True, exist_ok=True)
        all_parameter_data = self.model_reader.get_model_parameters(
            site=self.site,
            array_element_name="OBS-" + self.site,
            collection="sites",
            model_version=self.model_version,
        )
        if not all_parameter_data:
            logger.warning(f"No observatory parameters found for site {self.site}")
            return

        self._export_model_files(all_parameter_data)
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
        class_grouped_data = {}
        dict_tables = []

        for parameter, parameter_data in all_parameter_data.items():
            row, dict_table = self._build_calibration_row(
                parameter,
                parameter_data,
                calibration_descriptions.get(parameter) or telescope_descriptions.get(parameter),
                array_element,
            )
            if row is None:
                continue
            if dict_table is not None:
                dict_tables.append(dict_table)
            class_grouped_data.setdefault(row[0], []).append(row)

        data = []
        for class_name in sorted(class_grouped_data, reverse=True):
            data.extend(sorted(class_grouped_data[class_name], key=lambda x: x[1]))
        return data, dict_tables

    def _build_calibration_row(
        self, parameter, parameter_data, parameter_descriptions, array_element
    ):
        """Build one calibration table row and optional dict-table payload."""
        value_data = parameter_data.get("value")
        if value_data is None:
            return None, None

        parameter_version = parameter_data.get("parameter_version")
        unit = parameter_data.get("unit") or " "
        dict_table = (parameter, value_data, unit) if self._is_list_of_dicts(value_data) else None
        value = self._format_parameter_value(
            parameter, value_data, unit, parameter_data.get("file", False), parameter_version
        )
        description = parameter_descriptions.get("description")
        return [
            parameter_descriptions.get("inst_class"),
            *self._emphasize_specific(
                parameter,
                parameter_version,
                value,
                description,
                parameter_descriptions.get("short_description") or description,
                parameter_data["instrument"],
                array_element,
            ),
        ], dict_table

    def is_markdown_link(self, value):
        """
        Return True if the string is a Markdown-style link: [text](target).

        Parameters
        ----------
        value : str
            The string to check.

        Returns
        -------
        bool
            True if the string is a Markdown link, False otherwise.
        """
        return bool(re.fullmatch(r"\[[^\]]*\]\([^)]+\)", value.strip()))

    def produce_calibration_reports(self):
        """Write calibration reports."""
        for calibration_device in self._collect_calibration_array_elements():
            device_sites = names.get_site_from_array_element_name(calibration_device)
            site = device_sites[0] if isinstance(device_sites, list) else device_sites
            all_parameter_data = self.model_reader.get_model_parameters(
                site=site,
                array_element_name=calibration_device,
                collection="calibration_devices",
                model_version=self.model_version,
            )

            output_filename = Path(self.output_path / (f"{calibration_device}.md"))
            output_filename.parent.mkdir(parents=True, exist_ok=True)
            data, dict_tables = self.get_calibration_data(all_parameter_data, calibration_device)
            design_model = self.model_reader.get_design_model(
                self.model_version, calibration_device, "calibration_devices"
            )
            self._write_single_calibration_report(
                output_filename, calibration_device, data, design_model, dict_tables=dict_tables
            )

    def _collect_calibration_array_elements(self):
        """Return a list of calibration devices including their design models."""
        calibration_array_elements = self.model_reader.get_array_elements(
            self.model_version, collection="calibration_devices"
        )
        array_elements = calibration_array_elements.copy()
        for element in calibration_array_elements:
            design_model = self.model_reader.get_design_model(
                self.model_version, element, "calibration_devices"
            )
            if design_model and design_model not in array_elements:
                array_elements.append(design_model)
        return array_elements

    def _write_single_calibration_report(
        self, output_filename, calibration_device, data, design_model, dict_tables=None
    ):
        """Write a single calibration device markdown report file."""
        with output_filename.open("w", encoding="utf-8") as file:
            file.write(f"# {calibration_device}\n\n\n")
            if not names.is_design_type(calibration_device):
                file.write(
                    f"The design model can be found here: [{design_model}]({design_model}.md).\n\n"
                    "Parameters shown in ***bold and italics*** are specific"
                    " to each array element.\n"
                    "Parameters without emphasis are inherited from the design model.\n\n\n"
                )

            for class_name, group in groupby(data, key=lambda x: x[0]):
                file.write(f"## {class_name}\n\n")
                self._write_to_file(
                    [
                        [row[0], _display_name(row[1]), *row[2:]]
                        for row in sorted(group, key=lambda x: x[1])
                    ],
                    file,
                )

            for parameter, value_data, unit in dict_tables or []:
                self._write_dict_table(parameter, file, value_data, unit)

    def generate_model_parameter_reports_for_devices(self, array_elements):
        """Create model-parameter comparison reports for calibration devices."""
        for calibration_device in array_elements:
            device_sites = names.get_site_from_array_element_name(calibration_device)
            # Parameters are site independent, so read the first site's value.
            self.site = device_sites[0] if isinstance(device_sites, list) else device_sites
            self.array_element = calibration_device
            self.produce_model_parameter_reports(collection="calibration_devices")
