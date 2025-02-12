#!/usr/bin/python3

r"""Class to read and manage relevant model parameters for a given telescope model."""

import logging
import textwrap
from importlib.resources import files
from itertools import groupby
from operator import itemgetter
from pathlib import Path

from simtools.io_operations import io_handler
from simtools.model.telescope_model import TelescopeModel
from simtools.utils import general as gen
from simtools.utils import names
from simtools.visualization import plot_tables

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
        output_data_path = Path(self.output_path / self.telescope_model.name / "data_files")
        output_data_path.mkdir(parents=True, exist_ok=True)
        output_file_name = Path(input_file.stem + ".md")
        output_file = output_data_path / output_file_name

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

        return f"data_files/{output_file_name}"

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

        for instrument_class in names.instrument_classes("telescope"):
            for parameter, details in names.load_model_parameters(instrument_class).items():
                parameter_description[parameter] = details.get("description")
                short_description[parameter] = details.get("short_description")
                inst_class[parameter] = instrument_class

        return parameter_description, short_description, inst_class

    def get_telescope_parameter_data(self, args, telescope_model):
        """
        Get model parameter data.

        Returns
        -------
            list: A list of lists containing parameter names, values with units,
                  descriptions, and short descriptions.
        """
        all_params = telescope_model.db.get_model_parameters(
            site=args["site"], array_element_name=args["telescope"], collection="telescopes"
        )

        telescope_model.export_model_files()
        parameter_descriptions = self.get_all_parameter_descriptions()
        data = []

        for parameter in all_params:
            if all_params[parameter]["instrument"] == args["telescope"]:
                parameter_version = telescope_model.get_parameter_version(parameter)
                value = telescope_model.get_parameter_value_with_unit(parameter)
                if telescope_model.get_parameter_file_flag(parameter) and value:
                    try:
                        input_file_name = telescope_model.config_file_directory / Path(value)
                        output_file_name = self._convert_to_md(input_file_name)
                        value = f"[{Path(value).name}]({output_file_name})"
                    except FileNotFoundError:
                        value = f"File not found: {value}"
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

    def compare_parameter_across_versions(self, args, output_path, parameter_name):
        """
        Compare a parameter's value across different model versions.

        Parameters
        ----------
        parameter_name : str
            The name of the parameter to compare.
        telescope_model : TelescopeModel
            The telescope model instance.

        Returns
        -------
        list
            A list of dictionaries containing model version, parameter value, and description.
        """
        all_versions = self.telescope_model.db.get_model_versions()

        comparison_data = []

        for model_version in all_versions:
            telescope_model = TelescopeModel(
                site=args["site"],
                telescope_name=args["telescope"],
                model_version=model_version,
                label="reports",
                mongo_db_config=self.db_config,
            )
            value = telescope_model.get_parameter_value_with_unit(parameter_name)
            if isinstance(value, str) and value.endswith(".dat"):
                self.plot_parameter(output_path, parameter_name, telescope_model)
            parameter_data = self.get_telescope_parameter_data(args, telescope_model)
            for param in parameter_data:
                if param[1] == parameter_name:
                    comparison_data.append(
                        {
                            "model_version": model_version,
                            "parameter_version": param[2],
                            "value": param[3],
                            "description": param[4],
                        }
                    )
                    break
        return comparison_data

    def plot_parameter(self, output_path, parameter_name, telescope_model):
        """
        Produce plot of given parameter.

        Customize the config file according to command line input for
        producing reports and then use it to plot the tabular parameter data.

        Parameters
        ----------
        parameter_name : str
            The name of the parameter to compare.
        telescope_model : TelescopeModel
            The telescope model instance.

        """
        config_file_path = Path(files("simtools") / "reporting/plot_configuration_files")
        config_template = config_file_path / f"plot_{parameter_name}_parameter.yml"
        config_file = output_path / f"plot_{telescope_model.name}_{parameter_name}.yml"
        new_output_path = output_path / "images"
        new_output_path.mkdir(parents=True, exist_ok=True)
        try:
            with open(config_template, encoding="utf-8") as of:
                old_content = of.read()

            targets = ["site", "telescope"]
            replacements = [telescope_model.site, telescope_model.name]

            for k, v in zip(targets, replacements):
                old_content = old_content.replace(f"{{{k}}}", str(v))

            with open(config_file, "w", encoding="utf-8") as of:
                of.write(old_content)

            plot_config = gen.convert_keys_in_dict_to_lowercase(
                gen.collect_data_from_file(config_file)
            )

            plot_tables.plot(
                config=plot_config["cta_simpipe"]["plot"],
                output_file=io_handler.IOHandler().get_output_file(
                    new_output_path / f"{telescope_model.name}_{parameter_name}"
                ),
                db_config=self.db_config,
            )
        except FileNotFoundError:
            # self._logger.error('FileNotFoundError: ', parameter_name)
            pass

    def generate_array_element_report(self, args):
        """
        Generate a markdown report of all model parameters per array element.

        Parameters
        ----------
        output_path : str
            The folder where the markdown file will be saved.
        args_dict : dict
            Configuration arguments including model version and telescope name.
        """
        data = self.get_telescope_parameter_data(args, self.telescope_model)

        # Sort data by class to prepare for grouping
        data.sort(key=itemgetter(0, 1), reverse=True)

        output_filename = self.output_path / Path(args["telescope"] + ".md")

        # Start writing the Markdown file
        with Path(output_filename).open("w", encoding="utf-8") as file:
            # Group by class and write sections
            file.write(f"# {args['telescope']}\n")
            for class_name, group in groupby(data, key=itemgetter(0)):
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

    def generate_parameter_report(self, args):
        """
        Generate a markdown report per parameters per array element.

        Parameters
        ----------
        args: dict
            Configuration arguments including model version and telescope name.

        Output
        ----------
        One markdown report per model parameter of a given array element comparing
        values across model versions.
        """
        logger.info(
            f"Comparing parameters across model versions for Telescope: {args['telescope']}"
            f"and Site: {args['site']}."
        )
        io_handler_instance = io_handler.IOHandler()
        output_path = io_handler_instance.get_output_directory(
            label="reports", sub_dir=f"parameters/{args['telescope']}"
        )

        all_params = self.telescope_model.db.get_model_parameters(
            site=args["site"], array_element_name=args["telescope"], collection="telescopes"
        )

        for parameter in all_params:
            if all_params[parameter]["instrument"] == args["telescope"]:
                print("param: ", parameter, all_params[parameter]["instrument"])
                comparison_data = self.compare_parameter_across_versions(
                    args, output_path, parameter
                )
                if comparison_data:
                    output_filename = output_path / f"{parameter}.md"
                    with output_filename.open("w", encoding="utf-8") as file:
                        # Write header
                        file.write(
                            f"# {parameter}\n\n"
                            f"**Telescope**: {args['telescope']}\n\n"
                            f"**Description**: {comparison_data[0]['description']}\n\n"
                            "\n"
                        )

                        # Write table header
                        file.write(
                            "| Model Version      | Parameter Version      "
                            "| Value                |\n"
                            "|--------------------|------------------------"
                            "|----------------------|\n"
                        )

                        # Write table rows
                        for item in comparison_data:
                            file.write(
                                f"| {item['model_version']} |"
                                f" {item['parameter_version']} |"
                                f"{item['value']} |\n"
                            )

                        file.write("\n")
                        if isinstance(comparison_data[0]["value"], str) and comparison_data[0][
                            "value"
                        ].endswith(".md)"):
                            file.write(
                                f"![Parameter plot.](images/"
                                f"{args['telescope']}_{parameter}.png)"
                            )
