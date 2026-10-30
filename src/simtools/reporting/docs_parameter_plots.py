#!/usr/bin/python3

r"""Plot and file-reference helpers of the parameter reports (mixed into ReadParameters)."""

import logging
from pathlib import Path

from simtools.constants import DEFAULT_SIMULATIONS_REPO
from simtools.io import ascii_handler
from simtools.model_repository.asset_names import get_export_file_name
from simtools.reporting.figure_cache import figure_cache_key
from simtools.reporting.model_figures import (
    plot_camera_config,
    plot_mirror_config,
    plot_nsb_altitude_correction,
)
from simtools.utils import names
from simtools.visualization import plot_tables, visualize

logger = logging.getLogger()


def _get_figure_cache_key(plot_config, data, context):
    """Hash effective table-plot inputs and output settings for per-run figure reuse."""
    # The model_version selects the source data; resolved data is hashed separately.
    config_for_render = {
        **plot_config,
        "tables": [
            {k: v for k, v in t.items() if k != "model_version"}
            for t in plot_config.get("tables", [])
        ],
    }
    return figure_cache_key(
        renderer="plot_tables._plot_data",
        renderer_modules={"plot_tables": plot_tables.__file__, "visualize": visualize.__file__},
        plot_config=config_for_render,
        rendered_data=data,
        context=context,
    )


def scope_kwargs(telescope=None, site=None):
    """Return the non-empty telescope and site keyword arguments."""
    return {k: v for k, v in (("telescope", telescope), ("site", site)) if v}


def version_key(version):
    """Sort key for dotted version strings."""
    return tuple(map(int, version.split(".")))


# Mixin kept as a class: its helpers are patched by name in the unit tests.
class ParameterPlotMixin:  # pylint: disable=too-few-public-methods
    """Plot and file-reference methods of ReadParameters (relies on its attributes)."""

    def _generate_plots(
        self,
        parameter,
        parameter_version,
        input_file,
        outpath,
        telescope=None,
        site=None,
        model_version=None,
    ):
        """Generate plots based on the parameter type."""
        telescope_name = telescope or self.array_element
        site_name = site or self.site
        version = model_version or self.model_version
        if parameter == "correct_nsb_spectrum_to_telescope_altitude":
            return plot_nsb_altitude_correction(input_file, outpath, context=self.context)
        if parameter == "camera_pixel_layout":
            return plot_camera_config(
                parameter_version,
                outpath,
                telescope_name,
                site_name,
                version,
                self.model_reader,
                context=self.context,
            )
        if parameter in (
            "mirror_list",
            "primary_mirror_segmentation",
            "secondary_mirror_segmentation",
        ):
            return plot_mirror_config(
                parameter,
                parameter_version,
                input_file,
                outpath,
                telescope_name,
                site_name,
                version,
                context=self.context,
            )
        if parameter_version:
            return self._plot_parameter_tables(
                parameter,
                parameter_version,
                outpath,
                **scope_kwargs(telescope, site),
                model_version=model_version,
            )
        return []

    def _plot_parameter_tables(
        self, parameter, parameter_version, outpath, telescope=None, site=None, model_version=None
    ):
        """Generate plots for parameter tables."""
        config_data = plot_tables.generate_plot_configurations(
            parameter=parameter,
            parameter_version=parameter_version,
            site=site or self.site,
            telescope=self._get_telescope_identifier(model_version, telescope),
            output_path=outpath,
            plot_type="all",
            model_reader=self.model_reader,
            model_version=model_version or self.model_version,
            context=self.context,
        )
        if not config_data:
            return []

        plot_configs, output_files = config_data
        plot_names = []
        for plot_config, output_file in zip(plot_configs, output_files):
            plot_data = plot_tables.read_table_data(
                plot_config, model_reader=self.model_reader, context=self.context
            )
            figure_stem = (
                f"{output_file.stem.replace('.', '-')}_"
                f"{_get_figure_cache_key(plot_config, plot_data, self.context)}"
            )
            plot_names.append(figure_stem)
            if not (outpath / figure_stem).with_suffix(".png").exists():
                plot_tables._plot_data(  # pylint: disable=protected-access
                    plot_data, plot_config, outpath / figure_stem, context=self.context
                )
        return plot_names

    def _get_telescope_identifier(self, model_version=None, telescope=None):
        """Get the appropriate telescope design type for file naming (e.g., LSTN-design)."""
        model_version = model_version or self.model_version
        telescope = telescope or self.array_element
        cache_key = (model_version, telescope)
        if cache_key not in self._telescope_identifiers:
            self._telescope_identifiers[cache_key] = self.model_reader.get_design_model(
                model_version, telescope, collection="telescopes"
            )
        if not telescope:
            return None
        if not names.is_design_type(telescope):
            return self._telescope_identifiers[cache_key]
        return telescope

    def _get_parameter_repository_url(self, parameter, file_name, telescope=None, site=None):
        """Return the source URL for a model parameter asset."""
        telescope = telescope or self.array_element
        if telescope:
            scope = (
                self._get_telescope_identifier(telescope=telescope)
                if self.model_version
                else telescope
            )
        elif site or self.site:
            scope = f"OBS-{site or self.site}"
        else:
            scope = "global"
        return (
            f"{DEFAULT_SIMULATIONS_REPO}/simulation-model/simulation-models/-/blob/main/"
            f"simulation-models/model_parameters/{scope}/{parameter}/{Path(file_name).name}"
        )

    def _convert_to_md(
        self,
        parameter,
        parameter_version,
        input_file,
        telescope=None,
        site=None,
        model_version=None,
    ):
        """Convert a file to a Markdown file, preserving formatting."""
        input_file = Path(input_file)
        if not input_file.exists():
            raise FileNotFoundError(f"Data file not found: {input_file}")

        output_data_path = Path(self.output_path / "_data_files")
        output_data_path.mkdir(parents=True, exist_ok=True)
        output_file_name = Path(input_file.stem + ".md")
        markdown_output_file = output_data_path / output_file_name

        if not markdown_output_file.exists():
            outpath = self.context.paths.get_output_directory("_images")
            outpath.mkdir(parents=True, exist_ok=True)
            plot_names = self._generate_plots(
                parameter,
                parameter_version,
                input_file,
                outpath,
                **scope_kwargs(telescope, site),
                model_version=model_version or self.model_version,
            )
            file_contents = ascii_handler.read_file_encoded_in_utf_or_latin(input_file)
            repository_url = self._get_parameter_repository_url(
                parameter, input_file, telescope, site
            )
            markdown_output_file.write_text(
                f"# {input_file.stem}\n"
                + "".join(f"![Parameter plot.]({outpath}/{name}.png)\n\n" for name in plot_names)
                + "\n\nThe full file can be found in the Simulation Model repository [here]"
                f"({repository_url}).\n\n\n\n"
                "The first 30 lines of the file are:\n```\n"
                + "".join(file_contents[:30])
                + "\n```",
                encoding="utf-8",
            )

        return f"_data_files/{output_file_name}"

    def _write_file_flag_section(self, file, parameter, comparison_data, collection):
        """Write image/plot references when parameter entries include files."""
        outpath = self.context.paths.get_output_directory("_images")
        latest_item = max(
            comparison_data.get(parameter), key=lambda x: version_key(x["parameter_version"])
        )
        latest_parameter_version = latest_item["parameter_version"]
        latest_model_version = max(latest_item["model_version"].split(", "), key=version_key)
        parameter_data = self.model_reader.get_model_parameters(
            site=self.site,
            array_element_name=self.array_element,
            collection=collection,
            model_version=latest_model_version,
        )
        selected_parameter = parameter_data.get(parameter)
        if (
            not selected_parameter
            or selected_parameter.get("parameter_version") != latest_parameter_version
            or not selected_parameter.get("file")
        ):
            return

        model_path = Path(self.output_path) / "model"
        model_path.mkdir(parents=True, exist_ok=True)
        self.model_reader.export_model_files(parameters=parameter_data, dest=model_path)
        plot_names = self._generate_plots(
            parameter,
            latest_parameter_version,
            model_path / get_export_file_name(selected_parameter),
            outpath,
            telescope=self.array_element,
            site=self.site,
            model_version=latest_model_version,
        )
        image_paths = [p for p in (outpath / f"{name}.png" for name in plot_names) if p.is_file()]
        if not image_paths:
            logger.info(
                "No plot configuration or generated image found for %s version %s",
                parameter,
                latest_parameter_version,
            )
            return

        file.write("The latest parameter version is plotted below.\n\n")
        for image_path in image_paths:
            file.write(f"![Parameter plot.]({image_path.as_posix()})\n\n")

    def _write_array_layouts_section(self, file, layouts):
        """Write the array layouts section of the report."""
        file.write("\n## Array Layouts\n\n")
        version = self.model_version.replace(".", "-")
        for layout in layouts:
            layout_name = layout["name"]
            file.write(f"### {layout_name}\n\n| Element |\n|---------|\n")
            for element in sorted(layout["elements"]):
                file.write(f"| [{element}]({element}.md) |\n")
            filename = f"array_layout_{layout_name}_{self.site}_ground_{version}.png"
            file.write(f"\n![{layout_name} Layout](/_images/{filename})\n\n\n")
