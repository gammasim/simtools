#!/usr/bin/python3

r"""Plot and file-reference helpers of the parameter reports (mixed into ReadParameters)."""

import logging
from pathlib import Path

import numpy as np

from simtools.constants import DEFAULT_SIMULATIONS_REPO
from simtools.io import ascii_handler
from simtools.layout.array_layout_utils import read_layouts
from simtools.model import camera as camera_model
from simtools.model_repository.asset_names import get_export_file_name
from simtools.reporting.model_figures import (
    plot_camera_config,
    plot_mirror_config,
    plot_nsb_altitude_correction,
)
from simtools.utils import names
from simtools.visualization import plot_tables, plot_trigger_patches, visualize
from simtools.visualization.array_layout_cache import array_layout_figure_name
from simtools.visualization.matplotlib_backend import pyplot as plt
from simtools.visualization.visualize import save_figure

logger = logging.getLogger()


def _configured_figure_formats(context):
    """Return the figure formats that save_figure writes for this context (PNG by default)."""
    formats = visualize.figure_output_settings(context).get("figure_format")
    if not formats:
        return ["png"]
    return [formats] if isinstance(formats, str) else list(formats)


def _first_written_figure(output_dir, name, formats):
    """Return the first existing figure file of a plot name, in the order of the formats."""
    for figure_format in formats:
        path = Path(output_dir) / f"{name}.{figure_format}"
        if path.is_file():
            return path
    return None


def _count_trigger_groups_per_pixel(pixel_ids, trigger_groups, trigger_members):
    """Count distinct configured trigger groups containing each camera pixel."""
    pixel_ids = np.asarray(pixel_ids, dtype=int)
    if len(np.unique(pixel_ids)) != len(pixel_ids):
        raise ValueError("Camera pixel IDs must be unique")
    pixel_indices = {int(pixel_id): index for index, pixel_id in enumerate(pixel_ids)}
    try:
        group_ids = np.asarray(trigger_groups["group_id"], dtype=int)
        member_groups = np.asarray(trigger_members["group_id"], dtype=int)
        member_pixels = np.asarray(trigger_members["pixel_id"], dtype=int)
    except KeyError as exc:
        raise ValueError(f"Trigger configuration is missing required column {exc.args[0]}") from exc
    if len(np.unique(group_ids)) != len(group_ids):
        raise ValueError("Trigger group IDs must be unique")
    if group_ids.size == 0:
        raise ValueError("Camera configuration contains no trigger groups")
    unknown_groups = np.setdiff1d(np.unique(member_groups), group_ids)
    if len(unknown_groups):
        raise ValueError(f"Trigger member references unknown trigger group {unknown_groups[0]}")
    unknown_pixels = np.setdiff1d(np.unique(member_pixels), pixel_ids)
    if len(unknown_pixels):
        raise ValueError(f"Trigger member pixel {unknown_pixels[0]} is not in the camera")
    empty_groups = np.setdiff1d(group_ids, member_groups)
    if empty_groups.size:
        raise ValueError(f"Trigger groups have no members: {empty_groups.tolist()}")
    unique_memberships = np.unique(np.column_stack((member_groups, member_pixels)), axis=0)
    member_pixels, memberships_per_pixel = np.unique(unique_memberships[:, 1], return_counts=True)
    counts = np.zeros(len(pixel_ids), dtype=int)
    for pixel_id, count in zip(member_pixels, memberships_per_pixel):
        counts[pixel_indices[int(pixel_id)]] = count
    return counts


def _get_figure_cache_key(plot_config, data, context):
    """Hash rendered table data and plot options for cross-report image reuse."""
    # Table settings select and transform the data, which is represented in `data` already.
    nonvisual_options = {"tables", "type", "plot_name"}
    config_for_render = {
        key: value for key, value in plot_config.items() if key not in nonvisual_options
    }
    return visualize.figure_cache_key(
        renderer="plot_tables._plot_data",
        renderer_modules={
            "docs_parameter_plots": __file__,
            "plot_tables": plot_tables.__file__,
            "visualize": visualize.__file__,
        },
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


def _plot_signature(config):
    """Return the plot and column settings that must match for an overlay."""
    table_options = (
        "parameter",
        "column_x",
        "column_y",
        "column_x_err",
        "column_y_err",
        "normalize_y",
        "select_values",
    )
    tables = [
        {key: table[key] for key in table_options if key in table}
        for table in config.get("tables", [])
    ]
    return (
        {key: value for key, value in config.items() if key not in ("tables", "label")},
        tables,
    )


def _overlay_plot_data(plot_data, parameter_version):
    """Label each series with its parameter version for a comparison plot."""
    return {f"{label} ({parameter_version})": values for label, values in plot_data.items()}


def _table_axis_units(table_config, model_reader, context):
    """Return the rendered table's x/y units, or None when they cannot be resolved."""
    try:
        table = model_reader.export_model_file(
            parameter=table_config["parameter"],
            site=table_config["site"],
            array_element_name=table_config.get("telescope"),
            parameter_version=table_config.get("parameter_version"),
            model_version=table_config.get("model_version"),
            export_file_as_table=True,
            dest=table_config.get("model_export_path") or context.paths.get_output_directory(),
        )
        return tuple(
            str(getattr(table[column], "unit", None) or "")
            for column in (table_config["column_x"], table_config["column_y"])
        )
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        logger.debug(
            "Axis units of %s could not be resolved: %s", table_config.get("parameter"), exc
        )
        return None


def _parameter_plot_candidates(all_versions, parameter, parameter_version, model_version):
    """Return the three latest parameter versions available through model_version."""
    candidates = {}
    target_key = version_key(model_version)
    for selected_model_version, parameters in all_versions.items():
        if version_key(selected_model_version) > target_key:
            continue
        record = parameters.get(parameter)
        selected_parameter_version = (
            record.get("parameter_version") if isinstance(record, dict) else None
        )
        if selected_parameter_version:
            candidates[selected_parameter_version] = selected_model_version
    candidates[parameter_version] = model_version
    return [(version, candidates[version]) for version in sorted(candidates, key=version_key)[-3:]]


def _matching_version_plot_config(configs, current_signature, table_config, model_reader, context):
    """Find an equivalent plot config with matching axis units."""
    candidates = [
        config
        for config in configs
        if _plot_signature(config) == current_signature and len(config.get("tables", [])) == 1
    ]
    if not candidates:
        return None
    reference_units = _table_axis_units(table_config, model_reader, context)
    for config in candidates:
        units = _table_axis_units(config["tables"][0], model_reader, context)
        if units is not None and units == reference_units:
            return config
    return None


def _read_parameter_plot_version(
    parameter,
    current_parameter_version,
    selected_parameter_version,
    selected_model_version,
    plot_config,
    current_data,
    current_signature,
    site,
    telescope,
    outpath,
    model_reader,
    context,
):
    """Return data for a candidate version when an equivalent plot exists."""
    if selected_parameter_version == current_parameter_version:
        return current_data
    result = plot_tables.generate_plot_configurations(
        parameter=parameter,
        parameter_version=selected_parameter_version,
        site=site,
        telescope=telescope,
        output_path=outpath,
        plot_type="all",
        model_reader=model_reader,
        model_version=selected_model_version,
        context=context,
    )
    if not result:
        return None
    configs, _ = result
    matching = _matching_version_plot_config(
        configs,
        current_signature,
        plot_config["tables"][0],
        model_reader,
        context,
    )
    if matching is None:
        return None
    data = plot_tables.read_table_data(matching, model_reader=model_reader, context=context)
    return data if len(data) == 1 else None


def _plot_camera_trigger_membership(
    output_path, telescope, site, model_version, model_reader, context
):
    """Plot the number of trigger groups containing each camera pixel."""
    if not telescope or not site or not model_version:
        return []
    parameters = model_reader.get_model_parameters(
        site=site,
        array_element_name=telescope,
        collection="telescopes",
        model_version=model_version,
        parameter_names=(
            "camera_pixel_layout",
            "camera_pixel_types",
            "camera_rotate",
            "camera_trigger_groups",
            "camera_trigger_members",
        ),
    )
    required = (
        "camera_pixel_layout",
        "camera_pixel_types",
        "camera_trigger_groups",
        "camera_trigger_members",
    )
    if any(name not in parameters for name in required):
        return []
    layout_table = model_reader.get_parameter_table(parameters["camera_pixel_layout"])
    pixels = [
        {
            "pixel_id": int(pixel_id),
            "x_cm": float(getattr(x_coordinate, "value", x_coordinate)),
            "y_cm": float(getattr(y_coordinate, "value", y_coordinate)),
            "enabled": bool(enabled),
        }
        for pixel_id, x_coordinate, y_coordinate, enabled in zip(
            layout_table["pixel_id"],
            layout_table["x_cm"],
            layout_table["y_cm"],
            layout_table["enabled"],
        )
    ]
    components = {
        "pixels": pixels,
        "pixel_types": parameters["camera_pixel_types"]["value"],
        "rotate": parameters.get("camera_rotate", {}).get("value", 0.0),
    }
    camera = camera_model.Camera.from_configuration(telescope, components, 1.0)
    group_table = model_reader.get_parameter_table(parameters["camera_trigger_groups"])
    member_table = model_reader.get_parameter_table(parameters["camera_trigger_members"])
    pixel_ids = np.asarray(camera.pixels["pix_id"], dtype=int)
    group_counts = _count_trigger_groups_per_pixel(pixel_ids, group_table, member_table)
    rendered = {
        "pixel_ids": pixel_ids.tolist(),
        "x_cm": np.asarray(camera.pixels["x"].value, dtype=float).tolist(),
        "y_cm": np.asarray(camera.pixels["y"].value, dtype=float).tolist(),
        "group_counts": group_counts.tolist(),
    }
    cache_key = visualize.figure_cache_key(
        renderer="camera_trigger_groups.per_pixel_membership_count",
        renderer_modules={
            "model_figures": __file__,
            "camera": camera_model.__file__,
            "trigger_patches": plot_trigger_patches.__file__,
        },
        plot_config={"telescope": telescope, "units": "cm", "measurement": "distinct groups"},
        rendered_data=rendered,
        context=context,
    )
    plot_name = f"camera_trigger_membership_{telescope}_{cache_key}"
    formats = _configured_figure_formats(context)
    image_paths = [Path(output_path) / f"{plot_name}.{fmt}" for fmt in formats]
    if not all(path.is_file() for path in image_paths):
        figure, axes = plt.subplots(figsize=(8, 8))
        points = axes.scatter(
            np.asarray(camera.pixels["x"].value, dtype=float),
            np.asarray(camera.pixels["y"].value, dtype=float),
            c=group_counts,
            s=12,
            cmap="viridis",
        )
        axes.set_title(f"Trigger groups per pixel in {telescope}")
        axes.set_xlabel("Camera x [cm]")
        axes.set_ylabel("Camera y [cm]")
        axes.set_aspect("equal", adjustable="box")
        figure.colorbar(points, ax=axes, label="Distinct trigger groups per pixel")
        save_figure(figure, Path(output_path) / plot_name, close=True, context=context)
    return [plot_name] if all(path.is_file() for path in image_paths) else []


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
        if parameter in ("camera_trigger_groups", "camera_trigger_members"):
            return _plot_camera_trigger_membership(
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

        plot_configs, _output_files = config_data
        plot_names = []
        for plot_config in plot_configs:
            plot_data = plot_tables.read_table_data(
                plot_config, model_reader=self.model_reader, context=self.context
            )
            versions = self._get_compatible_parameter_plot_versions(
                parameter,
                parameter_version,
                plot_config,
                plot_data,
                site or self.site,
                self._get_telescope_identifier(model_version, telescope),
                model_version or self.model_version,
                outpath,
            )
            if versions:
                plot_data = {}
                for version, version_data in versions:
                    plot_data.update(_overlay_plot_data(version_data, version))
            figure_stem = (
                f"parameter_plot_{_get_figure_cache_key(plot_config, plot_data, self.context)}"
            )
            plot_names.append(figure_stem)
            if not (outpath / figure_stem).with_suffix(".png").exists():
                figure = visualize.plot_1d(plot_data, **plot_config)
                save_figure(figure, outpath / figure_stem, close=True, context=self.context)
        return plot_names

    def _get_compatible_parameter_plot_versions(
        self,
        parameter,
        parameter_version,
        plot_config,
        current_data,
        site,
        telescope,
        model_version,
        outpath,
    ):
        """Read up to three recent compatible 1D table versions for an overlay."""
        if (
            not telescope
            or not model_version
            or len(plot_config.get("tables", [])) != 1
            or len(current_data) != 1
        ):
            return []
        all_versions = self.model_reader.get_model_parameters_for_all_model_versions(
            site, telescope, collection="telescopes"
        )
        candidates = _parameter_plot_candidates(
            all_versions, parameter, parameter_version, model_version
        )
        compatible_data = []
        current_signature = _plot_signature(plot_config)
        for selected_parameter_version, selected_model_version in candidates:
            version_data = _read_parameter_plot_version(
                parameter,
                parameter_version,
                selected_parameter_version,
                selected_model_version,
                plot_config,
                current_data,
                current_signature,
                site,
                telescope,
                outpath,
                self.model_reader,
                self.context,
            )
            if version_data is not None:
                compatible_data.append((selected_parameter_version, version_data))
        return compatible_data

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

    def _get_parameter_repository_url(
        self, parameter, file_name, telescope=None, site=None, source_value=None
    ):
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
        source_name = Path(source_value or file_name).name
        return (
            f"{DEFAULT_SIMULATIONS_REPO}/simulation-model/simulation-models/-/blob/main/"
            f"simulation-models/model_parameters/{scope}/{parameter}/{source_name}"
        )

    def _convert_to_md(
        self,
        parameter,
        parameter_version,
        input_file,
        telescope=None,
        site=None,
        model_version=None,
        source_value=None,
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
                parameter, input_file, telescope, site, source_value
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
        formats = _configured_figure_formats(self.context)
        image_paths = [
            path
            for path in (_first_written_figure(outpath, name, formats) for name in plot_names)
            if path is not None
        ]
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
        resolved_layouts_by_name = {}
        if layouts:
            plot_args = {
                "model_version": self.model_version,
                "array_layout_name": None,
                "plot_all_layouts": True,
                "ignore_software_version": True,
                "array_layout_name_background": None,
                "array_layout_parameter_file": None,
                "array_layout_name_from_parameter_file": None,
                "array_layout_file": None,
                "array_element_list": None,
                "show_labels": False,
                "axes_range": None,
                "marker_scaling": 1.0,
                "grayed_out_array_elements": None,
                "highlighted_array_elements": None,
                "legend_location": "best",
                "bounds": "symmetric",
                "padding": 0.1,
                "x_lim": None,
                "y_lim": None,
                "coordinate_system": "ground",
                "site": self.site,
                "figure_format": self.context.args.get("figure_format", ["png"]),
                "figure_dpi": self.context.args.get("figure_dpi", 300),
                "figure_png_palette": self.context.args.get("figure_png_palette", False),
            }
            resolved_layouts, _background_layout = read_layouts(
                plot_args, model_reader=self.model_reader, context=self.context
            )
            resolved_layouts_by_name = {item["name"]: item for item in resolved_layouts}
        for layout in layouts:
            layout_name = layout["name"]
            file.write(f"### {layout_name}\n\n| Element |\n|---------|\n")
            for element in sorted(layout["elements"]):
                file.write(f"| [{element}]({element}.md) |\n")
            resolved_layout = resolved_layouts_by_name.get(layout_name)
            if resolved_layout is None:
                logger.warning(
                    "No resolved array layout for %s; its figure is not linked.", layout_name
                )
                continue
            figure_name = array_layout_figure_name(resolved_layout, plot_args)
            formats = _configured_figure_formats(self.context)
            figure_path = _first_written_figure(
                self.context.paths.get_output_directory("_images"), figure_name, formats
            )
            figure_format = figure_path.suffix[1:] if figure_path else formats[0]
            file.write(f"\n![{layout_name} Layout](/_images/{figure_name}.{figure_format})\n\n\n")
