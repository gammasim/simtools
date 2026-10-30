"""Generate figures attached to model reports."""

import hashlib
import inspect
import logging
import tempfile
from pathlib import Path

import numpy as np
from astropy.table import Table

from simtools.context import RunContext
from simtools.model import camera as camera_model
from simtools.model import mirrors as mirror_model
from simtools.model import telescope_geometry
from simtools.model_repository.asset_names import get_export_file_name
from simtools.reporting.figure_cache import figure_cache_key
from simtools.simtel import simtel_file_writer
from simtools.utils import names
from simtools.visualization import (
    camera_plot_utils,
    plot_mirrors,
    plot_pixels,
    plot_telescope_structure,
)
from simtools.visualization.matplotlib_backend import pyplot as plt
from simtools.visualization.plot_pixels import resolve_camera_components
from simtools.visualization.visualize import save_figure

_FLASHER_X_LABEL = "Time relative to peak [ns]"
_FLASHER_Y_LABEL = "Pulse-table amplitude"
_NSB_TITLE = "Atmospheric extinction for NSB altitude correction"
_NSB_X_LABEL = "Wavelength [nm]"
_NSB_Y_LABEL = "Emission altitude [km]"


def write_telescope_structure_figure(file, telescope_model, logger, context=None):
    """Generate and link the nominal telescope structure for this model version."""
    _use_exported_model_filenames(telescope_model)
    context = context or RunContext.from_environment()
    image_dir = context.paths.get_output_directory("_images")
    try:
        geometry = telescope_geometry.telescope_geometry(telescope_model)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        logger.warning(
            "Telescope structure plot unavailable for %s (%s): %s",
            telescope_model.name,
            telescope_model.model_version,
            exc,
        )
        file.write(f"**Telescope structure plot unavailable:** {exc}\n\n")
        return

    cache_key = figure_cache_key(
        renderer="plot_telescope_structure.plot_telescope_structure",
        renderer_modules={
            "plot_telescope_structure": plot_telescope_structure.__file__,
            "visualize": inspect.getsourcefile(plot_telescope_structure.save_figure),
        },
        plot_config={"name": telescope_model.name, "shares": None},
        rendered_data=geometry,
        context=context,
    )
    image_name = f"telescope_structure_{telescope_model.name}_{cache_key}"
    image_stem = image_dir / image_name
    image_path = image_stem.with_suffix(".png")

    if not image_path.is_file():
        plot_telescope_structure.plot_telescope_structure(geometry, image_stem)

    if image_path.is_file():
        file.write(f"![Telescope structure.]({image_path.as_posix()})\n\n")


def _use_exported_model_filenames(telescope_model):
    """Use qualified names of model files exported to the report's model directory."""
    parameters = getattr(telescope_model, "parameters", None)
    if not isinstance(parameters, dict):
        return
    for parameter in parameters.values():
        if not parameter.get("file") or not parameter.get("value"):
            continue
        file_name = get_export_file_name(parameter)
        if (telescope_model.config_file_directory / file_name).is_file():
            parameter["value"] = file_name


def write_flasher_pulse_shape_plots(
    file, comparison_items, collection, model_reader, site, calibration_device, context=None
):
    """Render Gauss-Exponential tables using the production pulse-table writer."""
    context = context or RunContext.from_environment()
    image_dir = Path(context.paths.get_output_directory("_images"))
    for item in comparison_items:
        _write_flasher_pulse_version(
            file,
            item,
            collection,
            model_reader,
            site,
            calibration_device,
            image_dir,
            context,
        )


def _write_flasher_pulse_version(
    file, item, collection, model_reader, site, calibration_device, image_dir, context
):
    """Write pulse plots for associated telescope designs at one selected model version."""
    model_version = _latest_model_version(item["model_version"])
    parameter_data = model_reader.get_model_parameters(
        site=site,
        array_element_name=calibration_device,
        collection=collection,
        model_version=model_version,
    )
    shape_data = parameter_data.get("flasher_pulse_shape")
    if not shape_data or shape_data.get("parameter_version") != item["parameter_version"]:
        return

    pulse = _supported_gauss_exponential(shape_data.get("value"))
    if pulse is None:
        return
    shape, width, exp_decay = pulse
    readout_models = _get_flasher_readout_models(
        model_reader, site, model_version, calibration_device
    )
    for telescope_model, fadc_bins in readout_models:
        _write_flasher_model_figure(
            file,
            image_dir,
            telescope_model,
            shape,
            width,
            exp_decay,
            fadc_bins,
            context,
        )


def _supported_gauss_exponential(value):
    """Return a supported positive-width Gauss-Exponential parameter tuple."""
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    shape, width, exp_decay = value
    if shape != "Gauss-Exponential" or width <= 0 or exp_decay <= 0:
        return None
    return shape, width, exp_decay


def _write_flasher_model_figure(
    file,
    image_dir,
    telescope_model,
    shape,
    width,
    exp_decay,
    fadc_bins,
    context,
):
    """Generate and link one telescope design's exact pulse table."""
    try:
        time, amplitude = _generate_pulse_samples(width, exp_decay, fadc_bins)
    except (TypeError, ValueError, OSError) as exc:
        logging.getLogger(__name__).info("Unable to generate flasher pulse table: %s", exc)
        return
    cache_key = figure_cache_key(
        renderer="simtel_file_writer.write_light_pulse_table_gauss_exp_conv",
        renderer_modules={
            "model_figures": __file__,
            "simtel_file_writer": simtel_file_writer.__file__,
        },
        plot_config={
            "shape": shape,
            "telescope_model": telescope_model,
            "title": f"Flasher pulse table for {telescope_model}",
            "x_label": _FLASHER_X_LABEL,
            "y_label": _FLASHER_Y_LABEL,
        },
        rendered_data={"time_ns": time, "amplitude": amplitude},
        context=context,
    )
    image_name = f"flasher_pulse_shape_{telescope_model}_{cache_key}"
    image_path = (Path(image_dir) / image_name).with_suffix(".png")
    if not image_path.is_file():
        _save_flasher_figure(image_path, telescope_model, time, amplitude)

    if image_path.is_file():
        file.write(
            "The figure shows the pulse table written by the current sim_telarray "
            "configuration generator for the selected model values.\n\n"
            f"![Flasher pulse table for {shape} on {telescope_model}.]"
            f"({image_path.as_posix()})\n\n"
        )


def _latest_model_version(model_versions):
    """Return the highest version from a comma-separated model-version cell."""
    versions = model_versions.split(", ")
    return max(versions, key=lambda version: tuple(map(int, version.split("."))))


def _get_flasher_readout_models(model_reader, site, model_version, calibration_device):
    """Find telescope designs that assign this calibration device and readout window."""
    readout_models = []
    production = model_reader.read_production_table("telescopes", model_version)
    telescope_models = production["parameters"]
    for telescope_model in telescope_models:
        if not names.is_design_type(telescope_model):
            continue
        parameters = model_reader.get_model_parameters(
            site=site,
            array_element_name=telescope_model,
            collection="telescopes",
            model_version=model_version,
            parameter_names=("calibration_devices", "fadc_sum_bins"),
        )
        devices = parameters.get("calibration_devices", {}).get("value") or {}
        fadc_bins = parameters.get("fadc_sum_bins", {}).get("value")
        if calibration_device in devices.values() and fadc_bins is not None:
            readout_models.append((telescope_model, fadc_bins))
    return readout_models


def _generate_pulse_samples(width, exp_decay, fadc_bins):
    """Return the time and amplitude samples written for sim_telarray."""
    with tempfile.TemporaryDirectory() as temporary_directory:
        table_path = Path(temporary_directory) / "pulse.dat"
        simtel_file_writer.write_light_pulse_table_gauss_exp_conv(
            file_path=table_path,
            width_ns=width,
            exp_decay_ns=exp_decay,
            fadc_sum_bins=fadc_bins,
            time_margin_ns=5.0,
        )
        return np.loadtxt(table_path, unpack=True)


def _save_flasher_figure(image_path, telescope_model, time, amplitude):
    """Save the table samples as a PNG."""
    figure, axes = plt.subplots(figsize=(8, 4.5))
    axes.plot(time, amplitude, label="Gauss-Exponential")
    axes.set_title(f"Flasher pulse table for {telescope_model}")
    axes.set_xlabel(_FLASHER_X_LABEL)
    axes.set_ylabel(_FLASHER_Y_LABEL)
    axes.grid(True, alpha=0.3)
    axes.legend()
    save_figure(figure, image_path.with_suffix(""), figure_format=["png"], close=True)


def plot_nsb_altitude_correction(input_file, output_path, context=None):
    """Plot the spectral extinction grid used to correct the NSB by altitude."""
    table = Table.read(input_file, format="ascii.ecsv")
    wavelength = np.unique(np.asarray(table["wavelength"], dtype=float))
    altitude = np.unique(np.asarray(table["altitude"], dtype=float))
    extinction = np.full((len(altitude), len(wavelength)), np.nan)
    wavelength_indices = {value: index for index, value in enumerate(wavelength)}
    altitude_indices = {value: index for index, value in enumerate(altitude)}
    for row in table:
        wavelength_index = wavelength_indices[float(row["wavelength"])]
        altitude_index = altitude_indices[float(row["altitude"])]
        extinction[altitude_index, wavelength_index] = float(row["extinction"])

    cache_key = figure_cache_key(
        renderer="plot_nsb_altitude_correction",
        renderer_modules={"model_figures": __file__},
        plot_config={
            "title": _NSB_TITLE,
            "xtitle": _NSB_X_LABEL,
            "ytitle": _NSB_Y_LABEL,
            "colorbar": "Extinction",
        },
        rendered_data=_file_digest(input_file),
        context=context,
    )
    image_name = f"{Path(input_file).stem.replace('.', '-')}_{cache_key}"
    image_path = (Path(output_path) / image_name).with_suffix(".png")
    if not image_path.is_file():
        figure, axes = plt.subplots(figsize=(9, 5.5))
        mesh = axes.pcolormesh(
            wavelength,
            altitude,
            np.ma.masked_invalid(extinction),
            shading="auto",
            cmap="viridis",
        )
        axes.set_title(_NSB_TITLE)
        axes.set_xlabel(_NSB_X_LABEL)
        axes.set_ylabel(_NSB_Y_LABEL)
        figure.colorbar(mesh, ax=axes, label="Extinction")
        save_figure(figure, image_path.with_suffix(""), figure_format=["png"], close=True)
    return [image_name] if image_path.is_file() else []


def plot_camera_config(
    parameter_version,
    output_path,
    telescope,
    site,
    model_version,
    model_reader,
    context=None,
):
    """Plot the camera layout file using the selected model version."""
    if not parameter_version:
        return []
    parameters = model_reader.get_model_parameters(
        site=site,
        array_element_name=telescope,
        collection="telescopes",
        model_version=model_version,
    )
    layout_parameter = model_reader.get_model_parameter(
        parameter="camera_pixel_layout",
        site=site,
        array_element_name=telescope,
        parameter_version=parameter_version,
        model_version=model_version,
    )
    parameters["camera_pixel_layout"] = layout_parameter["camera_pixel_layout"]
    camera_components = resolve_camera_components(model_reader, parameters)
    cache_key = figure_cache_key(
        renderer="plot_pixels.camera_layout_from_configuration",
        renderer_modules={
            "plot_pixels": plot_pixels.__file__,
            "camera": camera_model.__file__,
            "camera_plot_utils": camera_plot_utils.__file__,
            "visualize": inspect.getsourcefile(plot_pixels.visualize.save_figure),
        },
        plot_config={"telescope": telescope, "focal_length": 1.0},
        rendered_data=camera_components,
        context=context,
    )
    plot_name = f"camera_pixel_layout_{telescope}_{cache_key}"
    plot_path = (Path(output_path) / plot_name).with_suffix(".png")
    if not plot_path.is_file():
        camera = camera_model.Camera.from_configuration(telescope, camera_components, 1.0)
        plot_pixels.plot(
            output_file=Path(output_path) / plot_name,
            camera=camera,
            raw_pixels=plot_pixels.raw_pixels_from_configuration(camera_components),
        )
    return [plot_name]


def plot_mirror_config(
    parameter,
    parameter_version,
    input_file,
    output_path,
    telescope,
    site,
    model_version,
    context=None,
):
    """Plot the mirror configuration file using the selected model version."""
    if not parameter_version:
        return []
    cache_key = figure_cache_key(
        renderer="plot_mirrors.plot",
        renderer_modules={
            "plot_mirrors": plot_mirrors.__file__,
            "mirrors": mirror_model.__file__,
            "visualize": inspect.getsourcefile(plot_mirrors.visualize.save_figure),
        },
        plot_config={"parameter": parameter, "telescope": telescope},
        rendered_data=_file_digest(input_file),
        context=context,
    )
    plot_name = f"{parameter}_{telescope}_{cache_key}"
    plot_path = (Path(output_path) / plot_name).with_suffix(".png")
    if not plot_path.is_file():
        plot_mirrors.plot(
            config={
                "parameter": parameter,
                "telescope": telescope,
                "parameter_version": parameter_version,
                "site": site,
                "model_version": model_version,
            },
            output_file=Path(output_path) / plot_name,
            data_file_path=input_file,
            mirrors=(mirror_model.Mirrors(input_file) if parameter == "mirror_list" else None),
        )
    return [plot_name]


def _file_digest(input_file):
    """Return a SHA-256 digest of one exact renderer input file."""
    return hashlib.sha256(Path(input_file).read_bytes()).hexdigest()
