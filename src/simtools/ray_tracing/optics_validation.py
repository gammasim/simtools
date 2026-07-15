"""Validating telescope optical model parameters via ray tracing."""

import logging
from collections import OrderedDict
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.table import Table

import simtools.utils.general as gen
from simtools.model.model_utils import initialize_simulation_models
from simtools.ray_tracing.ray_tracing import RayTracing
from simtools.visualization import plot_ray_tracing_psf, visualize

logger = logging.getLogger(__name__)


def _find_psf_data_column(table, column_name, fallback_index):
    """Find a PSF data column by name, falling back to legacy column position."""
    matched_column = next((col for col in table.colnames if column_name in col.lower()), None)
    if matched_column is not None:
        return matched_column
    try:
        return table.colnames[fallback_index]
    except IndexError as exc:
        msg = f"Could not find required PSF data column '{column_name}' in {table.colnames}."
        raise ValueError(msg) from exc


def _radius_values_in_cm(radius_column):
    """Return radius values in centimeters, using table units when available."""
    if getattr(radius_column, "unit", None) is not None:
        return radius_column.to(u.cm).value
    return np.asarray(radius_column, dtype=float) * 0.1


def load_data(datafile):
    """
    Load the data file with the measured PSF vs radius [cm].

    Parameters
    ----------
    datafile : str or Path
        Path to the file containing the measured cumulative PSF.
        Expected columns: radial distance in mm, differential intensity, integral intensity.

    Returns
    -------
    numpy.ndarray
        Structured array with fields 'Radius [cm]' and 'Relative intensity'.
    """
    radius_cm = "Radius [cm]"
    relative_intensity = "Relative intensity"

    table = Table.read(datafile, format="ascii")
    radius_column = _find_psf_data_column(table, "radius", 0)
    integral_psf_column = _find_psf_data_column(table, "integral", 2)

    d_type = {"names": (radius_cm, relative_intensity), "formats": ("f8", "f8")}
    data = np.zeros(len(table), dtype=d_type)
    data[radius_cm] = _radius_values_in_cm(table[radius_column])
    data[relative_intensity] = np.asarray(table[integral_psf_column], dtype=float)
    data[relative_intensity] /= np.max(np.abs(data[relative_intensity]))
    return data


def validate_cumulative_psf(app_context):
    """
    Simulate PSF measurements and compare cumulative PSF with measured data if provided.

    Parameters
    ----------
    app_context : object
        Application context with ``args`` and ``io_handler`` attributes.

    Raises
    ------
    ValueError
        If no radius data is available to compute the cumulative PSF.
    """
    args_dict = app_context.args
    io_handler = app_context.io_handler
    label = args_dict.get("label") or Path("validate_cumulative_psf").stem

    tel_model, site_model, _ = initialize_simulation_models(
        label=label,
        site=args_dict["site"],
        telescope_name=args_dict["telescope"],
        model_version=args_dict["model_version"],
    )

    ray = RayTracing(
        telescope_model=tel_model,
        site_model=site_model,
        label=label,
        zenith_angle=args_dict["zenith_angle"],
        source_distance=args_dict["source_distance"],
        off_axis_angle=[0.0] * u.deg,
    )

    ray.simulate(test=args_dict["test"], force=False)
    ray.analyze(force=False)

    image = ray.images()[0]

    logger.info(f"d80 in cm = {image.get_psf()}")

    data_to_plot = OrderedDict()
    radius = None
    if args_dict.get("data", None):
        data_file = gen.find_file(args_dict["data"], args_dict["model_path"])
        data_to_plot["measured"] = load_data(data_file)
        radius = data_to_plot["measured"]["Radius [cm]"]

    if radius is not None:
        data_to_plot[r"sim$\_$telarray"] = image.get_cumulative_data(radius * u.cm)
    else:
        raise ValueError("Radius data is not available. Cannot compute cumulative PSF.")

    fig = visualize.plot_1d(data_to_plot)
    fig.gca().set_ylim(0, 1.05)

    plot_file_name = label + "_" + tel_model.name + "_cumulative_PSF"
    plot_file = io_handler.get_output_file(plot_file_name)
    visualize.save_figure(fig, plot_file, close=True)

    data_to_plot = image.get_image_data()
    fig, _ = plot_ray_tracing_psf.create_psf_image_figure(
        data_to_plot,
        containment_radius_cm=image.get_psf(0.8) / 2,
        center=(0, 0),
        bins=80,
        cmap="gist_heat_r",
        psf_kwargs={"color": "k", "fill": False, "lw": 2, "ls": "--"},
    )

    plot_file_name = label + "_" + tel_model.name + "_image"
    plot_file = io_handler.get_output_file(plot_file_name)
    visualize.save_figure(fig, plot_file, close=True)


def validate_optics(app_context):
    """
    Build telescope model, run ray tracing, plot PSF/area/focal-length results.

    Parameters
    ----------
    app_context : object
        Application context with ``args`` and ``io_handler`` attributes.
    """
    args_dict = app_context.args
    io_handler = app_context.io_handler

    label = args_dict.get("label") or Path("validate_optics").stem

    tel_model, site_model, _ = initialize_simulation_models(
        label=label,
        site=args_dict["site"],
        telescope_name=args_dict["telescope"],
        model_version=args_dict["model_version"],
    )

    logger.info(
        f"\nValidating telescope optics with ray tracing simulations for {tel_model.name}\n"
    )

    offset_directions = None
    if args_dict.get("offset_directions"):
        offset_directions = [d.strip().upper() for d in args_dict["offset_directions"].split(",")]

    max_offset = args_dict["max_offset"].to_value(u.deg)
    offset_steps = int(max_offset / args_dict["offset_step"].to_value(u.deg)) + 1

    ray = RayTracing(
        telescope_model=tel_model,
        site_model=site_model,
        label=args_dict.get("label") or Path("validate_optics").stem,
        zenith_angle=args_dict["zenith_angle"],
        source_distance=args_dict["source_distance"],
        off_axis_angle=np.linspace(0, max_offset, offset_steps) * u.deg,
        offset_file=args_dict.get("offset_file"),
        offset_directions=offset_directions,
    )
    ray.simulate(test=args_dict["test"], force=False)
    ray.analyze(force=True)

    for key in ["psf_deg", "psf_cm", "eff_area", "eff_flen"]:
        fig = ray.plot(key, marker="o", linestyle="none", color="k")
        plot_file_name = "_".join((label, tel_model.name, key))
        plot_file = io_handler.get_output_file(plot_file_name)
        visualize.save_figure(fig, plot_file, close=True)

    if args_dict["plot_images"]:
        plot_file_name = "_".join((label, tel_model.name, "images.pdf"))
        plot_file = io_handler.get_output_file(plot_file_name)

        logger.info(f"Plotting images into {plot_file}")

        images_dict = ray.psf_images
        max_x_extent = 0.0
        max_y_extent = 0.0

        for image in images_dict.values():
            data = image.get_image_data(centralized=True)
            if len(data) > 0:
                x_extent = np.max(np.abs(data["X"]))
                y_extent = np.max(np.abs(data["Y"]))
                max_x_extent = max(max_x_extent, x_extent)
                max_y_extent = max(max_y_extent, y_extent)

        max_extent = max(max_x_extent, max_y_extent)
        max_extent_rounded = np.ceil(max_extent * 2) / 2

        logger.info(f"Setting consistent image axes: x,y range = +-{max_extent_rounded} cm")

        figures = []
        for (off_x, off_y), image in images_dict.items():
            psf_cm = image.get_psf(fraction=0.8, unit="cm")
            figures.append(
                plot_ray_tracing_psf.create_annotated_psf_image_figure(
                    image.get_image_data(centralized=True),
                    off_x=off_x,
                    off_y=off_y,
                    psf_cm=psf_cm,
                    image_range=[
                        [-max_extent_rounded, max_extent_rounded],
                        [-max_extent_rounded, max_extent_rounded],
                    ],
                    bins=150,
                    cmap="gist_heat_r",
                    psf_kwargs={"color": "k", "fill": False, "lw": 2, "ls": "--"},
                )
            )
        visualize.save_figures_to_single_document(figures, plot_file, close=True)
