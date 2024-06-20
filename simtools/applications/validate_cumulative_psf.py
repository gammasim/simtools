#!/usr/bin/python3

"""
    Summary
    -------
    This application simulates the cumulative PSF and compare with data (if available).

    The telescope zenith angle and the source distance can be set by command line arguments.

    The measured cumulative PSF should be provided by using the command line argument data. \
    A file name is expected, in which the file should contains 3 columns: radial distance in mm, \
    differential value of photon intensity and its integral value.

    The MC model can be changed by providing a yaml file with the new parameter values using \
    the argument pars (see example below).

    Examples of the plots generated by this applications are shown below. On the left, \
    the cumulative PSF and on the right, the simulated PSF image.

    .. _validate_cumulative_psf_plot:
    .. image::  images/validate_cumulative_psf_North-LST-1_cumulativePSF.png
      :width: 49 %
    .. image::  images/validate_cumulative_psf_North-LST-1_image.png
      :width: 49 %

    Command line arguments
    ----------------------
    site (str, required)
        North or South.
    telescope (str, required)
        Telescope model name (e.g. LST-1, SST-D, ...).
    model_version (str, optional)
        Model version.
    src_distance (float, optional)
        Source distance in km.
    zenith (float, optional)
        Zenith angle in deg.
    data (str, optional)
        Name of the data file with the measured cumulative PSF.
    pars (str, optional)
        Yaml file with the new model parameters to replace the default ones.
    test (activation mode, optional)
        If activated, application will be faster by simulating fewer photons.
    verbosity (str, optional)
        Log level to print.

    Example
    -------
    LST-1 Prod5

    Runtime < 1 min.

    Get an example dataset from the DB:

    .. code-block:: console

        simtools-get-file-from-db --file_name PSFcurve_data_v2.txt

    Run the application:

    .. code-block:: console

        simtools-compare-cumulative-psf --site North --telescope LST-1 \
            --model_version prod5 --data PSFcurve_data_v2.txt

    The output is saved in simtools-output/validate_cumulative_psf

    Expected final print-out message:

    .. code-block:: console

        d80 in cm = 3.3662565358159013

"""

import logging
from collections import OrderedDict
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import yaml

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.io_operations import io_handler
from simtools.model.telescope_model import TelescopeModel
from simtools.ray_tracing import RayTracing
from simtools.visualization import visualize


def _parse(label):
    config = configurator.Configurator(
        label=label,
        description=(
            "Calculate and plot the PSF and eff. mirror area as a function of off-axis angle "
            "of the telescope requested."
        ),
    )
    config.parser.add_argument(
        "--src_distance",
        help="Source distance in km",
        type=float,
        default=10,
    )
    config.parser.add_argument(
        "--zenith",
        help="Zenith angle in deg",
        type=float,
        default=20.0,
    )
    config.parser.add_argument(
        "--data",
        help="Data file name with the measured PSF vs radius [cm]",
        type=str,
    )
    config.parser.add_argument(
        "--mc_parameter_file",
        help="Yaml file with the model parameters to be replaced",
        type=str,
    )
    return config.initialize(db_config=True, simulation_model="telescope")


def load_data(datafile):
    """
    Load the data file with the measured PSF vs radius [cm].

    """
    radius_cm = "Radius [cm]"
    relative_intensity = "Relative intensity"

    d_type = {"names": (radius_cm, relative_intensity), "formats": ("f8", "f8")}
    data = np.loadtxt(datafile, dtype=d_type, usecols=(0, 2))
    data[radius_cm] *= 0.1
    data[relative_intensity] /= np.max(np.abs(data[relative_intensity]))
    return data


def main():
    label = Path(__file__).stem
    args_dict, db_config = _parse(label)

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    # Output directory to save files related directly to this app
    _io_handler = io_handler.IOHandler()
    output_dir = _io_handler.get_output_directory(label, sub_dir="application-plots")

    tel_model = TelescopeModel(
        site=args_dict["site"],
        telescope_name=args_dict["telescope"],
        mongo_db_config=db_config,
        model_version=args_dict["model_version"],
        label=label,
    )

    # New parameters
    if args_dict.get("pars", None):
        with open(args_dict["pars"], encoding="utf-8") as file:
            new_pars = yaml.safe_load(file)
        tel_model.change_multiple_parameters(**new_pars)

    ray = RayTracing.from_kwargs(
        telescope_model=tel_model,
        simtel_path=args_dict["simtel_path"],
        source_distance=args_dict["src_distance"] * u.km,
        zenith_angle=args_dict["zenith"] * u.deg,
        off_axis_angle=[0.0 * u.deg],
    )

    ray.simulate(test=args_dict["test"], force=False)
    ray.analyze(force=False)

    # Plotting cumulative PSF
    im = ray.images()[0]

    print(f"d80 in cm = {im.get_psf()}")

    # Plotting cumulative PSF
    # Measured cumulative PSF
    data_to_plot = OrderedDict()
    radius = None
    if args_dict.get("data", None):
        data_file = gen.find_file(args_dict["data"], args_dict["model_path"])
        data_to_plot["measured"] = load_data(data_file)
        radius = data_to_plot["measured"]["Radius [cm]"]

    # Simulated cumulative PSF
    if radius is not None:
        data_to_plot[r"sim$\_$telarray"] = im.get_cumulative_data(radius * u.cm)
    else:
        raise ValueError("Radius data is not available. Cannot compute cumulative PSF.")

    fig = visualize.plot_1d(data_to_plot)
    fig.gca().set_ylim(0, 1.05)

    plot_file_name = label + "_" + tel_model.name + "_cumulative_PSF"
    plot_file = output_dir.joinpath(plot_file_name)
    for f in ["pdf", "png"]:
        plt.savefig(str(plot_file) + "." + f, format=f, bbox_inches="tight")
    fig.clf()

    # Plotting image
    data_to_plot = im.get_image_data()
    fig = visualize.plot_hist_2d(data_to_plot, bins=80)
    circle = plt.Circle((0, 0), im.get_psf(0.8) / 2, color="k", fill=False, lw=2, ls="--")
    fig.gca().add_artist(circle)
    fig.gca().set_aspect("equal")

    plot_file_name = label + "_" + tel_model.name + "_image"
    plot_file = output_dir.joinpath(plot_file_name)
    for f in ["pdf", "png"]:
        fig.savefig(str(plot_file) + "." + f, format=f, bbox_inches="tight")
    fig.clf()


if __name__ == "__main__":
    main()
