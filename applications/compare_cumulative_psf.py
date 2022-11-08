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

    .. _compare_cumulative_psf_plot:
    .. image::  images/compare_cumulative_psf_North-LST-1_cumulativePSF.png
      :width: 49 %
    .. image::  images/compare_cumulative_psf_North-LST-1_image.png
      :width: 49 %

    Command line arguments
    ----------------------
    site (str, required)
        North or South.
    telescope (str, required)
        Telescope model name (e.g. LST-1, SST-D, ...).
    model_version (str, optional)
        Model version (default=prod4).
    src_distance (float, optional)
        Source distance in km (default=10).
    zenith (float, optional)
        Zenith angle in deg (default=20).
    data (str, optional)
        Name of the data file with the measured cumulative PSF.
    pars (str, optional)
        Yaml file with the new model parameters to replace the default ones.
    test (activation mode, optional)
        If activated, application will be faster by simulating fewer photons.
    verbosity (str, optional)
        Log level to print (default=INFO).

    Example
    -------
    LST-1 Prod5

    Runtime < 1 min.

    First, create an yml file named lst_pars.yml with the following content:

    .. code-block:: yaml

        mirror_reflection_random_angle: '0.0075,0.15,0.035'
        mirror_align_random_horizontal: '0.0040,28.,0.0,0.0'
        mirror_align_random_vertical: '0.0040,28.,0.0,0.0'

    And then run:

    .. code-block:: console

        python applications/compare_cumulative_psf.py --site North --telescope LST-1 \
            --model_version prod5 --pars lst_pars.yml --data PSFcurve_data_v2.txt

    .. todo::

        * Change default model to default (after this feature is implemented in db_handler)
"""

import logging
from collections import OrderedDict
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import yaml

import simtools.configuration as configurator
import simtools.util.general as gen
from simtools import io_handler, visualize
from simtools.model.telescope_model import TelescopeModel
from simtools.ray_tracing import RayTracing


def load_data(datafile):
    d_type = {"names": ("Radius [cm]", "Relative intensity"), "formats": ("f8", "f8")}
    # test_data_file = io.get_test_data_file('PSFcurve_data_v2.txt')
    data = np.loadtxt(datafile, dtype=d_type, usecols=(0, 2))
    data["Radius [cm]"] *= 0.1
    data["Relative intensity"] /= np.max(np.abs(data["Relative intensity"]))
    return data


def main():

    label = Path(__file__).stem
    config = configurator.Configurator(
        label=label,
        description=(
            "Calculate and plot the PSF and eff. mirror area as a function of off-axis angle "
            "of the telescope requested."
        ),
    )
    config.parser.add_argument(
        "--src_distance",
        help="Source distance in km (default=10)",
        type=float,
        default=10,
    )
    config.parser.add_argument(
        "--zenith", help="Zenith angle in deg (default=20)", type=float, default=20.0
    )
    config.parser.add_argument(
        "--data", help="Data file name with the measured PSF vs radius [cm]", type=str
    )
    config.parser.add_argument(
        "--pars", help="Yaml file with the model parameters to be replaced", type=str
    )

    args_dict, db_config = config.initialize(db_config=True, telescope_model=True)

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    # Output directory to save files related directly to this app
    _io_handler = io_handler.IOHandler()
    output_dir = _io_handler.get_output_directory(label, dir_type="application-plots")

    tel_model = TelescopeModel(
        site=args_dict["site"],
        telescope_model_name=args_dict["telescope"],
        mongo_db_config=db_config,
        model_version=args_dict["model_version"],
        label=label,
    )

    # New parameters
    if args_dict.get("pars", None):
        with open(args_dict["pars"]) as file:
            new_pars = yaml.safe_load(file)
        tel_model.change_multiple_parameters(**new_pars)

    ray = RayTracing.from_kwargs(
        telescope_model=tel_model,
        simtel_source_path=args_dict["simtelpath"],
        source_distance=args_dict["src_distance"] * u.km,
        zenith_angle=args_dict["zenith"] * u.deg,
        off_axis_angle=[0.0 * u.deg],
    )

    ray.simulate(test=args_dict["test"], force=False)
    ray.analyze(force=False)

    # Plotting cumulative PSF
    im = ray.images()[0]

    print("d80 in cm = {}".format(im.get_psf()))

    # Plotting cumulative PSF
    # Measured cumulative PSF
    data_to_plot = OrderedDict()
    if args_dict.get("data", None):
        data_file = gen.find_file(args_dict["data"], args_dict["model_path"])
        data_to_plot["measured"] = load_data(data_file)
        radius = data_to_plot["measured"]["Radius [cm]"]

    # Simulated cumulative PSF
    data_to_plot[r"sim$\_$telarray"] = im.get_cumulative_data(radius * u.cm)

    fig = visualize.plot_1D(data_to_plot)
    fig.gca().set_ylim(0, 1.05)

    plot_file_name = label + "_" + tel_model.name + "_cumulative_PSF"
    plot_file = output_dir.joinpath(plot_file_name)
    for f in ["pdf", "png"]:
        plt.savefig(str(plot_file) + "." + f, format=f, bbox_inches="tight")
    fig.clf()

    # Plotting image
    data_to_plot = im.get_image_data()
    fig = visualize.plot_hist_2D(data_to_plot, bins=80)
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
