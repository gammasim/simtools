#!/usr/bin/python3

r"""
    Derives the mirror alignment parameters using cumulative PSF measurement.

    This includes parameters mirror_reflection_random_angle, \
    mirror_align_random_horizontal and mirror_align_random_vertical.

    The telescope zenith angle and the source distance can be set by command line arguments.

    The measured cumulative PSF should be provided by using the command line argument data. \
    A file name is expected, in which the file should contain 3 columns: radial distance in mm, \
    differential value of photon intensity and its integral value.

    The derivation is performed through a random search. A number of random combination of the \
    parameters are tested and the best ones are selected based on the minimum value of \
    the Root Mean Squared Deviation between data and simulations. The range in which the \
    parameter are drawn uniformly are defined based on the previous value on the telescope model.

    The assumption are:

    a) mirror_align_random_horizontal and mirror_align_random_vertical are the same.

    b) mirror_align_random_horizontal/vertical have no dependence on the zenith angle.

    One example of the plot generated by this applications are shown below.

    .. _derive_psf_parameters_plot:
    .. image::  images/derive_psf_parameters.png
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
    plot_all (activation mode, optional)
        If activated, plots will be generated for all values tested during tuning.
    fixed (activation mode, optional)
        Keep the first entry of mirror_reflection_random_angle fixed.
    test (activation mode, optional)
        If activated, application will be faster by simulating fewer photons.
    verbosity (str, optional)
        Log level to print.

    Example
    -------
    LST-1 Prod5

    Runtime < 3 min.

    Get PSF data from the DB:

    .. code-block:: console

        simtools-db-get-file-from-db --file_name PSFcurve_data_v2.txt

    Run the application:

    .. code-block:: console

        simtools-derive-psf-parameters --site North --telescope LSTN-01 \\
            --model_version prod6 --data PSFcurve_data_v2.txt --plot_all --test

    The output is saved in simtools-output/derive_psf_parameters.

    Expected final print-out message:

    .. code-block:: console

        Best parameters:
        mirror_reflection_random_angle = [0.006, 0.133, 0.005]
        mirror_align_random_horizontal = [0.005, 28.0, 0.0, 0.0]
        mirror_align_random_vertical = [0.005, 28.0, 0.0, 0.0]

"""
import logging
from collections import OrderedDict

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.io_operations import io_handler
from simtools.model.telescope_model import TelescopeModel
from simtools.ray_tracing import RayTracing
from simtools.visualization import visualize


def load_data(data_file):
    """
    Load data from a text file containing cumulative PSF measurements.

    Parameters
    ----------
    data_file : str
        Name of the data file with the measured cumulative PSF.

    Returns
    -------
    numpy.ndarray
        Loaded and processed data from the file.
    """
    radius_cm = "Radius [cm]"
    cumulative_psf = "Cumulative PSF"
    d_type = {"names": (radius_cm, cumulative_psf), "formats": ("f8", "f8")}
    data = np.loadtxt(data_file, dtype=d_type, usecols=(0, 2))
    data[radius_cm] *= 0.1
    data[cumulative_psf] /= np.max(np.abs(data[cumulative_psf]))
    return data


def _parse():
    config = configurator.Configurator(
        description=(
            "Derive mirror_reflection_random_angle, mirror_align_random_horizontal "
            "and mirror_align_random_vertical using cumulative PSF measurement."
        )
    )
    config.parser.add_argument(
        "--src_distance",
        help="Source distance in km",
        type=float,
        default=10,
    )
    config.parser.add_argument("--zenith", help="Zenith angle in deg", type=float, default=20)
    config.parser.add_argument(
        "--data", help="Data file name with the measured PSF vs radius [cm]", type=str
    )
    config.parser.add_argument(
        "--plot_all",
        help=(
            "On: plot cumulative PSF for all tested combinations, "
            "Off: plot it only for the best set of values"
        ),
        action="store_true",
    )
    config.parser.add_argument(
        "--fixed",
        help=("Keep the first entry of mirror_reflection_random_angle fixed."),
        action="store_true",
    )
    return config.initialize(db_config=True, simulation_model="telescope")


def add_parameters(
    all_parameters,
    mirror_reflection,
    mirror_align,
    mirror_reflection_fraction=0.15,
    mirror_reflection_2=0.035,
):
    """
    Transforms and add parameters to the all_parameters list.

    Parameters
    ----------
    mirror_reflection : float
        The random angle of mirror reflection.

    mirror_align : float
        The random angle for mirror alignment (both horizontal and vertical).

    mirror_reflection_fraction : float, optional
        The fraction of the mirror reflection. Default is 0.15.

    mirror_reflection_2 : float, optional
        A secondary random angle for mirror reflection. Default is 0.035.

    Returns
    -------
    None
        Updates the all_parameters list in place.
    """
    # If we want to start from values different than the ones currently in the model:
    # align = 0.0046
    # pars_to_change = {
    #     'mirror_reflection_random_angle': '0.0075 0.125 0.0037',
    #     'mirror_align_random_horizontal': f'{align} 28 0 0',
    #     'mirror_align_random_vertical': f'{align} 28 0 0',
    # }
    # tel_model.change_multiple_parameters(**pars_to_change)
    pars = {
        "mirror_reflection_random_angle": [
            mirror_reflection,
            mirror_reflection_fraction,
            mirror_reflection_2,
        ],
        "mirror_align_random_horizontal": [mirror_align, 28.0, 0.0, 0.0],
        "mirror_align_random_vertical": [mirror_align, 28.0, 0.0, 0.0],
    }
    all_parameters.append(pars)


def get_previous_values(tel_model, logger):
    """
    Retrieve previous parameter values from the telescope model.

    Parameters
    ----------
    tel_model : TelescopeModel
        Telescope model object.
    logger : logging.Logger
        Logger object for logging messages.

    Returns
    -------
    tuple
        Tuple containing the previous values of mirror_reflection_random_angle (first entry),
        mirror_reflection_fraction, second entry), mirror_reflection_random_angle (third entry),
        and mirror_align_random_horizontal/vertical.
    """
    split_par = tel_model.get_parameter_value("mirror_reflection_random_angle")
    mrra_0, mfr_0, mrra2_0 = split_par[0], split_par[1], split_par[2]
    mar_0 = tel_model.get_parameter_value("mirror_align_random_horizontal")[0]

    logger.debug(
        "Previous parameter values:\n"
        f"MRRA = {str(mrra_0)}\n"
        f"MRF = {str(mfr_0)}\n"
        f"MRRA2 = {str(mrra2_0)}\n"
        f"MAR = {str(mar_0)}\n"
    )

    return mrra_0, mfr_0, mrra2_0, mar_0


def generate_random_parameters(
    all_parameters, n_runs, args_dict, mrra_0, mfr_0, mrra2_0, mar_0, logger
):
    """
    Generate random parameters for tuning.

    Parameters
    ----------
    all_parameters : list
        List to store all parameter sets.
    n_runs : int
        Number of random parameter combinations to test.
    args_dict : dict
        Dictionary containing parsed command-line arguments.
    mrra_0 : float
        Initial value of mirror_reflection_random_angle.
    mfr_0 : float
        Initial value of mirror_reflection_fraction.
    mrra2_0 : float
        Initial value of the second mirror_reflection_random_angle.
    mar_0 : float
        Initial value of mirror_align_random_horizontal/vertical.
    logger : logging.Logger
        Logger object for logging messages.
    """
    # Range around the previous values are hardcoded
    # Number of runs is hardcoded
    if args_dict["fixed"]:
        logger.debug("fixed=True - First entry of mirror_reflection_random_angle is kept fixed.")

    for _ in range(n_runs):
        mrra_range = 0.004 if not args_dict["fixed"] else 0
        mrf_range = 0.1
        mrra2_range = 0.03
        mar_range = 0.005
        rng = np.random.default_rng()
        mrra = rng.uniform(max(mrra_0 - mrra_range, 0), mrra_0 + mrra_range)
        mrf = rng.uniform(max(mfr_0 - mrf_range, 0), mfr_0 + mrf_range)
        mrra2 = rng.uniform(max(mrra2_0 - mrra2_range, 0), mrra2_0 + mrra2_range)
        mar = rng.uniform(max(mar_0 - mar_range, 0), mar_0 + mar_range)
        add_parameters(all_parameters, mrra, mar, mrf, mrra2)


def load_and_process_data(args_dict):
    """
    Load and process data if specified in the command-line arguments.

    Returns
    -------
    - data_to_plot: OrderedDict containing loaded and processed data.
    - radius: Radius data from loaded data (if available).
    """
    data_to_plot = OrderedDict()
    radius = None

    if args_dict["data"] is not None:
        data_file = gen.find_file(args_dict["data"], args_dict["model_path"])
        data_to_plot["measured"] = load_data(data_file)
        radius = data_to_plot["measured"]["Radius [cm]"]

    return data_to_plot, radius


def calculate_rmsd(data, sim):
    """
    Calculates the Root Mean Squared Deviation to be used as metric to find the best parameters.
    """
    return np.sqrt(np.mean((data - sim) ** 2))


def run_pars(tel_model, args_dict, pars, data_to_plot, radius, pdf_pages):
    """
    Runs the tuning for one set of parameters, add a plot to the pdfPages and return RMSD and D80.

    Plotting is optional (if plot=True).
    """
    cumulative_psf = "Cumulative PSF"

    if pars is not None:
        tel_model.change_multiple_parameters(**pars)
    else:
        raise ValueError("No best parameters found")

    ray = RayTracing.from_kwargs(
        telescope_model=tel_model,
        simtel_source_path=args_dict["simtel_path"],
        source_distance=args_dict["src_distance"] * u.km,
        zenith_angle=args_dict["zenith"] * u.deg,
        off_axis_angle=[0.0 * u.deg],
    )

    ray.simulate(test=args_dict["test"], force=True)
    ray.analyze(force=True, use_rx=False)

    # Plotting cumulative PSF
    im = ray.images()[0]
    d80 = im.get_psf()

    if radius is not None:
        # Simulated cumulative PSF
        data_to_plot["simulated"] = im.get_cumulative_data(radius * u.cm)
    else:
        raise ValueError("Radius data is not available.")

    rmsd = calculate_rmsd(
        data_to_plot["measured"][cumulative_psf], data_to_plot["simulated"][cumulative_psf]
    )

    if args_dict["plot_all"]:
        fig = visualize.plot_1d(
            data_to_plot,
            plot_difference=True,
            no_markers=True,
        )
        ax = fig.get_axes()[0]
        ax.set_ylim(0, 1.05)
        ax.set_title(
            f"refl_rnd={pars['mirror_reflection_random_angle']}, "
            f"align_rnd={pars['mirror_align_random_vertical']}"
        )

        ax.text(
            0.8,
            0.3,
            f"D80 = {d80:.3f} cm\nRMSD = {rmsd:.4f}",
            verticalalignment="center",
            horizontalalignment="center",
            transform=ax.transAxes,
        )
        plt.tight_layout()
        pdf_pages.savefig(fig)
        plt.clf()

    return d80, rmsd


def find_best_parameters(all_parameters, tel_model, args_dict, data_to_plot, radius, pdf_pages):
    """
    Find the best parameters from all parameter sets.

    Returns
    -------
    - Tuple of best parameters and their D80 value.
    """
    min_rmsd = 100
    best_pars = None

    for pars in all_parameters:
        _, rmsd = run_pars(tel_model, args_dict, pars, data_to_plot, radius, pdf_pages)
        if rmsd < min_rmsd:
            min_rmsd = rmsd
            best_pars = pars

    return best_pars, min_rmsd


def main():
    args_dict, db_config = _parse()

    label = "tune_psf"
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

    all_parameters = []
    mrra_0, mfr_0, mrra2_0, mar_0 = get_previous_values(tel_model, logger)

    n_runs = 5 if args_dict["test"] else 50
    generate_random_parameters(
        all_parameters, n_runs, args_dict, mrra_0, mfr_0, mrra2_0, mar_0, logger
    )

    data_to_plot, radius = load_and_process_data(args_dict)

    # Preparing figure name
    plot_file_name = "_".join((label, tel_model.name + ".pdf"))
    plot_file = output_dir.joinpath(plot_file_name)
    pdf_pages = PdfPages(plot_file)

    best_pars, _ = find_best_parameters(
        all_parameters, tel_model, args_dict, data_to_plot, radius, pdf_pages
    )

    # Rerunning and plotting the best pars
    run_pars(tel_model, args_dict, best_pars, data_to_plot, radius, pdf_pages)
    plt.close()
    pdf_pages.close()

    # Printing the results
    print("Best parameters:")
    for par, value in best_pars.items():
        print(f"{par} = {value}")


if __name__ == "__main__":
    main()
