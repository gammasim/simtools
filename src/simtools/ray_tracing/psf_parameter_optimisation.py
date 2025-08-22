"""
PSF parameter optimisation and fitting routines for mirror alignment and reflection parameters.

This module provides functions for loading PSF data, generating random parameter sets,
running PSF simulations, calculating RMSD, and finding the best-fit parameters for a given
telescope model.

"""

import logging
from collections import OrderedDict

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from simtools.data_model import model_data_writer as writer
from simtools.model import model_utils
from simtools.ray_tracing.ray_tracing import RayTracing
from simtools.utils import general as gen
from simtools.visualization import visualize

logger = logging.getLogger(__name__)

# Constants
RADIUS_CM = "Radius [cm]"
CUMULATIVE_PSF = "Cumulative PSF"


def load_psf_data(data_file):
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
    d_type = {"names": (RADIUS_CM, CUMULATIVE_PSF), "formats": ("f8", "f8")}
    data = np.loadtxt(data_file, dtype=d_type, usecols=(0, 2))
    data[RADIUS_CM] *= 0.1
    data[CUMULATIVE_PSF] /= np.max(np.abs(data[CUMULATIVE_PSF]))
    return data


def calculate_rmsd(data, sim):
    """Calculate Root Mean Squared Deviation to be used as metric to find the best parameters."""
    return np.sqrt(np.mean((data - sim) ** 2))


def add_parameters(
    all_parameters,
    mirror_reflection,
    mirror_align,
    mirror_reflection_fraction=0.15,
    mirror_reflection_2=0.035,
):
    """
    Transform and add parameters to the all_parameters list.

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


def get_previous_values(tel_model):
    """
    Retrieve previous parameter values from the telescope model.

    Parameters
    ----------
    tel_model : TelescopeModel
        Telescope model object.

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
        f"MRRA = {mrra_0!s}\n"
        f"MRF = {mfr_0!s}\n"
        f"MRRA2 = {mrra2_0!s}\n"
        f"MAR = {mar_0!s}\n"
    )
    return mrra_0, mfr_0, mrra2_0, mar_0


def generate_random_parameters(
    all_parameters, n_runs, args_dict, mrra_0, mfr_0, mrra2_0, mar_0, tel_model
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
    tel_model : TelescopeModel
        Telescope model object to check if it's a dual mirror telescope.
    """
    # Range around the previous values are hardcoded
    # Number of runs is hardcoded
    if args_dict["fixed"]:
        logger.debug("fixed=True - First entry of mirror_reflection_random_angle is kept fixed.")

    # Check if telescope is dual mirror and set mar to 0 if so
    is_dual_mirror = model_utils.is_two_mirror_telescope(tel_model.name)
    if is_dual_mirror:
        mar_fixed_value = 0.0
    else:
        mar_fixed_value = None

    for _ in range(n_runs):
        mrra_range = 0.004 if not args_dict["fixed"] else 0
        mrf_range = 0.1
        mrra2_range = 0.03
        mar_range = 0.005
        rng = np.random.default_rng()
        mrra = rng.uniform(max(mrra_0 - mrra_range, 0), mrra_0 + mrra_range)
        mrf = rng.uniform(max(mfr_0 - mrf_range, 0), mfr_0 + mrf_range)
        mrra2 = rng.uniform(max(mrra2_0 - mrra2_range, 0), mrra2_0 + mrra2_range)

        # Set mar to 0 for dual mirror telescopes, otherwise use random value
        if mar_fixed_value is not None:
            mar = mar_fixed_value
        else:
            mar = rng.uniform(max(mar_0 - mar_range, 0), mar_0 + mar_range)

        add_parameters(all_parameters, mrra, mar, mrf, mrra2)


def _run_ray_tracing_simulation(tel_model, site_model, args_dict, pars):
    """
    Run the core ray tracing simulation for a given set of parameters.

    Parameters
    ----------
    tel_model : TelescopeModel
        Telescope model object.
    site_model : SiteModel
        Site model object.
    args_dict : dict
        Dictionary containing parsed command-line arguments.
    pars : dict
        Parameter set dictionary.

    Returns
    -------
    tuple
        (d80, simulated_data) - D80 value and simulated data from ray tracing.
    """
    if pars is not None:
        tel_model.change_multiple_parameters(**pars)
    else:
        raise ValueError("No best parameters found")

    ray = RayTracing(
        telescope_model=tel_model,
        site_model=site_model,
        simtel_path=args_dict["simtel_path"],
        zenith_angle=args_dict["zenith"] * u.deg,
        source_distance=args_dict["src_distance"] * u.km,
        off_axis_angle=[0.0] * u.deg,
    )
    ray.simulate(test=args_dict["test"], force=True)
    ray.analyze(force=True, use_rx=False)
    im = ray.images()[0]
    d80 = im.get_psf()

    return d80, im


def _create_psf_simulation_plot(data_to_plot, pars, d80, rmsd, is_best, pdf_pages):
    """
    Create a plot for PSF simulation results.

    Parameters
    ----------
    data_to_plot : dict
        Data dictionary for plotting.
    pars : dict
        Parameter set dictionary.
    d80 : float
        D80 value.
    rmsd : float
        RMSD value.
    is_best : bool
        Whether this is the best parameter set.
    pdf_pages : PdfPages
        PDF pages object for saving plots.
    """
    fig = visualize.plot_1d(
        data_to_plot,
        plot_difference=True,
        no_markers=True,
    )
    ax = fig.get_axes()[0]
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(CUMULATIVE_PSF)

    # Create title with asterisk for best parameters
    title_prefix = "* " if is_best else ""
    ax.set_title(
        f"{title_prefix}refl_rnd = "
        f"{pars['mirror_reflection_random_angle'][0]:.5f}, "
        f"{pars['mirror_reflection_random_angle'][1]:.5f}, "
        f"{pars['mirror_reflection_random_angle'][2]:.5f}\n"
        f"align_rnd = {pars['mirror_align_random_vertical'][0]:.5f}, "
        f"{pars['mirror_align_random_vertical'][1]:.5f}, "
        f"{pars['mirror_align_random_vertical'][2]:.5f}, "
        f"{pars['mirror_align_random_vertical'][3]:.5f}"
    )

    # Highlight D80 text for best parameters
    d80_color = "red" if is_best else "black"
    d80_weight = "bold" if is_best else "normal"
    d80_text = f"D80 = {d80:.5f} cm"

    ax.text(
        0.5,
        0.3,
        f"{d80_text}\nRMSD = {rmsd:.4f}",
        verticalalignment="center",
        horizontalalignment="left",
        transform=ax.transAxes,
        color=d80_color,
        weight=d80_weight,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "yellow", "alpha": 0.7}
        if is_best
        else None,
    )

    # Add footnote for best parameters
    if is_best:
        fig.text(
            0.02,
            0.02,
            "* Best parameter set (lowest RMSD)",
            fontsize=8,
            style="italic",
            color="red",
        )

    pdf_pages.savefig(fig, bbox_inches="tight")
    plt.clf()


def run_psf_simulation(
    tel_model,
    site_model,
    args_dict,
    pars,
    data_to_plot,
    radius,
    pdf_pages=None,
    is_best=False,
    return_simulated_data=False,
):
    """
    Run the simulation for one set of parameters and return D80, RMSD.

    Parameters
    ----------
    tel_model : TelescopeModel
        Telescope model object.
    site_model : SiteModel
        Site model object.
    args_dict : dict
        Dictionary containing parsed command-line arguments.
    pars : dict
        Parameter set dictionary.
    data_to_plot : dict
        Data dictionary for plotting.
    radius : array-like
        Radius data.
    pdf_pages : PdfPages, optional
        PDF pages object for plotting. If None, no plotting is done.
    is_best : bool, optional
        Whether this is the best parameter set for highlighting in plots.
    return_simulated_data : bool, optional
        If True, returns simulated data as third element in return tuple.

    Returns
    -------
    tuple
        (d80, rmsd) if return_simulated_data=False
        (d80, rmsd, simulated_data) if return_simulated_data=True
    """
    d80, im = _run_ray_tracing_simulation(tel_model, site_model, args_dict, pars)

    if radius is None:
        raise ValueError("Radius data is not available.")

    simulated_data = im.get_cumulative_data(radius * u.cm)
    rmsd = calculate_rmsd(data_to_plot["measured"][CUMULATIVE_PSF], simulated_data[CUMULATIVE_PSF])

    # Handle plotting if requested
    if pdf_pages is not None and args_dict.get("plot_all", False):
        data_to_plot["simulated"] = simulated_data
        _create_psf_simulation_plot(data_to_plot, pars, d80, rmsd, is_best, pdf_pages)
        del data_to_plot["simulated"]

    return (d80, rmsd, simulated_data) if return_simulated_data else (d80, rmsd)


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
        data_to_plot["measured"] = load_psf_data(data_file)
        radius = data_to_plot["measured"][RADIUS_CM]
    return data_to_plot, radius


def _create_plot_for_parameters(pars, rmsd, d80, simulated_data, data_to_plot, is_best, pdf_pages):
    """
    Create a single plot for a parameter set.

    Parameters
    ----------
    pars : dict
        Parameter set dictionary
    rmsd : float
        RMSD value for this parameter set
    d80 : float
        D80 value for this parameter set
    simulated_data : array
        Simulated data for plotting
    data_to_plot : dict
        Data dictionary for plotting
    is_best : bool
        Whether this is the best parameter set
    pdf_pages : PdfPages
        PDF pages object to save the plot
    """
    original_simulated = data_to_plot.get("simulated")
    data_to_plot["simulated"] = simulated_data

    fig = visualize.plot_1d(
        data_to_plot,
        plot_difference=True,
        no_markers=True,
    )
    ax = fig.get_axes()[0]
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(CUMULATIVE_PSF)

    title_prefix = "* " if is_best else ""

    ax.set_title(
        f"{title_prefix}reflection = "
        f"{pars['mirror_reflection_random_angle'][0]:.5f}, "
        f"{pars['mirror_reflection_random_angle'][1]:.5f}, "
        f"{pars['mirror_reflection_random_angle'][2]:.5f}\n"
        f"align_vertical = {pars['mirror_align_random_vertical'][0]:.5f}, "
        f"{pars['mirror_align_random_vertical'][1]:.5f}, "
        f"{pars['mirror_align_random_vertical'][2]:.5f}, "
        f"{pars['mirror_align_random_vertical'][3]:.5f}\n"
        f"align_horizonal = {pars['mirror_align_random_horizontal'][0]:.5f}, "
        f"{pars['mirror_align_random_horizontal'][1]:.5f}, "
        f"{pars['mirror_align_random_horizontal'][2]:.5f}, "
        f"{pars['mirror_align_random_horizontal'][3]:.5f}"
    )

    # Highlight D80 text for best parameters
    d80_color = "red" if is_best else "black"
    d80_weight = "bold" if is_best else "normal"

    ax.text(
        0.5,
        0.3,
        f"D80 = {d80:.5f} cm\nRMSD = {rmsd:.4f}",
        verticalalignment="center",
        horizontalalignment="left",
        transform=ax.transAxes,
        color=d80_color,
        weight=d80_weight,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "yellow", "alpha": 0.7}
        if is_best
        else None,
    )

    # Add footnote for best parameters
    if is_best:
        fig.text(
            0.02,
            0.02,
            "* Best parameter set (lowest RMSD)",
            fontsize=8,
            style="italic",
            color="red",
        )

    pdf_pages.savefig(fig, bbox_inches="tight")
    plt.clf()

    if original_simulated is not None:
        data_to_plot["simulated"] = original_simulated


def _run_all_simulations(all_parameters, tel_model, site_model, args_dict, data_to_plot, radius):
    """
    Run all simulations and collect results.

    Returns
    -------
    tuple
        (best_pars, best_d80, best_rmsd, results)
    """
    best_rmsd = float("inf")
    best_pars = None
    best_d80 = None
    results = []  # Store (pars, rmsd, d80, simulated_data)

    logger.info(f"Running {len(all_parameters)} simulations...")

    for i, pars in enumerate(all_parameters):
        try:
            logger.info(f"Running simulation {i + 1}/{len(all_parameters)}")
            d80, rmsd, simulated_data = run_psf_simulation(
                tel_model,
                site_model,
                args_dict,
                pars,
                data_to_plot,
                radius,
                return_simulated_data=True,
            )
        except (ValueError, RuntimeError) as e:
            logger.warning(f"Simulation failed for parameters {pars}: {e}")
            continue

        results.append((pars, rmsd, d80, simulated_data))
        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_pars = pars
            best_d80 = d80

    logger.info(f"Best RMSD found: {best_rmsd:.5f}")
    return best_pars, best_d80, best_rmsd, results


def _create_all_plots(results, best_pars, data_to_plot, pdf_pages):
    """
    Create plots for all parameter sets if requested.

    Parameters
    ----------
    results : list
        List of (pars, rmsd, d80, simulated_data) tuples
    best_pars : dict
        Best parameter set for highlighting
    data_to_plot : dict
        Data dictionary for plotting
    pdf_pages : PdfPages
        PDF pages object to save plots
    """
    logger.info("Creating plots for all parameter sets...")

    for i, (pars, rmsd, d80, simulated_data) in enumerate(results):
        is_best = pars == best_pars
        logger.info(f"Creating plot {i + 1}/{len(results)}{' (BEST)' if is_best else ''}")

        _create_plot_for_parameters(
            pars, rmsd, d80, simulated_data, data_to_plot, is_best, pdf_pages
        )


def find_best_parameters(
    all_parameters, tel_model, site_model, args_dict, data_to_plot, radius, pdf_pages=None
):
    """
    Find the best parameters by running simulations for all parameter sets.

    Loop over all parameter sets, run the simulation, compute RMSD,
    and return the best parameters and their RMSD.
    """
    # Run all simulations and store data
    best_pars, best_d80, _, results = _run_all_simulations(
        all_parameters, tel_model, site_model, args_dict, data_to_plot, radius
    )

    # Create all plots if requested
    if pdf_pages is not None and args_dict.get("plot_all", False) and results:
        _create_all_plots(results, best_pars, data_to_plot, pdf_pages)

    return best_pars, best_d80, results


def create_d80_vs_offaxis_plot(tel_model, site_model, args_dict, best_pars, output_dir):
    """
    Create D80 vs off-axis angle plot using the best parameters.

    Parameters
    ----------
    tel_model : TelescopeModel
        Telescope model object.
    site_model : SiteModel
        Site model object.
    args_dict : dict
        Dictionary containing parsed command-line arguments.
    best_pars : dict
        Best parameter set.
    output_dir : Path
        Output directory for saving plots.
    """
    logger.info("Creating D80 vs off-axis angle plot with best parameters...")

    # Apply best parameters to telescope model
    tel_model.change_multiple_parameters(**best_pars)

    # Create off-axis angle array
    max_offset = args_dict.get("max_offset", 4.5)
    offset_steps = args_dict.get("offset_steps", 0.1)
    off_axis_angles = np.linspace(
        0,
        max_offset,
        int(max_offset / offset_steps) + 1,
    )

    # Run ray tracing simulation for multiple off-axis angles
    ray = RayTracing(
        telescope_model=tel_model,
        site_model=site_model,
        simtel_path=args_dict["simtel_path"],
        zenith_angle=args_dict["zenith"] * u.deg,
        source_distance=args_dict["src_distance"] * u.km,
        off_axis_angle=off_axis_angles * u.deg,
    )

    logger.info(f"Running ray tracing for {len(off_axis_angles)} off-axis angles...")
    ray.simulate(test=args_dict["test"], force=True)
    ray.analyze(force=True)

    # Create D80 plots
    for key in ["d80_cm", "d80_deg"]:
        plt.figure(figsize=(10, 6), tight_layout=True)

        ray.plot(key, marker="o", linestyle="-", color="blue", linewidth=2, markersize=6)

        plt.title(
            f"PSF D80 vs Off-axis Angle - {tel_model.name}\n"
            f"Best Parameters: "
            f"refl=[{best_pars['mirror_reflection_random_angle'][0]:.4f}, "
            f"{best_pars['mirror_reflection_random_angle'][1]:.4f}, "
            f"{best_pars['mirror_reflection_random_angle'][2]:.4f}], "
            f"align={best_pars['mirror_align_random_horizontal'][0]:.4f}"
        )
        plt.xlabel("Off-axis Angle (degrees)")
        plt.ylabel("D80 (cm)" if key == "d80_cm" else "D80 (degrees)")
        plt.xticks(rotation=45)
        plt.xlim(0, max_offset)
        plt.grid(True, alpha=0.3)

        plot_file_name = f"tune_psf_{tel_model.name}_best_params_{key}.pdf"
        plot_file = output_dir.joinpath(plot_file_name)
        visualize.save_figure(plt, plot_file, log_title=f"D80 vs off-axis ({key})")

    plt.close("all")


def write_tested_parameters_to_file(results, best_pars, best_d80, output_dir):
    """
    Write all tested parameters and their metrics to a text file.

    Parameters
    ----------
    results : list
        List of (pars, rmsd, d80, simulated_data) tuples
    best_pars : dict
        Best parameter set
    best_d80 : float
        Best D80 value
    output_dir : Path
        Output directory path
    """
    param_file = output_dir.joinpath("tested_psf_parameters.txt")
    with open(param_file, "w", encoding="utf-8") as f:
        f.write("Tested parameter sets:\n")
        for i, (pars, rmsd, d80, _) in enumerate(results):
            is_best = pars == best_pars
            prefix = "*" if is_best else " "
            f.write(f"{prefix} Set {i + 1}: RMSD={rmsd:.5f}, D80={d80:.5f} cm\n")
            for par, value in pars.items():
                f.write(f"    {par} = {value}\n")
            if is_best:
                f.write("    <-- BEST\n")
        f.write("\nBest parameters:\n")
        for par, value in best_pars.items():
            f.write(f"{par} = {value}\n")
        f.write(f"Best D80: {best_d80:.5f} cm\n")
    return param_file


def _add_units_to_psf_parameters(best_pars):
    """
    Add proper astropy units to PSF parameters based on their schemas.

    Parameters
    ----------
    best_pars : dict
        Dictionary with PSF parameter names as keys and values as lists

    Returns
    -------
    dict
        Dictionary with same keys but values converted to astropy quantities with units
    """
    psf_pars_with_units = {}

    for param_name, param_values in best_pars.items():
        if param_name == "mirror_reflection_random_angle":
            # [deg, dimensionless, deg]
            psf_pars_with_units[param_name] = [
                param_values[0] * u.deg,
                param_values[1] * u.dimensionless_unscaled,
                param_values[2] * u.deg,
            ]
        elif param_name in ["mirror_align_random_horizontal", "mirror_align_random_vertical"]:
            # [deg, deg, dimensionless, dimensionless]
            psf_pars_with_units[param_name] = [
                param_values[0] * u.deg,
                param_values[1] * u.deg,
                param_values[2] * u.dimensionless_unscaled,
                param_values[3] * u.dimensionless_unscaled,
            ]
        else:
            # For any other parameters, keep as-is
            psf_pars_with_units[param_name] = param_values

    return psf_pars_with_units


def export_psf_parameters_as_json(best_pars, tel_model, parameter_version, output_dir, func_logger):
    """
    Export PSF parameters as JSON model parameter files.

    Parameters
    ----------
    best_pars : dict
        Best parameter set
    tel_model : TelescopeModel
        Telescope model object
    parameter_version : str
        Parameter version string
    output_dir : Path
        Output directory path
    func_logger : Logger
        Logger object
    """
    try:
        func_logger.info("Exporting best PSF parameters as JSON model parameter files")
        psf_pars_with_units = _add_units_to_psf_parameters(best_pars)
        parameter_output_path = output_dir / tel_model.name
        for parameter_name, parameter_value in psf_pars_with_units.items():
            writer.ModelDataWriter.dump_model_parameter(
                parameter_name=parameter_name,
                value=parameter_value,
                instrument=tel_model.name,
                parameter_version=parameter_version,
                output_file=f"{parameter_name}-{parameter_version}.json",
                output_path=parameter_output_path,
                use_plain_output_path=True,
            )
        func_logger.info(f"JSON model parameter files exported to {output_dir}")
    except ImportError as e:
        func_logger.warning(f"Could not export JSON parameters: {e}")
    except (ValueError, KeyError, OSError) as e:
        func_logger.error(f"Error exporting JSON parameters: {e}")


def run_psf_optimization_workflow(tel_model, site_model, args_dict, output_dir, func_logger):
    """
    Run the complete PSF parameter optimization workflow.

    This function consolidates the main optimization logic to make the application lighter.

    Parameters
    ----------
    tel_model : TelescopeModel
        Telescope model object
    site_model : SiteModel
        Site model object
    args_dict : dict
        Dictionary containing parsed command-line arguments
    output_dir : Path
        Output directory path
    func_logger : Logger
        Logger object

    Returns
    -------
    None
        All results are saved to files and printed to console
    """
    # Generate parameter sets
    all_parameters = []
    mrra_0, mfr_0, mrra2_0, mar_0 = get_previous_values(tel_model)

    n_runs = 5 if args_dict["test"] else 50
    generate_random_parameters(
        all_parameters, n_runs, args_dict, mrra_0, mfr_0, mrra2_0, mar_0, tel_model
    )

    data_to_plot, radius = load_and_process_data(args_dict)

    # Preparing figure name and PDF pages for plotting
    plot_file_name = "_".join(("tune_psf", tel_model.name + ".pdf"))
    plot_file = output_dir.joinpath(plot_file_name)
    pdf_pages = PdfPages(plot_file)

    # Find best parameters
    best_pars, best_d80, results = find_best_parameters(
        all_parameters, tel_model, site_model, args_dict, data_to_plot, radius, pdf_pages
    )

    plt.close()
    pdf_pages.close()

    # Write all tested parameters and their metrics to a file
    param_file = write_tested_parameters_to_file(results, best_pars, best_d80, output_dir)
    print(f"\nParameter results written to {param_file}")

    # Automatically create D80 vs off-axis angle plot for best parameters
    create_d80_vs_offaxis_plot(tel_model, site_model, args_dict, best_pars, output_dir)
    print("D80 vs off-axis angle plots created successfully")

    print("\nBest parameters:")
    for par, value in best_pars.items():
        print(f"{par} = {value}")

    # Export best parameters as JSON model parameter files (if flag is provided)
    if args_dict.get("write_psf_parameters", False):
        # Use the output_dir directly for proper tune_psf/telescope structure
        export_psf_parameters_as_json(
            best_pars,
            tel_model,
            args_dict.get("parameter_version", "0.0.0"),
            output_dir.parent,
            func_logger,
        )
