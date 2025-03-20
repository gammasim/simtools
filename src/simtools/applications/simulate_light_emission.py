#!/usr/bin/python3

r"""
Simulate calibration devices using the light emission package.

Run the application in the command line.
There are two ways this application can be executed:

1. Illuminator at varying distances.
2. Illuminator and telescopes at fixed positions as defined in the layout.

Example Usage
-------------

1. Simulate light emission with varying distances:

    .. code-block:: console

        simtools-simulate-light-emission --telescope MSTN-04 --site North \
        --illuminator ILLN-01 --light_source_setup variable \
        --model_version 6.0.0 --light_source_type led

2. Simulate light emission with telescopes at fixed positions according to the layout:

    .. code-block:: console

        simtools-simulate-light-emission --telescope MSTN-04 --site North \
        --illuminator ILLN-01 --light_source_setup layout \
        --model_version 6.0.0 \
        --light_source_type led

Command Line Arguments
----------------------
telescope (str, required)
    Telescope model name (e.g. LSTN-01, SSTS-design, SSTS-25, ...)
site (str, required)
    Site name (North or South).
illuminator (str, optional)
    Illuminator in array, e.g., ILLN-01.
light_source_setup (str, optional)
    Select calibration light source positioning/setup:
    - "variable" for varying distances.
    - "layout" for actual telescope positions.
model_version (str, optional)
    Version of the simulation model.
light_source_type (str, optional)
    Select calibration light source type: led (default) or laser.
    This changes the pre-compiled (simtel_array) application that is used to run the
    light emission package with. Currently we use xyzls (laser), and ls-beam can be
    accessed by using the laser option.
off_axis_angle (float, optional)
    Off axis angle for light source direction.
plot (flag, optional)
    Produce a multiple pages pdf file with the image plots.


Example
-------

Simulate isotropic light source at different distances for the MSTN-04:

.. code-block:: console

    simtools-simulate-light-emission --telescope MSTN-04 --site North \
    --illuminator ILLN-01 --light_source_setup variable \
    --model_version 6.0.0 --light_source_type led    ```

Expected Output:

.. code-block:: console

    light-emission package stage:
    File '/workdir/external/simtools/simtools-output/light_emission/model/
        atmprof_ecmwf_north_winter_fixed.dat' registered for atmospheric profile 99.
    Atmospheric profile 99 to be read from file '/workdir/external/simtools/
        simtools-output/light_emission/model/atmprof_ecmwf_north_winter_fixed.dat'.
    Atmospheric profile 99 with 55 levels read from file /workdir/external/
        simtools/simtools-output/light_emission/model/atmprof_ecmwf_north_winter_fixed.dat
    Initialize atmosphere ranging from 0.000 to 120.000 km a.s.l.
    IACT control parameter line: print_events 999 10 100 1000 0
    Case 1: 1 event with 1e+10 photons.
    Using IACT/ATMO package version 1.67 (2023-11-10) for CORSIKA 6.999
    Output file /workdir/external/simtools/simtools-output/light_emission/xyzls.iact.gz
        not yet created.
    Telescope output file: '/workdir/external/simtools/simtools-output/
        light_emission/xyzls.iact.gz'
    ....
    ....
    Sim_telarray stage:
    Telescope 1 triggered (1/0/0/0, mask 1), summation from 36 to 95 of 105
    Event end data has been found.
    Shower of 0.0000 TeV energy was seen in 1 of 1 cases.
    Photon statistics:
    All photons:               928518
    Used photons:              928518
    Not absorbed/max. Q.E.:    189560
    Reflected on mirror:        26815
    Camera hit:                 25574
    Pixel hit:                  25574
    Detected:                   20998
    Trigger statistics:
    Tel. triggered:             1
    Tel. + array:               1
    Early readout:              0
    Late readout:               0
    Finish data conversion ...
    Writing 13 histograms to output file.

"""

import logging
import subprocess
from pathlib import Path

import astropy.units as u

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.corsika.corsika_histograms_visualize import save_figs_to_pdf
from simtools.model.calibration_model import CalibrationModel
from simtools.model.site_model import SiteModel
from simtools.model.telescope_model import TelescopeModel
from simtools.simtel.simulator_light_emission import SimulatorLightEmission
from simtools.visualization.visualize import plot_simtel_ctapipe


def _parse(label):
    """Parse command line configuration."""
    config = configurator.Configurator(
        label=label,
        description=(
            "Simulate the light emission by an artificial light source for calibration purposes."
        ),
    )
    config.parser.add_argument(
        "--off_axis_angle",
        help="Off axis angle for light source direction",
        type=float,
        default=0.0,
        required=False,
    )
    config.parser.add_argument(
        "--plot",
        help="Produce a multiple pages pdf file with the image plots.",
        action="store_true",
    )
    config.parser.add_argument(
        "--light_source_type",
        help="Select calibration light source type: led or laser",
        type=str,
        default="led",
        choices=["led", "laser"],
        required=False,
    )
    config.parser.add_argument(
        "--light_source_setup",
        help="Select calibration light source positioning/setup: \
              varying distances (variable), layout positions (layout)",
        type=str,
        choices=["layout", "variable"],
        default=None,
        required=True,
    )
    config.parser.add_argument(
        "--distances_ls",
        help="Light source distance in m (Example --distances_ls 800 1200)",
        nargs="+",
        required=False,
    )
    config.parser.add_argument(
        "--illuminator",
        help="Illuminator in array, i.e. ILLN-design",
        type=str,
        default=None,
        required=True,
    )
    config.parser.add_argument(
        "--return_cleaned",
        help="ctapipe, if image should be cleaned, \
              notice as well image cleaning parameters",
        type=str,
        default=False,
        required=False,
    )
    config.parser.add_argument(
        "--picture_thresh",
        help="ctapipe, threshold above which all pixels are retained",
        type=int,
        required=False,
    )
    config.parser.add_argument(
        "--boundary_thresh",
        help="ctapipe, threshold above which pixels are retained if\
              they have a neighbor already above the picture_thresh",
        type=int,
        required=False,
    )
    config.parser.add_argument(
        "--min_neighbors",
        help="ctapipe, A picture pixel survives cleaning only if it has at\
              least this number of picture neighbors. This has no effect in\
              case keep_isolated_pixels is True",
        type=int,
        required=False,
    )
    config.parser.add_argument(
        "--level",
        help="read 5",
        type=int,
        default=5,
        required=False,
    )
    config.parser.add_argument(
        "--integration_window",
        help="ctapipe, A picture pixel survives cleaning only if it has at\
              least this number of picture neighbors. This has no effect in\
              case keep_isolated_pixels is True",
        nargs="*",
        default=["7", "3"],
        required=False,
    )
    return config.initialize(
        db_config=True,
        simulation_model=["telescope", "model_version"],
        require_command_line=True,
    )


def distance_list(arg):
    """
    Convert distance list to astropy quantities.

    Parameters
    ----------
    arg: list
        List of distances.

    Returns
    -------
    values: list
        List of distances as astropy quantities.
    """
    try:
        return [float(x) * u.m for x in arg]
    except ValueError as exc:
        raise ValueError("Distances must be numeric values") from exc


def default_le_configs(le_application, args_dict):
    """
    Define default light emission configurations.

    Predefined angular distribution names not requiring to read any table are
    "Isotropic", "Gauss:<rms>", "Rayleigh", "Cone:<angle>", and "FilledCone:<angle>", "Parallel",
    with all angles given in degrees, all with respect to the given direction vector
    (vertically downwards if missing). If the light source has a non-zero length and velocity
    (in units of the vacuum speed of light), it is handled as a moving source,
    in the given direction.

    Parameters
    ----------
    le_application: str
        Light emission application.

    args_dict: dict
        Dictionary with command line arguments.

    Returns
    -------
    default_config: dict
        Default light emission configuration.
    """
    if le_application in ("xyzls", "ls-beam") and args_dict["light_source_setup"] == "variable":
        return {
            "x_pos": {"len": 1, "unit": u.Unit("cm"), "default": 0 * u.cm, "names": ["x_position"]},
            "y_pos": {"len": 1, "unit": u.Unit("cm"), "default": 0 * u.cm, "names": ["y_position"]},
            "z_pos": {
                "len": 1,
                "unit": u.Unit("cm"),
                "default": [i * 100 for i in [200, 300, 400, 600, 800, 1200, 2000, 4000]] * u.cm,
                "names": ["z_position"],
            },
            "direction": {
                "len": 3,
                "unit": u.dimensionless_unscaled,
                "default": [0, 0, -1],
                "names": ["direction", "cx,cy,cz"],
            },
        }
    return {}


def select_application(args_dict):
    """
    Select sim_telarray application for light emission simulations.

    Parameters
    ----------
    args_dict: dict
        Dictionary with command line arguments.

    Returns
    -------
    le_application: str
        Light emission application.
    """
    if args_dict["light_source_type"] == "led":
        return "xyzls", args_dict["light_source_setup"]
    if args_dict["light_source_type"] == "laser":
        return "ls-beam", args_dict["light_source_setup"]
    return None, args_dict["light_source_setup"]


def prepare_light_source(
    args_dict, le_config, le_application, telescope_model, calibration_model, site_model
):
    """Prepare the SimulatorLightEmission object."""
    return SimulatorLightEmission(
        telescope_model=telescope_model,
        calibration_model=calibration_model,
        site_model=site_model,
        default_le_config=le_config,
        le_application=le_application,
        simtel_path=args_dict["simtel_path"],
        light_source_type=args_dict["light_source_type"],
    )


def run_light_emission_simulation(light_source, args_dict, figures, logger):
    """Run the light emission simulation."""
    run_script = light_source.prepare_script(generate_postscript=True, **args_dict)
    log_file = Path(light_source.output_directory) / "logfile.log"
    with open(log_file, "w", encoding="utf-8") as log_file:
        subprocess.run(
            run_script,
            shell=False,
            check=False,
            text=True,
            stdout=log_file,
            stderr=log_file,
        )
    process_simulation_output(light_source, args_dict, figures, logger)


def process_simulation_output(light_source, args_dict, figures, logger):
    """Process the simulation output, including plotting and saving figures."""
    try:
        filename = (
            f"{light_source.output_directory}/"
            f"{light_source.le_application[0]}_{light_source.le_application[1]}.simtel.gz"
        )

        try:
            distance = light_source.default_le_config["z_pos"]["default"]
        except KeyError:
            distance = round(light_source.distance, 2)

        fig = plot_simtel_ctapipe(
            filename,
            cleaning_args=[
                args_dict["boundary_thresh"],
                args_dict["picture_thresh"],
                args_dict["min_neighbors"],
            ],
            distance=distance,
            return_cleaned=args_dict["return_cleaned"],
        )
        figures.append(fig)

    except AttributeError:
        msg = (
            f"Telescope not triggered at distance of "
            f"{light_source.default_le_config['z_pos']['default']}"
        )
        logger.warning(msg)


def save_figures_to_pdf(figures, output_directory, telescope, le_application):
    """Save the generated figures to a PDF file."""
    save_figs_to_pdf(
        figures,
        f"{output_directory}/{telescope}_{le_application[0]}_{le_application[1]}.pdf",
    )


def simulate_variable_distances(
    args_dict,
    default_le_config,
    le_application,
    telescope_model,
    calibration_model,
    site_model,
    logger,
):
    """Simulate light emission for variable distances."""
    if args_dict["distances_ls"] is not None:
        default_le_config["z_pos"]["default"] = distance_list(args_dict["distances_ls"])
    logger.info(f"Simulating for distances of {default_le_config['z_pos']['default']}")

    figures = []
    for distance in default_le_config["z_pos"]["default"]:
        le_config = default_le_config.copy()
        le_config["z_pos"]["default"] = distance
        light_source = prepare_light_source(
            args_dict,
            le_config,
            le_application,
            telescope_model,
            calibration_model,
            site_model,
        )
        run_light_emission_simulation(light_source, args_dict, figures, logger)
    save_figures_to_pdf(
        figures, light_source.output_directory, args_dict["telescope"], le_application
    )


def simulate_layout_positions(
    args_dict,
    default_le_config,
    le_application,
    telescope_model,
    calibration_model,
    site_model,
    logger,
):
    """Simulate light emission for layout positions."""
    light_source = prepare_light_source(
        args_dict,
        default_le_config,
        le_application,
        telescope_model,
        calibration_model,
        site_model,
    )
    figures = []
    run_light_emission_simulation(light_source, args_dict, figures, logger)
    save_figures_to_pdf(
        figures, light_source.output_directory, args_dict["telescope"], le_application
    )


def main():
    """Simulate light emission."""
    label = Path(__file__).stem
    args_dict, db_config = _parse(label)
    le_application = select_application(args_dict)
    default_le_config = default_le_configs(le_application[0], args_dict)
    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    telescope_model = TelescopeModel(
        site=args_dict["site"],
        telescope_name=args_dict["telescope"],
        mongo_db_config=db_config,
        model_version=args_dict["model_version"],
        label=label,
    )

    calibration_model = CalibrationModel(
        site=args_dict["site"],
        calibration_device_model_name=args_dict["illuminator"],
        mongo_db_config=db_config,
        model_version=args_dict["model_version"],
        label=label,
    )

    site_model = SiteModel(
        site=args_dict["site"],
        mongo_db_config=db_config,
        model_version=args_dict["model_version"],
        label=label,
    )

    if args_dict["light_source_setup"] == "variable":
        simulate_variable_distances(
            args_dict,
            default_le_config,
            le_application,
            telescope_model,
            calibration_model,
            site_model,
            logger,
        )
    elif args_dict["light_source_setup"] == "layout":
        simulate_layout_positions(
            args_dict,
            default_le_config,
            le_application,
            telescope_model,
            calibration_model,
            site_model,
            logger,
        )


if __name__ == "__main__":
    main()
