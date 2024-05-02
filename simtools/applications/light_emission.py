import logging
import subprocess
from pathlib import Path

import astropy.units as u

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.corsika.corsika_histograms_visualize import save_figs_to_pdf
from simtools.layout.array_layout import ArrayLayout
from simtools.model.calibration_model import CalibrationModel
from simtools.model.site_model import SiteModel
from simtools.model.telescope_model import TelescopeModel
from simtools.simtel.simtel_light_emission import SimulatorLightEmission


def _parse(label):
    """
    Parse command line configuration
    """

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
        default="layout",
    )
    config.parser.add_argument(  # remove
        "--distance_ls",
        help="Light source distance in m (Example --distance_ls 800 1200)",
        nargs="+",
    )
    config.parser.add_argument(
        "--illuminator",
        help="Illuminator in array, i.e. ILLN-design",
        type=str,
        default=None,
    )
    config.parser.add_argument(
        "--telescope_file",
        help="Telescope position file (temporary)",
        type=str,
        default=None,
    )

    return config.initialize(
        db_config=True,
        simulation_model="telescope",
        require_command_line=False,
    )


def default_le_configs(le_application):
    """Predefined angular distribution names not requiring to read any table are
    "Isotropic", "Gauss:<rms>", "Rayleigh", "Cone:<angle>", and "FilledCone:<angle>", "Parallel",
     with all angles given in degrees, all with respect to the given direction vector
    (vertically downwards if missing). If the light source has a non-zero length and velocity
    (in units of the vacuum speed of light), it is handled as a moving source,
    in the given direction.
    """

    if le_application in ("xyzls", "ls_beam"):
        default_config = {
            "beam_shape": {
                "len": 1,
                "unit": str,
                "default": "Gauss",
                "names": ["beam_shape", "angular_distribution"],
            },
            "beam_width": {
                "len": 1,
                "unit": u.Unit("deg"),
                "default": 0.5 * u.deg,
                "names": ["rms"],
            },
            "pulse_shape": {
                "len": 1,
                "unit": str,
                "default": "Gauss",
                "names": ["pulse_shape"],
            },
            "pulse_width": {
                "len": 1,
                "unit": u.Unit("deg"),
                "default": 5 * u.ns,
                "names": ["rms"],
            },
            "x_pos": {
                "len": 1,
                "unit": u.Unit("cm"),
                "default": 0 * u.cm,
                "names": ["x_position"],
            },
            "y_pos": {
                "len": 1,
                "unit": u.Unit("cm"),
                "default": 0 * u.m,
                "names": ["y_position"],
            },
            "z_pos": {
                "len": 1,
                "unit": u.Unit("cm"),
                "default": [i * 100 for i in [200, 300, 400, 600, 800, 1200, 2000, 4000]] * u.cm,
                "names": ["z_position"],
            },
            "x_pos_ILLN-01": {
                "len": 1,
                "unit": u.Unit("m"),
                "default": 2.75 * u.m,
                "names": ["x_position"],
            },
            "y_pos_ILLN-01": {
                "len": 1,
                "unit": u.Unit("m"),
                "default": -587.18 * u.m,
                "names": ["y_position"],
            },
            "z_pos_ILLN-01": {
                "len": 1,
                "unit": u.Unit("m"),
                "default": 2295 * u.m,
                "names": ["z_position"],
            },
            "direction": {
                "len": 3,
                "unit": u.dimensionless_unscaled,
                "default": [0, 0.0, -1],
                "names": ["direction", "cx,cy,cz"],
            },
        }
    return default_config


def select_application(args_dict):
    if args_dict["light_source_type"] == "led":
        le_application = "xyzls"
    elif args_dict["light_source_type"] == "laser":
        le_application = "ls_beam"

    return le_application, args_dict["light_source_setup"]


def main():
    """
    Run the application in the command line.
    There are two ways how this can be run:
    1. Illuminator at varying distances
    2. Illuminator and telscopes at fixed positions as defined in the layout

    Example:
    simtools-simulate-light-emission --telescope MSTN-design --site North \
      --illuminator ILLN-design --light_source_setup variable --model_version prod6

      simtools-simulate-light-emission --telescope MSTN-04 --site North  \
        --illuminator ILLN-01 --light_source_setup layout --model_version prod6 \
        --telescope_file\
        /workdir/external/simtools/tests/resources/telescope_positions-North-ground.ecsv
    """

    label = Path(__file__).stem
    args_dict, db_config = _parse(label)
    le_application = select_application(args_dict)
    default_le_config = default_le_configs(le_application[0])
    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    # Create telescope model
    telescope_model = TelescopeModel(
        site=args_dict["site"],
        telescope_model_name=args_dict["telescope"],
        mongo_db_config=db_config,
        model_version=args_dict["model_version"],
        label=label,
    )
    # TODO: Use real coordinates from calibration_model or instance
    # we use now ILLN-01 coordinates in the default configuration

    # Create calibration model
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
        figures = []
        for distance in default_le_config["z_pos"]["default"]:
            le_config = default_le_config.copy()
            le_config["z_pos"]["default"] = distance
            le = SimulatorLightEmission.from_kwargs(
                telescope_model=telescope_model,
                calibration_model=calibration_model,
                site_model=site_model,
                default_le_config=le_config,
                le_application=le_application,
                simtel_source_path=args_dict["simtel_path"],
                light_source_type=args_dict["light_source_type"],
            )
            run_script = le.prepare_script(generate_postscript=True)
            subprocess.run(run_script, shell=False, check=False)
            # le.plot_simtel() #custom plots using eventio
            try:
                fig = le.plot_simtel_ctapipe()
                figures.append(fig)
            except AttributeError:
                msg = f"telescope not triggered at distance of {le.distance.to(u.meter)}"
                logger.warning(msg)

        save_figs_to_pdf(
            figures,
            f"{le.output_directory}/{args_dict['telescope']}_{le.le_application[0]}_"
            f"{le.le_application[1]}.pdf",
        )

    elif args_dict["light_source_setup"] == "layout":

        # TODO: Here we use coordinates from the telescope list, change as soon as
        # coordinates are in DB i.e. calibration_model.coordinate, telescope_model.coordinate
        layout = ArrayLayout(
            mongo_db_config=db_config,
            model_version=args_dict["model_version"],
            collection="telescopes",
            site=args_dict["site"],
            telescope_list_file=args_dict["telescope_file"],
            # telescope_list_metadata_file=args_dict["input_meta"],
            # validate=not args_dict["skip_input_validation"],
        )
        layout.convert_coordinates()
        layout.select_assets(args_dict["telescope"])

        for telescope in layout._telescope_list:  # pylint: disable=protected-access
            if telescope.name == args_dict["telescope"]:
                xx, yy, zz = telescope.get_coordinates(crs_name="ground")

        # telescope coordinates from list
        default_le_config["x_pos"]["real"] = xx
        default_le_config["y_pos"]["real"] = yy
        default_le_config["z_pos"]["real"] = zz

        le = SimulatorLightEmission.from_kwargs(
            telescope_model=telescope_model,
            calibration_model=calibration_model,
            site_model=site_model,
            default_le_config=default_le_config,
            le_application=le_application,
            simtel_source_path=args_dict["simtel_path"],
            light_source_type=args_dict["light_source_type"],
        )
        run_script = le.prepare_script(generate_postscript=True)
        subprocess.run(run_script, shell=False, check=False)


if __name__ == "__main__":
    main()
