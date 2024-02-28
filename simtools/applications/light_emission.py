import logging
import subprocess
from pathlib import Path

import astropy.units as u

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.corsika.corsika_histograms_visualize import save_figs_to_pdf
from simtools.io_operations import io_handler
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
    )
    config.parser.add_argument(
        "--plot",
        help="Produce a multiple pages pdf file with the image plots.",
        action="store_true",
    )
    config.parser.add_argument(
        "--light_source_type",
        help="Select calibration light source type: laser (1), other (2)",
        type=int,
        default=1,
    )
    config.parser.add_argument(
        "--distance_ls",
        help="Light source distance in m (Example --distance_ls 800 1200)",
        nargs="+",
    )
    return config.initialize(db_config=True, telescope_model=True, require_command_line=False)


def default_le_configs(le_application):
    """Predefined angular distribution names not requiring to read any table are
    "Isotropic", "Gauss:<rms>", "Rayleigh", "Cone:<angle>", and "FilledCone:<angle>", "Parallel",
     with all angles given in degrees, all with respect to the given direction vector
    (vertically downwards if missing). If the light source has a non-zero length and velocity
    (in units of the vaccum speed of light), it is handled as a moving source,
    in the given direction.
    """

    if le_application == "xyzls":
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
            "direction": {
                "len": 3,
                "unit": u.dimensionless_unscaled,
                "default": [0, 0.0, -1],
                "names": ["direction", "cx,cy,cz"],
            },
        }
    return default_config


def select_application(args_dict):
    if args_dict["light_source_type"] == 1:
        le_application = "xyzls"

    return le_application


def main():
    """
    Run the application in the command line.
    Example:
    simtools-simulate-light-emission --telescope MST-NectarCam-D --site North
    """

    label = Path(__file__).stem
    args_dict, db_config = _parse(label)
    le_application = select_application(args_dict)
    default_le_config = default_le_configs(le_application)
    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    # Output directory to save files related directly to this app
    _io_handler = io_handler.IOHandler()
    output_dir = _io_handler.get_output_directory(label)
    print(default_le_config["z_pos"]["default"])
    print(args_dict["distance_ls"])
    if args_dict["distance_ls"] is not None:
        default_le_config["z_pos"]["default"] = [
            100 * int(dist) for dist in args_dict["distance_ls"]
        ] * u.cm
        print(default_le_config["z_pos"]["default"])

    telescope_model = TelescopeModel(
        site=args_dict["site"],
        telescope_model_name=args_dict["telescope"],
        mongo_db_config=db_config,
        model_version=args_dict["model_version"],
        label=label,
    )
    # important for triggering with a single telescope
    telescope_model.remove_parameters("array_triggers")
    # set to 0 to have the optical axis of the MST aligned.
    telescope_model.remove_parameters("axes_offsets")

    figures = []
    for distance in default_le_config["z_pos"]["default"]:
        le_config = default_le_config.copy()
        le_config["z_pos"]["default"] = distance
        le = SimulatorLightEmission.from_kwargs(
            telescope_model=telescope_model,
            default_le_config=le_config,
            le_application=le_application,
            output_dir=output_dir,
            label=label,
            simtel_source_path=args_dict["simtel_path"],
        )
        # command = le._make_light_emission_script()
        # command = le._make_simtel_script(output_dir)
        run_script = le.prepare_script(plot=True)
        subprocess.run(run_script, shell=False, check=False)
        # le.plot_simtel() #custom plots using eventio

        try:
            fig = le.plot_simtel_ctapipe()
            figures.append(fig)
        except AttributeError:
            msg = f"telescope not triggered at distance of {distance.to(u.meter)}"
            logger.warning(msg)

    save_figs_to_pdf(figures, f"{le.output_dir}/{args_dict['telescope']}_{le.le_application}.pdf")


if __name__ == "__main__":
    main()
