import logging
import subprocess
from pathlib import Path

import astropy.units as u

import simtools.utils.general as gen
from simtools.configuration import configurator
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
        "--ls_distance",
        help="Light source distance in m",
        type=float,
        default=1000,
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
                "default": 0.1 * u.deg,
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
                "default": 3 * u.ns,
                "names": ["rms"],
            },
        }
    return default_config


def select_application(args_dict):
    if args_dict["light_source_type"] == 1:
        le_application = "xyzls"

    return le_application


def main():
    label = Path(__file__).stem
    args_dict, db_config = _parse(label)
    # TODO: following is passed as command-line parameters when running the app
    # args_dict["telescope"] = "LST-1"
    args_dict["telescope"] = "MST-NectarCam-D"
    args_dict["site"] = "north"
    le_application = select_application(args_dict)
    default_le_config = default_le_configs(le_application)
    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    # Output directory to save files related directly to this app
    _io_handler = io_handler.IOHandler()
    output_dir = _io_handler.get_output_directory(label)

    telescope_model = TelescopeModel(
        site=args_dict["site"],
        telescope_model_name=args_dict["telescope"],
        mongo_db_config=db_config,
        model_version=args_dict["model_version"],
        label=label,
    )
    le = SimulatorLightEmission.from_kwargs(
        telescope_model=telescope_model,
        default_le_config=default_le_config,
        le_application=le_application,
        output_dir=output_dir,
        label=label,
        simtel_source_path=args_dict["simtel_path"],
    )
    # command = le._make_light_emission_script()
    # command = le._make_simtel_script(output_dir)
    run_script = le.prepare_script(plot=True)

    subprocess.run(run_script, shell=False, check=False)


if __name__ == "__main__":
    main()
