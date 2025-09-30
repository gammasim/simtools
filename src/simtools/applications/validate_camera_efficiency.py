#!/usr/bin/python3

r"""
    Calculate on-axis telescope throughput and NSB pixels rates.

    Uses the sim_telarray tool "testeff" to calculate the camera efficiency.
    The results of telescope throughput including optical and camera components for Cherenkov (left)
    and NSB light (right) as a function of wavelength are plotted. See examples below.

    .. _validate_camera_eff_plot:
    .. image:: images/validate_camera_efficiency_North-MST-NectarCam-D_cherenkov.png
      :width: 49 %
    .. image:: images/validate_camera_efficiency_North-MST-NectarCam-D_nsb.png
      :width: 49 %

    Command line arguments
    ----------------------
    site (str, required)
        North or South.
    telescope (str, required)
        Telescope model name (e.g. LSTN-01, SSTS-15)
    model_version (str, optional)
        Simulation model version
    zenith_angle (float, optional)
        Zenith angle in degrees (between 0 and 180).
    azimuth_angle (float, optional)
        Telescope pointing direction in azimuth.
    nsb_spectrum (str, optional)
        File with NSB spectrum to use for the efficiency simulation.

    Example
    -------
    MSTN-01 5.0.0

    Runtime < 1 min.

    .. code-block:: console

        simtools-validate-camera-efficiency --site North \\
            --azimuth_angle 0 --zenith_angle 20 \\
            --nsb_spectrum average_nsb_spectrum_CTAO-N_ze20_az0.txt \\
            --telescope MSTN-01 --model_version 5.0.0

    The output is saved in simtools-output/validate_camera_efficiency.
"""

from pathlib import Path

import simtools.data_model.model_data_writer as writer
from simtools.application_control import get_application_label, startup_application
from simtools.camera.camera_efficiency import CameraEfficiency
from simtools.configuration import configurator


def _parse():
    """Parse command line configuration."""
    config = configurator.Configurator(
        label=get_application_label(__file__),
        description=(
            "Calculate the camera efficiency and NSB pixel rates. "
            "Plot the camera efficiency vs wavelength for Cherenkov and NSB light."
        ),
    )
    config.parser.add_argument(
        "--nsb_spectrum",
        help=(
            "File with NSB spectrum to use for the efficiency simulation."
            "The expected format is two columns with wavelength in nm and "
            "NSB flux with the units: [1e9 * ph/m2/s/sr/nm]."
            "If the file has more than two columns, the first and third are used,"
            "and the second is ignored (native sim_telarray behavior)."
        ),
        type=str,
        default=None,
        required=False,
    )
    config.parser.add_argument(
        "--skip_correction_to_nsb_spectrum",
        help=(
            "Skip correction to the NSB spectrum to account for the "
            "difference between the altitude used in the reference B&E spectrum and "
            "the observation level at the CTAO sites."
        ),
        required=False,
        action="store_true",
    )
    config.parser.add_argument(
        "--write_reference_nsb_rate_as_parameter",
        help=("Write the NSB pixel rate obtained for reference conditions as a model parameter "),
        action="store_true",
        required=False,
    )
    _args_dict, _db_config = config.initialize(
        db_config=True,
        simulation_model=["telescope", "model_version", "parameter_version"],
        simulation_configuration={"corsika_configuration": ["zenith_angle", "azimuth_angle"]},
    )
    if _args_dict["site"] is None or _args_dict["telescope"] is None:
        config.parser.print_help()
        print("\n\nSite and telescope must be provided\n\n")
        raise RuntimeError("Site and telescope must be provided")
    return _args_dict, _db_config


def main():
    """Calculate the camera efficiency and NSB pixel rates."""
    args_dict, db_config, _, _io_handler = startup_application(_parse)

    ce = CameraEfficiency(
        db_config=db_config,
        label=args_dict.get("label"),
        config_data=args_dict,
    )
    ce.simulate()
    ce.analyze(force=True)
    ce.plot_efficiency(efficiency_type="Cherenkov", save_fig=True)
    ce.plot_efficiency(efficiency_type="NSB", save_fig=True)

    writer.ModelDataWriter.dump_model_parameter(
        parameter_name="nsb_pixel_rate",
        value=ce.get_nsb_pixel_rate(
            reference_conditions=args_dict.get("write_reference_nsb_rate_as_parameter", False)
        ),
        instrument=args_dict["telescope"],
        parameter_version=args_dict.get("parameter_version", "0.0.0"),
        output_file=Path(f"nsb_pixel_rate-{args_dict.get('parameter_version', '0.0.0')}.json"),
        output_path=_io_handler.get_output_directory() / args_dict["telescope"] / "nsb_pixel_rate",
    )


if __name__ == "__main__":
    main()
