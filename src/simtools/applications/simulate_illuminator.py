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

        simtools-simulate-illuminator --telescope MSTN-04 --site North \
        --illuminator ILLN-01 --light_source_setup variable \
        --model_version 6.0.0

2. Simulate light emission with telescopes at fixed positions according to the layout:

    .. code-block:: console

        simtools-simulate-illuminator --telescope MSTN-04 --site North \
        --illuminator ILLN-01 --light_source_setup layout \
        --model_version 6.0.0

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
off_axis_angle (float, optional)
    Off axis angle for light source direction.
number_events (int, optional)
    Number of events to simulate.


Example
-------

Simulate isotropic light source at different distances for the MSTN-04:

.. code-block:: console

    simtools-simulate-illuminator --telescope MSTN-04 --site North \
    --illuminator ILLN-01 --light_source_setup variable \
    --model_version 6.0.0    ```

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
from pathlib import Path

import astropy.units as u

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.model.calibration_model import CalibrationModel
from simtools.model.model_utils import initialize_simulation_models
from simtools.simtel.simulator_light_emission import SimulatorLightEmission


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
        "--number_events",
        help="Number of events to simulate",
        type=int,
        default=1,
        required=False,
    )
    config.parser.add_argument(
        "--output_prefix",
        help="Prefix for output files (default: empty)",
        type=str,
        default=None,
        required=False,
    )
    return config.initialize(
        db_config=True,
        simulation_model=["telescope", "model_version"],
        require_command_line=True,
    )


def light_emission_configs(args_dict):
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
    args_dict: dict
        Dictionary with command line arguments.

    args_dict: dict
        Dictionary with command line arguments.

    Returns
    -------
    default_config: dict
        Default light emission configuration.
    """
    if args_dict["light_source_setup"] == "variable":
        cfg = {
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
        args_dict.update(cfg)
        return args_dict
    return args_dict


def main():
    """Simulate light emission."""
    label = Path(__file__).stem
    args_dict, db_config = _parse(label)
    light_emission_config = light_emission_configs(args_dict)
    print(light_emission_config)
    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    telescope_model, site_model = initialize_simulation_models(
        label=label,
        db_config=db_config,
        site=args_dict["site"],
        telescope_name=args_dict["telescope"],
        model_version=args_dict["model_version"],
    )

    calibration_model = CalibrationModel(
        site=args_dict["site"],
        calibration_device_model_name=args_dict["illuminator"],
        mongo_db_config=db_config,
        model_version=args_dict["model_version"],
        label=label,
    )

    light_source = SimulatorLightEmission(
        telescope_model=telescope_model,
        calibration_model=calibration_model,
        site_model=site_model,
        light_emission_config=light_emission_config,
        light_source_setup=args_dict["light_source_setup"],
        simtel_path=args_dict["simtel_path"],
        light_source_type="illuminator",
        label=label,
        test=args_dict["test"],
    )

    if args_dict["light_source_setup"] == "variable":
        outputs = light_source.simulate_variable_distances(args_dict)
    elif args_dict["light_source_setup"] == "layout":
        outputs = light_source.simulate_layout_positions(args_dict)
    else:
        outputs = []

    if outputs:
        logger.info("Simulation outputs:\n%s", "\n".join(str(p) for p in outputs))


if __name__ == "__main__":
    main()
