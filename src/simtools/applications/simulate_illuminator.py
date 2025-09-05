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

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.simtel.simulator_light_emission import SimulatorLightEmission


def _parse(label):
    """Parse command line configuration."""
    config = configurator.Configurator(
        label=label,
        description=(
            "Simulate the light emission by a calibration light source "
            "(not attached to a telescope)."
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
        "--light_source_position",
        help="Light source position (x,y,z) relative to the array center (ground coordinates) in m",
        nargs=3,
        required=False,
    )
    config.parser.add_argument(
        "--light_source_pointing",
        help=(
            "Light source pointing direction "
            "(Example for pointing downwards: --light_source_pointing 0 0 -1)"
        ),
        nargs=3,
        required=False,
    )
    config.parser.add_argument(
        "--light_source",
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


def main():
    """Simulate light emission."""
    label = Path(__file__).stem
    logger = logging.getLogger()

    args_dict, db_config = _parse(label)
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    light_source = SimulatorLightEmission(
        light_emission_config=args_dict,
        db_config=db_config,
        label=label,
    )

    light_source.simulate()


if __name__ == "__main__":
    main()
