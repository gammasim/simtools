#!/usr/bin/python3

r"""
Generate scan grids for NSB and proton telescope trigger bias curves.

This application always prepares both curves:

- NSB curve
- Proton curve with proton primary

For each curve, the application creates a production grid and
parameter-scan grid:

- ``base_grid.ecsv``
- ``scan_config.yaml``
- ``scan_grid.ecsv``

The resulting scan grids can be consumed by a backend-specific submission generator,
for example ``simtools-simulate-prod-htcondor-generator``.

Command line arguments
----------------------
site (str, required)
    Observation site (e.g., North, South).
model_version (str, required)
    Simulation model version.
telescope (str, required)
    Telescope name for simulations.
simulation_software (str)
    Simulation software (default: corsika_sim_telarray).
azimuth_angle (float, required)
    Azimuth angle in degrees.
zenith_angle (float, required)
    Zenith angle in degrees.
showers_per_run (int, required)
    Number of showers per run.
core_scatter (str, required)
    Core scatter (e.g., "20 1900 m").
view_cone (str, required)
    View cone (e.g., "0 deg 5 deg").
number_of_runs (int, required)
    Number of runs.
corsika_le_interaction (str)
    CORSIKA low-energy interaction model (default: urqmd).
corsika_he_interaction (str)
    CORSIKA high-energy interaction model (default: epos).
corsika_hadronic_transition_energy (Quantity)
    Transition energy between low- and high-energy hadronic models. If omitted, use the CORSIKA
    build default.
nsb_energy_range (str)
    NSB gamma energy range (default: "20 MeV 25 MeV").
proton_energy_range (str)
    Proton energy range (default: "2 GeV 2000 GeV").
nsb_scaling_factor (float)
    NSB scaling factor used for both curves (default: 2).
trigger_thresholds (float, int, float, optional)
    Three values defining the trigger-threshold scan used for both curves:
    minimum threshold, number of thresholds, and step size. Trigger-dependent
    defaults are used when omitted.
output_path (Path)
    Root output directory; nsb/ and proton/ sub-dirs are created inside it
    (provided by framework, default: ./simtools-output/).

Example
-------
.. code-block:: console

    simtools-generate-bias-curve-submissions \
        --site North \
        --model_version 7.0.0 \
        --telescope LSTN-01 \
        --azimuth_angle 0.0 \
        --zenith_angle 20.0 \
        --showers_per_run 10000 \
        --core_scatter "20 1900 m" \
        --view_cone "0 deg 5 deg" \
        --number_of_runs 10 \
        --nsb_energy_range "20 MeV 25 MeV" \
        --proton_energy_range "2 GeV 2000 GeV" \
        --nsb_scaling_factor 2 \
        --trigger_thresholds 220 3 10 \
        --output_path ./bias_curves

Submit files can be generated explicitly for a chosen backend, for
example:

.. code-block:: console

    simtools-simulate-prod-htcondor-generator \
        --job_grid_file ./bias_curves/nsb/scan_grid.ecsv \
        --output_path ./bias_curves/nsb/htcondor_submit \
        --apptainer_image /path/to/image.sif \
        --label nsb

"""

from simtools.application_control import build_application
from simtools.configuration.commandline_argument_helpers import (
    azimuth_angle,
    parse_integer_and_quantity,
    parse_quantity_pair,
    positive_quantity,
    zenith_angle,
)
from simtools.job_execution import bias_curve_submissions


def _add_arguments(parser):
    """Application-specific arguments."""
    parser.add_argument(
        "--site",
        help="Observation site (e.g., North, South).",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model_version",
        help="Simulation model version.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--telescope",
        help="Telescope name.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--simulation_software",
        help="Simulation software.",
        type=str,
        default="corsika_sim_telarray",
    )
    parser.add_argument(
        "--azimuth_angle",
        help="Azimuth angle in degrees.",
        type=azimuth_angle,
        required=True,
    )
    parser.add_argument(
        "--zenith_angle",
        help="Zenith angle in degrees.",
        type=zenith_angle,
        required=True,
    )
    parser.add_argument(
        "--showers_per_run",
        help="Number of showers per run.",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--core_scatter",
        help="Core scatter, e.g. '20 1900 m'.",
        type=parse_integer_and_quantity,
        required=True,
    )
    parser.add_argument(
        "--view_cone",
        help="View cone, e.g. '0 deg 5 deg'.",
        type=parse_quantity_pair,
        required=True,
    )
    parser.add_argument(
        "--number_of_runs",
        help="Number of runs.",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--corsika_le_interaction",
        help="CORSIKA low-energy interaction model.",
        type=str,
        default="urqmd",
    )
    parser.add_argument(
        "--corsika_he_interaction",
        help="CORSIKA high-energy interaction model.",
        type=str,
        default="epos",
    )
    parser.add_argument(
        "--corsika_hadronic_transition_energy",
        help=(
            "Transition energy between low- and high-energy hadronic models. "
            "If omitted, use the CORSIKA build default."
        ),
        type=positive_quantity("GeV"),
        default=None,
    )
    parser.add_argument(
        "--nsb_energy_range",
        help="Energy range for the NSB gamma curve.",
        type=parse_quantity_pair,
        default=parse_quantity_pair("20 MeV 25 MeV"),
    )
    parser.add_argument(
        "--proton_energy_range",
        help="Energy range for the proton curve.",
        type=parse_quantity_pair,
        default=parse_quantity_pair("2 GeV 2000 GeV"),
    )
    parser.add_argument(
        "--nsb_scaling_factor",
        help="NSB scaling factor used for both curves.",
        type=float,
        default=2,
    )
    parser.add_argument(
        "--trigger_thresholds",
        help=(
            "Define evenly spaced trigger thresholds for both curves as "
            "MIN_THRESHOLD NUMBER_OF_THRESHOLDS STEP_SIZE."
        ),
        type=float,
        nargs=3,
        metavar=("MIN_THRESHOLD", "NUMBER_OF_THRESHOLDS", "STEP_SIZE"),
        default=None,
    )


def main():
    """See CLI description."""
    app_context = build_application(
        application_path=__file__,
        description=__doc__,
        add_arguments_function=_add_arguments,
        initialization_kwargs={"db_config": True, "output": False},
    )
    bias_curve_submissions.generate_scan_grids(app_context.args, app_context.io_handler)


if __name__ == "__main__":
    main()
