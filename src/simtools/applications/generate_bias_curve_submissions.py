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

The resulting scan grids can be submitted directly with
``simtools-simulate-prod --backend htcondor``.

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

    simtools-simulate-prod --backend htcondor \
        --job_grid_file ./bias_curves/nsb/scan_grid.ecsv \
        --output_path ./bias_curves/nsb/htcondor_submit \
        --backend_config htcondor.yml \
        --label nsb

"""

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.configuration.argument_helpers import parse_quantity_pair
from simtools.job_execution import bias_curve_submissions

_ARGUMENTS = (
    cli.SITE(required=True),
    cli.MODEL_VERSION(required=True, nargs=None),
    cli.TELESCOPE(required=True),
    cli.SIMULATION_SOFTWARE,
    cli.AZIMUTH_ANGLE(required=True, action="store", nargs=None, default=None),
    cli.ZENITH_ANGLE(required=True, action="store", nargs=None, default=None),
    cli.SHOWERS_PER_RUN(required=True),
    cli.CORE_SCATTER(required=True),
    cli.VIEW_CONE(required=True),
    cli.NUMBER_OF_RUNS(required=True),
    cli.CORSIKA_LE_INTERACTION(action="store", nargs=None, default="urqmd"),
    cli.CORSIKA_HE_INTERACTION(action="store", nargs=None, default="epos"),
    cli.CORSIKA_HADRONIC_TRANSITION_ENERGY,
    cli.ArgumentDefinition(
        "nsb_energy_range",
        help="Energy range for the NSB gamma curve.",
        type=parse_quantity_pair,
        default=parse_quantity_pair("20 MeV 25 MeV"),
    ),
    cli.ArgumentDefinition(
        "proton_energy_range",
        help="Energy range for the proton curve.",
        type=parse_quantity_pair,
        default=parse_quantity_pair("2 GeV 2000 GeV"),
    ),
    cli.ArgumentDefinition(
        "nsb_scaling_factor", help="NSB scaling factor used for both curves.", type=float, default=2
    ),
    cli.ArgumentDefinition(
        "trigger_thresholds",
        help=(
            "Define evenly spaced trigger thresholds for both curves as "
            "MIN_THRESHOLD NUMBER_OF_THRESHOLDS STEP_SIZE."
        ),
        type=float,
        nargs=3,
        metavar=("MIN_THRESHOLD", "NUMBER_OF_THRESHOLDS", "STEP_SIZE"),
        default=None,
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        *cli.OUTPUT_PATH_ARGUMENTS,
    ),
    database=True,
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()
    bias_curve_submissions.generate_scan_grids(app_context.args, app_context.io_handler)


if __name__ == "__main__":
    main()
