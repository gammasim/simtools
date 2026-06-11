#!/usr/bin/python3

r"""
Generate HTCondor submit scripts for NSB (gamma) and proton telescope trigger bias curves.

This application always generates both curves:

- NSB curve with gamma primary
- Proton curve with proton primary

The trigger-threshold scan values are built into
``simtools.job_execution.bias_curve_submissions``. The user does not provide
threshold values through the CLI.

The telescope is resolved from ``array_layout_name`` using the site model. This
application currently supports only single-telescope layouts. The threshold
parameter is chosen automatically from the telescope type:

- LST: ``asum_threshold``
- MST/SST: ``dsum_threshold``

Energy ranges are fixed internally:

- NSB gamma curve: ``20 MeV 25 MeV``
- Proton curve: ``800 GeV 2000 GeV``

Overwrite model-parameter files are generated dynamically from scratch. No
external overwrite templates are needed.

Command line arguments
----------------------
site (str, required)
    Observation site (e.g., North, South).
model_version (str, required)
    Simulation model version.
array_layout_name (str, required)
    Single-telescope array layout name for simulations.
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
nsb_scaling_factor (float, required)
    NSB scaling factor for the gamma NSB curve.
htcondor_output_path (str)
    Sub-directory inside each curve output dir for HTCondor submit files
    (default: htcondor_submit).
priority (int)
    HTCondor job priority (default: 1).
label (str)
    Job label (provided by framework, optional).
apptainer_image (str, required)
    Apptainer/Singularity image path (provided by framework).
output_path (Path)
    Root output directory; nsb/ and proton/ sub-dirs are created inside it
    (provided by framework, default: ./simtools-output/).

Example
-------
.. code-block:: console

    simtools-generate-bias-curve-submissions \
        --site North \
        --model_version 7.0.0 \
        --array_layout_name LSTN-01 \
        --azimuth_angle 0.0 \
        --zenith_angle 20.0 \
        --showers_per_run 10000 \
        --core_scatter "20 1900 m" \
        --view_cone "0 deg 5 deg" \
        --number_of_runs 10 \
        --nsb_scaling_factor 2 \
        --apptainer_image ./simtools-output/simtools.sif \
        --label LSTN1_7-0-0_dark \
        --output_path ./bias_curves \
        --priority 5

"""

from simtools.application_control import build_application
from simtools.job_execution import bias_curve_submissions


def _add_arguments(parser):
    """Register application-specific command line arguments."""
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
        "--array_layout_name",
        help="Single-telescope array layout name.",
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
        type=float,
        required=True,
    )
    parser.add_argument(
        "--zenith_angle",
        help="Zenith angle in degrees.",
        type=float,
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
        type=str,
        required=True,
    )
    parser.add_argument(
        "--view_cone",
        help="View cone, e.g. '0 deg 5 deg'.",
        type=str,
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
        "--htcondor_output_path",
        help="Sub-directory inside each curve output dir for HTCondor submit files.",
        type=str,
        default="htcondor_submit",
    )
    parser.add_argument(
        "--priority",
        help="HTCondor job priority.",
        type=int,
        default=1,
    )


def main():
    """See CLI description."""
    app_context = build_application(
        initialization_kwargs={"db_config": True, "output": False},
    )
    bias_curve_submissions.generate_bias_curve_submissions(app_context.args)


if __name__ == "__main__":
    main()
