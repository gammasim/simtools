#!/usr/bin/python3

r"""
Generate a simulation job grid from production choices.

``simtools-production-generate-grid`` expands a production definition into an ECSV
table in which each row is one executable simulation run. The expansion combines
particle, model, interaction, shower, pointing, source-offset, model-version
dependent night-sky background rate, energy, and run choices. The generated grid
is intended as input for local production execution or workload-management
submission tools.

The application supports two pointing-grid modes:

- horizontal grids, configured with ``azimuth`` and ``zenith`` axes;
- equatorial grids, configured with ``ra`` and ``dec`` axes and converted to
  local horizontal coordinates using ``site`` and ``time_of_observation``.

Direction axes can be specified explicitly by the number of bins or derived from a
direction-grid density. If a CORSIKA limits lookup table is supplied, the grid is
interpolated at each direction point and the configured energy, core-scatter, and
view-cone values act as absolute bounds.

NSB handling
------------
NSB is not configured as a production-grid axis. The generated rows contain
``nsb_rate``, derived from the site model through the selected ``site`` and
``model_version``. This means NSB values are directly dependent on the model
version; when multiple model versions are requested, simtools resolves the NSB
rate separately for each version. The resolved ``nsb_rate`` is also used for
CORSIKA-limit lookup interpolation.

Production context
------------------
site (str)
    Observatory site, for example ``North`` or ``South``. Required for resolving
    the model-version-dependent NSB rate and for RA/Dec-to-horizontal coordinate
    conversion.
model_version (str or list)
    Simulation model version. Multiple values expand the job grid over all
    requested versions.
array_layout_name (str or list)
    Array layout name, for example ``CTAO-North-Alpha`` or ``LSTN-01``.
    Multiple layouts expand the grid.
primary (str or list)
    Primary particle to simulate, for example ``gamma`` or ``proton``. Values are
    common particle names by default; use ``primary_id_type`` for CORSIKA7 or PDG
    particle identifiers.
primary_id_type (str, optional)
    Identifier type for ``primary``. Allowed values are ``common_name``,
    ``corsika7_id``, and ``pdg_id``. Default is ``common_name``.
corsika_le_interaction, corsika_he_interaction (str or list, optional)
    Low- and high-energy CORSIKA hadronic interaction models. Missing values use
    the simtools defaults. Multiple values expand the grid.
simulation_software (str, optional)
    Simulation steps represented by the grid. Typical values are ``corsika``,
    ``sim_telarray``, and ``corsika_sim_telarray``. The executable versions are
    determined by the runtime environment, not by this grid file.

Shower and CORSIKA parameters
-----------------------------
energy_range (quantity pair or list of pairs, optional)
    Primary-particle energy range. Provide one min/max pair, for example
    ``--energy_range 30 GeV 300 TeV``. Configuration files may also provide a
    list of min/max pairs to generate several fixed-energy or binned-energy
    ranges. A fixed energy is represented by equal minimum and maximum values.
    Default is ``3 GeV 330 TeV``.
eslope (float, optional)
    CORSIKA spectral index. Default is ``-2.0``.
core_scatter (integer and quantity, optional)
    CORSIKA core-scatter configuration as ``<reuse_count> <max_radius> <unit>``,
    for example ``--core_scatter 10 500 m``. The reuse count is written as
    ``core_scatter_number`` and the radius as ``core_scatter_max``. Default is
    ``10 10000 m``.
view_cone (quantity pair, optional)
    Minimum and maximum view-cone radius for primary arrival directions, for
    example ``--view_cone 0 deg 10 deg``. Default is ``0 deg 0 deg``.
curved_atmosphere_min_zenith_angle (quantity, optional)
    Zenith angle above which curved-atmosphere CORSIKA binaries are selected.
    This is normally a production-policy setting. The default is configured in
    simtools defaults.

Pointing and grid axes
----------------------
axis (repeatable)
    Compact axis definition:
    ``--axis <name> <min> <unit> <max> <unit> <binning> [scaling]``.
    Supported axis names are ``azimuth``, ``zenith``, ``ra``, ``dec``, and
    ``offset``. Supported scaling modes are ``linear``, ``log``, and
    ``1/cos``; the default is ``linear``.

    **Examples**
    - ``--axis azimuth 310 deg 20 deg 3 linear``
    - ``--axis zenith 0 deg 70 deg 8 linear``
    - ``--axis offset 0 deg 10 deg 2 linear``

    For explicit grids, ``binning`` is the number of points on that axis.
    Directed angular ranges such as azimuth ``310 deg`` to ``20 deg`` may cross
    the 0 deg boundary. For density-based direction grids, direction-axis
    binning is derived from ``direction_grid_density`` and the axis min/max
    values define the accepted range.
direction_grid_density (float or quantity, optional)
    Target density for direction points, normally in ``1/deg^2``. When set,
    simtools derives the number of horizontal ``azimuth``/``zenith`` or
    equatorial ``ra``/``dec`` direction points from the axis ranges and this
    density. Non-direction axes such as ``offset`` still use their explicit
    axis binning.
time_of_observation (str, optional)
    UTC observation time in ``YYYY-MM-DD HH:MM:SS`` format. Required when RA/Dec
    axes are used because the conversion to local azimuth and zenith depends on
    time and site. Ignored for horizontal grids.
local_zenith_range (quantity pair, optional)
    Local zenith range used to keep or reject points generated for density-based
    RA/Dec grids, for example ``--local_zenith_range 0 deg 70 deg``.
local_azimuth_range (quantity pair, optional)
    Directed local azimuth range used to keep or reject points generated for
    density-based RA/Dec grids, for example
    ``--local_azimuth_range 300 deg 60 deg``. The interval may cross 0 deg.

CORSIKA limits lookup
---------------------
corsika_limits (str, optional)
    Path to an ECSV lookup table with direction-dependent CORSIKA simulation
    limits for the selected array layout. When provided, simtools interpolates
    limits at each grid point using zenith, azimuth, and the model-version
    dependent ``nsb_rate``. The configured ``energy_range``, ``core_scatter``,
    and ``view_cone`` values remain absolute bounds:

    - the lower energy is raised when the lookup threshold is above the
      configured minimum;
    - the upper energy is clipped by configured maximum energy and optional
      zenith scaling;
    - ``core_scatter_max`` and ``view_cone_max`` are clipped by the configured
      maxima and lookup limits;
    - grid points whose selected lower energy is above the selected upper energy
      are skipped.

Run statistics
--------------
showers_per_run (int)
    Baseline number of CORSIKA showers per run. This value can be modified per
    energy bin with ``showers_per_run_power_law`` and per zenith angle with
    ``showers_per_run_scaling``.
number_of_runs (int, optional)
    Number of runs generated for each production grid point and energy range.
    Use either ``number_of_runs`` or ``total_showers``.
total_showers (int, optional)
    Requested total shower count per production grid point and energy range.
    simtools derives the number of runs from ``showers_per_run``. If the total
    is not divisible by the selected showers per run, the number of runs is
    rounded up so all generated runs have the same shower count.
total_showers_scaling (str, optional)
    Scaling mode for ``total_showers`` before deriving the number of runs.
    ``fixed`` keeps the configured total. ``zenith_scaled`` applies
    ``total_showers * exp(factor * (cos(zenith_angle) - 1))``. Default is
    ``fixed``.
zenith_angle_scaling_factor (float, optional)
    Factor used by ``total_showers_scaling=zenith_scaled``. The default is
    configured in simtools defaults.
showers_per_run_power_law (tuple, optional)
    Energy-dependent showers-per-run scaling as
    ``<power_index> <reference_energy_value> <reference_energy_unit>``. simtools
    uses the logarithmic energy-bin midpoint and applies
    ``showers_per_run * (E_mid / E_ref) ** power_index``. Example:
    ``--showers_per_run_power_law -2.0 1 TeV``.
showers_per_run_scaling (str, optional)
    Zenith-dependent showers-per-run scaling mode. ``fixed`` keeps the current
    showers-per-run value. ``cosine_zenith`` applies
    ``showers_per_run * cos(zenith_angle)`` and rounds up to at least one
    shower. Default is ``fixed``.
run_number_offset (int, optional)
    Offset added to generated run numbers. Default is ``0``.

Energy scaling
--------------
energy_max_scaling (tuple, optional)
    Zenith-dependent maximum-energy scaling as
    ``<power_index> <reference_energy_value> <reference_energy_unit>``.
    simtools applies
    ``E_max = reference_energy * cos(zenith_angle) ** power_index`` and clips
    the result to the configured ``energy_range`` maximum. Example:
    ``--energy_max_scaling -2.5 300 TeV``.
energy_max_scaling_index (float, optional)
    Legacy hidden option for zenith-dependent maximum-energy scaling. Prefer
    ``energy_max_scaling`` for new configurations.

Output
------
output_file (str, optional)
    Output ECSV file for the generated job grid. The path is resolved through
    the simtools output handler. The output rows include ``nsb_rate`` resolved
    from the selected ``site`` and ``model_version``. Default is
    ``job_grid.ecsv``.


Example
-------
To generate a standard zenith/azimuth grid of simulation points, execute:

.. code-block:: console

        simtools-production-generate-grid --site North --model_version 6.0.2 \
            --array_layout_name alpha \
            --axis azimuth 310 deg 20 deg 3 linear \
            --axis zenith 30 deg 40 deg 2 linear \
            --axis offset 0 deg 10 deg 2 linear \
            --corsika_limits tests/resources/corsika_simulation_limits/merged_corsika_limits.ecsv

To generate an all-sky RA/Dec direction grid and serialize output in RA/Dec,
execute:

.. code-block:: console

        simtools-production-generate-grid --site North --model_version 6.0.2 \
            --array_layout_name alpha \
            --axis ra 0 deg 360 deg 36 linear \
            --axis dec -90 deg 90 deg 18 linear \
            --axis offset 0 deg 10 deg 2 linear \
            --time_of_observation "2017-09-16 00:00:00" \
            --corsika_limits tests/resources/corsika_simulation_limits/merged_corsika_limits.ecsv

To generate an RA/Dec density grid constrained to local sky ranges (for example
full zenith coverage from 0 to 70 deg and a directed azimuth window), execute:

.. code-block:: console

        simtools-production-generate-grid --site North --model_version 6.0.2 \
            --array_layout_name alpha \
            --axis ra 0 deg 360 deg 1 linear \
            --axis dec -40 deg 80 deg 1 linear \
            --axis offset 0 deg 10 deg 2 linear \
            --direction_grid_density 0.25 1/deg^2 \
            --local_zenith_range 0 deg 70 deg \
            --local_azimuth_range 300 deg 60 deg \
            --time_of_observation "2017-09-16 00:00:00" \
            --corsika_limits tests/resources/corsika_simulation_limits/merged_corsika_limits.ecsv
"""

import argparse

from simtools.application_control import build_application
from simtools.configuration import defaults
from simtools.production_configuration.job_grid_io import serialize_job_grid_stream
from simtools.production_configuration.simulation_jobs import (
    build_job_grid_metadata,
    iter_simulation_jobs,
)


def _add_arguments(parser):
    """Register application-specific command line arguments."""
    parser.add_argument(
        "--axis",
        action="append",
        nargs="+",
        required=False,
        help=(
            "Compact axis definition: --axis <name> <min> <unit> <max> <unit> <binning> "
            "[scaling]. May be repeated. "
            "Supported axes: azimuth, zenith, ra, dec, offset. "
            "Options for scaling are: linear, log, 1/cos"
        ),
    )
    parser.add_argument(
        "--time_of_observation",
        type=str,
        required=False,
        help=(
            "Observation time in UTC (format: 'YYYY-MM-DD HH:MM:SS'). "
            "Used when RA/Dec axes are configured."
        ),
    )
    parser.add_argument(
        "--direction_grid_density",
        nargs="+",
        required=False,
        default=None,
        help=(
            "Direction-grid density in 1/deg^2. If set, direction-axis binning is "
            "derived from axis ranges and this density. With RA/Dec axes, use "
            "local_zenith_range/local_azimuth_range to filter generated points."
        ),
    )
    parser.add_argument(
        "--local_zenith_range",
        nargs="+",
        required=False,
        default=None,
        help=(
            "Local zenith range (quantity pair) used to filter RA/Dec density points, "
            "for example: --local_zenith_range 0 deg 70 deg"
        ),
    )
    parser.add_argument(
        "--local_azimuth_range",
        nargs="+",
        required=False,
        default=None,
        help=(
            "Local azimuth range (quantity pair) used to filter RA/Dec density points, "
            "for example: --local_azimuth_range 300 deg 60 deg"
        ),
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="job_grid.ecsv",
        help="Output file for the generated executable job grid.",
    )
    parser.add_argument(
        "--corsika_limits",
        type=str,
        required=False,
        help="Path to the lookup table for simulation limits. ",
    )
    parser.add_argument(
        "--number_of_runs",
        help="Number of runs to be simulated.",
        type=parser.scientific_int,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--total_showers",
        help="Total number of showers to simulate.",
        type=parser.scientific_int,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--total_showers_scaling",
        help="Scaling mode for total showers.",
        type=str,
        choices=["fixed", "zenith_scaled"],
        required=False,
        default="fixed",
    )
    parser.add_argument(
        "--zenith_angle_scaling_factor",
        help=(
            "Scaling factor for zenith-dependent total_showers scaling. "
            "Used only when --total_showers_scaling is 'zenith_scaled'."
        ),
        type=float,
        required=False,
        default=defaults.ZENITH_ANGLE_SCALING_FACTOR_DEFAULT,
    )
    parser.add_argument(
        "--showers_per_run_power_law",
        help=(
            "Scale showers_per_run by (E_mid / E_ref) ** power_index using the bin midpoint: "
            "<power_index> <reference_energy_value> <reference_energy_unit> "
            "(for example: --showers_per_run_power_law -2.0 1 TeV)."
        ),
        nargs=3,
        type=str,
        metavar=("POWER_INDEX", "REFERENCE_ENERGY_VALUE", "REFERENCE_ENERGY_UNIT"),
        required=False,
        default=None,
    )
    parser.add_argument(
        "--showers_per_run_scaling",
        help=(
            "Zenith-angle scaling mode for showers_per_run: "
            "'fixed' keeps the baseline value, "
            "'cosine_zenith' applies showers_per_run * cos(zenith_angle)."
        ),
        type=str,
        choices=["fixed", "cosine_zenith"],
        required=False,
        default="fixed",
    )
    parser.add_argument(
        "--energy_max_scaling",
        help=(
            "Scale max energy with zenith angle as "
            "energy_max_scaling_reference * cos(zenith_angle) ** power_index. "
            "Provide: <power_index> <reference_energy_value> <reference_energy_unit> "
            "(for example: --energy_max_scaling -2.5 300 TeV). "
            "Max energy is limited by the configured energy_range."
        ),
        nargs=3,
        type=str,
        metavar=("POWER_INDEX", "REFERENCE_ENERGY_VALUE", "REFERENCE_ENERGY_UNIT"),
        required=False,
        default=None,
    )
    parser.add_argument(
        "--energy_max_scaling_index",
        help=argparse.SUPPRESS,
        type=float,
        required=False,
        default=None,
    )


def main():
    """See CLI description."""
    app_context = build_application(
        initialization_kwargs={
            "db_config": True,
            "preserve_by_version_keys": ["array_layout_name"],
            "simulation_model": ["site", "layout", "telescope", "model_version"],
            "simulation_configuration": {"software": None, "corsika_configuration": ["all"]},
        },
    )

    serialize_job_grid_stream(
        job_rows=iter_simulation_jobs(app_context.args),
        output_file=app_context.io_handler.get_output_file(app_context.args["output_file"]),
        metadata=build_job_grid_metadata(app_context.args),
    )


if __name__ == "__main__":
    main()
