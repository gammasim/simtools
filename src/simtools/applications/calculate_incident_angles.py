#!/usr/bin/env python
r"""
Calculate photon incident angles on focal plane and primary/secondary mirrors.

Creates photon files with additional columns for incident angles calculation.
Outputs files and histograms of the incidence angles at
the focal plane, primary mirror, and if available, secondary mirror.
Optional debug plots can be also generated.
Note that this application does not include a full raytracing of telescope structures,
and their non symmetric shadowing at off-axis angles.

Example usage
-------------

.. code-block:: console

    simtools-calculate-incident-angles \
        --off_axis_angles 0 1 2 3 4 \
        --source_distance 10 \
        --number_of_photons 1000000 \
        --model_version 6.0.0 \
        --telescope MSTN-04 \
        --site North

Command line arguments
----------------------

- off_axis_angles (float, optional)
    One or more off-axis angles in degrees (space-separated). Default: [0.0].
- source_distance (float, optional)
    Source distance in kilometers. Default: 10.0.
- number_of_photons (int, optional)
    Number of photons of the light source to trace per run. Default: 10000.
- perfect_mirror (flag, optional)
    Assume perfect mirror shape/alignment/reflection.
- debug_plots (flag, optional)
    Generate additional debug plots (radius histograms, XY heatmaps, radius vs angle).
- calculate_primary_secondary_angles / no-calculate_primary_secondary_angles
    Include or skip angles on primary/secondary mirrors. Default: include.

The application writes:

- imaging list (photons) file
- stars list file
- a histogram of incident angles (PNG)
- a results table in ECSV format

Example of a focal-plane incident angle plot for a SST:

.. _plot_calculate_incident_angles_plot:
.. image:: images/incident_angles_multi_calculate_incident_angles_SSTS-04.png
    :width: 49 %

Example of a primary mirror incident angle plot for a SST:

.. _plot_calculate_incident_angles_plot_primary:
.. image:: images/incident_angles_primary_multi_calculate_incident_angles_SSTS-04.png
    :width: 49 %

Note also the relation between radius and primary mirror incident angles, and how this relates to
the peak seen in the primary mirror incident angle distribution:

.. _plot_calculate_incident_angles_plot_angle_vs_radius:
.. image:: images/primary_angle_vs_radius.png
    :width: 49 %

Example of a secondary mirror incident angle plot for a SST:

.. _plot_calculate_incident_angles_plot_secondary:
.. image:: images/incident_angles_secondary_multi_calculate_incident_angles_SSTS-04.png
    :width: 49 %
"""

from simtools.application_startup import get_application_label, startup_application
from simtools.configuration import configurator
from simtools.ray_tracing.incident_angles import IncidentAnglesCalculator
from simtools.visualization.plot_incident_angles import plot_incident_angles


def _parse():
    """Parse command line configuration."""
    config = configurator.Configurator(
        label=get_application_label(__file__),
        description=(
            "Calculate photon incident angles on focal plane and primary/secondary mirrors."
        ),
    )
    config.parser.add_argument(
        "--off_axis_angles",
        help="One or more off-axis angles in degrees (space-separated)",
        type=float,
        nargs="+",
        required=False,
    )
    config.parser.add_argument(
        "--source_distance",
        help="Source distance in kilometers",
        type=float,
        default=10.0,
        required=False,
    )
    config.parser.add_argument(
        "--number_of_photons",
        help="Number of star photons to trace (per run)",
        type=int,
        default=10000,
        required=False,
    )
    config.parser.add_argument(
        "--perfect_mirror",
        help="Assume perfect mirror shape/alignment/reflection",
        action="store_true",
        required=False,
    )
    config.parser.add_argument(
        "--debug_plots",
        dest="debug_plots",
        help="Generate additional debug plots (radius histograms, XY heatmaps, radius vs angle)",
        action="store_true",
        required=False,
    )
    config.parser.add_argument(
        "--calculate_primary_secondary_angles",
        dest="calculate_primary_secondary_angles",
        help="Also compute angles of incidence on primary and secondary mirrors",
        required=False,
        action="store_true",
    )
    return config.initialize(
        db_config=True,
        simulation_model=["telescope", "site", "model_version"],
    )


def main():
    """Calculate photon incident angles on focal plane and primary/secondary mirrors."""
    args_dict, db_config, logger, _io_handler = startup_application(_parse)

    logger.info("Starting calculation of incident angles")

    output_dir = _io_handler.get_output_path(args_dict)
    base_label = args_dict.get("label", get_application_label(__file__))
    telescope_name = args_dict["telescope"]
    label_with_telescope = f"{base_label}_{telescope_name}"

    calculator = IncidentAnglesCalculator(
        simtel_path=args_dict["simtel_path"],
        db_config=db_config,
        config_data=args_dict,
        output_dir=output_dir,
        label=base_label,
    )
    offsets = [float(v) for v in args_dict.get("off_axis_angles", [0.0])]

    results_by_offset = calculator.run_for_offsets(offsets)
    plot_incident_angles(
        results_by_offset,
        output_dir,
        label_with_telescope,
        debug_plots=args_dict.get("debug_plots", False),
    )
    total = sum(len(t) for t in results_by_offset.values())
    summary_msg = (
        f"Calculated incident angles for {len(results_by_offset)} offsets,\n"
        f"total photon statistics {total}"
    )
    if total < 1_000_000:
        summary_msg += " (below 1e6; results may be statistically unstable)"
    logger.info(summary_msg)


if __name__ == "__main__":
    main()
