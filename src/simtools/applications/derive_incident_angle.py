#!/usr/bin/env python
r"""
Derive photon incident angles on focal plane and primary/secondary mirrors.

Creates photon files with additional columns for incident angles calculation.
Outputs files and histograms of the incidence angles at
the focal plane, primary mirror, and if available, secondary mirror.
Optional debug plots can be also generated.
Note that this application does not include a full raytracing of telescope structures,
and their non symmetric shadowing at off-axis angles.

Example usage
-------------

.. code-block:: console

    simtools-derive-incident-angle \
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

.. _plot_derive_incident_angle_plot:
.. image:: images/incident_angles_multi_derive_incident_angle_SSTS-04.png
    :width: 49 %

Example of a primary mirror incident angle plot for a SST:

.. _plot_derive_incident_angle_plot_primary:
.. image:: images/incident_angles_primary_multi_derive_incident_angle_SSTS-04.png
    :width: 49 %

Note also the relation between radius and primary mirror incident angles, and how this relates to
the peak seen in the primary mirror incident angle distribution:

.. _plot_derive_incident_angle_plot_angle_vs_radius:
.. image:: images/primary_angle_vs_radius.png
    :width: 49 %

Example of a secondary mirror incident angle plot for a SST:

.. _plot_derive_incident_angle_plot_secondary:
.. image:: images/incident_angles_secondary_multi_derive_incident_angle_SSTS-04.png
    :width: 49 %
"""

import astropy.units as u

from simtools.application_control import build_application
from simtools.ray_tracing.incident_angles import IncidentAnglesCalculator
from simtools.visualization.plot_incident_angles import plot_incident_angles


def _add_arguments(parser):
    """Register application-specific command line arguments."""
    parser.initialize_application_arguments(
        ["off_axis_angles", "source_distance", "number_of_photons"]
    )
    parser.add_argument(
        "--perfect_mirror",
        help="Assume perfect mirror shape/alignment/reflection",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--debug_plots",
        dest="debug_plots",
        help="Generate additional debug plots (radius histograms, XY heatmaps, radius vs angle)",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--calculate_primary_secondary_angles",
        dest="calculate_primary_secondary_angles",
        help="Compute angles of incidence on primary and secondary mirrors",
        required=False,
        action="store_true",
    )


def main():
    """See CLI description."""
    app_context = build_application(
        __file__,
        description=__doc__,
        add_arguments_function=_add_arguments,
        initialization_kwargs={
            "db_config": True,
            "simulation_model": ["telescope", "site", "model_version"],
        },
    )

    app_context.logger.info("Starting derivation of incident angles")

    output_dir = app_context.io_handler.get_output_directory()
    base_label = app_context.args.get("label") or app_context.args["application_label"]
    telescope_name = app_context.args["telescope"]
    label_with_telescope = f"{base_label}_{telescope_name}"

    calculator = IncidentAnglesCalculator(
        config_data=app_context.args,
        output_dir=output_dir,
        label=base_label,
    )
    offsets = [
        value.to_value(u.deg) for value in app_context.args.get("off_axis_angles", [0.0 * u.deg])
    ]

    results_by_offset = calculator.run_for_offsets(offsets)
    plot_incident_angles(
        results_by_offset,
        output_dir,
        label_with_telescope,
        debug_plots=app_context.args.get("debug_plots", False),
        model_version=app_context.args.get("model_version", None),
    )
    calculator.save_model_parameters(results_by_offset)
    total = sum(len(t) for t in results_by_offset.values())
    summary_msg = (
        f"Derived incident angles for {len(results_by_offset)} offsets,\n"
        f"total photon statistics {total}"
    )
    if total < 1_000_000:
        summary_msg += " (below 1e6; results may be statistically unstable)"
    app_context.logger.info(summary_msg)


if __name__ == "__main__":
    main()
