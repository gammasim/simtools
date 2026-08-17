#!/usr/bin/env python
"""Derive photon incident angles on focal plane and primary/secondary mirrors."""

import astropy.units as u

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.ray_tracing.incident_angles import IncidentAnglesCalculator
from simtools.visualization.plot_incident_angles import plot_incident_angles

_ARGUMENTS = (
    cli.OFF_AXIS_ANGLES,
    cli.SOURCE_DISTANCE,
    cli.NUMBER_OF_PHOTONS,
    cli.ArgumentDefinition(
        "perfect_mirror",
        help="Assume perfect mirror shape/alignment/reflection",
        action="store_true",
        required=False,
    ),
    cli.ArgumentDefinition(
        "debug_plots",
        dest="debug_plots",
        help="Generate additional debug plots (radius histograms, XY heatmaps, radius vs angle)",
        action="store_true",
        required=False,
    ),
    cli.ArgumentDefinition(
        "calculate_primary_secondary_angles",
        dest="calculate_primary_secondary_angles",
        help="Compute angles of incidence on primary and secondary mirrors",
        required=False,
        action="store_true",
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        cli.MODEL_VERSION,
        cli.OVERWRITE_MODEL_PARAMETERS,
        cli.SITE,
        cli.TELESCOPE,
        *cli.OUTPUT_PATH_ARGUMENTS,
    ),
    database=True,
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()

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
