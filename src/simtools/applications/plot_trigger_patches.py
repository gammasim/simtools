"""Create an offline camera trigger-patch configuration report."""

import hashlib
import io
import json

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.model.model_utils import read_overwrite_model_parameter_dict
from simtools.model.telescope_model import TelescopeModel
from simtools.visualization.plot_trigger_patches import (
    PATCH_COUNT,
    PIXEL_IDS,
    analyze_trigger_patch_configuration,
    write_trigger_patch_report,
)

SITE = "site"
TELESCOPE = "telescope"
MODEL_VERSION = "model_version"
OUTPUT_FILE = "output_file"
OVERWRITE_MODEL_PARAMETERS = "overwrite_model_parameters"
ECSV_FORMAT = "ascii.ecsv"
TEXT_ENCODING = "utf-8"

APPLICATION = ApplicationDefinition.for_module(
    __name__,
    model_repository=True,
    arguments=(
        cli.ArgumentDefinition(
            OUTPUT_FILE,
            aliases=("output_name",),
            type=str,
            default=None,
            help="Base name output files; defaults: trigger-patches-<telescope>.",
        ),
        cli.MODEL_VERSION(required=True, nargs=None),
        cli.OVERWRITE_MODEL_PARAMETERS,
        cli.SITE(required=True),
        cli.TELESCOPE(required=True),
        *cli.OUTPUT_PATH_ARGUMENTS,
    ),
    needs=("model",),
)


def run(app_context):
    """Write a configuration-only PDF and machine-readable diagnostics."""
    args = app_context.args
    telescope = TelescopeModel(
        site=args[SITE],
        telescope_name=args[TELESCOPE],
        model_version=args[MODEL_VERSION],
        label="plot_trigger_patches",
        model_reader=app_context.model_reader,
        overwrite_model_parameter_dict=read_overwrite_model_parameter_dict(
            args[OVERWRITE_MODEL_PARAMETERS]
        ),
    )
    pixel_table, group_table = analyze_trigger_patch_configuration(
        telescope.camera,
        telescope.get_parameter_table("camera_trigger_groups"),
        telescope.get_parameter_table("camera_trigger_members"),
    )

    output_name = args.get(OUTPUT_FILE) or f"trigger-patches-{telescope.name}"
    pdf_path = app_context.paths.get_output_file(f"{output_name}.pdf")
    page_summary = write_trigger_patch_report(
        telescope.camera,
        pixel_table,
        group_table,
        pdf_path,
    )
    pixels_path = pdf_path.with_suffix(".pixels.ecsv")
    groups_path = pdf_path.with_suffix(".groups.ecsv")
    summary_path = pdf_path.with_suffix(".summary.json")
    pixel_table.write(pixels_path, format=ECSV_FORMAT, overwrite=True)
    group_table.write(groups_path, format=ECSV_FORMAT, overwrite=True)
    summary = _build_summary(args, pixel_table, group_table)
    summary_path.write_text(json.dumps(summary, indent=2), encoding=TEXT_ENCODING)
    app_context.logger.info("%s", page_summary)
    app_context.logger.info("Wrote trigger-patch configuration report to %s", pdf_path)
    return {
        "pdf_file": pdf_path,
        "pixels_file": pixels_path,
        "groups_file": groups_path,
        "summary_file": summary_path,
    }


def _build_summary(args, pixel_table, group_table):
    """Build reproducibility and status metadata from the analyzed configuration."""
    uncovered = pixel_table[PATCH_COUNT] == 0
    configuration_digest = hashlib.sha256()
    for table in (pixel_table, group_table):
        output = io.StringIO()
        table.write(output, format=ECSV_FORMAT)
        configuration_digest.update(output.getvalue().encode(TEXT_ENCODING))
    return {
        SITE: args[SITE],
        TELESCOPE: args[TELESCOPE],
        MODEL_VERSION: args[MODEL_VERSION],
        "pixel_count": len(pixel_table),
        "trigger_group_count": len(group_table),
        "coordinate_unit": "cm",
        "patch_configuration_sha256": configuration_digest.hexdigest(),
        "unclassified_pixel_ids": pixel_table[PIXEL_IDS][uncovered].tolist(),
        "membership_coverage_status": "review" if uncovered.any() else "pass",
        "runtime_trigger_status": "not tested",
        "simulation_executed": False,
    }


def main():
    """Start the application and generate its report."""
    run(APPLICATION.start())


if __name__ == "__main__":
    main()
