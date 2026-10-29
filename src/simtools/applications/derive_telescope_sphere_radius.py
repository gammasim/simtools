#!/usr/bin/env python3
"""Derive the CORSIKA fiducial sphere radius from telescope model geometry."""

import astropy.units as u

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.data_model.model_data_writer import ModelDataWriter
from simtools.model.model_utils import initialize_simulation_models
from simtools.model.telescope_sphere import derive_telescope_sphere_radius

_TELESCOPE = "telescope"
_TELESCOPE_SPHERE_RADIUS = "telescope_sphere_radius"

APPLICATION = ApplicationDefinition.for_module(
    __name__,
    model_repository=True,
    arguments=(
        cli.MODEL_VERSION,
        cli.PARAMETER_VERSION,
        cli.OVERWRITE_MODEL_PARAMETERS,
        cli.SITE,
        cli.TELESCOPE,
        cli.IGNORE_EXISTING_PARAMETER_VERSION,
        *cli.OUTPUT_PATH_ARGUMENTS,
    ),
    needs=("model", "corsika"),
)


def main():
    """Derive and write the telescope sphere radius with application metadata."""
    context = APPLICATION.start()
    args = context.args
    label = args.get("label") or args["application_label"]
    model, _, _ = initialize_simulation_models(
        label=label,
        model_version=args["model_version"],
        site=args["site"],
        telescope_name=args[_TELESCOPE],
        context=context,
    )
    radius = derive_telescope_sphere_radius(model)
    ModelDataWriter.write_model_parameter(
        parameter_name=_TELESCOPE_SPHERE_RADIUS,
        value=radius.to_value(u.m),
        instrument=args[_TELESCOPE],
        parameter_version=args["parameter_version"],
        output_file=f"{_TELESCOPE_SPHERE_RADIUS}.json",
        output_path=args.get("output_path"),
        metadata_input_dict=args,
        unit="m",
        context=context,
    )
    context.logger.info(f"Derived telescope sphere radius: {radius:.3f}")


if __name__ == "__main__":
    main()
