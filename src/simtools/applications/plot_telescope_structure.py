#!/usr/bin/python3

"""Plot the optical structure of a telescope: mirrors, camera housing and telescope structure.

The nominal geometry comes from the model parameters, without random alignment
errors. With ``--shadowing``, the telescope structure is coloured by the light each
element stops on its own; this needs the optional obdeect package.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.model import telescope_geometry
from simtools.model.telescope_model import TelescopeModel
from simtools.ray_tracing.telescope_shadowing import shadowing_shares
from simtools.visualization import plot_telescope_structure as structure

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "shadowing",
        help="Colour the structure by the light each element stops alone (needs obdeect).",
        action="store_true",
        default=False,
    ),
    cli.ArgumentDefinition(
        "zenith_angle",
        help="Zenith angle of the star for --shadowing [deg].",
        type=float,
        default=20.0,
    ),
    cli.ArgumentDefinition(
        "photons",
        help="Number of star photons traced for --shadowing.",
        type=int,
        default=1_000_000,
    ),
    cli.ArgumentDefinition(
        "seed",
        help="Random seed of the star photons for --shadowing.",
        type=int,
        default=11,
    ),
    cli.ArgumentDefinition(
        "telescope_structure_file",
        help="ECSV file defining the telescope structure (overrides the model file).",
        type=Path,
        default=None,
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    model_repository=True,
    arguments=(
        *_ARGUMENTS,
        cli.MODEL_VERSION,
        cli.OVERWRITE_MODEL_PARAMETERS,
        cli.SITE,
        cli.TELESCOPE,
        *cli.OUTPUT_PATH_ARGUMENTS,
    ),
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()
    args = app_context.args
    label = "plot_telescope_structure"
    with TemporaryDirectory(prefix="simtools-plot-telescope-structure-") as model_directory:
        tel_model = TelescopeModel(
            site=args["site"],
            telescope_name=args["telescope"],
            model_version=args["model_version"],
            label=label,
            model_reader=app_context.model_reader,
            model_directory=model_directory,
        )
        parameters = tel_model.parameters
        if args["shadowing"]:
            original_structure = None
            if args["telescope_structure_file"] is not None:
                original_structure = parameters.pop("telescope_structure", None)
            tel_model.export_model_files()
            if original_structure is not None:
                parameters["telescope_structure"] = original_structure
        model_files = {}
        for parameter in ("mirror_list", "telescope_structure"):
            record = parameters.get(parameter)
            if record and record.get("value"):
                model_files[parameter] = record
        if args["telescope_structure_file"] is not None:
            structure_parameter = parameters.get("telescope_structure", {"unit": None})
            parameters["telescope_structure"] = {
                **structure_parameter,
                "value": str(args["telescope_structure_file"].resolve()),
                "file": True,
            }
            model_files.pop("telescope_structure", None)
        if model_files:
            tel_model.model_reader.export_model_files(
                parameters=model_files,
                dest=tel_model.config_file_directory,
            )
        geometry = telescope_geometry.telescope_geometry(tel_model)
        shares = None
        if args["shadowing"]:
            shares = shadowing_shares(
                tel_model, args["zenith_angle"], args["photons"], args["seed"]
            )
        output_dir = app_context.io_handler.get_output_directory()
        structure.plot_telescope_structure(
            geometry, output_dir.joinpath(f"{label}_{tel_model.name}"), shares
        )


if __name__ == "__main__":
    main()
