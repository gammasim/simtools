#!/usr/bin/python3

"""Plot the optical structure of a telescope: mirrors, camera housing and telescope structure.

The nominal geometry comes from the model parameters, without random alignment
errors. With ``--shadowing``, the telescope structure is coloured by the light each
element stops on its own; this needs the optional obdeect package.
"""

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.model.telescope_model import TelescopeModel
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
    tel_model = TelescopeModel(
        site=args["site"],
        telescope_name=args["telescope"],
        model_version=args["model_version"],
        label=label,
        model_reader=app_context.model_reader,
    )
    tel_model.export_model_files()
    geometry = structure.telescope_geometry(tel_model)
    shares = None
    if args["shadowing"]:
        shares = structure.shadowing_shares(
            tel_model, args["zenith_angle"], args["photons"], args["seed"]
        )
    output_dir = app_context.io_handler.get_output_directory()
    structure.plot_telescope_structure(
        geometry, output_dir.joinpath(f"{label}_{tel_model.name}"), shares
    )


if __name__ == "__main__":
    main()
