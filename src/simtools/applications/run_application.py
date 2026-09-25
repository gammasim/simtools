#!/usr/bin/python3

"""Run several simtools applications using a configuration file."""

from pathlib import Path

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.io import ascii_handler
from simtools.runners.simtools_runner import run_applications

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "config_file", help="Application configuration.", type=str, required=True, default=None
    ),
    cli.ArgumentDefinition(
        "steps",
        type=int,
        nargs="+",
        help="List of steps to be execution (e.g., '--steps 7 8 9'; do not specify to run all).",
    ),
    cli.ArgumentDefinition(
        "context_file",
        type=str,
        help="YAML mapping containing workflow placeholder replacements.",
    ),
    cli.ArgumentDefinition(
        "replace",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Replace one workflow placeholder. May be repeated.",
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(*_ARGUMENTS,),
    database=True,
    setup_io_handler=False,
    resolve_sim_software_executables=False,
    usage="simtools-run-application --config_file config_file_name",
)


def main():
    """Run several simtools applications using a configuration file."""
    app_context = APPLICATION.start()
    replacements = _load_replacements(app_context.args)

    run_applications(
        app_context.args,
        run_time=app_context.run_time,
        replacements=replacements,
    )


def _load_replacements(args):
    """Load context-file values and command-line replacements."""
    replacements = {}
    context_file = args.get("context_file")
    if context_file:
        context = ascii_handler.collect_data_from_file(context_file)
        if not isinstance(context, dict):
            raise ValueError(f"Context file must contain a mapping: {context_file}")
        replacements.update(context)

    for item in args.get("replace") or []:
        if "=" not in item:
            raise ValueError(f"Replacement must use KEY=VALUE syntax: {item!r}")
        key, value = item.split("=", maxsplit=1)
        key = key.strip()
        if not key:
            raise ValueError(f"Replacement key cannot be empty: {item!r}")
        replacements[key] = value

    config_file = args.get("config_file")
    if config_file:
        replacements.setdefault("__CONFIG_DIRECTORY__", str(Path(config_file).resolve().parent))
    return {str(key): str(value) for key, value in replacements.items()}


if __name__ == "__main__":
    main()
