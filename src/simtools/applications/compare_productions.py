#!/usr/bin/python3

r"""Compare simulation productions at different comparison levels.

The application accepts repeated production descriptors and dispatches to
comparison-level specific implementations.

Command line arguments
----------------------
production (repeated, required)
    Production descriptor in two fields:
    1) label
    2) comma-separated event data file patterns
comparison_level (str, optional)
    Comparison level selector. Supported values are:
    - events
    - signals
    - compute
output_path (str, required)
    Output directory for generated comparison plots.
"""

from simtools.application_control import build_application
from simtools.sim_events.production_comparison import (
    ProductionDescriptor,
    collect_production_metrics,
)
from simtools.utils.general import resolve_file_patterns
from simtools.visualization import plot_event_level_production_comparison


def _add_arguments(parser):
    """Register application-specific command line arguments."""
    parser.initialize_application_arguments(["output_path"])
    parser.add_argument(
        "--production",
        action="append",
        nargs="+",
        metavar=("LABEL", "EVENT_DATA_FILES"),
        required=True,
        help=(
            "Production descriptor. "
            "Use as: --production <label> <comma-separated file patterns>. "
            "Repeat this argument as needed for multiple productions."
        ),
    )
    parser.add_argument(
        "--comparison_level",
        choices=["events", "signals", "compute"],
        default="events",
        help="Comparison level to execute.",
    )


def parse_production_arguments(production_arguments):
    """Parse repeated production arguments into validated descriptors.

    Parameters
    ----------
    production_arguments : list[list[str]]
        Repeated ``--production`` arguments in the shape ``[label, patterns]``.

    Returns
    -------
    list[ProductionDescriptor]
        Validated and normalized production descriptors.

    Raises
    ------
    ValueError
        If configuration is malformed or does not contain any production.
    """
    parsed_productions = _normalize_production_arguments(production_arguments)
    if not parsed_productions:
        raise ValueError("At least one production is required.")

    labels = [label for label, _ in parsed_productions]
    if len(set(labels)) != len(labels):
        raise ValueError("Production labels must be unique.")

    descriptors = []
    for label, pattern_list in parsed_productions:
        patterns = [pattern.strip() for pattern in pattern_list.split(",") if pattern.strip()]
        if len(patterns) == 0:
            raise ValueError(f"Production '{label}' has no event_data_file pattern.")

        resolved_files = [str(path) for path in resolve_file_patterns(patterns)]
        if len(resolved_files) == 0:
            raise ValueError(f"Production '{label}' does not resolve to any files.")
        descriptors.append(ProductionDescriptor(label=label, event_data_files=resolved_files))

    return descriptors


def _normalize_production_arguments(production_arguments):
    """Normalize raw production arguments into ``[(label, files), ...]``."""
    if not production_arguments:
        return []

    normalized = []
    if all(isinstance(item, str) for item in production_arguments):
        return _pairwise_label_file_arguments(production_arguments)

    for item in production_arguments:
        normalized.extend(_normalize_single_production_argument(item))

    return normalized


def _pairwise_label_file_arguments(flat_arguments):
    """Convert a flat list of strings into ``[(label, files), ...]`` pairs."""
    if len(flat_arguments) % 2 != 0:
        _raise_invalid_production_arguments()
    return [
        (flat_arguments[index], flat_arguments[index + 1])
        for index in range(0, len(flat_arguments), 2)
    ]


def _normalize_single_production_argument(argument):
    """Normalize one nested production argument into label/file pairs."""
    if not isinstance(argument, list | tuple):
        _raise_invalid_production_arguments()
    if not all(isinstance(value, str) for value in argument):
        _raise_invalid_production_arguments()
    if len(argument) == 2:
        return [(argument[0], argument[1])]
    return _pairwise_label_file_arguments(list(argument))


def _raise_invalid_production_arguments():
    """Raise a standardized parser error for malformed production arguments."""
    raise ValueError("Production arguments must be provided as label/file pairs.")


def main():
    """See CLI description."""
    app_context = build_application(
        initialization_kwargs={"db_config": False, "output": True},
    )

    output_directory = app_context.io_handler.get_output_directory()
    production_descriptors = parse_production_arguments(app_context.args["production"])

    comparison_level = app_context.args["comparison_level"]
    if comparison_level == "events":
        metrics_per_production = collect_production_metrics(production_descriptors)
        plot_event_level_production_comparison.plot(
            metrics_per_production,
            output_path=output_directory,
        )
    else:
        raise NotImplementedError(f"Comparison level '{comparison_level}' is not implemented yet.")


if __name__ == "__main__":
    main()
