#!/usr/bin/python3
"""Compare telescope-throughput results of two releases.

Reads the combined summaries and wavelength tables written by
``simtools-validate-telescope-throughput`` and compares them telescope by telescope
without rerunning any calculation. Writes ``science-comparison.json`` and ``report.pdf``
to the output path.
"""

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.science_tests import telescope_throughput as throughput
from simtools.visualization import plot_science_comparison as plots
from simtools.visualization.visualize import pdf_report

_BASELINE_PATH = "baseline_path"
_CANDIDATE_PATH = "candidate_path"
_BASELINE_LABEL = "baseline_label"
_CANDIDATE_LABEL = "candidate_label"
_TELESCOPES = "telescopes"
_COMPARE_BY = "compare_by"
_TOLERANCE = "tolerance"
_REQUIRE_TABLES = "require_wavelength_tables"
_APPEND = "append"
_PATH_HELP = "Combined summary, telescope result directory, or directory of telescope results"
_REPORT_FILE = "report.pdf"
_STATUS = "status"
_DIAGNOSTICS = "diagnostics"
_COMPARISONS = "comparisons"
_LABEL = "label"

APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        cli.ArgumentDefinition(_BASELINE_PATH, type=str, required=True, help=f"{_PATH_HELP}."),
        cli.ArgumentDefinition(_CANDIDATE_PATH, type=str, required=True, help=f"{_PATH_HELP}."),
        cli.ArgumentDefinition(_BASELINE_LABEL, type=str, default=throughput.BASELINE),
        cli.ArgumentDefinition(_CANDIDATE_LABEL, type=str, default=throughput.CANDIDATE),
        cli.ArgumentDefinition(
            _TELESCOPES,
            type=str,
            nargs="+",
            default=[],
            help="Restrict the comparison to these telescopes.",
        ),
        cli.ArgumentDefinition(
            _COMPARE_BY,
            action=_APPEND,
            default=[],
            help="Summary metadata key allowed to differ; repeat for each key.",
        ),
        cli.ArgumentDefinition(
            _TOLERANCE,
            action=_APPEND,
            nargs="+",
            default=[],
            help="Advisory relative tolerances as METRIC=VALUE.",
        ),
        cli.ArgumentDefinition(
            _REQUIRE_TABLES,
            action="store_true",
            help="Mark the comparison incomplete if wavelength tables cannot be compared.",
        ),
        *cli.OUTPUT_PATH_ARGUMENTS,
    ),
    initialize_output=True,
    initialize_model_reader=False,
    excluded_standard_arguments=("test",),
)


def main():
    """Compare baseline and candidate results and write the report and JSON summary."""
    context = APPLICATION.start()
    args = context.args
    output = context.io_handler.get_output_directory()
    result = {
        _STATUS: throughput.INCOMPLETE,
        throughput.BASELINE: {_LABEL: args[_BASELINE_LABEL]},
        throughput.CANDIDATE: {_LABEL: args[_CANDIDATE_LABEL]},
        _COMPARISONS: {},
        _DIAGNOSTICS: [],
        "report": _REPORT_FILE,
    }
    try:
        baselines = throughput.load_results(args[_BASELINE_PATH])
        candidates = throughput.load_results(args[_CANDIDATE_PATH])
        tolerances = throughput.parse_tolerances(args[_TOLERANCE])
        with pdf_report(output / _REPORT_FILE) as pdf_pages:
            for telescope in args[_TELESCOPES] or sorted(set(baselines) | set(candidates)):
                result[_COMPARISONS][telescope] = _compare_telescope(
                    telescope, baselines, candidates, tolerances, args, pdf_pages
                )
    except (OSError, ValueError) as exc:
        result[_DIAGNOSTICS].append(str(exc))
    result[_DIAGNOSTICS].extend(
        f"{telescope}: {diagnostic}"
        for telescope, comparison in result[_COMPARISONS].items()
        for diagnostic in comparison[_DIAGNOSTICS]
    )
    if result[_COMPARISONS] and not result[_DIAGNOSTICS]:
        result[_STATUS] = throughput.COMPLETE
    throughput.write_json(output / "science-comparison.json", result)


def _compare_telescope(telescope, baselines, candidates, tolerances, args, pdf_pages):
    """Compare and plot one telescope present in both result sets."""
    if telescope not in baselines or telescope not in candidates:
        return {_STATUS: throughput.INCOMPLETE, _DIAGNOSTICS: ["Missing baseline or candidate."]}
    baseline, candidate = baselines[telescope], candidates[telescope]
    comparison = throughput.compare_results(baseline, candidate, tolerances, args[_COMPARE_BY])
    labels = (args[_BASELINE_LABEL], args[_CANDIDATE_LABEL])
    title = f"{telescope}: {labels[0]} vs {labels[1]}"
    if comparison[throughput.CHANGES]:
        plots.plot_metric_changes(comparison[throughput.CHANGES], title, pdf_pages)
    table_diagnostics = plots.plot_spectra(
        baseline[throughput.TABLES], candidate[throughput.TABLES], title, labels, pdf_pages
    )
    channels = sorted(set(baseline[throughput.TABLES]) | set(candidate[throughput.TABLES]))
    for channel in channels:
        if channel in baseline[throughput.TABLES] and channel in candidate[throughput.TABLES]:
            diagnostic = plots.plot_components(
                baseline[throughput.TABLES][channel],
                candidate[throughput.TABLES][channel],
                f"{title} ({channel})",
                labels,
                pdf_pages,
            )
        else:
            diagnostic = f"Wavelength table for {channel} is missing in baseline or candidate."
        if diagnostic:
            table_diagnostics.append(diagnostic)
    if not channels:
        table_diagnostics.append("No wavelength tables found.")
    if args[_REQUIRE_TABLES]:
        comparison[_DIAGNOSTICS].extend(table_diagnostics)
        if table_diagnostics:
            comparison[_STATUS] = throughput.INCOMPLETE
    else:
        comparison["table_diagnostics"] = table_diagnostics
    return comparison


if __name__ == "__main__":
    main()
