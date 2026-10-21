#!/usr/bin/python3
"""Compare exported science-test result artifacts."""

import math
from pathlib import Path

from matplotlib.backends.backend_pdf import PdfPages

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.science_tests.camera_efficiency_results import (
    compare_manifests,
    load_manifest,
    write_json,
)
from simtools.visualization.plot_science_comparison import (
    _plot_efficiency_components_for_channel,
    plot_comparison,
    plot_wavelength_comparisons,
)

_TOLERANCE_ARGUMENT = "tolerance"
_TELESCOPES_ARGUMENT = "telescopes"
_COMPARE_BY_ARGUMENT = "compare_by"
_REQUIRE_WAVELENGTH_TABLES_ARGUMENT = "require_wavelength_tables"
_APPEND_ACTION = "append"
_BASELINE_ARGUMENT = "baseline_path"
_CANDIDATE_ARGUMENT = "candidate_path"
_BASELINE_LABEL_ARGUMENT = "baseline_label"
_CANDIDATE_LABEL_ARGUMENT = "candidate_label"
_MANIFEST_FILE_NAME = "telescope_throughput_manifest.json"
_BASELINE_KEY = "baseline"
_CANDIDATE_KEY = "candidate"
_LABEL_KEY = "label"
_CASE_KEY = "case"
_SOFTWARE_KEY = "software"
_TELESCOPE_KEY = "telescope"
_CHANGES_KEY = "changes"
_DIAGNOSTICS_KEY = "diagnostics"
_PLOTS_KEY = "plots"
_STATUS_KEY = "status"
_INCOMPLETE_STATUS = "incomplete"
_SCHEMA_VERSION_KEY = "schema_version"
_REPORT_PDF = "report.pdf"
_WAVELENGTH_TABLE_SUFFIX = "_wavelength_table"
_LEGACY_WAVELENGTH_TABLE_KEY = "wavelength_table"
_SHOWER_CHANNEL = "shower"
_WAVELENGTH_CHANNELS = (_SHOWER_CHANNEL, "muon", "nsb")

_ARGUMENTS = (
    cli.ArgumentDefinition(
        _BASELINE_ARGUMENT,
        type=str,
        required=True,
        help="Camera-efficiency manifest, legacy summary, or artifact directory for baseline.",
    ),
    cli.ArgumentDefinition(
        _CANDIDATE_ARGUMENT,
        type=str,
        required=True,
        help="Camera-efficiency manifest, legacy summary, or artifact directory for candidate.",
    ),
    cli.ArgumentDefinition(_BASELINE_LABEL_ARGUMENT, type=str, default=_BASELINE_KEY),
    cli.ArgumentDefinition(_CANDIDATE_LABEL_ARGUMENT, type=str, default=_CANDIDATE_KEY),
    cli.ArgumentDefinition(
        _TELESCOPES_ARGUMENT,
        type=str,
        nargs="+",
        default=[],
        help="Restrict a directory comparison to these telescope names.",
    ),
    cli.ArgumentDefinition(
        _COMPARE_BY_ARGUMENT,
        action=_APPEND_ACTION,
        default=[],
        help="Exact manifest field path allowed to differ; repeat for each field.",
    ),
    cli.ArgumentDefinition(
        _TOLERANCE_ARGUMENT,
        action=_APPEND_ACTION,
        nargs="+",
        default=[],
        help="Advisory relative tolerances as METRIC=VALUE; provide one or more values.",
    ),
    cli.ArgumentDefinition(
        _REQUIRE_WAVELENGTH_TABLES_ARGUMENT,
        action="store_true",
        help="Require wavelength ECSV tables for both cases.",
    ),
)

APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(*_ARGUMENTS, *cli.OUTPUT_PATH_ARGUMENTS),
    initialize_output=True,
    initialize_model_reader=False,
    excluded_standard_arguments=("test", "ignore_existing_parameter_version"),
)


def main():
    """Compare two camera-efficiency result artifacts and write plots and JSON."""
    context = APPLICATION.start()
    args = context.args
    output = context.io_handler.get_output_directory()
    try:
        baselines = _load_manifest_set(args[_BASELINE_ARGUMENT])
        candidates = _load_manifest_set(args[_CANDIDATE_ARGUMENT])
        tolerances = _parse_tolerances(args.get(_TOLERANCE_ARGUMENT, []))
        comparison = _compare_manifest_sets(baselines, candidates, args, tolerances, output)
    except (OSError, ValueError) as exc:
        comparison = {
            _SCHEMA_VERSION_KEY: 1,
            _STATUS_KEY: _INCOMPLETE_STATUS,
            _BASELINE_KEY: {_LABEL_KEY: args[_BASELINE_LABEL_ARGUMENT]},
            _CANDIDATE_KEY: {_LABEL_KEY: args[_CANDIDATE_LABEL_ARGUMENT]},
            _CHANGES_KEY: {},
            _DIAGNOSTICS_KEY: [str(exc)],
            _PLOTS_KEY: {},
        }
    write_json(output / "science-comparison.json", comparison)


def _parse_tolerances(values):
    """Parse repeated METRIC=VALUE advisory tolerances."""
    tolerances = {}
    flattened_values = [
        item for value in values for item in (value if isinstance(value, list) else [value])
    ]
    for item in flattened_values:
        metric, separator, raw_value = item.partition("=")
        if not separator or metric not in {
            "reflectivity",
            "cam_eff",
            "tel_eff",
            "tot_sens",
            "nsb_rate",
            "nsb_ref",
            "muon_frac",
        }:
            raise ValueError(f"Invalid metric tolerance: {item}")
        value = float(raw_value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Tolerance must be finite and non-negative: {item}")
        tolerances[metric] = value
    return tolerances


def _load_manifest_set(path):
    """Load a single manifest or all telescope manifests below a result directory."""
    root = Path(path)
    direct = root if root.is_file() else root / _MANIFEST_FILE_NAME
    manifests = ([direct] if direct.is_file() else []) + sorted(
        root.glob(f"*/{_MANIFEST_FILE_NAME}")
    )
    if not manifests:
        manifests = [root]
    loaded = {}
    timestamps = {}
    for manifest_path in manifests:
        manifest = load_manifest(manifest_path)
        telescope = manifest.get(_CASE_KEY, {}).get(_TELESCOPE_KEY) or manifest_path.parent.name
        if not isinstance(telescope, str):
            raise ValueError(f"Manifest telescope must be a name: {manifest_path}")
        timestamp = manifest_path.stat().st_mtime_ns
        if telescope not in timestamps or timestamp >= timestamps[telescope]:
            loaded[telescope] = manifest
            timestamps[telescope] = timestamp
    return loaded


def _compare_manifest_sets(baselines, candidates, args, tolerances, output):
    """Compare every telescope pair and write all plot pages into one PDF."""
    telescopes = args.get(_TELESCOPES_ARGUMENT) or sorted(set(baselines) | set(candidates))
    missing = sorted(set(telescopes) - set(baselines) | set(telescopes) - set(candidates))
    if missing:
        raise ValueError(f"Baseline and candidate telescope sets differ: {', '.join(missing)}.")
    comparisons = {}
    with PdfPages(output / _REPORT_PDF) as report:
        for telescope in telescopes:
            baseline, candidate = baselines[telescope], candidates[telescope]
            current = compare_manifests(
                baseline, candidate, args.get(_COMPARE_BY_ARGUMENT), tolerances
            )
            if current[_CHANGES_KEY]:
                plot_comparison(
                    current[_CHANGES_KEY],
                    _plot_identity(baseline, args[_BASELINE_LABEL_ARGUMENT]),
                    _plot_identity(candidate, args[_CANDIDATE_LABEL_ARGUMENT]),
                    output,
                    figure_format=(),
                    pdf_pages=report,
                )
            _plot_tables(baseline, candidate, current, output, args, report)
            comparisons[telescope] = current
    if len(comparisons) == 1:
        result = next(iter(comparisons.values()))
    else:
        statuses = {item[_STATUS_KEY] for item in comparisons.values()}
        result = {
            _SCHEMA_VERSION_KEY: 1,
            _STATUS_KEY: _INCOMPLETE_STATUS if _INCOMPLETE_STATUS in statuses else "complete",
            _BASELINE_KEY: {_LABEL_KEY: args[_BASELINE_LABEL_ARGUMENT]},
            _CANDIDATE_KEY: {_LABEL_KEY: args[_CANDIDATE_LABEL_ARGUMENT]},
            _CHANGES_KEY: {},
            _DIAGNOSTICS_KEY: [
                f"{telescope}: {diagnostic}"
                for telescope, item in comparisons.items()
                for diagnostic in item[_DIAGNOSTICS_KEY]
            ],
            "comparisons": comparisons,
            _PLOTS_KEY: {},
        }
    result[_PLOTS_KEY]["report"] = str(output / _REPORT_PDF)
    return result


def _plot_identity(manifest, label):
    """Return the provenance shown in a comparison plot."""
    return {
        _LABEL_KEY: label,
        _CASE_KEY: manifest.get(_CASE_KEY, {}),
        _SOFTWARE_KEY: manifest.get(_SOFTWARE_KEY, {}),
    }


def _plot_tables(baseline, candidate, comparison, output, args, report):
    """Add spectrum and component pages using the same artifacts and plot labels."""
    baseline_paths = _wavelength_paths(baseline)
    candidate_paths = _wavelength_paths(candidate)
    required = args.get(_REQUIRE_WAVELENGTH_TABLES_ARGUMENT, False)
    missing = [
        channel
        for channel in _WAVELENGTH_CHANNELS
        if not baseline_paths[channel] or not candidate_paths[channel]
    ]
    if missing:
        requirement = "Required " if required else ""
        _record_plot_diagnostic(
            comparison,
            f"{requirement}wavelength tables are missing for: {', '.join(missing)}.",
            required,
        )
    plots = []
    if len(missing) < len(_WAVELENGTH_CHANNELS):
        plots.append((plot_wavelength_comparisons, baseline_paths, candidate_paths))
    for channel in _WAVELENGTH_CHANNELS:
        if channel in missing:
            continue
        plots.append(
            (
                _plot_efficiency_components_for_channel,
                baseline_paths[channel],
                candidate_paths[channel],
                channel,
            )
        )
    for entry in plots:
        plot, before, after, *channel = entry
        try:
            _, diagnostic = plot(
                before,
                after,
                output,
                baseline_label=args[_BASELINE_LABEL_ARGUMENT],
                candidate_label=args[_CANDIDATE_LABEL_ARGUMENT],
                pdf_pages=report,
                telescope=baseline.get(_CASE_KEY, {}).get(_TELESCOPE_KEY),
                **({"channel": channel[0]} if channel else {}),
            )
        except (OSError, ValueError) as exc:
            diagnostic = f"Cannot plot efficiency tables: {exc}"
        _record_plot_diagnostic(comparison, diagnostic, required)


def _record_plot_diagnostic(comparison, diagnostic, required):
    """Record table problems without discarding the scalar comparison."""
    if diagnostic:
        comparison[_DIAGNOSTICS_KEY].append(diagnostic)
        if required:
            comparison[_STATUS_KEY] = _INCOMPLETE_STATUS


def _wavelength_paths(manifest):
    """Resolve spectrum tables, including the legacy shower-table artifact."""
    paths = {
        channel: _artifact_path(manifest, f"{channel}{_WAVELENGTH_TABLE_SUFFIX}")
        for channel in _WAVELENGTH_CHANNELS
    }
    if not paths[_SHOWER_CHANNEL]:
        paths[_SHOWER_CHANNEL] = _artifact_path(manifest, _LEGACY_WAVELENGTH_TABLE_KEY)
    return paths


def _artifact_path(manifest, name):
    """Resolve a declared result artifact relative to its source manifest."""
    artifacts = manifest.get("artifacts", {})
    artifact = artifacts.get(name)
    if artifact is None and name.endswith(_WAVELENGTH_TABLE_SUFFIX):
        channel = name.removesuffix(_WAVELENGTH_TABLE_SUFFIX)
        artifact = next(
            (path for path in artifacts.get("wavelength_tables", []) if channel in path.lower()),
            None,
        )
    root = manifest.get("_root")
    if artifact is None or root is None:
        return None
    path = Path(root) / artifact
    return path if path.is_file() else None


if __name__ == "__main__":
    main()
