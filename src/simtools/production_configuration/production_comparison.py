"""Compare trigger-histogram products from simulation productions."""

import csv
import json
import logging
from collections import Counter
from pathlib import Path

from simtools.production_configuration.production_file_selection import (
    check_manifest,
    discover_product_manifests,
    filter_manifests,
    normalize_for_comparison,
    stable_configuration_hash,
)
from simtools.production_configuration.trigger_histograms import _validate_derived_products
from simtools.sim_events.production_comparison import (
    ProductionDescriptor,
    collect_production_metrics,
    parse_production_arguments,
)
from simtools.visualization import plot_event_level_production_comparison
from simtools.visualization.plot_event_level_production_comparison import _pdf_report, _PdfReport

_COMPARISON_INVENTORY_FILE = "comparison_inventory.json"
_CONFIGURATION_KEY = "configuration"
_OUTPUT_PATH_KEY = "output_path"
_REPORT_MODE_KEY = "report_mode"
_SELECT_KEY = "select"
_ENCODING = "utf-8"

BASELINE_KEY = "baseline"
COMPARISON_SETS_KEY = "comparison_sets"
PLOT_STATISTICS_KEY = "plot_statistics"
COMPARISONS_KEY = "comparisons"

COMPARISON_CONTEXT_KEY = "comparison_context"

ARRAY_LAYOUT_KEY = "array_layout_name"
REPORT_PAGE_COLUMN = "report_page"

_logger = logging.getLogger(__name__)


class _ProductionPairingError(ValueError):
    """Report unmatched metadata while retaining the pairs that can be compared."""

    def __init__(self, message, descriptor_pairs):
        super().__init__(message)
        self.descriptor_pairs = descriptor_pairs


def write_production_comparison(args_dict, output_directory):
    """Compare selected trigger-histogram productions and write their statistics.

    Parameters
    ----------
    args_dict : dict
        Application arguments, including either explicit production descriptors or metadata paths.
    output_directory : pathlib.Path
        Directory in which comparison products are written.
    """
    array_layout_names = args_dict.get(ARRAY_LAYOUT_KEY) or [None]
    if args_dict.get("baseline_path"):
        _write_metadata_comparisons(args_dict, output_directory, array_layout_names)
        return

    production_descriptors = parse_production_arguments(args_dict["production"])
    statistics_files = _write_array_layout_comparisons(
        production_descriptors,
        args_dict,
        output_directory,
        array_layout_names,
    )
    _write_comparison_summary(statistics_files, output_directory)


def _write_metadata_comparisons(args_dict, output_directory, array_layout_names):
    """Write comparisons for trigger-histogram metadata pairs."""
    pairing_error = None
    try:
        descriptor_pairs = _production_descriptor_pairs_from_metadata(args_dict)
    except _ProductionPairingError as exc:
        if not exc.descriptor_pairs:
            raise
        descriptor_pairs = exc.descriptor_pairs
        pairing_error = exc
    output_stems = [
        _comparison_pair_output_stem(production_descriptors)
        for _, production_descriptors in descriptor_pairs
    ]
    stem_counts = Counter(output_stems)
    statistics_files = []
    comparison_labels = {}
    report_pages = {}
    with _pdf_report(
        output_directory / "trigger-report.pdf", args_dict.get(_REPORT_MODE_KEY) == "compact"
    ) as compact_figures:
        for (pairing_key, production_descriptors), output_stem in zip(
            descriptor_pairs, output_stems
        ):
            pair_output_directory = _metadata_pair_output_directory(
                output_directory,
                pairing_key,
                output_stem,
                stem_counts,
            )
            pair_statistics = _write_array_layout_comparisons(
                production_descriptors,
                args_dict,
                pair_output_directory,
                array_layout_names,
                compact_figures=compact_figures,
                comparison_context=pairing_key,
                report_pages=report_pages if compact_figures is not None else None,
            )
            statistics_files.extend(pair_statistics or [])
            if array_layout_names == [None]:
                comparison_labels.update(
                    dict.fromkeys(pair_statistics or [], (pair_output_directory.name, ""))
                )
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / _COMPARISON_INVENTORY_FILE).write_text(
        json.dumps(
            {
                "format_version": 1,
                "matched_pair_count": len(descriptor_pairs),
                "pairing_error_count": int(pairing_error is not None),
            },
            indent=2,
        )
        + "\n",
        encoding=_ENCODING,
    )
    if pairing_error is not None:
        _logger.warning(str(pairing_error))
    if comparison_labels:
        _write_comparison_summary(
            statistics_files,
            output_directory,
            report_pages,
            comparison_labels=comparison_labels,
        )
    else:
        _write_comparison_summary(statistics_files, output_directory, report_pages)


def _metadata_pair_output_directory(output_directory, pairing_key, output_stem, stem_counts):
    """Return the output directory for one metadata comparison pair."""
    if stem_counts[output_stem] > 1:
        hash_key = pairing_key.get("pairing_key", pairing_key)
        output_stem = f"{output_stem}-{stable_configuration_hash(hash_key)}"
    return output_directory / output_stem


def _write_array_layout_comparisons(
    production_descriptors,
    args_dict,
    output_directory,
    array_layout_names,
    compact_figures=None,
    comparison_context=None,
    report_pages=None,
):
    """Write comparison products for all selected array layouts."""
    statistics_files = []
    for array_layout_name in array_layout_names:
        first_figure = len(compact_figures) if report_pages is not None else None
        statistics_file = _compare_array_layout(
            production_descriptors,
            args_dict,
            array_layout_name,
            output_directory,
            compact_figures,
            comparison_context,
        )
        statistics_files.append(statistics_file)
        if report_pages is not None:
            report_pages[statistics_file] = _figure_pages(compact_figures, first_figure)
    return statistics_files


def _figure_pages(figures, first_figure):
    """Map compact plot names to their one-based page numbers in the combined PDF."""
    pages = {}
    if isinstance(figures, _PdfReport):
        labels = figures.page_labels[first_figure:]
    else:
        labels = [figure.get_label() for figure in figures[first_figure:]]
    for page_number, plot_name in enumerate(labels, start=first_figure + 1):
        if plot_name:
            pages.setdefault(plot_name, []).append(page_number)
    return pages


def _compare_array_layout(
    production_descriptors,
    args_dict,
    array_layout_name,
    output_directory,
    compact_figures=None,
    comparison_context=None,
):
    """Compare one selected array layout and write its statistics file."""
    metrics_per_production = collect_production_metrics(
        production_descriptors,
        array_names=array_layout_name,
    )
    plot_args = {
        _OUTPUT_PATH_KEY: output_directory,
        ARRAY_LAYOUT_KEY: array_layout_name,
        "figure_format": args_dict.get("figure_format"),
        _REPORT_MODE_KEY: args_dict.get(_REPORT_MODE_KEY, "diagnostic"),
        "comparison_title": _comparison_title(comparison_context, array_layout_name),
    }
    if compact_figures is not None:
        plot_args["compact_figures"] = compact_figures
    if comparison_context is not None:
        plot_args[COMPARISON_CONTEXT_KEY] = comparison_context
    return plot_event_level_production_comparison.plot(
        metrics_per_production,
        **plot_args,
    )


def _comparison_title(comparison_context, array_layout_name):
    """Format pair configuration values for plot headings."""
    if not isinstance(comparison_context, dict) or "configurations" not in comparison_context:
        return f"Layout: {array_layout_name}" if array_layout_name else None

    configurations = comparison_context["configurations"]
    labels = list(configurations)
    baseline = configurations[labels[0]]
    candidate = configurations[labels[-1]]
    identity_fields = []
    for key, label in (("primary", "Primary"), ("site", "Site")):
        value = _shared_configuration_value(baseline, candidate, key)
        if value is not None:
            identity_fields.append(f"{label}: {value}")

    energy = _format_energy_range(baseline, candidate, labels)
    geometry = _format_pair_value("zenith_angle", "Zenith", baseline, candidate, labels, unit="deg")
    azimuth = _format_pair_value(
        "azimuth_angle", "Azimuth", baseline, candidate, labels, unit="deg"
    )
    interaction = _shared_configuration_value(baseline, candidate, "corsika_he_interaction")
    first_line = [f"Layout: {array_layout_name}" if array_layout_name else None]
    first_line.extend(identity_fields)
    first_line.extend(value for value in (energy, geometry) if value)
    second_line = [value for value in (azimuth,) if value]
    if interaction is not None:
        second_line.append(f"Interaction model: {interaction}")
    first_line_text = " | ".join(value for value in first_line if value)
    return first_line_text + "\n" + " | ".join(second_line)


def _format_energy_range(baseline, candidate, labels):
    """Format matching energy limits as one readable energy range."""
    baseline_min = _format_configuration_value(baseline.get("energy_min"), "energy")
    baseline_max = _format_configuration_value(baseline.get("energy_max"), "energy")
    candidate_min = _format_configuration_value(candidate.get("energy_min"), "energy")
    candidate_max = _format_configuration_value(candidate.get("energy_max"), "energy")
    if baseline_min == "None" and baseline_max == "None":
        return None
    baseline_range = _join_energy_limits(baseline_min, baseline_max)
    candidate_range = _join_energy_limits(candidate_min, candidate_max)
    if baseline_range == candidate_range:
        return f"Energy: {baseline_range}"
    return f"Energy: {labels[0]} {baseline_range}; {labels[-1]} {candidate_range}"


def _join_energy_limits(minimum, maximum):
    """Join minimum and maximum energies without repeating identical limits."""
    if minimum == maximum:
        return minimum
    return f"{minimum} to {maximum}"


def _format_pair_value(key, label, baseline, candidate, labels, unit=None):
    """Format one configuration value, labeling both sides when they differ."""
    baseline_value = _format_configuration_value(baseline.get(key), unit)
    candidate_value = _format_configuration_value(candidate.get(key), unit)
    if baseline_value == "None" and candidate_value == "None":
        return None
    if baseline_value == candidate_value:
        return f"{label}: {baseline_value}"
    return f"{label}: {labels[0]} {baseline_value} -> {labels[-1]} {candidate_value}"


def _shared_configuration_value(baseline, candidate, key):
    """Return a compact value only when both pair configurations agree."""
    baseline_value = _format_configuration_value(baseline.get(key))
    candidate_value = _format_configuration_value(candidate.get(key))
    if baseline_value == candidate_value and baseline_value != "None":
        return baseline_value
    return None


def _format_configuration_value(value, quantity=None):
    """Format manifest values, including quantities, for a compact plot heading."""
    if isinstance(value, dict) and set(value) == {"value", "unit"}:
        magnitude = value["value"]
        value_unit = value["unit"]
        if quantity == "energy" and str(value_unit).lower() == "gev" and float(magnitude) >= 1000:
            return f"{float(magnitude) / 1000:g} TeV"
        return (
            f"{magnitude:g} {value_unit}"
            if isinstance(magnitude, int | float)
            else f"{magnitude} {value_unit}"
        )
    if quantity and isinstance(value, int | float):
        return f"{value:g} {quantity}"
    return str(value)


def _write_comparison_summary(
    statistics_files, output_directory, report_pages=None, comparison_labels=None
):
    """Combine per-pair statistics into one summary and remove the individual files."""
    baseline = None
    comparison_sets = {}
    compact_statistics = []
    metric_scopes = {}
    for statistics_file in statistics_files:
        statistics = json.loads(statistics_file.read_text(encoding=_ENCODING))
        baseline = _add_production_summary(baseline, statistics[BASELINE_KEY])
        comparison_sets = _add_comparison_sets(comparison_sets, statistics[COMPARISON_SETS_KEY])
        label = _comparison_coordinates(
            statistics_file, output_directory, (comparison_labels or {}).get(statistics_file)
        )
        pair, layout = label
        metric_scopes.setdefault(pair, {})[layout] = (
            statistics[PLOT_STATISTICS_KEY]
            .get("telescope_participation_fraction", {})
            .get("telescope_types", [])
        )
        compact_statistics.extend(
            _compact_statistic_rows(
                statistics_file,
                output_directory,
                statistics,
                (report_pages or {}).get(statistics_file, {}),
                label,
            )
        )
        statistics_file.unlink()
        _remove_empty_comparison_directories(statistics_file.parent, output_directory)

    summary = {
        "format_version": 2,
        BASELINE_KEY: baseline
        or {
            "label": None,
            "simulated_event_count": 0,
            "triggered_event_count": 0,
        },
        COMPARISON_SETS_KEY: list(comparison_sets.values()),
        PLOT_STATISTICS_KEY: {},
        COMPARISONS_KEY: [],
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "comparison_statistics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding=_ENCODING
    )
    _write_compact_statistics_csv(compact_statistics, output_directory)
    inventory = output_directory / _COMPARISON_INVENTORY_FILE
    if inventory.is_file():
        payload = json.loads(inventory.read_text(encoding=_ENCODING))
        payload["metric_scopes"] = metric_scopes
        inventory.write_text(json.dumps(payload, indent=2) + "\n", encoding=_ENCODING)


def _compact_statistic_rows(
    statistics_file, output_directory, statistics, report_pages=None, comparison_label=None
):
    """Extract scalar comparison results while excluding histogram arrays."""
    pair, array_layout = _comparison_coordinates(
        statistics_file, output_directory, comparison_label
    )
    report_pages = report_pages or {}
    rows = []
    for plot_name, plot_statistics in statistics[PLOT_STATISTICS_KEY].items():
        rows.extend(
            _compact_plot_statistic_rows(
                pair,
                array_layout,
                plot_name,
                plot_statistics,
                report_pages.get(plot_name, []),
            )
        )
    return rows


def _comparison_coordinates(statistics_file, output_directory, comparison_label):
    """Identify one pair/layout consistently for scalar rows and coverage checks."""
    path_parts = statistics_file.relative_to(output_directory).parts
    if comparison_label is not None:
        pair, array_layout = comparison_label
        return pair, array_layout
    if len(path_parts) > 2:
        return path_parts[0], path_parts[1]
    if len(path_parts) == 1:
        return "", ""
    return "", path_parts[0]


def _compact_plot_statistic_rows(pair, array_layout, plot_name, plot_statistics, report_pages):
    """Build compact statistic rows for one named plot and its rendered page."""
    scope, telescope_type = _statistic_scope(plot_name)
    rows = []
    for comparison in plot_statistics.get("comparisons", []):
        rows.extend(
            _compact_comparison_rows(
                pair,
                array_layout,
                plot_name,
                scope,
                telescope_type,
                comparison,
                report_pages,
            )
        )
    return rows


def _compact_comparison_rows(
    pair, array_layout, plot_name, scope, telescope_type, comparison, report_pages
):
    """Build scalar rows for the population statistics in one comparison."""
    rows = []
    for population in ("simulated", "triggered"):
        result = comparison.get(population)
        if result is not None:
            rows.append(
                _compact_statistic_row(
                    pair,
                    array_layout,
                    plot_name,
                    scope,
                    telescope_type,
                    population,
                    result,
                    report_pages,
                )
            )
    if "value" in comparison:
        rows.append(
            _compact_statistic_row(
                pair,
                array_layout,
                plot_name,
                scope,
                telescope_type,
                "triggered",
                comparison,
                report_pages,
            )
        )
    return rows


def _statistic_scope(plot_name):
    """Map a plot name to whole-array or telescope-type scope."""
    suffix = plot_name.rsplit("_", 1)[-1]
    return (
        ("telescope_type", suffix)
        if suffix.startswith(("LST", "MST", "SST"))
        else ("whole_array", "")
    )


def _compact_statistic_row(
    pair, array_layout, plot_name, scope, telescope_type, population, result, report_pages
):
    """Build one scalar result row for machine-readable acceptance checks."""
    return {
        "pair": pair,
        "array_layout": array_layout,
        "plot": plot_name,
        REPORT_PAGE_COLUMN: ", ".join(str(page) for page in report_pages),
        "scope": scope,
        "telescope_type": telescope_type,
        "population": population,
        "metric": result.get("metric"),
        "value": result.get("value"),
        "normalized_value": result.get("normalized_value"),
        "valid": result.get("valid"),
        "reason": result.get("reason"),
        "baseline_count": result.get("baseline_total"),
        "candidate_count": result.get("candidate_total"),
    }


def _write_compact_statistics_csv(rows, output_directory):
    """Write scalar comparison results without count arrays or expanded JSON."""
    fields = [
        "pair",
        "array_layout",
        "plot",
        REPORT_PAGE_COLUMN,
        "scope",
        "telescope_type",
        "population",
        "metric",
        "value",
        "normalized_value",
        "valid",
        "reason",
        "baseline_count",
        "candidate_count",
    ]
    output_directory.mkdir(parents=True, exist_ok=True)
    with (output_directory / "comparison_metrics.csv").open(
        "w", encoding=_ENCODING, newline=""
    ) as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _remove_empty_comparison_directories(directory, output_directory):
    """Remove emptied per-pair directories without touching plots or the report root."""
    while directory != output_directory and directory.is_relative_to(output_directory):
        try:
            directory.rmdir()
        except OSError:
            break
        directory = directory.parent


def _add_comparison_sets(comparison_sets, productions):
    """Merge event counts for each candidate production label."""
    for production in productions:
        label = production["label"]
        comparison_sets[label] = _add_production_summary(comparison_sets.get(label), production)
    return comparison_sets


def _add_production_summary(total, production):
    """Add event counts from one production summary."""
    if total is None:
        return dict(production)
    total["simulated_event_count"] += production["simulated_event_count"]
    total["triggered_event_count"] += production["triggered_event_count"]
    return total


def _production_descriptor_pairs_from_metadata(args_dict):
    """Build one descriptor pair per matched trigger-histogram configuration."""
    baseline = _selected_trigger_histogram_manifests(
        args_dict["baseline_path"], args_dict.get("select")
    )
    candidate = _selected_trigger_histogram_manifests(
        args_dict["candidate_path"], args_dict.get("select")
    )
    compare_by = {field.removeprefix("configuration.") for field in args_dict.get("compare_by", [])}
    baseline_by_key = _unique_manifests_by_pairing_key(baseline, compare_by, BASELINE_KEY)
    candidate_by_key = _unique_manifests_by_pairing_key(candidate, compare_by, "candidate")

    missing_candidates = sorted(set(baseline_by_key) - set(candidate_by_key))
    missing_baselines = sorted(set(candidate_by_key) - set(baseline_by_key))
    descriptor_pairs = _matched_descriptor_pairs(
        baseline_by_key,
        candidate_by_key,
        args_dict.get("baseline_label", BASELINE_KEY),
        args_dict.get("candidate_label", "candidate"),
    )
    if missing_candidates or missing_baselines:
        raise _ProductionPairingError(
            "Trigger-histogram metadata pairing failed: "
            f"missing candidates={len(missing_candidates)}, "
            f"missing baselines={len(missing_baselines)}.",
            descriptor_pairs,
        )

    return descriptor_pairs


def _matched_descriptor_pairs(baseline_by_key, candidate_by_key, baseline_label, candidate_label):
    """Build descriptor pairs for configurations present in both productions."""
    return [
        (
            {
                "pairing_key": key,
                "configurations": {
                    baseline_label: baseline_by_key[key].data.get(_CONFIGURATION_KEY, {}),
                    candidate_label: candidate_by_key[key].data.get(_CONFIGURATION_KEY, {}),
                },
            },
            [
                ProductionDescriptor(
                    label=baseline_label,
                    input_files=[str(_single_trigger_histogram_file(baseline_by_key[key]))],
                ),
                ProductionDescriptor(
                    label=candidate_label,
                    input_files=[str(_single_trigger_histogram_file(candidate_by_key[key]))],
                ),
            ],
        )
        for key in sorted(set(baseline_by_key) & set(candidate_by_key), key=str)
    ]


def _comparison_pair_output_stem(production_descriptors):
    """Return a readable directory stem from the paired histogram filenames."""
    input_stems = [Path(descriptor.input_files[0]).stem for descriptor in production_descriptors]
    if len(set(input_stems)) == 1:
        return input_stems[0]
    return "-vs-".join(input_stems)


def _selected_trigger_histogram_manifests(path, selections):
    """Return checked trigger-histogram manifests matching selections."""
    manifests = discover_product_manifests(path, "trigger_histograms")
    selected = filter_manifests(manifests, selections or [])
    if not selected:
        raise ValueError(f"No trigger-histogram metadata files matched in {path}.")
    for manifest in selected:
        check_manifest(manifest)
    return selected


def _unique_manifests_by_pairing_key(manifests, compare_by, label):
    """Return manifests keyed by comparable simulation configuration."""
    keyed = {}
    for manifest in manifests:
        key = _pairing_key(manifest.data, compare_by)
        if key in keyed:
            raise ValueError(
                f"More than one {label} trigger-histogram file matches configuration {key}."
            )
        keyed[key] = manifest
    return keyed


def _pairing_key(manifest, compare_by):
    """Return normalized configuration fields used for baseline/candidate pairing."""
    configuration = manifest.get(_CONFIGURATION_KEY, {})
    paired_configuration = {
        key: value for key, value in configuration.items() if key not in compare_by
    }
    layout = paired_configuration.get(ARRAY_LAYOUT_KEY)
    if isinstance(layout, (list, tuple)) and len(layout) == 1:
        paired_configuration[ARRAY_LAYOUT_KEY] = layout[0]
    return normalize_for_comparison(
        {
            _CONFIGURATION_KEY: paired_configuration,
            "histogram_settings": manifest.get("histogram_settings", {}),
            "array_selection": manifest.get("array_selection", []),
        }
    )


def _single_trigger_histogram_file(manifest):
    """Return the single trigger-histogram file referenced by a manifest."""
    files = manifest.data.get("files", {}).get("trigger_histograms", [])
    if len(files) != 1:
        raise ValueError(
            f"Expected exactly one trigger-histogram file in {manifest.path}, found {len(files)}."
        )
    path = Path(files[0])
    if path.is_absolute():
        return path
    return manifest.directory / path


def _validate_comparison_inputs(applications, production_paths):
    """Check source coverage and freshness before trigger-product comparisons."""
    comparison = next(
        (
            application.get(_CONFIGURATION_KEY, {})
            for application in applications
            if application.get("application") == "simtools-compare-productions"
            and application.get(_CONFIGURATION_KEY, {}).get("comparison_level", "events")
            == "events"
        ),
        None,
    )
    if comparison is None:
        return
    for production_path in production_paths:
        configuration = {
            "production_path": production_path,
            _OUTPUT_PATH_KEY: str(Path(production_path) / "derived/trigger_histograms"),
        }
        if _SELECT_KEY in comparison:
            configuration[_SELECT_KEY] = comparison[_SELECT_KEY]
        _validate_derived_products(configuration)
