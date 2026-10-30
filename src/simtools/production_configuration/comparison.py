"""Compare trigger-histogram products from simulation productions."""

import csv
import json
import logging
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from simtools.camera import trace_analysis as trace
from simtools.histogram.histogram import rebin_counts
from simtools.io import table_handler
from simtools.io.file_type import product_filename
from simtools.production_configuration.production_file_selection import (
    check_manifest,
    discover_product_manifests,
    filter_manifests,
    normalize_for_comparison,
    stable_configuration_hash,
)
from simtools.production_configuration.trigger_histograms import (
    TRIGGER_HISTOGRAM_METADATA_TABLE,
    TRIGGER_SUBSET_HISTOGRAMS_TABLE,
    TRIGGER_TOPOLOGY_COUNTS_TABLE,
    load_dense_histogram_payloads,
    validate_derived_products,
)
from simtools.simtel.simtel_event_reader import read_events_for_telescopes
from simtools.simtel.simtel_io_metadata import (
    get_sim_telarray_telescope_id_to_telescope_name_mapping,
)
from simtools.utils.general import resolve_file_patterns
from simtools.utils.value_conversion import ensure_string_lists
from simtools.visualization import plot_production_comparison
from simtools.visualization.visualize import pdf_report

_COMPARISON_INVENTORY_FILE = product_filename("comparison_inventory")
_COMPARISON_STATISTICS_FILE = product_filename("comparison_statistics")
_COMPARISON_METRICS_FILE = product_filename("comparison_metrics")
_CONFIGURATION_KEY = "configuration"
_REPORT_MODE_KEY = "report_mode"
_ENCODING = "utf-8"
_INVALID_PRODUCTION_ARGUMENTS = "Production arguments must be provided as label/file pairs."

BASELINE_KEY = "baseline"
COMPARISON_SETS_KEY = "comparison_sets"
PLOT_STATISTICS_KEY = "plot_statistics"
COMPARISONS_KEY = "comparisons"
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
        production_descriptors, args_dict, output_directory, array_layout_names
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
    report_context = (
        pdf_report(output_directory / "trigger-report.pdf")
        if args_dict.get(_REPORT_MODE_KEY) == "compact"
        else nullcontext(None)
    )
    with report_context as compact_figures:
        for (pairing_key, production_descriptors), output_stem in zip(
            descriptor_pairs, output_stems
        ):
            pair_directory = output_directory / output_stem
            if stem_counts[output_stem] > 1:
                hash_key = pairing_key.get("pairing_key", pairing_key)
                pair_directory = (
                    output_directory / f"{output_stem}-{stable_configuration_hash(hash_key)}"
                )
            pair_statistics = _write_array_layout_comparisons(
                production_descriptors,
                args_dict,
                pair_directory,
                array_layout_names,
                compact_figures=compact_figures,
                comparison_context=pairing_key,
                report_pages=report_pages if compact_figures is not None else None,
            )
            statistics_files.extend(pair_statistics)
            if array_layout_names == [None]:
                comparison_labels.update(dict.fromkeys(pair_statistics, (pair_directory.name, "")))

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
    _write_comparison_summary(
        statistics_files, output_directory, report_pages, comparison_labels=comparison_labels
    )


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
        statistics_file = plot_production_comparison.plot_event_level_comparison(
            collect_production_metrics(production_descriptors, array_names=array_layout_name),
            output_path=output_directory,
            array_layout_name=array_layout_name,
            figure_format=args_dict.get("figure_format"),
            report_mode=args_dict.get(_REPORT_MODE_KEY, "diagnostic"),
            compact_figures=compact_figures,
            comparison_title=_comparison_title(comparison_context, array_layout_name),
            comparison_context=comparison_context,
        )
        statistics_files.append(statistics_file)
        if report_pages is not None:
            report_pages[statistics_file] = _figure_pages(compact_figures, first_figure)
    return statistics_files


def _figure_pages(figures, first_figure):
    """Map compact plot names to their one-based page numbers in the combined PDF."""
    labels = (
        figures.page_labels[first_figure:]
        if hasattr(figures, "page_labels")
        else [figure.get_label() for figure in figures[first_figure:]]
    )
    pages = {}
    for page_number, plot_name in enumerate(labels, start=first_figure + 1):
        if plot_name:
            pages.setdefault(plot_name, []).append(page_number)
    return pages


def _comparison_title(comparison_context, array_layout_name):
    """Format pair configuration values for plot headings."""
    layout = f"Layout: {array_layout_name}" if array_layout_name else None
    if not isinstance(comparison_context, dict) or "configurations" not in comparison_context:
        return layout

    configurations = comparison_context["configurations"]
    labels = list(configurations)
    baseline = configurations[labels[0]]
    candidate = configurations[labels[-1]]
    first_line = [layout]
    for key, label in (("primary", "Primary"), ("site", "Site")):
        value = _shared_configuration_value(baseline, candidate, key)
        if value is not None:
            first_line.append(f"{label}: {value}")
    first_line.append(_format_energy_range(baseline, candidate, labels))
    first_line.append(
        _format_pair_value("zenith_angle", "Zenith", baseline, candidate, labels, unit="deg")
    )
    second_line = [
        _format_pair_value("azimuth_angle", "Azimuth", baseline, candidate, labels, unit="deg")
    ]
    interaction = _shared_configuration_value(baseline, candidate, "corsika_he_interaction")
    if interaction is not None:
        second_line.append(f"Interaction model: {interaction}")
    return (
        " | ".join(value for value in first_line if value)
        + "\n"
        + " | ".join(value for value in second_line if value)
    )


def _format_energy_range(baseline, candidate, labels):
    """Format matching energy limits as one readable energy range."""
    baseline_min = _format_configuration_value(baseline.get("energy_min"), "energy")
    baseline_max = _format_configuration_value(baseline.get("energy_max"), "energy")
    candidate_min = _format_configuration_value(candidate.get("energy_min"), "energy")
    candidate_max = _format_configuration_value(candidate.get("energy_max"), "energy")
    if baseline_min == "None" and baseline_max == "None":
        return None
    baseline_range = _join_limits(baseline_min, baseline_max)
    candidate_range = _join_limits(candidate_min, candidate_max)
    if baseline_range == candidate_range:
        return f"Energy: {baseline_range}"
    return f"Energy: {labels[0]} {baseline_range}; {labels[-1]} {candidate_range}"


def _join_limits(minimum, maximum):
    """Join minimum and maximum energies without repeating identical limits."""
    return minimum if minimum == maximum else f"{minimum} to {maximum}"


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
    report_pages = report_pages or {}
    comparison_labels = comparison_labels or {}
    for statistics_file in statistics_files:
        statistics = json.loads(statistics_file.read_text(encoding=_ENCODING))
        baseline = _add_production_summary(baseline, statistics[BASELINE_KEY])
        comparison_sets = _add_comparison_sets(comparison_sets, statistics[COMPARISON_SETS_KEY])
        pair, layout = _comparison_coordinates(
            statistics_file, output_directory, comparison_labels.get(statistics_file)
        )
        metric_scopes.setdefault(pair, {})[layout] = (
            statistics[PLOT_STATISTICS_KEY]
            .get("telescope_participation_fraction", {})
            .get("telescope_types", [])
        )
        compact_statistics.extend(
            _compact_statistic_rows(pair, layout, statistics, report_pages.get(statistics_file, {}))
        )
        statistics_file.unlink()
        _remove_empty_comparison_directories(statistics_file.parent, output_directory)

    summary = {
        "format_version": 2,
        BASELINE_KEY: baseline
        or {"label": None, "simulated_event_count": 0, "triggered_event_count": 0},
        COMPARISON_SETS_KEY: list(comparison_sets.values()),
        PLOT_STATISTICS_KEY: {},
        COMPARISONS_KEY: [],
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / _COMPARISON_STATISTICS_FILE).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding=_ENCODING
    )
    _write_compact_statistics_csv(compact_statistics, output_directory)
    inventory = output_directory / _COMPARISON_INVENTORY_FILE
    if inventory.is_file():
        payload = json.loads(inventory.read_text(encoding=_ENCODING))
        payload["metric_scopes"] = metric_scopes
        inventory.write_text(json.dumps(payload, indent=2) + "\n", encoding=_ENCODING)


def _compact_statistic_rows(pair, array_layout, statistics, report_pages):
    """Extract scalar comparison results, without histogram arrays, as one row per population."""
    rows = []
    for plot_name, plot_statistics in statistics[PLOT_STATISTICS_KEY].items():
        suffix = plot_name.rsplit("_", 1)[-1]
        scope, telescope_type = (
            ("telescope_type", suffix)
            if suffix.startswith(("LST", "MST", "SST"))
            else ("whole_array", "")
        )
        for comparison in plot_statistics.get("comparisons", []):
            results = [
                (population, comparison[population])
                for population in ("simulated", "triggered")
                if comparison.get(population) is not None
            ]
            if "value" in comparison:
                results.append(("triggered", comparison))
            for population, result in results:
                rows.append(
                    {
                        "pair": pair,
                        "array_layout": array_layout,
                        "plot": plot_name,
                        REPORT_PAGE_COLUMN: ", ".join(
                            str(page) for page in report_pages.get(plot_name, [])
                        ),
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
    with (output_directory / _COMPARISON_METRICS_FILE).open(
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
    baseline_label = args_dict.get("baseline_label", BASELINE_KEY)
    candidate_label = args_dict.get("candidate_label", "candidate")

    descriptor_pairs = [
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
    missing_candidates = sorted(set(baseline_by_key) - set(candidate_by_key))
    missing_baselines = sorted(set(candidate_by_key) - set(baseline_by_key))
    if missing_candidates or missing_baselines:
        raise _ProductionPairingError(
            "Trigger-histogram metadata pairing failed: "
            f"missing candidates={len(missing_candidates)}, "
            f"missing baselines={len(missing_baselines)}.",
            descriptor_pairs,
        )
    return descriptor_pairs


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
    if isinstance(layout, list | tuple) and len(layout) == 1:
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


def validate_comparison_inputs(applications, production_paths):
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
            "output_path": str(Path(production_path) / "derived/trigger_histograms"),
        }
        if "select" in comparison:
            configuration["select"] = comparison["select"]
        validate_derived_products(configuration)


@dataclass
class ProductionDescriptor:
    """Descriptor for one production input provided."""

    label: str
    input_files: list[str]


@dataclass
class ProductionEventMetrics:
    """Aggregated event-level metrics for one production."""

    label: str
    trigger_combinations: Counter
    telescope_participation: Counter
    simulated_event_count: int
    triggered_event_count: int
    simulated_energies: np.ndarray = field(default_factory=lambda: np.array([]))
    triggered_energies: np.ndarray = field(default_factory=lambda: np.array([]))
    simulated_core_distances: np.ndarray = field(default_factory=lambda: np.array([]))
    triggered_core_distances: np.ndarray = field(default_factory=lambda: np.array([]))
    simulated_angular_distances: np.ndarray = field(default_factory=lambda: np.array([]))
    triggered_angular_distances: np.ndarray = field(default_factory=lambda: np.array([]))
    trigger_multiplicity: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    per_type: dict = field(default_factory=dict)
    quantity_histograms: dict = field(default_factory=dict)
    trigger_multiplicity_histogram: tuple = field(default_factory=tuple)


@dataclass
class ProductionSignalMetrics:
    """Aggregated signal metrics for one production and telescope."""

    label: str
    pedestals: np.ndarray
    signals: np.ndarray
    peak_timing: np.ndarray
    triggered_pixels: np.ndarray


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
        If configuration is malformed or does not contain at least two productions.
    """
    parsed_productions = _normalize_production_arguments(production_arguments)
    if len(parsed_productions) < 2:
        raise ValueError("At least two productions are required for comparison.")

    labels = [label for label, _ in parsed_productions]
    if len(set(labels)) != len(labels):
        raise ValueError("Production labels must be unique.")

    descriptors = []
    for label, pattern_list in parsed_productions:
        patterns = [pattern.strip() for pattern in pattern_list.split(",") if pattern.strip()]
        if not patterns:
            raise ValueError(f"Production '{label}' has no input_file pattern.")
        resolved_files = [str(path) for path in resolve_file_patterns(patterns)]
        if not resolved_files:
            raise ValueError(f"Production '{label}' does not resolve to any files.")
        descriptors.append(ProductionDescriptor(label=label, input_files=resolved_files))
    return descriptors


def _normalize_production_arguments(production_arguments):
    """Normalize raw production arguments into ``[(label, files), ...]``."""
    if not production_arguments:
        return []
    if all(isinstance(item, str) for item in production_arguments):
        return _pairwise_label_file_arguments(production_arguments)

    normalized = []
    for item in production_arguments:
        if not isinstance(item, list | tuple) or not all(isinstance(value, str) for value in item):
            raise ValueError(_INVALID_PRODUCTION_ARGUMENTS)
        if len(item) == 2:
            normalized.append((item[0], item[1]))
        else:
            normalized.extend(_pairwise_label_file_arguments(list(item)))
    return normalized


def _pairwise_label_file_arguments(flat_arguments):
    """Convert a flat list of strings into ``[(label, files), ...]`` pairs."""
    if len(flat_arguments) % 2 != 0:
        raise ValueError(_INVALID_PRODUCTION_ARGUMENTS)
    return [
        (flat_arguments[index], flat_arguments[index + 1])
        for index in range(0, len(flat_arguments), 2)
    ]


def collect_production_metrics(production_descriptors, array_names=None):
    """Collect comparison metrics from trigger histogram files for each production.

    Parameters
    ----------
    production_descriptors : list[ProductionDescriptor]
        Input descriptor for each production.
    array_names : list[str] or str, optional
        Restrict loaded histogram references to these array layout names.

    Returns
    -------
    list[ProductionEventMetrics]
        Aggregated metrics per production.
    """
    return [
        _collect_single_production_histogram_metrics(descriptor, array_names=array_names)
        for descriptor in production_descriptors
    ]


def _collect_single_production_histogram_metrics(production_descriptor, array_names=None):
    """Collect comparison metrics for one trigger-histogram production descriptor."""
    selected_array_names = ensure_string_lists(array_names)
    accumulators = {
        "quantity_histograms": {
            "simulated": {},
            "triggered": {},
            "subset_triggered": defaultdict(dict),
        },
        "trigger_multiplicity": Counter(),
        "trigger_combinations": Counter(),
        "telescope_participation": Counter(),
        "subset_multiplicity": defaultdict(Counter),
        "simulated_event_count": 0,
        "triggered_event_count": 0,
    }
    matched_references = sum(
        _collect_metrics_from_trigger_histogram_file(
            trigger_histogram_file, accumulators, array_names=selected_array_names
        )
        for trigger_histogram_file in production_descriptor.input_files
    )
    if selected_array_names and matched_references == 0:
        raise ValueError(
            "Array layout selection did not match any trigger-histogram references for "
            f"production '{production_descriptor.label}'."
        )

    simulated_histograms = accumulators["quantity_histograms"]["simulated"]
    triggered_histograms = accumulators["quantity_histograms"]["triggered"]
    return ProductionEventMetrics(
        label=production_descriptor.label,
        trigger_combinations=accumulators["trigger_combinations"],
        telescope_participation=accumulators["telescope_participation"],
        simulated_event_count=accumulators["simulated_event_count"],
        triggered_event_count=accumulators["triggered_event_count"],
        per_type=_build_per_type_histogram_metrics(
            production_descriptor.label, simulated_histograms, accumulators
        ),
        quantity_histograms={
            quantity: {
                "simulated": simulated_histogram,
                "triggered": triggered_histograms[quantity],
            }
            for quantity, simulated_histogram in simulated_histograms.items()
        },
        trigger_multiplicity_histogram=_counter_to_histogram(accumulators["trigger_multiplicity"]),
    )


def _collect_metrics_from_trigger_histogram_file(
    trigger_histogram_file, accumulators, array_names=None
):
    """Collect metrics from one trigger-histogram HDF5 file."""
    dense_payloads = load_dense_histogram_payloads(trigger_histogram_file)
    tables = table_handler.read_tables(
        trigger_histogram_file,
        [
            TRIGGER_HISTOGRAM_METADATA_TABLE,
            TRIGGER_TOPOLOGY_COUNTS_TABLE,
            TRIGGER_SUBSET_HISTOGRAMS_TABLE,
        ],
        file_type="HDF5",
    )
    metadata = tables[TRIGGER_HISTOGRAM_METADATA_TABLE]
    topology_counts = tables[TRIGGER_TOPOLOGY_COUNTS_TABLE]
    subset_histograms = tables[TRIGGER_SUBSET_HISTOGRAMS_TABLE]
    topology_rows_by_reference = table_handler.group_table_rows(topology_counts, "reference_id")
    subset_rows_by_reference = table_handler.group_table_rows(subset_histograms, "reference_id")

    matched_references = 0
    for row in metadata:
        if array_names and str(row["array_name"]) not in array_names:
            continue
        matched_references += 1
        reference_id = str(row["reference_id"])
        _accumulate_quantity_histograms_from_dense_payload(
            dense_payloads.get(reference_id, {}), accumulators
        )
        _accumulate_topology_counts_for_reference(
            topology_rows_by_reference.get(row["reference_id"], topology_counts[:0]),
            accumulators,
        )
        _accumulate_subset_histograms_for_reference(
            subset_rows_by_reference.get(row["reference_id"], subset_histograms[:0]),
            accumulators,
        )
        accumulators["simulated_event_count"] += int(row["total_simulated_events"])
        accumulators["triggered_event_count"] += int(row["total_triggered_events"])
    return matched_references


def _accumulate_quantity_histograms_from_dense_payload(histograms, accumulators):
    """Accumulate base simulated and triggered histograms from dense payloads."""
    histogram_map = {
        "energy_mc": ("energy", "simulated"),
        "energy": ("energy", "triggered"),
        "core_distance_mc": ("core_distance", "simulated"),
        "core_distance": ("core_distance", "triggered"),
        "angular_distance_mc": ("angular_distance", "simulated"),
        "angular_distance": ("angular_distance", "triggered"),
    }
    for histogram_name, (quantity, event_kind) in histogram_map.items():
        histogram = histograms.get(histogram_name)
        if histogram is None or histogram.ndim != 1:
            continue
        _add_histogram(
            accumulators["quantity_histograms"][event_kind],
            quantity,
            np.asarray(histogram.values, dtype=float),
            np.asarray(histogram.edges[0], dtype=float),
        )


def _add_histogram(target, quantity, counts, bin_edges):
    """Add histogram counts into a target mapping, requiring consistent bin edges."""
    if quantity not in target:
        target[quantity] = (np.asarray(counts, dtype=float), np.asarray(bin_edges, dtype=float))
        return
    existing_counts, existing_edges = target[quantity]
    bin_edges = np.asarray(bin_edges, dtype=float)
    if np.array_equal(existing_edges, bin_edges):
        target[quantity] = (existing_counts + counts, existing_edges)
        return

    merged_edges = np.union1d(existing_edges, bin_edges)
    if merged_edges.size < 2:
        raise ValueError(f"Inconsistent bin edges for quantity '{quantity}'.")
    target[quantity] = (
        rebin_counts(existing_counts, existing_edges, merged_edges)
        + rebin_counts(np.asarray(counts, dtype=float), bin_edges, merged_edges),
        merged_edges,
    )


def _accumulate_topology_counts_for_reference(topology_rows, accumulators):
    """Accumulate trigger topology count rows."""
    for row in topology_rows:
        count_type = str(row["count_type"])
        key = str(row["key"])
        count = int(row["count"])
        if count_type == "trigger_multiplicity":
            accumulators["trigger_multiplicity"][int(key)] += count
        elif count_type == "trigger_combinations":
            accumulators["trigger_combinations"][key] += count
        elif count_type == "telescope_participation":
            accumulators["telescope_participation"][key] += count
        elif count_type == "subset_multiplicity":
            accumulators["subset_multiplicity"][str(row["subset"])][int(key)] += count


def _accumulate_subset_histograms_for_reference(subset_rows, accumulators):
    """Accumulate per-subset triggered quantity histograms."""
    for subset_name, subset_selected in table_handler.group_table_rows(
        subset_rows, "subset"
    ).items():
        for quantity, rows in table_handler.group_table_rows(subset_selected, "quantity").items():
            rows.sort("bin_index")
            counts = np.asarray(rows["count"], dtype=float)
            bin_edges = np.concatenate(
                [
                    np.asarray(rows["bin_low"][:1], dtype=float),
                    np.asarray(rows["bin_high"], dtype=float),
                ]
            )
            _add_histogram(
                accumulators["quantity_histograms"]["subset_triggered"][str(subset_name)],
                str(quantity),
                counts,
                bin_edges,
            )


def _counter_to_histogram(counter):
    """Convert integer-key count data to histogram counts and bin edges."""
    if not counter:
        return np.array([], dtype=float), np.array([], dtype=float)
    max_key = max(int(key) for key in counter)
    bin_edges = np.arange(1, max_key + 2)
    counts = np.array([counter.get(index, 0) for index in range(1, max_key + 1)], dtype=float)
    return counts, bin_edges


def _build_per_type_histogram_metrics(label, simulated_histograms, accumulators):
    """Build per-subset metrics from histogram-backed accumulators."""
    per_type = {}
    for subset_name, triggered_histograms in accumulators["quantity_histograms"][
        "subset_triggered"
    ].items():
        subset_multiplicity = accumulators["subset_multiplicity"].get(subset_name, Counter())
        per_type[subset_name] = ProductionEventMetrics(
            label=label,
            trigger_combinations=Counter(),
            telescope_participation=Counter(),
            simulated_event_count=accumulators["simulated_event_count"],
            triggered_event_count=int(sum(subset_multiplicity.values())),
            quantity_histograms={
                quantity: {
                    "simulated": simulated_histograms[quantity],
                    "triggered": triggered_histograms[quantity],
                }
                for quantity in triggered_histograms
                if quantity in simulated_histograms
            },
            trigger_multiplicity_histogram=_counter_to_histogram(subset_multiplicity),
        )
    return per_type


def collect_signal_metrics(production_descriptors, telescope_names=None):
    """Collect telescope-level signal metrics for each production.

    Parameters
    ----------
    production_descriptors : list[ProductionDescriptor]
        Production descriptors containing sim_telarray input files.
    telescope_names : str or list[str], optional
        Restrict the metrics to one or more telescope names. If omitted, all
        telescopes shared by the input files are processed.

    Returns
    -------
    dict[str, list[ProductionSignalMetrics]]
        Metrics grouped by telescope name, with one entry per production.

    Raises
    ------
    ValueError
        If a required telescope is absent or no event data is available for a
        telescope.
    """
    selected_telescope_names = _discover_telescope_names(production_descriptors, telescope_names)
    if not selected_telescope_names:
        raise ValueError("The sim_telarray inputs contain no telescopes.")

    metrics_by_telescope = {name: [] for name in selected_telescope_names}
    for production in production_descriptors:
        metrics = _collect_production_signal_metrics(production, selected_telescope_names)
        for telescope_name, telescope_metrics in metrics.items():
            metrics_by_telescope[telescope_name].append(telescope_metrics)
    return metrics_by_telescope


def _discover_telescope_names(production_descriptors, requested_telescope_names=None):
    """Discover and validate the telescope set represented by all input files."""
    input_files = [
        input_file for production in production_descriptors for input_file in production.input_files
    ]
    if not input_files:
        raise ValueError("Signal comparison has no sim_telarray input files.")

    available_by_file = {
        input_file: {
            str(name)
            for name in get_sim_telarray_telescope_id_to_telescope_name_mapping(input_file).values()
        }
        for input_file in input_files
    }
    if requested_telescope_names is not None:
        requested = set(ensure_string_lists(requested_telescope_names))
        missing = {
            input_file: sorted(requested - available)
            for input_file, available in available_by_file.items()
            if requested - available
        }
        if missing:
            raise ValueError(f"Requested telescope(s) not found in input files: {missing}.")
        return sorted(requested)

    expected = next(iter(available_by_file.values()))
    for input_file, available in available_by_file.items():
        if available != expected:
            raise ValueError(
                f"Input '{input_file}' has telescope set {sorted(available)}; expected "
                f"the shared telescope set {sorted(expected)}."
            )
    return sorted(expected)


def _collect_production_signal_metrics(production, telescope_names):
    """Collect all telescope metrics for one production."""
    values = {
        name: {"pedestals": [], "signals": [], "peak_timing": [], "triggered_pixels": []}
        for name in telescope_names
    }
    for input_file in production.input_files:
        _collect_file_metrics(input_file, telescope_names, values)

    result = {}
    for telescope_name, telescope_values in values.items():
        if not any(telescope_values.values()):
            raise ValueError(
                f"Production '{production.label}' has no event data for telescope "
                f"'{telescope_name}'."
            )
        result[telescope_name] = ProductionSignalMetrics(
            label=production.label,
            **{
                key: np.concatenate(items) if items else np.array([])
                for key, items in telescope_values.items()
            },
        )
    return result


def _collect_file_metrics(input_file, telescope_names, values):
    """Collect metrics from one file for all requested telescopes."""
    _event_ids, _telescope_descriptions, events_by_telescope = read_events_for_telescopes(
        input_file,
        telescope_names,
        event_ids=None,
        max_events=None,
    )
    if events_by_telescope is None:
        raise ValueError(f"Telescope '{telescope_names[0]}' was not found in input '{input_file}'.")

    for telescope_name in telescope_names:
        for event in events_by_telescope[telescope_name]:
            try:
                samples, pedestals, signals = trace.get_trace_data(event["adc_samples"])
                peak_samples, _pixel_ids, _found_count = trace.trace_maxima(
                    samples, sum_threshold=trace.DEFAULT_SUM_THRESHOLD
                )
                trigger_pixels = _get_triggered_pixel_count(event)
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Input '{input_file}' has incomplete signal data for telescope "
                    f"'{telescope_name}'."
                ) from exc

            values[telescope_name]["pedestals"].append(np.asarray(pedestals))
            values[telescope_name]["signals"].append(np.asarray(signals))
            if peak_samples is not None:
                values[telescope_name]["peak_timing"].append(np.asarray(peak_samples))
            values[telescope_name]["triggered_pixels"].append(np.asarray([trigger_pixels]))


def _get_triggered_pixel_count(event):
    """Return the triggered pixel count, falling back to the selected pixel list."""
    pixel_lists = event["pixel_lists"]
    pixel_list = pixel_lists.get(0)
    if pixel_list is None:
        pixel_list = pixel_lists.get(1)
    if pixel_list is None:
        return 0
    return int(pixel_list["pixels"])
