"""Collect and summarize CORSIKA and sim_telarray resource records."""

import json
from collections.abc import Mapping
from functools import partial
from numbers import Real
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.table import Table

from simtools.io.file_type import product_filename
from simtools.production_configuration.production_file_selection import (
    discover_manifests,
    filter_manifests,
    resolve_relative_manifest_path,
)
from simtools.production_configuration.production_metadata import get_sim_telarray_event_counts

_BASELINE = "baseline"
_CANDIDATE = "candidate"
_PRODUCTION_LABEL = "production_label"
_JOB_ID = "job_id"
_MESSAGE = "message"
_ENCODING = "utf-8"
_RECORD_ENVIRONMENTS = "_record_environments"
_ROLES = ("corsika", "sim_telarray")
_UNKNOWN = "unknown"
_ENVIRONMENT_COLUMNS = ("cpu_model", "logical_cpu_count", "available_cpu_count", "container_image")
_ENVIRONMENT_GROUP_COLUMNS = (_PRODUCTION_LABEL, "role", *_ENVIRONMENT_COLUMNS)
_COVERAGE_DIMENSIONS = (
    "role",
    "primary",
    "site",
    "array_layout_name",
    "model_version",
    "energy_min_gev",
    "energy_max_gev",
    "zenith_angle_deg",
)
_STATISTIC_FUNCTIONS = {
    "mean": np.mean,
    "median": np.median,
    "std": np.std,
    "min": np.min,
    "max": np.max,
    "p05": partial(np.percentile, q=5),
    "p95": partial(np.percentile, q=95),
}
_STORAGE_OUTPUT_TYPES = ("sim_telarray", "reduced_event_data", "sim_telarray_histogram")
_OUTPUT_TYPES = ("corsika", *_STORAGE_OUTPUT_TYPES)
_COMPACT_METRICS = (
    ("wall_time_seconds_per_event", "Wall time (s/event)"),
    ("cpu_time_seconds_per_event", "CPU time (s/event)"),
    ("peak_rss_bytes", "Peak RSS (bytes)"),
    ("sim_telarray_storage_bytes_per_event", "Storage (bytes/event)"),
)


def collect_resource_requirements(
    production_path, selections=None, label=_BASELINE, comparison_role=None
):
    """Collect one resource row per recorded simulation process.

    Parameters
    ----------
    production_path : str or pathlib.Path
        Production directory containing ``job-*`` directories.
    selections : list[str], optional
        Manifest selection expressions accepted by ``filter_manifests``.
    label : str, optional
        Label identifying this production in combined results.
    comparison_role : str, optional
        Stable comparison role, either ``"baseline"`` or ``"candidate"``.

    Returns
    -------
    tuple[list[dict], list[dict]]
        Valid resource rows and diagnostics for resource records that cannot
        be used. Missing CORSIKA output files are represented by ``None``.
    """
    manifests = filter_manifests(discover_manifests(production_path), selections or [])
    if not manifests:
        raise ValueError(f"No production jobs matched in {production_path}.")
    rows = []
    diagnostics = []
    for manifest in manifests:
        for role in _ROLES:
            try:
                rows.append(_resource_row(manifest, role, label, comparison_role))
            except (AttributeError, OSError, TypeError, ValueError) as exc:
                diagnostics.append(
                    {
                        _PRODUCTION_LABEL: label,
                        _JOB_ID: manifest.data.get(_JOB_ID),
                        "run_number": manifest.run_number,
                        "role": role,
                        _MESSAGE: str(exc),
                    }
                )
    if not rows:
        raise ValueError(
            f"No valid CORSIKA or sim_telarray resource records found in {production_path}."
        )
    return rows, diagnostics


def write_resource_requirements(args_dict, output_directory, plotter):
    """Collect, write, and plot resource requirements for one or two productions.

    Parameters
    ----------
    args_dict : dict
        Compute-comparison application arguments.
    output_directory : pathlib.Path
        Destination directory for result products.
    plotter : callable
        Plot function accepting rows, output path, and figure formats.

    Returns
    -------
    dict
        Paths of the ECSV, JSON, Markdown, and plot outputs plus diagnostics.
    """
    output_directory = Path(output_directory)
    selections = list(args_dict.get("select") or [])
    layouts = args_dict.get("array_layout_name") or []
    if len(layouts) > 1:
        raise ValueError("Computing comparison supports one selected array layout at a time.")
    if layouts:
        selections.append(f"configuration.array_layout_name={layouts[0]}")
    rows, diagnostics = collect_resource_requirements(
        args_dict["baseline_path"],
        selections,
        label=args_dict.get("baseline_label") or _BASELINE,
        comparison_role=_BASELINE,
    )
    if args_dict.get("candidate_path"):
        candidate_rows, candidate_diagnostics = collect_resource_requirements(
            args_dict["candidate_path"],
            selections,
            label=args_dict.get("candidate_label") or _CANDIDATE,
            comparison_role=_CANDIDATE,
        )
        rows.extend(candidate_rows)
        diagnostics.extend(candidate_diagnostics)
        _validate_resource_coverage(rows, diagnostics)

    report_mode = args_dict.get("report_mode", "diagnostic")
    table_file = None
    if report_mode == "compact":
        summary = _summarize_compact_resource_requirements(rows)
    else:
        table_file = output_directory / "resource_requirements.ecsv"
        resource_table(rows).write(table_file, format="ascii.ecsv", overwrite=True)
        summary = summarize_resource_requirements(rows)
    environments = _execution_environments(rows)
    json_file = output_directory / product_filename("resource_requirements")
    json_file.write_text(
        json.dumps(
            {
                "format_version": 1,
                "resource_row_count": len(rows),
                "diagnostic_count": len(diagnostics),
                "summary": summary,
                "diagnostics": diagnostics,
                "configurations": _configuration_resource_summary(rows),
                "execution_environments": environments,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding=_ENCODING,
    )
    report_file = output_directory / "resource_requirements.md"
    if report_mode == "compact":
        _write_compact_markdown_report(summary, diagnostics, report_file)
    else:
        write_markdown_report(summary, diagnostics, report_file)
    _append_environment_report(environments, report_file)
    plot_files = plotter(
        rows,
        output_directory,
        figure_format=args_dict.get("figure_format"),
        report_mode=report_mode,
    )
    return {
        "table_file": table_file,
        "json_file": json_file,
        "report_file": report_file,
        "plot_files": plot_files,
        "diagnostics": diagnostics,
    }


def resource_table(rows):
    """Return resource rows as an ECSV-compatible Astropy table."""
    return Table(
        rows=[
            {key: value for key, value in row.items() if key != _RECORD_ENVIRONMENTS}
            for row in rows
        ]
    )


def _group_rows(rows, group_columns):
    """Group rows by the requested columns."""
    grouped = {}
    for row in rows:
        grouped.setdefault(tuple(row[column] for column in group_columns), []).append(row)
    return grouped


def _summarize_compact_resource_requirements(rows):
    """Summarize key resource metrics across each production and process role."""
    summary = []
    for (production_label, role), group_rows in sorted(
        _group_rows(rows, (_PRODUCTION_LABEL, "role")).items()
    ):
        item = {_PRODUCTION_LABEL: production_label, "role": role, "job_count": len(group_rows)}
        for metric, _ in _COMPACT_METRICS:
            values = [row[metric] for row in group_rows if row[metric] is not None]
            item[f"{metric}_median"] = float(np.median(values)) if values else None
            item[f"{metric}_p05"] = float(np.percentile(values, 5)) if values else None
            item[f"{metric}_p95"] = float(np.percentile(values, 95)) if values else None
        summary.append(item)
    return summary


def _skipped_records_lines(diagnostics):
    """Return the Markdown lines listing skipped resource records."""
    if not diagnostics:
        return []
    return [
        "",
        "## Skipped records",
        "",
        "| Job | Role | Reason |",
        "| --- | --- | --- |",
        *(f"| {item[_JOB_ID]} | {item['role']} | {item[_MESSAGE]} |" for item in diagnostics),
    ]


def _write_compact_markdown_report(summary, diagnostics, output_file):
    """Write a short resource summary for release review."""
    lines = [
        "# Production resource requirements",
        "",
        f"Valid process records: {sum(item['job_count'] for item in summary)}",
        f"Skipped process records: {len(diagnostics)}",
        "",
        "Median and 5th/95th percentiles across jobs, grouped by production and process role.",
        "Hardware and concurrency are not controlled; timing ratios require review.",
        "",
        "| Production | Role | Jobs | Metric | Median | P05 | P95 |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for item in summary:
        for metric, label in _COMPACT_METRICS:
            values = [_format_value(item[f"{metric}_{name}"]) for name in ("median", "p05", "p95")]
            lines.append(
                f"| {item[_PRODUCTION_LABEL]} | {item['role']} | {item['job_count']} "
                f"| {label} | " + " | ".join(values) + " |"
            )
    lines.extend(_skipped_records_lines(diagnostics))
    Path(output_file).write_text("\n".join(lines) + "\n", encoding=_ENCODING)


def summarize_resource_requirements(rows):
    """Return grouped summary statistics for resource rows.

    The grouping dimensions deliberately exclude azimuth and run number, but
    preserve separate CORSIKA and sim_telarray values.
    """
    group_columns = (
        _PRODUCTION_LABEL,
        "role",
        "primary",
        "site",
        "array_layout_name",
        "model_version",
        "simtools_version",
        "energy_min_gev",
        "energy_max_gev",
        "energy_midpoint_gev",
        "zenith_angle_deg",
    )
    metric_columns = (
        "wall_time_seconds",
        "cpu_time_seconds",
        "triggered_events",
        "wall_time_seconds_per_event",
        "cpu_time_seconds_per_event",
        "wall_time_seconds_per_triggered_event",
        "cpu_time_seconds_per_triggered_event",
        "peak_rss_bytes",
        "corsika_output_bytes",
        "sim_telarray_output_bytes",
        "reduced_event_data_bytes",
        "sim_telarray_histogram_bytes",
        "sim_telarray_storage_bytes",
        "corsika_output_bytes_per_event",
        "sim_telarray_output_bytes_per_event",
        "sim_telarray_output_bytes_per_triggered_event",
        "reduced_event_data_bytes_per_event",
        "reduced_event_data_bytes_per_triggered_event",
        "sim_telarray_histogram_bytes_per_event",
        "sim_telarray_histogram_bytes_per_triggered_event",
        "sim_telarray_storage_bytes_per_event",
        "sim_telarray_storage_bytes_per_triggered_event",
    )
    summary = []
    for key, group_rows in _group_rows(rows, group_columns).items():
        item = dict(zip(group_columns, key, strict=True))
        item["job_count"] = len(group_rows)
        for metric in metric_columns:
            item.update(_metric_statistics(metric, group_rows))
        summary.append(item)
    return sorted(summary, key=lambda item: tuple(str(item[column]) for column in group_columns))


def _metric_statistics(metric, rows):
    """Return aggregate statistics for one metric (``None`` when no values)."""
    values = [row[metric] for row in rows if row[metric] is not None]
    return {
        f"{metric}_{name}": float(func(values)) if values else None
        for name, func in _STATISTIC_FUNCTIONS.items()
    }


def write_markdown_report(summary, diagnostics, output_file):
    """Write a compact human-readable resource requirements report."""
    output_file = Path(output_file)
    dimension_specs = (
        (_PRODUCTION_LABEL, "Production"),
        ("role", "Role"),
        ("primary", "Primary"),
        ("site", "Site"),
        ("array_layout_name", "Layout"),
        ("model_version", "Model version"),
        ("simtools_version", "simtools version"),
        ("energy_midpoint_gev", "Energy (GeV)"),
        ("zenith_angle_deg", "Zenith (deg)"),
        ("job_count", "Jobs"),
    )
    varying_dimensions = {
        key for key, _ in dimension_specs if len({str(item[key]) for item in summary}) > 1
    }
    dimensions = tuple(
        key
        for key, _ in dimension_specs
        if key in varying_dimensions or key in {_PRODUCTION_LABEL, "role", "job_count"}
    )
    metrics = (
        ("wall_time_seconds", "Wall time (s/run)"),
        ("cpu_time_seconds", "CPU time (s/run)"),
        ("wall_time_seconds_per_event", "Wall time (s/event)"),
        ("cpu_time_seconds_per_event", "CPU time (s/event)"),
        ("peak_rss_bytes", "Peak RSS (bytes)"),
        ("corsika_output_bytes", "CORSIKA output (bytes/run)"),
        ("sim_telarray_output_bytes", "sim_telarray output (bytes/run)"),
        ("reduced_event_data_bytes", "reduced event data (bytes/run)"),
        ("sim_telarray_histogram_bytes", "sim_telarray histogram (bytes/run)"),
        ("sim_telarray_storage_bytes", "sim_telarray Storage (bytes/run)"),
        ("corsika_output_bytes_per_event", "CORSIKA output (bytes/event)"),
        ("sim_telarray_output_bytes_per_event", "sim_telarray output (bytes/event)"),
        ("reduced_event_data_bytes_per_event", "reduced event data (bytes/event)"),
        ("sim_telarray_histogram_bytes_per_event", "sim_telarray histogram (bytes/event)"),
        ("sim_telarray_storage_bytes_per_event", "sim_telarray Storage (bytes/event)"),
        ("triggered_events", "Triggered events (run)"),
        ("wall_time_seconds_per_triggered_event", "Wall time (s/triggered event)"),
        ("cpu_time_seconds_per_triggered_event", "CPU time (s/triggered event)"),
        (
            "sim_telarray_storage_bytes_per_triggered_event",
            "sim_telarray Storage (bytes/triggered event)",
        ),
    )
    headers = [header for key, header in dimension_specs if key in dimensions] + [
        label for _, label in metrics
    ]
    lines = [
        "# Production resource requirements",
        "",
        f"Valid process records: {sum(item['job_count'] for item in summary)}",
        f"Skipped process records: {len(diagnostics)}",
        "",
        "Each metric contains mean, median, standard deviation, minimum, and maximum.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---:" for _ in headers) + " |",
    ]
    for item in summary:
        values = [
            _format_value(item[dimension])
            if dimension in {"energy_midpoint_gev", "zenith_angle_deg"}
            else item[dimension]
            for dimension in dimensions
        ]
        values.extend(_format_statistic(item, metric) for metric, _ in metrics)
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    lines.extend(_skipped_records_lines(diagnostics))
    output_file.write_text("\n".join(lines) + "\n", encoding=_ENCODING)


def _resource_row(manifest, role, label, comparison_role=None):
    """Build one validated normalized resource row."""
    record_paths = _resource_record_paths(manifest, role)
    records = [json.loads(path.read_text(encoding=_ENCODING)) for path in record_paths]
    for record, record_path in zip(records, record_paths, strict=True):
        _validate_record(record, record_path, manifest.run_number, role)
    record = _combine_records(records)
    configuration = manifest.data["configuration"]
    showers = configuration["showers_per_run"]
    energy_min = _quantity_value(configuration["energy_min"], u.GeV)
    energy_max = _quantity_value(configuration["energy_max"], u.GeV)
    output_sizes = _output_sizes(manifest, role)
    triggered_events = _event_count(manifest.data.get("statistics", {}).get("triggered_events"))
    if role == "sim_telarray" and triggered_events is None:
        event_counts = get_sim_telarray_event_counts(manifest.directory, manifest.data["files"])
        triggered_events = _event_count(
            event_counts.get("triggered_events") if event_counts else None
        )
    if role != "sim_telarray":
        triggered_events = None
    cpu_time = _sum_or_none(record.get("user_cpu_seconds"), record.get("system_cpu_seconds"))
    storage_sizes = [output_sizes[key] for key in _STORAGE_OUTPUT_TYPES]
    storage_size = (
        sum(size for size in storage_sizes if size is not None)
        if any(size is not None for size in storage_sizes)
        else None
    )
    wall_time = record.get("wall_time_seconds")
    corsika, sim_tel, reduced, histogram = (
        output_sizes[key] for key in ("corsika", *_STORAGE_OUTPUT_TYPES)
    )
    return {
        _PRODUCTION_LABEL: label,
        "comparison_role": comparison_role or label,
        _JOB_ID: manifest.data[_JOB_ID],
        "run_number": manifest.run_number,
        "role": role,
        "production_id": manifest.data.get("production_id") or _UNKNOWN,
        "primary": configuration["primary"],
        "site": configuration["site"],
        "array_layout_name": _layout_name(configuration["array_layout_name"]),
        "model_version": str(configuration["model_version"]),
        "simtools_version": record.get("simtools_version") or _UNKNOWN,
        "energy_min_gev": energy_min,
        "energy_max_gev": energy_max,
        "energy_midpoint_gev": (
            float(np.sqrt(energy_min * energy_max)) if energy_min > 0 and energy_max > 0 else None
        ),
        "zenith_angle_deg": _quantity_value(configuration["zenith_angle"], u.deg),
        "azimuth_angle_deg": _quantity_value(configuration["azimuth_angle"], u.deg),
        "showers_per_run": showers,
        "triggered_events": triggered_events,
        "wall_time_seconds": wall_time,
        "cpu_time_seconds": cpu_time,
        "peak_rss_bytes": record.get("peak_rss_bytes"),
        "user_cpu_seconds": record.get("user_cpu_seconds"),
        "system_cpu_seconds": record.get("system_cpu_seconds"),
        "start_time": record.get("start_time"),
        "end_time": record.get("end_time"),
        "measurement_method": record.get("measurement_method"),
        "platform": record.get("platform"),
        **_execution_environment(record),
        _RECORD_ENVIRONMENTS: [_execution_environment(item) for item in records],
        "wall_time_seconds_per_event": _per_event(wall_time, showers),
        "cpu_time_seconds_per_event": _per_event(cpu_time, showers),
        "wall_time_seconds_per_triggered_event": _per_event(wall_time, triggered_events),
        "cpu_time_seconds_per_triggered_event": _per_event(cpu_time, triggered_events),
        "corsika_output_bytes": corsika,
        "sim_telarray_output_bytes": sim_tel,
        "reduced_event_data_bytes": reduced,
        "sim_telarray_histogram_bytes": histogram,
        "corsika_output_bytes_per_event": _per_event(corsika, showers),
        "sim_telarray_output_bytes_per_event": _per_event(sim_tel, showers),
        "sim_telarray_output_bytes_per_triggered_event": _per_event(sim_tel, triggered_events),
        "reduced_event_data_bytes_per_event": _per_event(reduced, showers),
        "reduced_event_data_bytes_per_triggered_event": _per_event(reduced, triggered_events),
        "sim_telarray_histogram_bytes_per_event": _per_event(histogram, showers),
        "sim_telarray_histogram_bytes_per_triggered_event": _per_event(histogram, triggered_events),
        "sim_telarray_storage_bytes": storage_size,
        "sim_telarray_storage_bytes_per_event": _per_event(storage_size, showers),
        "sim_telarray_storage_bytes_per_triggered_event": _per_event(
            storage_size, triggered_events
        ),
        "resource_record": ";".join(str(path) for path in record_paths),
    }


def _execution_environment(record):
    """Return host, processor, and container identity from a resource record."""
    return {
        "hostname": record.get("hostname") or _UNKNOWN,
        "cpu_model": record.get("cpu_model") or _UNKNOWN,
        "logical_cpu_count": record.get("logical_cpu_count"),
        "available_cpu_count": record.get("available_cpu_count"),
        "container_image": record.get("container_image") or _UNKNOWN,
    }


def _resource_record_paths(manifest, role):
    """Find resource records for a manifest role and run number."""
    suffix = f".{'simtel' if role == 'sim_telarray' else role}.resources"
    records = sorted(manifest.directory.rglob(f"*{suffix}*.json"))
    matching = [path for path in records if f"run{manifest.run_number:06d}" in path.parts]
    if not matching:
        raise FileNotFoundError(
            f"Expected one {role} resource record for run {manifest.run_number} "
            f"in {manifest.directory}, found {len(matching)}."
        )
    return matching


def _combine_records(records):
    """Combine split process records using additive time and peak RSS semantics."""
    if len(records) == 1:
        return records[0]
    combined = dict(records[0])
    for field in ("wall_time_seconds", "user_cpu_seconds", "system_cpu_seconds"):
        values = [record[field] for record in records]
        combined[field] = sum(values) if all(value is not None for value in values) else None
    rss_values = [record["peak_rss_bytes"] for record in records]
    combined["peak_rss_bytes"] = (
        max(rss_values) if all(value is not None for value in rss_values) else None
    )
    combined["end_time"] = records[-1].get("end_time")
    combined["return_code"] = max(record.get("return_code", 0) for record in records)
    return combined


def _validate_record(record, record_path, run_number, role):
    """Validate resource-record fields required by this report."""
    if not isinstance(record, Mapping):
        raise ValueError(f"Malformed resource record in {record_path}: expected a mapping.")
    if record.get("role") != role:
        raise ValueError(f"Resource record role mismatch in {record_path}: {record.get('role')!r}.")
    if str(record.get("run_id")) != str(run_number):
        raise ValueError(
            f"Resource record run ID mismatch in {record_path}: {record.get('run_id')!r}."
        )
    if record.get("return_code") != 0:
        raise ValueError(
            f"Resource record failed in {record_path}: return code {record.get('return_code')!r}."
        )
    required_fields = (
        "start_time",
        "end_time",
        "wall_time_seconds",
        "user_cpu_seconds",
        "system_cpu_seconds",
        "peak_rss_bytes",
        "measurement_method",
        "platform",
    )
    missing = [field for field in required_fields if field not in record]
    if missing:
        raise ValueError(f"Resource record missing fields in {record_path}: {', '.join(missing)}.")
    for field in ("wall_time_seconds", "user_cpu_seconds", "system_cpu_seconds", "peak_rss_bytes"):
        value = record[field]
        if value is not None and (not isinstance(value, Real) or isinstance(value, bool)):
            raise TypeError(f"Resource record field {field!r} is not numeric in {record_path}.")


def _output_sizes(manifest, role):
    """Return output sizes attributable to a CORSIKA or sim_telarray process."""
    sizes = {}
    for output_type in _OUTPUT_TYPES:
        belongs_to_role = (output_type == "corsika") == (role == "corsika")
        if not belongs_to_role:
            sizes[output_type] = None
            continue
        files = manifest.data["files"].get(output_type, [])
        paths = [resolve_relative_manifest_path(manifest.directory, path) for path in files]
        sizes[output_type] = sum(path.stat().st_size for path in paths) if paths else None
    return sizes


def _quantity_value(value, unit):
    """Return a manifest quantity converted to the requested unit."""
    quantity = value if isinstance(value, u.Quantity) else value["value"] * u.Unit(value["unit"])
    return float(quantity.to_value(unit))


def _layout_name(value):
    """Return a stable display value for a manifest layout selection."""
    return ",".join(value) if isinstance(value, list) else str(value)


def _sum_or_none(first, second):
    """Return a sum only when both process CPU components are measured."""
    return None if first is None or second is None else first + second


def _event_count(value):
    """Return a non-negative integer event count or ``None``."""
    if value is None:
        return None
    try:
        count = int(value)
    except TypeError, ValueError:
        return None
    return count if count >= 0 else None


def _per_event(value, event_count):
    """Normalize an optional scalar by a positive event count."""
    return None if value is None or event_count is None or event_count <= 0 else value / event_count


def _format_value(value):
    """Format an optional numeric report value."""
    return "-" if value is None else f"{value:.6g}"


def _format_statistic(summary, metric):
    """Format all aggregate statistics for one metric in the Markdown report."""
    statistics = {name: summary[f"{metric}_{name}"] for name in _STATISTIC_FUNCTIONS}
    if statistics["median"] is None:
        return "-"
    return "; ".join(f"{name}={value:.6g}" for name, value in statistics.items())


def _configuration_resource_summary(rows):
    """Keep small configuration-resolved resource summaries without per-job records."""
    dimensions = (
        "role",
        "primary",
        "site",
        "array_layout_name",
        "model_version",
        "energy_midpoint_gev",
        "zenith_angle_deg",
    )
    metrics = [metric for metric, _ in _COMPACT_METRICS]
    groups = {}
    for row in rows:
        key = tuple(row.get(name) for name in dimensions)
        role = row.get("comparison_role") or row[_PRODUCTION_LABEL]
        groups.setdefault(key, {}).setdefault(role, []).append(row)
    output = []
    for key, productions in sorted(groups.items(), key=lambda item: str(item[0])):
        item = dict(zip(dimensions, key, strict=True))
        item["productions"] = {
            production: _summarize_production_samples(samples, metrics)
            for production, samples in productions.items()
        }
        if {_BASELINE, _CANDIDATE} <= productions.keys():
            item["ratios"] = _baseline_candidate_ratios(
                item["productions"][_BASELINE], item["productions"][_CANDIDATE], metrics
            )
        output.append(item)
    return output


def _summarize_production_samples(samples, metrics):
    """Return the sample count and median/p95 per metric for one production."""
    values = {"count": len(samples)}
    for metric in metrics:
        observations = [row[metric] for row in samples if row.get(metric) is not None]
        if observations:
            values[metric] = {
                "median": float(np.median(observations)),
                "p95": float(np.percentile(observations, 95)),
            }
    return values


def _baseline_candidate_ratios(baseline, candidate, metrics):
    """Return candidate/baseline median ratios per metric."""
    ratios = {}
    for metric in metrics:
        base_median = baseline.get(metric, {}).get("median")
        cand_median = candidate.get(metric, {}).get("median")
        ratios[metric] = (
            cand_median / base_median if base_median and cand_median is not None else None
        )
    return ratios


def _validate_resource_coverage(rows, diagnostics=None):
    """Reject omitted configuration/role pairs instead of silently intersecting them."""
    keys = {
        role: {
            tuple(row.get(key) for key in _COVERAGE_DIMENSIONS)
            for row in rows
            if row.get("comparison_role") == role
        }
        for role in (_BASELINE, _CANDIDATE)
    }
    if keys[_BASELINE] == keys[_CANDIDATE]:
        return
    details = [
        f"{role} only: {_format_coverage_keys(keys[role] - keys[other])}"
        for role, other in ((_BASELINE, _CANDIDATE), (_CANDIDATE, _BASELINE))
        if keys[role] - keys[other]
    ]
    if diagnostics:
        details.append(
            "skipped records: "
            + "; ".join(
                f"{item[_PRODUCTION_LABEL]} {item[_JOB_ID]} {item['role']}: {item[_MESSAGE]}"
                for item in diagnostics
            )
        )
    raise ValueError(
        "Resource configuration/role coverage differs between baseline and candidate ("
        + "; ".join(details)
        + ")."
    )


def _format_coverage_keys(keys):
    """Format configuration/role coverage keys for an error message."""
    return ", ".join(
        "{"
        + ", ".join(
            f"{name}={value}" for name, value in zip(_COVERAGE_DIMENSIONS, key, strict=True)
        )
        + "}"
        for key in sorted(keys, key=str)
    )


def _execution_environments(rows):
    """Count resource records and hosts per production, role, and execution environment."""
    groups = {}
    for row in rows:
        for record_environment in row.get(_RECORD_ENVIRONMENTS, [row]):
            environment_row = {**row, **record_environment}
            key = tuple(environment_row.get(column) for column in _ENVIRONMENT_GROUP_COLUMNS)
            groups.setdefault(key, []).append(environment_row)
    environments = []
    for key, group_rows in sorted(groups.items(), key=lambda item: str(item[0])):
        environment = dict(zip(_ENVIRONMENT_GROUP_COLUMNS, key, strict=True))
        environment["host_count"] = len({row.get("hostname") for row in group_rows})
        environment["record_count"] = len(group_rows)
        environments.append(environment)
    return environments


def _append_environment_report(environments, output_file):
    """Append the execution environments behind the reported measurements."""
    headers = (
        "Production",
        "Role",
        "CPU model",
        "Logical CPUs",
        "Available CPUs",
        "Container",
        "Hosts",
        "Records",
    )
    columns = (*_ENVIRONMENT_GROUP_COLUMNS, "host_count", "record_count")
    lines = [
        "",
        "## Execution environments",
        "",
        "Timing comparisons are only meaningful between matching environments.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for item in environments:
        values = ("-" if item[column] is None else str(item[column]) for column in columns)
        lines.append("| " + " | ".join(values) + " |")
    with Path(output_file).open("a", encoding=_ENCODING) as report:
        report.write("\n".join(lines) + "\n")
