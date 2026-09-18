"""Collect and summarize CORSIKA and sim_telarray resource records."""

import json
from collections.abc import Mapping
from numbers import Real
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.table import Table

from simtools.production_configuration.job_metadata import get_sim_telarray_event_counts
from simtools.production_configuration.production_file_selection import (
    _resolve_relative_manifest_path,
    discover_manifests,
    filter_manifests,
)

_ROLES = ("corsika", "sim_telarray")
_OUTPUT_TYPES = (
    "corsika",
    "sim_telarray",
    "reduced_event_data",
    "sim_telarray_histogram",
)


def collect_resource_requirements(
    production_path, selections=None, label="baseline", comparison_role=None
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
                        "production_label": label,
                        "job_id": manifest.data.get("job_id"),
                        "run_number": manifest.run_number,
                        "role": role,
                        "message": str(exc),
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
        Paths of the ECSV, Markdown, and plot outputs plus diagnostics.
    """
    output_directory = Path(output_directory)
    baseline_label = args_dict.get("baseline_label") or "baseline"
    candidate_label = args_dict.get("candidate_label") or "candidate"
    rows, diagnostics = collect_resource_requirements(
        args_dict["baseline_path"],
        args_dict.get("select"),
        label=baseline_label,
        comparison_role="baseline",
    )
    if args_dict.get("candidate_path"):
        candidate_rows, candidate_diagnostics = collect_resource_requirements(
            args_dict["candidate_path"],
            args_dict.get("select"),
            label=candidate_label,
            comparison_role="candidate",
        )
        rows.extend(candidate_rows)
        diagnostics.extend(candidate_diagnostics)
    table_file = output_directory / "resource_requirements.ecsv"
    resource_table(rows).write(table_file, format="ascii.ecsv", overwrite=True)
    summary = summarize_resource_requirements(rows)
    report_file = output_directory / "resource_requirements.md"
    write_markdown_report(summary, diagnostics, report_file)
    plot_files = plotter(rows, output_directory, figure_format=args_dict.get("figure_format"))
    return {
        "table_file": table_file,
        "report_file": report_file,
        "plot_files": plot_files,
        "diagnostics": diagnostics,
    }


def resource_table(rows):
    """Return resource rows as an ECSV-compatible Astropy table."""
    return Table(rows=rows)


def summarize_resource_requirements(rows):
    """Return grouped summary statistics for resource rows.

    The grouping dimensions deliberately exclude azimuth and run number, but
    preserve separate CORSIKA and sim_telarray values.
    """
    group_columns = (
        "production_label",
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
        "triggered_events",
        "wall_time_seconds_per_event",
        "cpu_time_seconds_per_event",
        "wall_time_seconds_per_triggered_event",
        "cpu_time_seconds_per_triggered_event",
        "peak_rss_bytes",
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
    grouped = _group_rows(rows, group_columns)
    summary = [
        _summarize_group(key, grouped_rows, group_columns, metric_columns)
        for key, grouped_rows in grouped.items()
    ]
    return sorted(summary, key=lambda item: tuple(str(item[column]) for column in group_columns))


def _group_rows(rows, group_columns):
    """Group rows by the requested configuration columns."""
    grouped = {}
    for row in rows:
        key = tuple(row[column] for column in group_columns)
        grouped.setdefault(key, []).append(row)
    return grouped


def _summarize_group(key, rows, group_columns, metric_columns):
    """Build summary statistics for one resource group."""
    result = dict(zip(group_columns, key, strict=True))
    result["job_count"] = len(rows)
    for metric in metric_columns:
        result.update(_metric_statistics(metric, rows))
    return result


def _metric_statistics(metric, rows):
    """Return aggregate statistics for one metric."""
    values = [row[metric] for row in rows if row[metric] is not None]
    if not values:
        return {
            f"{metric}_{statistic}": None for statistic in ("mean", "median", "std", "min", "max")
        }
    return {
        f"{metric}_mean": float(np.mean(values)),
        f"{metric}_median": float(np.median(values)),
        f"{metric}_std": float(np.std(values)),
        f"{metric}_min": float(np.min(values)),
        f"{metric}_max": float(np.max(values)),
    }


def write_markdown_report(summary, diagnostics, output_file):
    """Write a compact human-readable resource requirements report."""
    output_file = Path(output_file)
    dimension_specs = (
        ("production_label", "Production"),
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
        if key in varying_dimensions or key in {"production_label", "role", "job_count"}
    )
    metrics = (
        "wall_time_seconds_per_event",
        "cpu_time_seconds_per_event",
        "peak_rss_bytes",
        "corsika_output_bytes_per_event",
        "sim_telarray_output_bytes_per_event",
        "reduced_event_data_bytes_per_event",
        "sim_telarray_histogram_bytes_per_event",
        "sim_telarray_storage_bytes_per_event",
        "triggered_events",
        "wall_time_seconds_per_triggered_event",
        "cpu_time_seconds_per_triggered_event",
        "sim_telarray_storage_bytes_per_triggered_event",
    )
    headers = [header for key, header in dimension_specs if key in dimensions] + [
        metric.removesuffix("_per_event").replace("_", " ").title() for metric in metrics
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
            item[dimension]
            if dimension not in {"energy_midpoint_gev", "zenith_angle_deg"}
            else _format_value(item[dimension])
            for dimension in dimensions
        ]
        values.extend(_format_statistic(item, metric) for metric in metrics)
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    if diagnostics:
        lines.extend(
            ["", "## Skipped records", "", "| Job | Role | Reason |", "| --- | --- | --- |"]
        )
        lines.extend(
            f"| {item['job_id']} | {item['role']} | {item['message']} |" for item in diagnostics
        )
    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resource_row(manifest, role, label, comparison_role=None):
    """Build one validated normalized resource row."""
    record_paths = _resource_record_paths(manifest, role)
    records = [json.loads(path.read_text(encoding="utf-8")) for path in record_paths]
    for record, record_path in zip(records, record_paths, strict=True):
        _validate_record(record, record_path, manifest.run_number, role)
    record = _combine_records(records)
    configuration = manifest.data["configuration"]
    showers = configuration["showers_per_run"]
    energy_min = _quantity_value(configuration["energy_min"], u.GeV)
    energy_max = _quantity_value(configuration["energy_max"], u.GeV)
    output_sizes = _output_sizes(manifest, role)
    statistics = manifest.data.get("statistics", {})
    triggered_events = _event_count(statistics.get("triggered_events"))
    if role == "sim_telarray" and triggered_events is None:
        event_counts = get_sim_telarray_event_counts(manifest.directory, manifest.data["files"])
        triggered_events = _event_count(
            event_counts.get("triggered_events") if event_counts else None
        )
    if role != "sim_telarray":
        triggered_events = None
    cpu_time = _sum_or_none(record.get("user_cpu_seconds"), record.get("system_cpu_seconds"))
    storage_components = [
        size
        for key, size in output_sizes.items()
        if key in {"sim_telarray", "reduced_event_data", "sim_telarray_histogram"}
        and size is not None
    ]
    storage_size = sum(storage_components) if storage_components else None
    return {
        "production_label": label,
        "comparison_role": comparison_role or label,
        "job_id": manifest.data["job_id"],
        "run_number": manifest.run_number,
        "role": role,
        "production_id": manifest.data.get("production_id") or "unknown",
        "primary": configuration["primary"],
        "site": configuration["site"],
        "array_layout_name": _layout_name(configuration["array_layout_name"]),
        "model_version": str(configuration["model_version"]),
        "simtools_version": record.get("simtools_version") or "unknown",
        "energy_min_gev": energy_min,
        "energy_max_gev": energy_max,
        "energy_midpoint_gev": (
            float(np.sqrt(energy_min * energy_max)) if energy_min > 0 and energy_max > 0 else None
        ),
        "zenith_angle_deg": _quantity_value(configuration["zenith_angle"], u.deg),
        "azimuth_angle_deg": _quantity_value(configuration["azimuth_angle"], u.deg),
        "showers_per_run": showers,
        "triggered_events": triggered_events,
        "wall_time_seconds": record.get("wall_time_seconds"),
        "cpu_time_seconds": cpu_time,
        "peak_rss_bytes": record.get("peak_rss_bytes"),
        "user_cpu_seconds": record.get("user_cpu_seconds"),
        "system_cpu_seconds": record.get("system_cpu_seconds"),
        "start_time": record.get("start_time"),
        "end_time": record.get("end_time"),
        "measurement_method": record.get("measurement_method"),
        "platform": record.get("platform"),
        "wall_time_seconds_per_event": _per_event(record.get("wall_time_seconds"), showers),
        "cpu_time_seconds_per_event": _per_event(cpu_time, showers),
        "wall_time_seconds_per_triggered_event": _per_event(
            record.get("wall_time_seconds"), triggered_events
        ),
        "cpu_time_seconds_per_triggered_event": _per_event(cpu_time, triggered_events),
        "corsika_output_bytes": output_sizes["corsika"],
        "sim_telarray_output_bytes": output_sizes["sim_telarray"],
        "reduced_event_data_bytes": output_sizes["reduced_event_data"],
        "sim_telarray_histogram_bytes": output_sizes["sim_telarray_histogram"],
        "corsika_output_bytes_per_event": _per_event(output_sizes["corsika"], showers),
        "sim_telarray_output_bytes_per_event": _per_event(output_sizes["sim_telarray"], showers),
        "sim_telarray_output_bytes_per_triggered_event": _per_event(
            output_sizes["sim_telarray"], triggered_events
        ),
        "reduced_event_data_bytes_per_event": _per_event(
            output_sizes["reduced_event_data"], showers
        ),
        "reduced_event_data_bytes_per_triggered_event": _per_event(
            output_sizes["reduced_event_data"], triggered_events
        ),
        "sim_telarray_histogram_bytes_per_event": _per_event(
            output_sizes["sim_telarray_histogram"], showers
        ),
        "sim_telarray_histogram_bytes_per_triggered_event": _per_event(
            output_sizes["sim_telarray_histogram"], triggered_events
        ),
        "sim_telarray_storage_bytes": storage_size,
        "sim_telarray_storage_bytes_per_event": _per_event(storage_size, showers),
        "sim_telarray_storage_bytes_per_triggered_event": _per_event(
            storage_size, triggered_events
        ),
        "resource_record": ";".join(str(path) for path in record_paths),
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
    if (
        record.get("schema_name") != "process_resource_record"
        or record.get("schema_version") != "1.0.0"
    ):
        raise ValueError(f"Unsupported resource record schema in {record_path}.")
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
        if role == "corsika" and output_type != "corsika":
            sizes[output_type] = None
            continue
        if role == "sim_telarray" and output_type == "corsika":
            sizes[output_type] = None
            continue
        files = manifest.data["files"].get(output_type, [])
        paths = [_resolve_relative_manifest_path(manifest.directory, path) for path in files]
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
    """Format an optional report value."""
    return "-" if value is None else f"{value:.6g}"


def _format_statistic(summary, metric):
    """Format all aggregate statistics for one metric in the Markdown report."""
    statistics = {
        name: summary[f"{metric}_{name}"] for name in ("mean", "median", "std", "min", "max")
    }
    if statistics["median"] is None:
        return "-"
    return "; ".join(f"{name}={value:.6g}" for name, value in statistics.items())
