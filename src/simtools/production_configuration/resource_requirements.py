"""Collect and summarize CORSIKA and sim_telarray resource records."""

import json
from pathlib import Path

import numpy as np
from astropy.table import Table

from simtools.production_configuration.production_file_selection import (
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


def collect_resource_requirements(production_path, selections=None, label="baseline"):
    """Collect one resource row per recorded simulation process.

    Parameters
    ----------
    production_path : str or pathlib.Path
        Production directory containing ``job-*`` directories.
    selections : list[str], optional
        Manifest selection expressions accepted by ``filter_manifests``.
    label : str, optional
        Label identifying this production in combined results.

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
                rows.append(_resource_row(manifest, role, label))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
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
    rows, diagnostics = collect_resource_requirements(
        args_dict["baseline_path"], args_dict.get("select"), label="baseline"
    )
    if args_dict.get("candidate_path"):
        candidate_rows, candidate_diagnostics = collect_resource_requirements(
            args_dict["candidate_path"], args_dict.get("select"), label="candidate"
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
        "wall_time_seconds_per_event",
        "cpu_time_seconds_per_event",
        "peak_rss_bytes",
        "corsika_output_bytes_per_event",
        "sim_telarray_output_bytes_per_event",
        "reduced_event_data_bytes_per_event",
        "sim_telarray_histogram_bytes_per_event",
        "sim_telarray_storage_bytes_per_event",
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
        return {f"{metric}_{statistic}": None for statistic in ("mean", "median", "min", "max")}
    return {
        f"{metric}_mean": float(np.mean(values)),
        f"{metric}_median": float(np.median(values)),
        f"{metric}_min": float(np.min(values)),
        f"{metric}_max": float(np.max(values)),
    }


def write_markdown_report(summary, diagnostics, output_file):
    """Write a compact human-readable resource requirements report."""
    output_file = Path(output_file)
    lines = [
        "# Production resource requirements",
        "",
        f"Valid process records: {sum(item['job_count'] for item in summary)}",
        f"Skipped process records: {len(diagnostics)}",
        "",
    ]
    lines.extend(
        [
            (
                "| Production | Role | Energy (GeV) | Zenith (deg) | Jobs | "
                "Wall time (s/event, median) | CPU time (s/event, median) | "
                "Peak RSS (bytes, median) | Storage (bytes/event, median) |"
            ),
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in summary:
        lines.append(
            f"| {item['production_label']} | {item['role']} | "
            f"{item['energy_midpoint_gev']:.6g} | {item['zenith_angle_deg']:.6g} | "
            f"{item['job_count']} | {_format_value(item['wall_time_seconds_per_event_median'])} | "
            f"{_format_value(item['cpu_time_seconds_per_event_median'])} | "
            f"{_format_value(item['peak_rss_bytes_median'])} | "
            f"{_format_value(item['sim_telarray_storage_bytes_per_event_median'])} |"
        )
    if diagnostics:
        lines.extend(
            ["", "## Skipped records", "", "| Job | Role | Reason |", "| --- | --- | --- |"]
        )
        lines.extend(
            f"| {item['job_id']} | {item['role']} | {item['message']} |" for item in diagnostics
        )
    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resource_row(manifest, role, label):
    """Build one validated normalized resource row."""
    record_path = _resource_record_path(manifest, role)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    _validate_record(record, record_path, manifest.run_number, role)
    configuration = manifest.data["configuration"]
    showers = configuration["showers_per_run"]
    energy_min = _quantity_value(configuration["energy_min"])
    energy_max = _quantity_value(configuration["energy_max"])
    output_sizes = _output_sizes(manifest, role)
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
        "job_id": manifest.data["job_id"],
        "run_number": manifest.run_number,
        "role": role,
        "primary": configuration["primary"],
        "site": configuration["site"],
        "array_layout_name": _layout_name(configuration["array_layout_name"]),
        "model_version": str(configuration["model_version"]),
        "simtools_version": record.get("simtools_version")
        or str(manifest.data.get("production_id") or "unknown"),
        "energy_min_gev": energy_min,
        "energy_max_gev": energy_max,
        "energy_midpoint_gev": float(np.sqrt(energy_min * energy_max)),
        "zenith_angle_deg": _quantity_value(configuration["zenith_angle"]),
        "azimuth_angle_deg": _quantity_value(configuration["azimuth_angle"]),
        "showers_per_run": showers,
        "wall_time_seconds": record.get("wall_time_seconds"),
        "cpu_time_seconds": cpu_time,
        "peak_rss_bytes": record.get("peak_rss_bytes"),
        "wall_time_seconds_per_event": _per_event(record.get("wall_time_seconds"), showers),
        "cpu_time_seconds_per_event": _per_event(cpu_time, showers),
        "corsika_output_bytes": output_sizes["corsika"],
        "sim_telarray_output_bytes": output_sizes["sim_telarray"],
        "reduced_event_data_bytes": output_sizes["reduced_event_data"],
        "sim_telarray_histogram_bytes": output_sizes["sim_telarray_histogram"],
        "corsika_output_bytes_per_event": _per_event(output_sizes["corsika"], showers),
        "sim_telarray_output_bytes_per_event": _per_event(output_sizes["sim_telarray"], showers),
        "reduced_event_data_bytes_per_event": _per_event(
            output_sizes["reduced_event_data"], showers
        ),
        "sim_telarray_histogram_bytes_per_event": _per_event(
            output_sizes["sim_telarray_histogram"], showers
        ),
        "sim_telarray_storage_bytes": storage_size,
        "sim_telarray_storage_bytes_per_event": _per_event(storage_size, showers),
        "resource_record": str(record_path),
    }


def _resource_record_path(manifest, role):
    """Find one resource record for a manifest role and run number."""
    suffix = f".{'simtel' if role == 'sim_telarray' else role}.resources.json"
    records = sorted(manifest.directory.rglob(f"*{suffix}"))
    matching = [path for path in records if f"run{manifest.run_number:06d}" in path.parts]
    if len(matching) != 1:
        raise FileNotFoundError(
            f"Expected one {role} resource record for run {manifest.run_number} "
            f"in {manifest.directory}, found {len(matching)}."
        )
    return matching[0]


def _validate_record(record, record_path, run_number, role):
    """Validate resource-record fields required by this report."""
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
        sizes[output_type] = (
            sum((manifest.directory / path).stat().st_size for path in files) or None
        )
    return sizes


def _quantity_value(value):
    """Return a manifest quantity in its configured unit."""
    return float(value["value"])


def _layout_name(value):
    """Return a stable display value for a manifest layout selection."""
    return ",".join(value) if isinstance(value, list) else str(value)


def _sum_or_none(first, second):
    """Return a sum only when both process CPU components are measured."""
    return None if first is None or second is None else first + second


def _per_event(value, showers):
    """Normalize an optional scalar by the positive number of simulated showers."""
    return None if value is None else value / showers


def _format_value(value):
    """Format an optional report value."""
    return "-" if value is None else f"{value:.6g}"
