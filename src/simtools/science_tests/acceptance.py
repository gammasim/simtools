"""Evaluate science-test products against their acceptance contracts.

Expected products are file patterns that must match non-empty files. Acceptance rules
apply numerical limits to values read from JSON documents or from selected rows of CSV
result tables. A valid value outside a limit is a warning; missing or invalid evidence is
an error.
"""

import csv
import fnmatch
import json
import math
from collections import Counter
from pathlib import Path

from simtools.io.file_type import _get_product_filename
from simtools.utils.general import replace_placeholders_recursively

STATUS_KEY = "status"
REASON_KEY = "reason"
WARNINGS_KEY = "acceptance_warnings"
_ENCODING = "utf-8"
_PASS = "pass"
_SCOPE = "scope"
_TELESCOPE_TYPE = "telescope_type"
_ARRAY_LAYOUT = "array_layout"
_PLOT = "plot"
_CHANGES = "changes"
_ADVISORY = "advisory"
_RESULT_STATUS_FILE = "result_status_file"
_REPORT_PAGE = "report_page"
# Columns identifying the compared configuration and telescope type of a CSV row.
_CSV_CONTEXT = ("pair", _ARRAY_LAYOUT, _SCOPE, _TELESCOPE_TYPE)
# Columns identifying one measurement; duplicates are errors.
_CSV_IDENTITY = ("pair", _ARRAY_LAYOUT, _PLOT, _SCOPE, _TELESCOPE_TYPE, "population", "metric")
_COMPARISON_INVENTORY_FILE = _get_product_filename("comparison_inventory")


class AcceptanceError(ValueError):
    """A valid measurement exceeds a configured acceptance limit."""


def matched_files(pattern):
    """Return files matching a glob pattern, requiring at least one and no empty file.

    Parameters
    ----------
    pattern : str
        Absolute or relative glob pattern.

    Returns
    -------
    list of pathlib.Path
        Matching files.
    """
    path = Path(pattern)
    root = Path(path.anchor or ".")
    files = sorted(p for p in root.glob(str(path.relative_to(root))) if p.is_file())
    if not files or any(file.stat().st_size == 0 for file in files):
        raise ValueError(f"Missing or empty required product: {pattern}")
    return files


def nonempty(path):
    """Return whether a path is a non-empty file or a directory containing one.

    Parameters
    ----------
    path : pathlib.Path
        File or directory.

    Returns
    -------
    bool
        True for a non-empty file or a directory with a non-empty file.
    """
    if path.is_file():
        return path.stat().st_size > 0
    return path.is_dir() and any(p.is_file() and p.stat().st_size > 0 for p in path.rglob("*"))


def evaluate_products(definition, replacements):
    """Check expected products, the result-status file, and numerical acceptance rules.

    Parameters
    ----------
    definition : dict
        Test definition with ``products``, ``rule`` and optional ``result_status_file``.
    replacements : dict
        Placeholder values.

    Returns
    -------
    dict
        ``status`` (pass or warn) with reason and warnings for a warning.
    """
    for pattern in definition.get("products", []):
        matched_files(replace_placeholders_recursively(pattern, replacements))
    warnings = []
    if definition.get(_RESULT_STATUS_FILE):
        path = Path(replace_placeholders_recursively(definition[_RESULT_STATUS_FILE], replacements))
        warnings.extend(_result_status_warnings(path))
    rule = definition.get("rule", {})
    for metric in rule.get("metrics", []):
        pattern = replace_placeholders_recursively(metric["file"], replacements)
        for path in matched_files(pattern):
            try:
                warnings.extend(_evaluate_metric(path, metric))
            except AcceptanceError as exc:
                warnings.append(str(exc))
    if rule.get("mode") == _ADVISORY:
        warnings.append("Numerical acceptance is advisory; review required.")
    if not warnings:
        return {STATUS_KEY: _PASS}
    return {
        STATUS_KEY: "warn",
        REASON_KEY: f"Acceptance warning ({len(warnings)}): {warnings[0]}",
        WARNINGS_KEY: warnings,
    }


def _result_status_warnings(path):
    """Require a complete comparison result and report advisory metric changes."""
    try:
        result = json.loads(path.read_text(encoding=_ENCODING))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read result status file {path}: {exc}") from exc
    if result.get(STATUS_KEY) != "complete":
        raise ValueError(f"Science-test result is {result.get(STATUS_KEY)}: {path}")
    comparisons = result.get("comparisons") or {}
    if not comparisons or any(not item.get(_CHANGES) for item in comparisons.values()):
        raise ValueError(f"Science-test result has no metric changes: {path}")
    return [
        f"{name}: advisory tolerance exceeded for {metric}."
        for name, comparison in comparisons.items()
        for metric, change in comparison[_CHANGES].items()
        if change.get(_ADVISORY)
    ]


def _evaluate_metric(path, metric):
    """Evaluate one configured metric in one product file."""
    if metric.get("format") == "csv":
        return _evaluate_csv_metric(path, metric)
    value = json.loads(path.read_text(encoding=_ENCODING))
    try:
        for key in metric["key"].split("."):
            value = value[int(key)] if isinstance(value, list) else value[key]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing metric {metric['key']} in {path}") from exc
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"Invalid metric {metric['key']} in {path}: {value}")
    _check_limits(value, metric, metric["key"])
    return []


def _evaluate_csv_metric(path, metric):
    """Apply limits to the selected rows of a CSV result table."""
    with path.open(encoding=_ENCODING, newline="") as stream:
        rows = list(csv.DictReader(stream))
    duplicates = [
        key
        for key, count in Counter(_columns(row, _CSV_IDENTITY) for row in rows).items()
        if count > 1
    ]
    if duplicates:
        raise ValueError(f"Duplicate comparison metric rows in {path}: {duplicates[:5]}.")
    selectors = metric.get("select", {})
    selected = [row for row in rows if _matches(row, selectors)]
    warnings = _missing_csv_rows(path, rows, selected, selectors)
    if not selected and not warnings:
        warnings.append(f"No comparison rows match {selectors} in {path}.")
    field = metric.get("field", metric["key"])
    for row in selected:
        try:
            warning = _evaluate_csv_row(row, metric, field, path)
        except AcceptanceError as exc:
            warning = str(exc)
        if warning:
            warnings.append(warning)
    return warnings


def _evaluate_csv_row(row, metric, field, path):
    """Evaluate one CSV row and return a warning for invalid or limited statistics."""
    label = f"{row.get(_PLOT)} {row.get(_SCOPE)} {row.get('pair')}"
    page = f" (PDF page {row[_REPORT_PAGE]})" if row.get(_REPORT_PAGE) else ""
    if row.get("valid", "").casefold() != "true":
        return f"{label}: invalid statistic ({row.get(REASON_KEY)}){page}."
    invalid = f"Invalid acceptance value in {path}: {row}"
    try:
        value = float(row[field])
        counts = [int(float(row[key])) for key in ("baseline_count", "candidate_count")]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(invalid) from exc
    if not math.isfinite(value):
        raise ValueError(invalid)
    if min(counts) < metric.get("minimum_entries", 0):
        return f"{label}: insufficient statistics ({counts}){page}."
    _check_limits(value, metric, f"{label}: {field}", page)
    warning_limit = metric.get("warning_maximum", math.inf)
    if value > warning_limit:
        return f"{label}: {field}={value:.4g} exceeds warning limit {warning_limit:.4g}{page}."
    return None


def _check_limits(value, metric, label, suffix=""):
    """Raise an acceptance error for a value outside the configured minimum and maximum."""
    minimum, maximum = metric.get("minimum", -math.inf), metric.get("maximum", math.inf)
    if not minimum <= value <= maximum:
        raise AcceptanceError(
            f"Acceptance failed for {label}={value:.4g}: "
            f"outside [{minimum:.4g}, {maximum:.4g}]{suffix}."
        )


def _missing_csv_rows(path, rows, selected, selectors):
    """Warn for each expected configuration and telescope type without a selected row."""
    observed = {_columns(row, _CSV_CONTEXT) for row in selected}
    context_selectors = {key: value for key, value in selectors.items() if key in _CSV_CONTEXT}
    plots = selectors.get(_PLOT, [])
    plots = [plots] if isinstance(plots, str) else plots
    warnings = []
    for context in sorted(_expected_csv_contexts(path, rows) - observed):
        row = dict(zip(_CSV_CONTEXT, context))
        telescope = row[_TELESCOPE_TYPE]
        if not _matches(row, context_selectors) or (
            telescope
            and plots
            and not any(fnmatch.fnmatchcase(telescope, plot.rsplit("_", 1)[-1]) for plot in plots)
        ):
            continue
        warnings.append(f"Missing comparison metric rows for {' '.join(context)}: {selectors}.")
    return warnings


def _expected_csv_contexts(path, rows):
    """Return the configurations declared in the comparison inventory, or those in the rows."""
    inventory = path.with_name(_COMPARISON_INVENTORY_FILE)
    scopes = None
    if inventory.is_file():
        scopes = json.loads(inventory.read_text(encoding=_ENCODING)).get("metric_scopes")
    if scopes is None:
        return {_columns(row, _CSV_CONTEXT) for row in rows}
    return {
        (pair, layout, _TELESCOPE_TYPE if telescope else "whole_array", telescope)
        for pair, layouts in scopes.items()
        for layout, telescopes in layouts.items()
        for telescope in ("", *telescopes)
    }


def _columns(row, keys):
    return tuple(row.get(key, "") for key in keys)


def _matches(row, selectors):
    """Match each selector value or list of glob patterns against a row."""
    return all(
        any(fnmatch.fnmatchcase(row.get(key, ""), str(choice)) for choice in _as_list(expected))
        for key, expected in selectors.items()
    )


def _as_list(value):
    return value if isinstance(value, list) else [value]
