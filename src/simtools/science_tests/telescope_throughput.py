"""Read and compare telescope-throughput validation results.

A result is the directory written by ``simtools-validate-telescope-throughput`` for one
telescope: a combined YAML summary with the case metadata (``meta``) and scalar metrics,
plus one ECSV wavelength table per efficiency channel (for example shower, muon, nsb).
"""

import json
import math
from pathlib import Path

import astropy.units as u

from simtools.io import ascii_handler

SUMMARY_PATTERN = "*_summary_*.yml"
TABLE_PATTERN = "*.ecsv"
_META_KEY = "meta"
_TELESCOPE_META_KEY = "tel"
_VALUE_KEY = "value"
_UNIT_KEY = "unit"
_STATUS_KEY = "status"
_DIAGNOSTICS_KEY = "diagnostics"
COMPLETE = "complete"
INCOMPLETE = "incomplete"
TELESCOPE = "telescope"
METRICS = "metrics"
TABLES = "tables"
CHANGES = "changes"
BASELINE = "baseline"
CANDIDATE = "candidate"


def load_results(path):
    """Load telescope results from a summary file, a result directory, or its parent.

    Parameters
    ----------
    path : str or pathlib.Path
        Combined summary file, one telescope result directory, or a directory with one
        result subdirectory per telescope.

    Returns
    -------
    dict
        Results keyed by telescope name.

    Raises
    ------
    ValueError
        No or ambiguous summaries were found.
    """
    path = Path(path)
    if path.is_file() or any(path.glob(SUMMARY_PATTERN)):
        summaries = [path]
    else:
        summaries = sorted(child for child in path.iterdir() if any(child.glob(SUMMARY_PATTERN)))
    results = {}
    for summary in summaries:
        result = read_result(summary)
        results[result[TELESCOPE]] = result
    if not results:
        raise ValueError(f"No telescope-throughput results found in {path}.")
    return results


def read_result(path):
    """Read one combined summary and the wavelength tables written beside it.

    Parameters
    ----------
    path : str or pathlib.Path
        Combined summary file or the directory containing exactly one.

    Returns
    -------
    dict
        Telescope name, case metadata, metrics as quantities, and tables by channel.
    """
    path = Path(path)
    if path.is_dir():
        summaries = sorted(path.glob(SUMMARY_PATTERN))
        if len(summaries) != 1:
            raise ValueError(f"Expected one summary matching {SUMMARY_PATTERN} in {path}.")
        path = summaries[0]
    summary = ascii_handler.collect_data_from_file(path)
    if not isinstance(summary, dict):
        raise ValueError(f"Telescope-throughput summary must be a mapping: {path}")
    case = summary.pop(_META_KEY, {})
    if not isinstance(case, dict):
        raise ValueError(f"Telescope-throughput summary metadata must be a mapping: {path}")
    return {
        TELESCOPE: case.get(_TELESCOPE_META_KEY) or path.parent.name,
        "case": case,
        METRICS: {name: _quantity(item) for name, item in summary.items()},
        TABLES: {
            table.stem.rsplit("_", 1)[-1]: table
            for table in sorted(path.parent.glob(TABLE_PATTERN))
        },
    }


def compare_results(baseline, candidate, tolerances=None, compare_by=None):
    """Compare metrics of two results with identical case settings.

    Parameters
    ----------
    baseline, candidate : dict
        Results returned by :func:`read_result`.
    tolerances : dict or None
        Advisory relative tolerances by metric name.
    compare_by : iterable of str or None
        Case metadata keys allowed to differ.

    Returns
    -------
    dict
        Status, both cases, per-metric changes, and diagnostics.
    """
    tolerances = tolerances or {}
    allowed = set(compare_by or [])
    diagnostics = [
        f"Case setting {key} differs: {baseline['case'].get(key)} != {candidate['case'].get(key)}."
        for key in sorted(set(baseline["case"]) | set(candidate["case"]))
        if key not in allowed and baseline["case"].get(key) != candidate["case"].get(key)
    ]
    metrics = set(baseline[METRICS]) | set(candidate[METRICS])
    diagnostics.extend(
        f"Tolerance given for unknown metric {name}." for name in sorted(set(tolerances) - metrics)
    )
    changes = {}
    for name in sorted(metrics):
        try:
            changes[name] = _metric_change(
                baseline[METRICS].get(name), candidate[METRICS].get(name), tolerances.get(name)
            )
        except (TypeError, ValueError, u.UnitConversionError) as exc:
            diagnostics.append(f"Metric {name}: {exc}")
    return {
        _STATUS_KEY: INCOMPLETE if diagnostics else COMPLETE,
        "case": {BASELINE: baseline["case"], CANDIDATE: candidate["case"]},
        CHANGES: changes,
        _DIAGNOSTICS_KEY: diagnostics,
    }


def parse_tolerances(values):
    """Parse ``METRIC=VALUE`` advisory relative tolerances.

    Parameters
    ----------
    values : iterable of str or list of str
        Tolerance strings, possibly nested by repeated command-line options.

    Returns
    -------
    dict
        Non-negative relative tolerances by metric name.
    """
    tolerances = {}
    for item in _flatten(values):
        name, separator, raw_value = item.partition("=")
        value = float(raw_value) if separator else math.nan
        if not name or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid tolerance {item!r}; expected METRIC=VALUE with VALUE >= 0.")
        tolerances[name] = value
    return tolerances


def write_json(path, value):
    """Write a JSON document, creating its parent directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def _metric_change(before, after, tolerance):
    """Return the change of one metric in the baseline unit."""
    if before is None or after is None:
        raise ValueError("missing in baseline or candidate")
    unit = before.unit.to_string()
    before, after = float(before.value), float(after.to_value(before.unit))
    delta = after - before
    relative = delta / before if before else None
    if not math.isfinite(delta) or (relative is not None and not math.isfinite(relative)):
        raise ValueError("change is not finite")
    return {
        BASELINE: before,
        CANDIDATE: after,
        _UNIT_KEY: unit,
        "absolute": delta,
        "relative": relative,
        "tolerance": tolerance,
        "advisory": bool(None not in (relative, tolerance) and abs(relative) > tolerance),
    }


def _quantity(item):
    """Convert a summary entry ``{value: number | {value, unit}}`` to a finite quantity."""
    value = item.get(_VALUE_KEY) if isinstance(item, dict) else item
    if isinstance(value, dict):
        value, unit = value.get(_VALUE_KEY), value.get(_UNIT_KEY, "")
    else:
        unit = ""
    invalid = f"Invalid metric value: {item!r}"
    if isinstance(value, bool):
        raise ValueError(invalid)
    try:
        quantity = u.Quantity(value, u.Unit(unit), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(invalid) from exc
    if not quantity.isscalar or not math.isfinite(quantity.value):
        raise ValueError(f"Metric value must be a finite scalar: {item!r}")
    return quantity


def _flatten(values):
    for value in values or []:
        if isinstance(value, list | tuple):
            yield from _flatten(value)
        else:
            yield value
