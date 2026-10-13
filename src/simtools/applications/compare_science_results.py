#!/usr/bin/python3
"""Compare exported science-test result artifacts."""

import math
from pathlib import Path

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.science_tests.camera_efficiency_results import (
    compare_manifests,
    load_manifest,
    write_json,
)
from simtools.visualization.plot_science_comparison import (
    plot_comparison,
    plot_wavelength_comparison,
)

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "baseline_path",
        type=str,
        required=True,
        help="Camera-efficiency manifest, legacy summary, or artifact directory for baseline.",
    ),
    cli.ArgumentDefinition(
        "candidate_path",
        type=str,
        required=True,
        help="Camera-efficiency manifest, legacy summary, or artifact directory for candidate.",
    ),
    cli.ArgumentDefinition("baseline_label", type=str, default="baseline"),
    cli.ArgumentDefinition("candidate_label", type=str, default="candidate"),
    cli.ArgumentDefinition(
        "compare_by",
        action="append",
        default=[],
        help="Exact manifest field path allowed to differ; repeat for each field.",
    ),
    cli.ArgumentDefinition(
        "tolerance",
        action="append",
        default=[],
        help="Advisory relative tolerance as METRIC=VALUE; repeat for each metric.",
    ),
    cli.ArgumentDefinition(
        "require_wavelength_tables",
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
        baseline = load_manifest(args["baseline_path"])
        candidate = load_manifest(args["candidate_path"])
        tolerances = _parse_tolerances(args.get("tolerance", []))
        comparison = compare_manifests(baseline, candidate, args.get("compare_by"), tolerances)
        plots, diagnostics = plot_comparison(
            comparison["changes"],
            {
                "label": args["baseline_label"],
                "case": baseline.get("case", {}),
                "software": baseline.get("software", {}),
            },
            {
                "label": args["candidate_label"],
                "case": candidate.get("case", {}),
                "software": candidate.get("software", {}),
            },
            output,
        )
        comparison["plots"].update(plots)
        comparison["diagnostics"].extend(diagnostics)
        _plot_wavelengths(
            baseline,
            candidate,
            comparison,
            output,
            args.get("require_wavelength_tables", False),
        )
    except (OSError, ValueError) as exc:
        comparison = {
            "schema_version": 1,
            "status": "incomplete",
            "baseline": {"label": args["baseline_label"]},
            "candidate": {"label": args["candidate_label"]},
            "changes": {},
            "diagnostics": [str(exc)],
            "plots": {},
        }
    write_json(output / "science-comparison.json", comparison)


def _parse_tolerances(values):
    """Parse repeated METRIC=VALUE advisory tolerances."""
    tolerances = {}
    for item in values:
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


def _plot_wavelengths(baseline, candidate, comparison, output, required):
    """Plot wavelength tables when both manifests reference them."""
    baseline_path = _artifact_path(baseline, "wavelength_table")
    candidate_path = _artifact_path(candidate, "wavelength_table")
    if baseline_path is None or candidate_path is None:
        if required:
            comparison["status"] = "incomplete"
            comparison["diagnostics"].append("Required wavelength tables are missing.")
        else:
            comparison["diagnostics"].append(
                "Wavelength comparison unavailable for legacy artifact."
            )
        return
    path, diagnostic = plot_wavelength_comparison(baseline_path, candidate_path, output)
    if diagnostic:
        comparison["diagnostics"].append(diagnostic)
        if required:
            comparison["status"] = "incomplete"
    else:
        comparison["plots"]["wavelength"] = path


def _artifact_path(manifest, name):
    """Resolve a declared result artifact relative to its source manifest."""
    artifact = manifest.get("artifacts", {}).get(name)
    root = manifest.get("_root")
    if artifact is None or root is None:
        return None
    path = Path(root) / artifact
    return path if path.is_file() else None


if __name__ == "__main__":
    main()
