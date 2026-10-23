#!/usr/bin/python3
"""Compare atmospheric profile and transmission model parameters."""

import json
import logging
from pathlib import Path

from simtools.application.definition import ApplicationDefinition
from simtools.application.model_reader import require_model_reader
from simtools.atmosphere.atmosphere_characterization import characterize_atmospheres
from simtools.configuration import arguments as cli
from simtools.configuration.argument_helpers import site as validate_site
from simtools.data_model import schema
from simtools.data_model.table_asset import read_ecsv_asset

_ATMOSPHERE = "atmosphere"
_PROFILE = "atmospheric_profile"
_TRANSMISSION = "atmospheric_transmission"
_RELATIVE_DIFFERENCE_LIMIT = "relative_difference_limit"
_DEFAULT_RELATIVE_DIFFERENCE_LIMIT = 35.0
_FALLBACK_TO_CANDIDATE = "fallback_to_candidate"
_CANDIDATE_COMMON_WAVELENGTH_COUNT = "candidate_common_wavelength_count"
_ECSV_SUFFIX = ".ecsv"
_SUMMARY_FILE = "atmosphere_summary.json"
_WARNINGS = "warnings"
_ATMOSPHERES = "atmospheres"
_TRANSMISSION_DIFFERENCE = "transmission_difference"
_COMMON_WAVELENGTH_COUNT = "common_wavelength_count"

logger = logging.getLogger(__name__)

APPLICATION = ApplicationDefinition.for_module(
    __name__,
    model_repository=True,
    arguments=(
        cli.ArgumentDefinition(
            _ATMOSPHERE,
            action="append",
            nargs=3,
            metavar=("SITE", "PROFILE", "TRANSMISSION"),
            required=True,
            help=(
                "Atmosphere input; repeat to compare. The first input is the reference. "
                "PROFILE and TRANSMISSION are each a parameter version or a path to an "
                "ECSV file."
            ),
        ),
        cli.ArgumentDefinition(
            _RELATIVE_DIFFERENCE_LIMIT,
            type=float,
            default=_DEFAULT_RELATIVE_DIFFERENCE_LIMIT,
            help="Symmetric relative-difference axis limit in percent.",
        ),
        cli.ArgumentDefinition(
            _FALLBACK_TO_CANDIDATE,
            action="store_true",
            default=False,
            help=(
                "Plot candidate atmospheres when the reference ECSV is missing, and record "
                "the fallback in the summary."
            ),
        ),
        *cli.OUTPUT_PATH_ARGUMENTS,
    ),
)


def _is_ecsv_file(value):
    """Return True if an atmosphere input value refers to an ECSV file."""
    return str(value).lower().endswith(_ECSV_SUFFIX)


def _read_ecsv_file(parameter, file_name):
    """Read and validate an atmospheric ECSV table against its parameter schema."""
    path = Path(file_name)
    if not path.is_file():
        raise FileNotFoundError(f"Atmosphere table for {parameter} not found: {path}")
    entries = [
        entry
        for entry in schema.get_model_parameter_schema(parameter).get("data", [])
        if entry.get("type") == "file"
    ]
    return read_ecsv_asset(path, schema_entry=entries[0] if entries else None)


def _read_table(context, parameter, site, value):
    """Read an atmospheric table from an ECSV file or the model repository."""
    if _is_ecsv_file(value):
        return _read_ecsv_file(parameter, value)
    return require_model_reader(context.model_reader).export_model_file(
        parameter=parameter,
        site=site,
        array_element_name=None,
        parameter_version=value,
        export_file_as_table=True,
        dest=context.io_handler.get_output_directory(),
    )


def _version(value, table):
    """Return the parameter version from its input value or ECSV metadata."""
    if _is_ecsv_file(value):
        return table.meta.get("parameter_version")
    return value


def _label(site, index):
    """Return a short role label for an atmosphere within its site comparison."""
    if index == 0:
        return f"{site} reference"
    if index == 1:
        return f"{site} candidate"
    return f"{site} candidate {index}"


def _candidate_common_wavelength_count(summary):
    """Return the smallest common-wavelength count across candidate comparisons."""
    candidates = summary[_ATMOSPHERES][1:] or summary[_ATMOSPHERES]
    counts = []
    for item in candidates:
        difference = item[_TRANSMISSION_DIFFERENCE]
        counts.append(difference[_COMMON_WAVELENGTH_COUNT] if difference else 0)
    return min(counts, default=0)


def main():
    """Compare atmospheres using the configured model repository or ECSV files."""
    context = APPLICATION.start()
    output_path = context.io_handler.get_output_directory()
    atmospheres = []
    site_counts = {}
    fallback_warnings = []
    inputs = context.args[_ATMOSPHERE]
    for input_index, (site, profile, transmission) in enumerate(inputs):
        site = validate_site(site)
        index = site_counts.get(site, 0)
        site_counts[site] = index + 1
        try:
            profile_table = _read_table(context, _PROFILE, site, profile)
            transmission_table = _read_table(context, _TRANSMISSION, site, transmission)
        except FileNotFoundError as exc:
            has_candidate = any(
                validate_site(candidate[0]) == site for candidate in inputs[input_index + 1 :]
            )
            if index != 0 or not context.args[_FALLBACK_TO_CANDIDATE] or not has_candidate:
                raise
            warning = (
                f"Reference atmosphere for {site} is unavailable ({exc}); "
                "generating plots from candidate atmosphere only."
            )
            logger.warning(warning)
            fallback_warnings.append(warning)
            continue
        atmospheres.append(
            {
                "label": _label(site, index),
                "site": site,
                "profile_version": _version(profile, profile_table),
                "transmission_version": _version(transmission, transmission_table),
                "profile": profile_table,
                "transmission": transmission_table,
            }
        )
    summary = characterize_atmospheres(
        atmospheres,
        output_path,
        relative_difference_limit=context.args[_RELATIVE_DIFFERENCE_LIMIT],
    )
    summary[_FALLBACK_TO_CANDIDATE] = int(bool(fallback_warnings))
    summary[_CANDIDATE_COMMON_WAVELENGTH_COUNT] = _candidate_common_wavelength_count(summary)
    if fallback_warnings:
        summary[_WARNINGS] = fallback_warnings
    (Path(output_path) / _SUMMARY_FILE).write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
