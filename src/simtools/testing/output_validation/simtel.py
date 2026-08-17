"""sim_telarray output and configuration validators."""

import logging
import re

from simtools.simtel import simtel_validate_metadata
from simtools.testing import assertions

_logger = logging.getLogger(__name__)
_CFG_IGNORE_KEYS = ("config_release", "Label", "simtools_")


def validate_output(path, rule):
    """Validate sim_telarray event data and metadata."""
    expected_output = {"event_type": rule.get("event_type", "shower")}
    expected_output.update(
        {name: expectation["range"] for name, expectation in rule.get("event", {}).items()}
    )
    config = {"expected_sim_telarray_output": expected_output}
    if "metadata" in rule:
        config["expected_sim_telarray_metadata"] = rule["metadata"]
    if not assertions.check_output_from_sim_telarray(path, config):
        raise AssertionError(f"Output '{path}' failed sim_telarray validation.")


def _assignment_metadata_keys():
    registry = simtel_validate_metadata.get_meta_parameter_registry(validate=False)
    return {
        name
        for name, definition in registry["meta_parameters"].items()
        if definition["mode"] == "assign"
    }


def _split_lines(lines, file_label):
    parameters = {}
    control_lines = []
    metadata_keys = _assignment_metadata_keys()
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("%"):
            continue
        if any(ignore_key in line for ignore_key in _CFG_IGNORE_KEYS):
            _logger.debug(f"Ignoring line in {file_label}: {line}")
            continue
        if re.match(r"metaparam (global|telescope) (add|set)\b", line):
            continue
        key, separator, value = line.partition("=")
        key = key.strip()
        if separator and key.replace("_", "").isalnum():
            if key not in metadata_keys:
                parameters[key] = value.strip()
            continue
        control_lines.append(line)
    return parameters, control_lines


def compare_config_files(reference_file, output_file):
    """Compare sim_telarray parameters and control lines."""
    with (
        open(reference_file, encoding="utf-8") as reference,
        open(output_file, encoding="utf-8") as output,
    ):
        reference_parts = _split_lines(reference, reference_file)
        output_parts = _split_lines(output, output_file)
    if reference_parts != output_parts:
        _logger.error(f"sim_telarray configuration differs: {reference_file} != {output_file}")
        return False
    return True
