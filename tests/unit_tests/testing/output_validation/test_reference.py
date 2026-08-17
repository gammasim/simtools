"""Tests for reference-file comparison validators."""

import json
from pathlib import Path

from simtools.testing.output_validation import reference


def test_compare_json_reference_ignores_schema_version(tmp_test_directory):
    """Ignore schema-version changes when comparing JSON references."""
    first = Path(tmp_test_directory) / "first.json"
    second = Path(tmp_test_directory) / "second.json"
    first.write_text(json.dumps({"schema_version": "1", "value": [1.0]}), encoding="utf-8")
    second.write_text(json.dumps({"schema_version": "2", "value": [1.0]}), encoding="utf-8")

    assert reference.compare_json_or_yaml_files(first, second)
