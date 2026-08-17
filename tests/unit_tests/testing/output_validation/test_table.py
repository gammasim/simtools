"""Tests for ECSV table validators."""

from pathlib import Path

import pytest
from astropy.table import Table

from simtools.testing.output_validation import table


def test_validate_table_rejects_too_few_rows(tmp_test_directory):
    """Reject a table that does not meet its minimum row count."""
    output = Path(tmp_test_directory) / "table.ecsv"
    Table({"id": [1]}).write(output, format="ascii.ecsv")

    with pytest.raises(AssertionError, match="rows"):
        table.validate_table(output, {"minimum_rows": 2})
