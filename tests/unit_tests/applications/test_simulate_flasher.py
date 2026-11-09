"""Tests for flasher application argument validation."""

import argparse

import pytest

from simtools.applications.simulate_flasher import _validate_max_photoelectrons


@pytest.mark.parametrize("value", [None, 1000, 499999999])
def test_max_photoelectrons_accepts_limits(value):
    _validate_max_photoelectrons({"max_photoelectrons": value}, {}, argparse.ArgumentParser())


@pytest.mark.parametrize("value", [999, 500000000])
def test_max_photoelectrons_rejects_outside_limits(value, capsys):
    with pytest.raises(SystemExit, match="2"):
        _validate_max_photoelectrons({"max_photoelectrons": value}, {}, argparse.ArgumentParser())
    assert "must be at least 1000 and less than 500000000" in capsys.readouterr().err
