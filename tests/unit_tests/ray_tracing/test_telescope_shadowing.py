"""Tests for telescope shadowing calculations."""

import sys
from types import SimpleNamespace

import pytest

from simtools.ray_tracing import telescope_shadowing


def test_shadowing_without_obdeect_gives_a_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "simtools.ray_tracing.backends.obdeect", None)
    with pytest.raises(ImportError, match="obdeect"):
        telescope_shadowing.shadowing_shares(SimpleNamespace(), 20.0, 10, 1)
