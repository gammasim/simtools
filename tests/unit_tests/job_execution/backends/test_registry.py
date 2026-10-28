"""Tests for execution backend registration."""

import pytest

import simtools.job_execution.backends.registry as registry
from simtools.job_execution.backends.base import BackendConfigurationError
from simtools.job_execution.backends.htcondor import HTCondorBackend
from simtools.job_execution.backends.local import LocalBackend


def test_builtin_backends_are_available():
    """The standard local and HTCondor backends are registered."""
    assert {"local", "htcondor"}.issubset(registry.available_backends())


def test_builtin_backend_factories_return_expected_implementations():
    """Built-in registrations remain lazy factories for their concrete backends."""
    assert isinstance(registry.get_backend("local"), LocalBackend)
    assert isinstance(registry.get_backend("htcondor"), HTCondorBackend)


def test_get_backend_reports_available_names():
    """Unknown backend names produce an actionable configuration error."""
    with pytest.raises(BackendConfigurationError, match="Available backends"):
        registry.get_backend("missing")
