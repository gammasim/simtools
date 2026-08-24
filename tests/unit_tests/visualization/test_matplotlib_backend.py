"""Tests for the shared Matplotlib backend configuration."""

from simtools.visualization.matplotlib_backend import pyplot


def test_matplotlib_backend_is_non_interactive():
    """The shared pyplot module uses the non-interactive Agg backend."""
    assert pyplot.get_backend().lower() == "agg"
