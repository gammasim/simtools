"""Shared fixtures for visualization unit tests."""

import matplotlib.figure
import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def close_all_figures():
    """Automatically close all matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture(autouse=True)
def _fast_savefig(monkeypatch):
    """Override savefig DPI to 10 to speed up visualization unit tests."""
    original = matplotlib.figure.Figure.savefig

    def fast_savefig(self, fname, *args, **kwargs):
        kwargs["dpi"] = 10
        return original(self, fname, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", fast_savefig)
