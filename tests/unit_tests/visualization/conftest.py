"""Shared fixtures for visualization unit tests."""

import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def close_all_figures():
    """Automatically close all matplotlib figures after each test."""
    yield
    plt.close("all")
