"""Unit tests for simtools.visualization.plot_ray_tracing_psf."""

import astropy.units as u
import numpy as np
import pytest

from simtools.visualization import plot_ray_tracing_psf


def test_create_cumulative_psf_figure_draws_curve_and_markers():
    data = np.array(
        [(0.0, 0.0), (1.0, 0.5), (2.0, 1.0)],
        dtype=[("Radius [cm]", "f8"), ("Cumulative PSF", "f8")],
    )
    _fig, ax = plot_ray_tracing_psf.create_cumulative_psf_figure(
        data,
        radius_key="Radius [cm]",
        cumulative_key="Cumulative PSF",
        containment_radius_cm=1.5,
        psf_diameter_cm=4.0,
        color="blue",
    )
    assert ax.get_xlabel() == "Radius (cm)"
    assert ax.get_ylabel() == "Contained light %"
    # One curve plus two vertical markers (containment radius and PSF diameter / 2)
    assert len(ax.lines) == 3
    assert ax.lines[1].get_xdata()[0] == pytest.approx(1.5)
    assert ax.lines[2].get_xdata()[0] == pytest.approx(2.0)
    assert ax.get_xlim()[1] >= 2.0


def test_create_psf_image_figure_draws_histogram_circle_and_axes():
    data = np.array([(0.0, 0.0), (1.0, -1.0)], dtype=[("X", "f8"), ("Y", "f8")])
    _fig, ax = plot_ray_tracing_psf.create_psf_image_figure(
        data,
        containment_radius=2.0,
        center=(0.0, 0.0),
        bins=10,
        image_range=[[-3.0, 3.0], [-3.0, 3.0]],
        psf_kwargs={"color": "k", "fill": False},
    )
    assert ax.get_xlabel() == "X Position (cm)"
    assert ax.get_ylabel() == "Y Position (cm)"
    assert len(ax.patches) == 1  # containment circle
    assert len(ax.lines) == 2  # reference axes


def test_create_annotated_psf_image_figure_adds_text():
    data = np.array([(0.0, 0.0), (1.0, -1.0)], dtype=[("X", "f8"), ("Y", "f8")])
    fig = plot_ray_tracing_psf.create_annotated_psf_image_figure(
        data,
        off_x=0.5,
        off_y=-0.5,
        psf=4.2 * u.cm,
        containment_radius=2.1 * u.cm,
        image_range=[[-3.0, 3.0], [-3.0, 3.0]],
        cmap="gist_heat_r",
        telescope_name="LSTN-01",
    )
    ax = fig.axes[0]
    assert ax.get_xlabel() == "X Position (cm)"
    assert ax.texts[0].get_text() == ("LSTN-01\nOffset: (+0.50 deg, -0.50 deg)\nPSF: 4.20 cm")


def test_create_annotated_psf_image_figure_without_telescope_name():
    data = np.array([(0.0, 0.0)], dtype=[("X", "f8"), ("Y", "f8")])
    fig = plot_ray_tracing_psf.create_annotated_psf_image_figure(
        data,
        off_x=0.0,
        off_y=0.0,
        psf=0.05 * u.deg,
        containment_radius=0.02 * u.deg,
        image_range=[[-1.0, 1.0], [-1.0, 1.0]],
    )
    text = fig.axes[0].texts[0].get_text()
    assert text == "Offset: (+0.00 deg, +0.00 deg)\nPSF: 0.0500 deg"
