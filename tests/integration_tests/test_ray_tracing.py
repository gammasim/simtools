#!/usr/bin/python3

import logging
from copy import copy

import astropy.units as u
import matplotlib.pyplot as plt
import pytest

from simtools.model.telescope_model import TelescopeModel
from simtools.ray_tracing import RayTracing

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.mark.parametrize("telescope_model_name", ["sst-1M", "sst-ASTRI", "sst-GCT"])
def test_ssts(telescope_model_name, db_config, simtel_path_no_mock, io_handler):
    # Test with 3 SSTs
    version = "prod3"
    config_data = {
        "source_distance": 10 * u.km,
        "zenith_angle": 20 * u.deg,
        "off_axis_angle": [0, 1.0, 2.0, 3.0, 4.0] * u.deg,
    }
    tel = TelescopeModel(
        site="south",
        telescope_model_name=telescope_model_name,
        model_version=version,
        label="test-sst",
        mongo_db_config=db_config,
    )

    ray = RayTracing(
        telescope_model=tel, simtel_source_path=simtel_path_no_mock, config_data=config_data
    )
    ray.simulate(test=True, force=True)
    ray.analyze(force=True)


def test_rx(db_config, simtel_path_no_mock, io_handler):
    version = "current"
    label = "test-lst"

    config_data = {
        "source_distance": 10 * u.km,
        "zenith_angle": 20 * u.deg,
        "off_axis_angle": [0, 2.5, 5.0] * u.deg,
    }

    tel = TelescopeModel(
        site="north",
        telescope_model_name="lst-1",
        model_version=version,
        label=label,
        mongo_db_config=db_config,
    )

    ray = RayTracing(
        telescope_model=tel, simtel_source_path=simtel_path_no_mock, config_data=config_data
    )

    ray.simulate(test=True, force=True)
    ray_rx = copy(ray)

    ray.analyze(force=True)
    ray_rx.analyze(force=True, use_rx=True)

    # Plotting d80
    plt.figure(figsize=(8, 6), tight_layout=True)
    ax = plt.gca()
    ax.set_xlabel("off-axis")
    ax.set_ylabel("d80")

    ray.plot("d80_deg", marker="o", linestyle=":")
    ray_rx.plot("d80_deg", marker="s", linestyle="--")

    plot_file_PSF = io_handler.get_output_file(
        file_name="d80_test_rx.pdf", sub_dir="plots", dir_type="test"
    )
    plt.savefig(plot_file_PSF)

    # Plotting eff_area
    plt.figure(figsize=(8, 6), tight_layout=True)
    ax = plt.gca()
    ax.set_xlabel("off-axis")
    ax.set_ylabel("eff. area")

    ray.plot("eff_area", marker="o", linestyle=":")
    ray_rx.plot("d80_deg", marker="s", linestyle="--")

    plot_file_area = io_handler.get_output_file(
        file_name="eff_area_test_rx.pdf", sub_dir="plots", dir_type="test"
    )
    plt.savefig(plot_file_area)


def test_plot_image(db_config, simtel_path_no_mock, io_handler):
    version = "prod3"
    label = "test-astri"
    config_data = {
        "source_distance": 10 * u.km,
        "zenith_angle": 20 * u.deg,
        "off_axis_angle": [0, 2.5, 5.0] * u.deg,
    }

    tel = TelescopeModel(
        site="south",
        telescope_model_name="sst-D",
        model_version=version,
        label=label,
        mongo_db_config=db_config,
    )

    ray = RayTracing(
        telescope_model=tel, simtel_source_path=simtel_path_no_mock, config_data=config_data
    )

    ray.simulate(test=True, force=True)
    ray.analyze(force=True)

    # Plotting images
    for ii, image in enumerate(ray.images()):
        plt.figure(figsize=(8, 6), tight_layout=True)
        ax = plt.gca()
        ax.set_xlabel("X [cm]")
        ax.set_ylabel("Y [cm]")
        image.plot_image(psf_color="b")
        plot_file = io_handler.get_output_file(
            file_name=f"test_plot_image_{ii}.pdf", sub_dir="plots", dir_type="test"
        )
        plt.savefig(plot_file)


def test_single_mirror(db_config, simtel_path_no_mock, io_handler, plot=False):
    # Test MST, single mirror PSF simulation
    version = "prod3"
    config_data = {"mirror_numbers": list(range(1, 5)), "single_mirror_mode": True}

    tel = TelescopeModel(
        site="north",
        telescope_model_name="mst-FlashCam-D",
        model_version=version,
        label="test-mst",
        mongo_db_config=db_config,
    )

    ray = RayTracing(
        telescope_model=tel, simtel_source_path=simtel_path_no_mock, config_data=config_data
    )
    ray.simulate(test=True, force=True)
    ray.analyze(force=True)

    # Plotting d80 histogram
    plt.figure(figsize=(8, 6), tight_layout=True)
    ax = plt.gca()
    ax.set_xlabel("d80")

    ray.plot_histogram("d80_cm", color="r", bins=10)
    plot_file = io_handler.get_output_file(
        file_name="d80_hist_test.pdf", sub_dir="plots", dir_type="test"
    )
    plt.savefig(plot_file)


def test_integral_curve(db_config, simtel_path_no_mock, io_handler):
    version = "prod4"
    label = "lst_integral"

    config_data = {
        "source_distance": 10 * u.km,
        "zenith_angle": 20 * u.deg,
        "off_axis_angle": [0] * u.deg,
    }

    tel = TelescopeModel(
        site="north",
        telescope_model_name="mst-FlashCam-D",
        model_version=version,
        label=label,
        mongo_db_config=db_config,
    )

    ray = RayTracing(
        telescope_model=tel, simtel_source_path=simtel_path_no_mock, config_data=config_data
    )

    ray.simulate(test=True, force=True)
    ray.analyze(force=True)

    # Plotting cumulative curve for each image
    plt.figure(figsize=(8, 6), tight_layout=True)
    ax = plt.gca()
    ax.set_xlabel("radius [cm]")
    ax.set_ylabel("relative intensity")
    for im in ray.images():
        im.plot_cumulative(color="b")
    plot_file = io_handler.get_output_file(
        file_name="test_cumulative_psf.pdf", sub_dir="plots", dir_type="test"
    )
    plt.savefig(plot_file)
