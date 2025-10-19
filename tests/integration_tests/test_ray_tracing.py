#!/usr/bin/python3

import logging
from copy import copy

import astropy.units as u
import matplotlib.pyplot as plt
import pytest

from simtools.model.telescope_model import TelescopeModel
from simtools.ray_tracing.ray_tracing import RayTracing

logger = logging.getLogger()


@pytest.mark.parametrize("telescope_model_name", ["SSTS-design"])
def test_ssts(
    telescope_model_name,
    db_config,
    simtel_path_no_mock,
    io_handler,
    model_version,
    site_model_south,
):
    # Test with 3 SSTs
    tel = TelescopeModel(
        site="south",
        telescope_name=telescope_model_name,
        model_version=model_version,
        label="test-sst",
        db_config=db_config,
    )

    ray = RayTracing(
        telescope_model=tel,
        site_model=site_model_south,
        simtel_path=simtel_path_no_mock,
        zenith_angle=20.0 * u.deg,
        source_distance=10.0 * u.km,
        off_axis_angle=[0, 1.0, 2.0, 3.0, 4.0] * u.deg,
    )
    ray.simulate(test=True, force=True)
    ray.analyze(force=True)


def test_rx(simtel_path_no_mock, io_handler, telescope_model_lst, site_model_north):
    ray = RayTracing(
        telescope_model=telescope_model_lst,
        site_model=site_model_north,
        simtel_path=simtel_path_no_mock,
        zenith_angle=20 * u.deg,
        source_distance=10 * u.km,
        off_axis_angle=[0, 2.5, 5.0] * u.deg,
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

    plot_file_psf = io_handler.get_output_file(file_name="d80_test_rx.pdf", sub_dir="plots")
    plt.savefig(plot_file_psf)
    plt.close()

    # Plotting eff_area
    plt.figure(figsize=(8, 6), tight_layout=True)
    ax = plt.gca()
    ax.set_xlabel("off-axis")
    ax.set_ylabel("eff. area")

    ray.plot("eff_area", marker="o", linestyle=":")
    ray_rx.plot("d80_deg", marker="s", linestyle="--")

    plot_file_area = io_handler.get_output_file(file_name="eff_area_test_rx.pdf", sub_dir="plots")
    plt.savefig(plot_file_area)
    plt.close()


def test_plot_image(simtel_path_no_mock, io_handler, telescope_model_sst, site_model_south):
    ray = RayTracing(
        telescope_model=telescope_model_sst,
        site_model=site_model_south,
        simtel_path=simtel_path_no_mock,
        zenith_angle=20 * u.deg,
        source_distance=10 * u.km,
        off_axis_angle=[0, 2.5, 5.0] * u.deg,
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
            file_name=f"test_plot_image_{ii}.pdf", sub_dir="plots"
        )
        plt.savefig(plot_file)
        plt.close()


def test_single_mirror(simtel_path_no_mock, io_handler, telescope_model_mst, site_model_south):
    """Test MST, single mirror PSF simulation"""

    telescope_model_mst.write_sim_telarray_config_file(site_model_south)
    ray = RayTracing(
        telescope_model=telescope_model_mst,
        site_model=site_model_south,
        simtel_path=simtel_path_no_mock,
        mirror_numbers=list(range(1, 5)),
        single_mirror_mode=True,
    )
    ray.simulate(test=True, force=True)
    ray.analyze(force=True)

    # Plotting d80 histogram
    plt.figure(figsize=(8, 6), tight_layout=True)
    ax = plt.gca()
    ax.set_xlabel("d80")

    ray.plot_histogram("d80_cm", color="r", bins=10)
    plot_file = io_handler.get_output_file(file_name="d80_hist_test.pdf", sub_dir="plots")
    plt.savefig(plot_file)
    plt.close()


def test_integral_curve(simtel_path_no_mock, io_handler, telescope_model_lst, site_model_north):
    ray = RayTracing(
        telescope_model=telescope_model_lst,
        site_model=site_model_north,
        simtel_path=simtel_path_no_mock,
        zenith_angle=20 * u.deg,
        source_distance=10 * u.km,
        off_axis_angle=[0] * u.deg,
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
    plot_file = io_handler.get_output_file(file_name="test_cumulative_psf.pdf", sub_dir="plots")
    plt.savefig(plot_file)
    plt.close()
