#!/usr/bin/python3

import logging

import numpy as np
import pytest

from simtools.model.camera import Camera

logger = logging.getLogger()


def test_get_number_of_pixels(telescope_model_lst):
    telescope_model_lst.export_model_files()
    assert telescope_model_lst.camera.get_number_of_pixels() == 1855  # Value for LST


def test_pixel_solid_angle(telescope_model_lst):
    tel_model = telescope_model_lst
    telescope_model_lst.export_model_files()
    pix_solid_angle = tel_model.camera.get_pixel_active_solid_angle()
    logger.debug(f"Pixel solid angle is {pix_solid_angle}")

    assert pix_solid_angle * 1e6 == pytest.approx(2.43, 0.01)  # Value for LST


def test_find_neighbors_square():
    x_pos = np.array([0, 0, 1, 1])
    y_pos = np.array([0, 1, 0, 1])

    # Test with radius 1
    radius_1 = 1.0
    expected_neighbors_radius_1 = [
        [1, 2],  # Neighbors of point (0, 0)
        [0, 3],  # Neighbors of point (0, 1)
        [0, 3],  # Neighbors of point (1, 0)
        [1, 2],  # Neighbors of point (1, 1)
    ]
    neighbors_radius_1 = Camera._find_neighbors(x_pos, y_pos, radius_1)
    assert neighbors_radius_1 == expected_neighbors_radius_1

    # Test with radius sqrt(2)
    radius_sqrt_2 = np.sqrt(2)
    expected_neighbors_radius_sqrt_2 = [
        [1, 2, 3],  # Neighbors of point (0, 0)
        [0, 2, 3],  # Neighbors of point (0, 1)
        [0, 1, 3],  # Neighbors of point (1, 0)
        [0, 1, 2],  # Neighbors of point (1, 1)
    ]
    neighbors_radius_sqrt_2 = Camera._find_neighbors(x_pos, y_pos, radius_sqrt_2)
    assert neighbors_radius_sqrt_2 == expected_neighbors_radius_sqrt_2
