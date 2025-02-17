import numpy as np
import pytest
from astropy.table import Table

from simtools.production_configuration.limits_calculation import LimitCalculator


@pytest.fixture
def hdf5_file():
    angle_table = Table(
        data={"angle": [0.1, 0.2, 0.3]},
        meta={
            "Title": "angle_to_observing_position__triggered_showers_",
            "x_bin_edges": [0, 0.1, 0.2, 0.3, 0.4],
        },
    )
    event_table = Table(
        data={"weight": [1, 2, 3]},
        meta={
            "Title": "event_weight__ra3d__log10_e__",
            "x_bin_edges": [0, 1, 2, 3, 4],
            "y_bin_edges": [0, 1, 2, 3, 4],
        },
    )
    return [angle_table, event_table]


@pytest.fixture
def limit_calculator(hdf5_file):
    return LimitCalculator(hdf5_file)


def test_compute_threshold_lower(limit_calculator):
    event_weight_array = np.array([[1, 2, 3], [4, 5, 6]])
    bin_edges = np.array([0, 1, 2, 3])
    loss_fraction = 0.5
    bin_index, bin_edge_value = limit_calculator.compute_threshold(
        event_weight_array, bin_edges, loss_fraction, axis=0, limit_type="lower"
    )
    assert bin_index == 1
    assert bin_edge_value == 3


def test_compute_threshold_upper(limit_calculator):
    event_weight_array = np.array([[1, 2, 3], [4, 5, 6]])
    bin_edges = np.array([0, 1, 2, 3])
    loss_fraction = 0.5
    bin_index, bin_edge_value = limit_calculator.compute_threshold(
        event_weight_array, bin_edges, loss_fraction, axis=1, limit_type="upper"
    )
    assert bin_index == 1
    assert bin_edge_value == 1


def test_compute_lower_energy_limit(limit_calculator):
    loss_fraction = 0.5
    lower_energy_limit = limit_calculator.compute_lower_energy_limit(loss_fraction)
    assert lower_energy_limit == 1


def test_compute_upper_radial_distance(limit_calculator):
    loss_fraction = 0.2
    upper_radial_distance = limit_calculator.compute_upper_radial_distance(loss_fraction)
    assert upper_radial_distance == 2


def test_compute_viewcone(limit_calculator):
    loss_fraction = 0.1
    viewcone = limit_calculator.compute_viewcone(loss_fraction)
    assert viewcone == 0
