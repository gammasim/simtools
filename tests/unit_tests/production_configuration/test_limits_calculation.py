import astropy.units as u
import h5py
import numpy as np
import pytest

from simtools.production_configuration.limits_calculation import LimitCalculator


@pytest.fixture
def hdf5_file(tmp_path):
    file_path = tmp_path / "test_data.h5"
    vlen_int_type = h5py.special_dtype(vlen=np.int16)
    with h5py.File(file_path, "w") as f:
        grp = f.create_group("data")
        grp.create_dataset("core_x", data=np.array([0.1, 0.2, 0.3, 0.1, 0.2, 0.3]))
        grp.create_dataset("core_y", data=np.array([0.1, 0.2, 0.3, 0.1, 0.2, 0.3]))
        grp.create_dataset("simulated", data=np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0]))
        grp.create_dataset("shower_id_triggered", data=np.array([0, 1, 2, 0, 1, 2]))
        grp.create_dataset(
            "file_names", data=np.array(["file1", "file2"], dtype=h5py.string_dtype())
        )
        grp.create_dataset("shower_sim_azimuth", data=np.array([0.1, 0.5, 1.0, 1.5, 2.0, 2.5]))
        grp.create_dataset("shower_sim_altitude", data=np.array([0.3, 0.4, 1.1, 1.3, 1.4, 1.5]))
        grp.create_dataset("array_azimuths", data=np.array([0.1, 0.5, 1.0, 1.5, 2.0, 2.5]))
        grp.create_dataset("array_altitudes", data=np.array([0.2, 0.6, 1.0, 1.2, 1.3, 1.4]))
        grp.create_dataset(
            "trigger_telescope_list_list",
            (6,),
            dtype=vlen_int_type,
            data=[
                np.array([1, 2, 3], dtype=np.int16),
                np.array([1, 2], dtype=np.int16),
                np.array([2, 3], dtype=np.int16),
                np.array([1, 3], dtype=np.int16),
                np.array([1, 2, 3], dtype=np.int16),
                np.array([1, 2, 3], dtype=np.int16),
            ],
        )
    return file_path


@pytest.fixture
def limit_calculator(hdf5_file):
    return LimitCalculator(hdf5_file)


@pytest.fixture
def limit_calculator_with_telescopes(hdf5_file):
    return LimitCalculator(hdf5_file, telescope_list=[1, 2])


def test_compute_lower_energy_limit(limit_calculator):
    loss_fraction = 0.5
    lower_energy_limit = limit_calculator.compute_lower_energy_limit(loss_fraction)
    assert lower_energy_limit.unit == u.TeV
    assert lower_energy_limit.value > 0


def test_compute_lower_energy_limit_with_telescopes(limit_calculator_with_telescopes):
    loss_fraction = 0.5
    lower_energy_limit = limit_calculator_with_telescopes.compute_lower_energy_limit(loss_fraction)
    assert lower_energy_limit.unit == u.TeV
    assert lower_energy_limit.value > 0


def test_compute_upper_radial_distance(limit_calculator):
    loss_fraction = 0.2
    upper_radial_distance = limit_calculator.compute_upper_radial_distance(loss_fraction)
    assert upper_radial_distance.unit == u.m
    assert upper_radial_distance.value > 0


def test_compute_upper_radial_distance_with_telescopes(limit_calculator_with_telescopes):
    loss_fraction = 0.2
    upper_radial_distance = limit_calculator_with_telescopes.compute_upper_radial_distance(
        loss_fraction
    )
    assert upper_radial_distance.unit == u.m
    assert upper_radial_distance.value > 0


def test_compute_viewcone(limit_calculator):
    loss_fraction = 0.001
    viewcone = limit_calculator.compute_viewcone(loss_fraction)
    assert viewcone.unit == u.deg
    assert viewcone.value > 0


def test_compute_viewcone_with_telescopes(limit_calculator_with_telescopes):
    loss_fraction = 0.001
    viewcone = limit_calculator_with_telescopes.compute_viewcone(loss_fraction)
    assert viewcone.unit == u.deg
    assert viewcone.value > 0


def test_plot_data(limit_calculator):
    limit_calculator.plot_data()


def test_plot_data_with_telescopes(limit_calculator_with_telescopes):
    limit_calculator_with_telescopes.plot_data()


@pytest.fixture
def hdf5_file_missing_datasets(tmp_path):
    file_path = tmp_path / "test_data_missing.h5"
    with h5py.File(file_path, "w") as f:
        grp = f.create_group("data")
        grp.create_dataset("core_x", data=np.array([0.1, 0.2, 0.3]))
    return file_path


def test_read_event_data_missing_datasets(hdf5_file_missing_datasets):
    with pytest.raises(
        KeyError, match="One or more required datasets are missing from the 'data' group."
    ):
        LimitCalculator(hdf5_file_missing_datasets)


@pytest.fixture
def hdf5_file_missing_datagroup(tmp_path):
    file_path = tmp_path / "test_datagroup_missing.h5"
    with h5py.File(file_path, "w") as f:
        grp = f.create_group("wrong_data")
        grp.create_dataset("core_x", data=np.array([0.1, 0.2, 0.3]))
    return file_path


def test_read_event_data_missing_datagroup(hdf5_file_missing_datagroup):
    with pytest.raises(KeyError, match="data group is missing from the HDF5 file."):
        LimitCalculator(hdf5_file_missing_datagroup)


def test_adjust_shower_ids_across_files():
    shower_id_triggered = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    num_files = 3
    showers_per_file = 3
    expected_adjusted_ids = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    adjusted_ids = LimitCalculator.adjust_shower_ids_across_files(
        shower_id_triggered, num_files, showers_per_file
    )
    assert np.all(adjusted_ids == expected_adjusted_ids)
