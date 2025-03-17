import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pytest
import tables

from simtools.production_configuration.derive_corsika_limits import LimitCalculator


@pytest.fixture
def hdf5_file(tmp_path):
    file_path = tmp_path / "test_data.h5"
    with tables.open_file(file_path, mode="w") as f:
        group = f.create_group("/", "data")
        reduced_data = f.create_table(
            group,
            "reduced_data",
            {
                "core_x": tables.Float32Col(),
                "core_y": tables.Float32Col(),
                "simulated": tables.Float32Col(),
                "shower_sim_azimuth": tables.Float32Col(),
                "shower_sim_altitude": tables.Float32Col(),
                "array_azimuths": tables.Float32Col(),
                "array_altitudes": tables.Float32Col(),
            },
        )
        triggered_data = f.create_table(
            group, "triggered_data", {"shower_id_triggered": tables.Int32Col()}
        )
        file_names = f.create_table(group, "file_names", {"file_names": tables.StringCol(16)})

        reduced_data.append(
            [
                (0.1, 0.1, 1.0, 0.1, 0.3, 0.1, 0.2),
                (0.2, 0.2, 2.0, 0.5, 0.4, 0.5, 0.6),
                (0.3, 0.3, 3.0, 1.0, 1.1, 1.0, 1.0),
                (0.1, 0.1, 1.0, 1.5, 1.3, 1.5, 1.2),
                (0.2, 0.2, 2.0, 2.0, 1.4, 2.0, 1.3),
                (0.3, 0.3, 3.0, 2.5, 1.5, 2.5, 1.4),
            ]
        )
        triggered_data.append([(0,), (1,), (2,), (0,), (1,), (2,)])
        file_names.append([("file1",), ("file2",)])
        f.create_vlarray(
            group,
            "trigger_telescope_list_list",
            tables.Int16Atom(),
            "List of triggered telescope IDs",
        )
        f.root.data.trigger_telescope_list_list.append(np.array([1, 2, 3], dtype=np.int16))
        f.root.data.trigger_telescope_list_list.append(np.array([1, 2], dtype=np.int16))
        f.root.data.trigger_telescope_list_list.append(np.array([2, 3], dtype=np.int16))
        f.root.data.trigger_telescope_list_list.append(np.array([1, 3], dtype=np.int16))
        f.root.data.trigger_telescope_list_list.append(np.array([1, 2, 3], dtype=np.int16))
        f.root.data.trigger_telescope_list_list.append(np.array([1, 2, 3], dtype=np.int16))
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
    assert plt.gcf().number == 1
    assert len(plt.gcf().get_axes()) > 0
    plt.close()


def test_plot_data_with_telescopes(limit_calculator_with_telescopes):
    limit_calculator_with_telescopes.plot_data()
    assert plt.gcf().number == 1
    assert len(plt.gcf().get_axes()) > 0
    plt.close()


@pytest.fixture
def hdf5_file_missing_datasets(tmp_path):
    file_path = tmp_path / "test_data_missing.h5"
    with tables.open_file(file_path, mode="w") as f:
        group = f.create_group("/", "data")
        reduced_data = f.create_table(group, "reduced_data", {"core_x": tables.Float32Col()})
        reduced_data.append([(0.1,), (0.2,), (0.3,)])
        # Add the triggered_data group but leave it empty
        f.create_group(group, "triggered_data")
        # Add the file_names group but leave it empty
        f.create_group(group, "file_names")
    return file_path


def test_read_event_data_missing_datasets(hdf5_file_missing_datasets):
    with pytest.raises(
        tables.NoSuchNodeError,
        match="group ``/data`` does not have a child named ``trigger_telescope_list_list``",
    ):
        LimitCalculator(hdf5_file_missing_datasets)


@pytest.fixture
def hdf5_file_missing_datagroup(tmp_path):
    file_path = tmp_path / "test_datagroup_missing.h5"
    with tables.open_file(file_path, mode="w") as f:
        group = f.create_group("/", "wrong_data")
        reduced_data = f.create_table(group, "reduced_data", {"core_x": tables.Float32Col()})
        reduced_data.append([(0.1,), (0.2,), (0.3,)])
    return file_path


def test_read_event_data_missing_datagroup(hdf5_file_missing_datagroup):
    with pytest.raises(
        tables.NoSuchNodeError, match="group ``/`` does not have a child named ``data``"
    ):
        LimitCalculator(hdf5_file_missing_datagroup)
