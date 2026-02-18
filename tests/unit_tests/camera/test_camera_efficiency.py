#!/usr/bin/python3

import logging
from pathlib import Path
from unittest.mock import MagicMock

import astropy.units as u
import numpy as np
import pytest
from astropy.table import Table

from simtools.camera.camera_efficiency import CameraEfficiency
from simtools.simtel.simulator_camera_efficiency import SimulatorCameraEfficiency

logger = logging.getLogger()


@pytest.fixture
def config_data_lst(model_version_prod5):
    return {
        "telescope": "LSTN-01",
        "site": "North",
        "model_version": model_version_prod5,
        "zenith_angle": 20 * u.deg,
        "azimuth_angle": 0 * u.deg,
    }


@pytest.fixture
def camera_efficiency_lst(config_data_lst, mocker):
    camera_eff = CameraEfficiency(
        config_data=config_data_lst, efficiency_type="shower", label="validate_camera_efficiency"
    )
    mock_camera = MagicMock()
    mock_camera.get_camera_fill_factor.return_value = 0.8
    mock_camera.get_pixel_active_solid_angle.return_value = 1.0e-6
    mocker.patch.object(
        type(camera_eff.telescope_model),
        "camera",
        new_callable=mocker.PropertyMock,
        return_value=mock_camera,
    )
    mocker.patch.object(
        camera_eff.telescope_model,
        "get_on_axis_eff_optical_area",
        return_value=100.0 * u.m**2,
    )
    return camera_eff


@pytest.fixture
def prepare_results_file(camera_efficiency_lst, mocker):
    test_resource_file = Path(
        "tests/resources/"
        "camera_efficiency_table_North_LSTN-01_za20.0deg_azm000deg_validate_camera_efficiency.ecsv"
    )
    mocker.patch.object(
        camera_efficiency_lst,
        "_file",
        {
            "results": test_resource_file,
            "sim_telarray": camera_efficiency_lst._file["sim_telarray"],
            "log": camera_efficiency_lst._file["log"],
        },
    )
    return test_resource_file


def test_report(camera_efficiency_lst):
    assert str(camera_efficiency_lst) == "CameraEfficiency(label=validate_camera_efficiency)\n"


def test_load_files(camera_efficiency_lst):
    # The code generates files with efficiency_type="shower" in the name
    _name = "camera_efficiency_North_LSTN-01_za20.0deg_azm000deg_shower"
    assert camera_efficiency_lst._file["results"].name == _name + ".ecsv"
    assert camera_efficiency_lst._file["sim_telarray"].name == _name + ".dat"
    assert camera_efficiency_lst._file["log"].name == _name + ".log"


def test_simulate(camera_efficiency_lst, caplog, mocker):
    mock_run = mocker.patch.object(SimulatorCameraEfficiency, "run")
    with caplog.at_level(logging.INFO):
        camera_efficiency_lst.simulate()
        assert "Simulating CameraEfficiency" in caplog.text
    mock_run.assert_called_once()


def test_read_results(camera_efficiency_lst, prepare_results_file):
    camera_efficiency_lst._read_results()
    assert isinstance(camera_efficiency_lst._results, Table)
    assert camera_efficiency_lst._has_results is True


def test_calc_camera_efficiency(camera_efficiency_lst, prepare_results_file):
    camera_efficiency_lst._read_results()
    result = camera_efficiency_lst.calc_camera_efficiency()
    assert isinstance(result, float)
    assert 0.0 < result < 1.0  # Efficiency should be between 0 and 1


def test_calc_tel_efficiency(camera_efficiency_lst, prepare_results_file):
    camera_efficiency_lst._read_results()
    result = camera_efficiency_lst.calc_tel_efficiency()
    assert isinstance(result, float)
    assert 0.0 < result < 1.0  # Efficiency should be between 0 and 1


def test_calc_tot_efficiency(camera_efficiency_lst, prepare_results_file):
    camera_efficiency_lst._read_results()
    tel_efficiency = camera_efficiency_lst.calc_tel_efficiency()
    tot_efficiency = camera_efficiency_lst.calc_tot_efficiency(tel_efficiency)
    assert isinstance(tot_efficiency, float)
    assert 0.0 < tot_efficiency < 1.0  # Total efficiency should be between 0 and 1


def test_calc_reflectivity(camera_efficiency_lst, prepare_results_file):
    camera_efficiency_lst._read_results()
    assert camera_efficiency_lst.calc_reflectivity() == pytest.approx(
        0.9167918392938349
    )  # Value for Prod5 LST-1


def test_calc_nsb_rate(camera_efficiency_lst, prepare_results_file):
    camera_efficiency_lst._read_results()
    nsb_rate, nsb_rate_ref_conditions = camera_efficiency_lst.calc_nsb_rate()
    assert nsb_rate is not None
    assert nsb_rate_ref_conditions.unit == u.GHz
    assert nsb_rate_ref_conditions.value > 0  # Should be positive rate


def test_export_results(mocker, camera_efficiency_lst, caplog, prepare_results_file):
    # no results available yet
    with caplog.at_level(logging.ERROR):
        camera_efficiency_lst.export_results()
    assert "Cannot export results because they do not exist" in caplog.text

    # results available
    mocker.patch.object(camera_efficiency_lst, "results_summary", return_value={"test": "data"})
    camera_efficiency_lst._read_results()
    with caplog.at_level(logging.INFO):
        camera_efficiency_lst.export_results()
    assert "Exporting summary results" in caplog.text


def test_analyze_has_results(camera_efficiency_lst, prepare_results_file):
    camera_efficiency_lst._read_results()
    camera_efficiency_lst.analyze()
    assert camera_efficiency_lst._has_results is True


def test_analyze_from_file(camera_efficiency_lst, mocker):
    from pathlib import Path

    camera_efficiency_lst._file["sim_telarray"] = Path(
        "tests/resources/"
        "camera_efficiency_North_MSTx-NectarCam_za20.0deg_azm000deg_validate_camera_efficiency.dat"
    )
    mocker.patch.object(CameraEfficiency, "results_summary", return_value="summary")
    camera_efficiency_lst.analyze(export=False, force=True)
    assert camera_efficiency_lst._has_results is True
    assert isinstance(camera_efficiency_lst._results, Table)
    assert len(camera_efficiency_lst._results) > 1
    assert "N4" in camera_efficiency_lst._results.colnames


def test_results_summary(camera_efficiency_lst, prepare_results_file):
    camera_efficiency_lst._read_results()
    camera_efficiency_lst.calc_nsb_rate()
    summary = camera_efficiency_lst.results_summary()
    assert summary["meta"]["tel"] == "LSTN-01"
    assert "meta" in summary


def test_plot_efficiency(camera_efficiency_lst, mocker, prepare_results_file):
    camera_efficiency_lst._read_results()
    camera_efficiency_lst.efficiency_type = "nsb"
    plot_table_mock = mocker.patch("simtools.visualization.visualize.plot_table")
    camera_efficiency_lst.plot_efficiency()
    plot_table_mock.assert_called_once()


def test_save_plot(camera_efficiency_lst, mocker, caplog):
    fig_mock = mocker.MagicMock()
    with caplog.at_level(logging.INFO):
        camera_efficiency_lst._save_plot(fig_mock, "test_plot")
    assert "Saved plot test_plot efficiency to" in caplog.text


def test_get_nsb_pixel_rate_provided_spectrum(camera_efficiency_lst, mocker):
    mocker.patch.object(
        camera_efficiency_lst.telescope_model, "get_parameter_value", return_value=10
    )
    camera_efficiency_lst.nsb_pixel_pe_per_ns = 5.0
    nsb_pixel_rate = camera_efficiency_lst.get_nsb_pixel_rate()
    assert nsb_pixel_rate.unit == u.GHz
    assert len(nsb_pixel_rate) == 10
    assert nsb_pixel_rate[0].value == pytest.approx(5.0)

    camera_efficiency_lst.nsb_pixel_pe_per_ns = 6.0 * u.GHz
    nsb_pixel_rate = camera_efficiency_lst.get_nsb_pixel_rate()
    assert nsb_pixel_rate.unit == u.GHz
    assert len(nsb_pixel_rate) == 10
    assert nsb_pixel_rate[0].value == pytest.approx(6.0)


def test_get_nsb_pixel_rate_reference_conditions(camera_efficiency_lst, mocker):
    mocker.patch.object(
        camera_efficiency_lst.telescope_model, "get_parameter_value", return_value=20
    )
    camera_efficiency_lst.nsb_rate_ref_conditions = 7.0
    nsb_pixel_rate = camera_efficiency_lst.get_nsb_pixel_rate(reference_conditions=True)
    assert nsb_pixel_rate.unit == u.GHz
    assert len(nsb_pixel_rate) == 20
    assert nsb_pixel_rate[0].value == pytest.approx(7.0)


def test_get_x_max_for_efficiency_type_shower(camera_efficiency_lst, caplog):
    camera_efficiency_lst.config["efficiency_type"] = "shower"
    with caplog.at_level(logging.INFO):
        x_max = camera_efficiency_lst._get_x_max_for_efficiency_type()
    assert x_max == pytest.approx(300.0)
    assert "Using X-max for shower efficiency" in caplog.text


def test_get_x_max_for_efficiency_type_muon(camera_efficiency_lst, mocker, caplog):
    camera_efficiency_lst.efficiency_type = "muon"
    mock_atmo = mocker.MagicMock()
    mock_atmo.interpolate.return_value = 850.5
    mocker.patch(
        "simtools.camera.camera_efficiency.AtmosphereProfile",
        return_value=mock_atmo,
    )
    mocker.patch.object(
        camera_efficiency_lst.site_model,
        "get_parameter_value_with_unit",
        return_value=5 * u.km,
    )
    mocker.patch.object(
        camera_efficiency_lst.site_model,
        "get_parameter_value",
        return_value="atmosphere.txt",
    )
    with caplog.at_level(logging.INFO):
        x_max = camera_efficiency_lst._get_x_max_for_efficiency_type()
    assert x_max == pytest.approx(850.5)
    assert "Using X-max for muon efficiency" in caplog.text


def test_dump_nsb_pixel_rate(camera_efficiency_lst, mocker, caplog):
    camera_efficiency_lst.nsb_pixel_pe_per_ns = 5.0
    mocker.patch.object(
        camera_efficiency_lst,
        "get_nsb_pixel_rate",
        return_value=u.Quantity(np.full(10, 5.0), u.GHz),
    )
    mock_dump = mocker.patch(
        "simtools.data_model.model_data_writer.ModelDataWriter.dump_model_parameter"
    )
    mock_config = mocker.MagicMock()
    mock_config.args = {"telescope": "LSTN-01", "parameter_version": "1.0.0"}
    mocker.patch("simtools.settings.config", mock_config)
    with caplog.at_level(logging.INFO):
        camera_efficiency_lst.dump_nsb_pixel_rate()
    mock_dump.assert_called_once()
    call_kwargs = mock_dump.call_args[1]
    assert call_kwargs["parameter_name"] == "nsb_pixel_rate"
    assert call_kwargs["instrument"] == "LSTN-01"
    assert call_kwargs["parameter_version"] == "1.0.0"


def test_dump_nsb_pixel_rate_reference_conditions(camera_efficiency_lst, mocker):
    camera_efficiency_lst.nsb_rate_ref_conditions = 7.0
    camera_efficiency_lst.config["write_reference_nsb_rate_as_parameter"] = True
    mocker.patch.object(
        camera_efficiency_lst,
        "get_nsb_pixel_rate",
        return_value=u.Quantity(np.full(20, 7.0), u.GHz),
    )
    mock_dump = mocker.patch(
        "simtools.data_model.model_data_writer.ModelDataWriter.dump_model_parameter"
    )
    mock_config = mocker.MagicMock()
    mock_config.args = {
        "telescope": "LSTN-01",
        "parameter_version": "2.0.0",
        "write_reference_nsb_rate_as_parameter": True,
    }
    mocker.patch("simtools.settings.config", mock_config)
    camera_efficiency_lst.dump_nsb_pixel_rate()
    mock_dump.assert_called_once()
    call_kwargs = mock_dump.call_args[1]
    assert call_kwargs["parameter_version"] == "2.0.0"
    camera_efficiency_lst.get_nsb_pixel_rate.assert_called_once_with(reference_conditions=True)


def test_calc_partial_efficiency(camera_efficiency_lst, prepare_results_file):
    camera_efficiency_lst._read_results()
    result = camera_efficiency_lst.calc_partial_efficiency(350, 450)
    assert isinstance(result, (float, np.floating))
    assert 0 <= result <= 1


def test_calc_partial_efficiency_full_range(camera_efficiency_lst, prepare_results_file):
    camera_efficiency_lst._read_results()
    result = camera_efficiency_lst.calc_partial_efficiency(200, 999)
    assert isinstance(result, (float, np.floating))
    assert result > 0


def test_calc_partial_efficiency_narrow_range(camera_efficiency_lst, prepare_results_file):
    camera_efficiency_lst._read_results()
    result = camera_efficiency_lst.calc_partial_efficiency(400, 410)
    assert isinstance(result, (float, np.floating))
    assert 0 <= result <= 1


def test_calc_partial_efficiency_logging(camera_efficiency_lst, prepare_results_file, caplog):
    camera_efficiency_lst._read_results()
    with caplog.at_level(logging.INFO):
        camera_efficiency_lst.calc_partial_efficiency(300, 500)
    assert "Fraction of light in the wavelength range 300-500 nm:" in caplog.text


def test_results_summary_shower_type(camera_efficiency_lst, prepare_results_file):
    camera_efficiency_lst._read_results()
    camera_efficiency_lst.efficiency_type = "shower"
    summary = camera_efficiency_lst.results_summary()
    assert summary["meta"]["tel"] == "LSTN-01"
    assert summary["meta"]["zen"] == pytest.approx(20.0)
    assert summary["meta"]["az"] == pytest.approx(0.0)
    assert "reflectivity" in summary
    assert summary["reflectivity"]["description"] == "Spectrum weighted reflectivity"
    assert "cam_eff" in summary
    assert summary["cam_eff"]["description"] == "Camera nominal efficiency with gaps (B-TEL-1170)"
    assert "tel_eff" in summary
    assert "Telescope total efficiency" in summary["tel_eff"]["description"]
    assert "tot_sens" in summary
    assert "Telescope total Cherenkov light efficiency" in summary["tot_sens"]["description"]


def test_results_summary_nsb_type(camera_efficiency_lst, prepare_results_file):
    camera_efficiency_lst._read_results()
    camera_efficiency_lst.efficiency_type = "nsb"
    camera_efficiency_lst.nsb_pixel_pe_per_ns = pytest.approx(0.5)
    camera_efficiency_lst.nsb_rate_ref_conditions = pytest.approx(0.25)
    summary = camera_efficiency_lst.results_summary()
    assert summary["meta"]["tel"] == "LSTN-01"
    assert "nsb_rate" in summary
    assert summary["nsb_rate"]["value"] == pytest.approx(0.5)
    assert (
        summary["nsb_rate"]["description"]
        == "Expected NSB pixel rate for the provided NSB spectrum"
    )
    assert "nsb_ref" in summary
    assert summary["nsb_ref"]["value"] == pytest.approx(0.25)
    assert summary["nsb_ref"]["description"] == "Expected NSB pixel rate for the reference NSB"


def test_results_summary_muon_type(camera_efficiency_lst, prepare_results_file):
    camera_efficiency_lst._read_results()
    camera_efficiency_lst.efficiency_type = "muon"
    summary = camera_efficiency_lst.results_summary()
    assert summary["meta"]["tel"] == "LSTN-01"
    assert "muon_frac" in summary
    assert (
        "Fraction of light (from muons) in the wavelength range 200-290 nm (B-TEL-0095)"
        in summary["muon_frac"]["description"]
    )


def test_results_summary_with_custom_nsb_spectrum(camera_efficiency_lst, prepare_results_file):
    camera_efficiency_lst._read_results()
    camera_efficiency_lst.config["nsb_spectrum"] = "custom_spectrum.fits"
    summary = camera_efficiency_lst.results_summary()
    assert summary["meta"]["nsb"] == "custom_spectrum.fits"


def test_results_summary_without_nsb_spectrum(camera_efficiency_lst, prepare_results_file):
    camera_efficiency_lst._read_results()
    camera_efficiency_lst.config["nsb_spectrum"] = None
    summary = camera_efficiency_lst.results_summary()
    assert summary["meta"]["nsb"] == "default sim_telarray spectrum"
