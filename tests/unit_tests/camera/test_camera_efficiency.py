#!/usr/bin/python3

import logging
from pathlib import Path
from unittest.mock import MagicMock

import astropy.units as u
import numpy as np
import pytest
from astropy.table import Table

from simtools.camera.camera_efficiency import CameraEfficiency
from simtools.camera.camera_efficiency_calculator import CameraEfficiencyCalculator
from simtools.simulation.configuration import get_model_writer

logger = logging.getLogger()
_TELESCOPE_THROUGHPUT_FILE_PREFIX = "telescope_throughput_"


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
def camera_efficiency_lst(config_data_lst, mocker, run_context):
    camera_eff = CameraEfficiency(
        config_data=config_data_lst,
        efficiency_type="shower",
        label="validate_camera_efficiency",
        context=run_context,
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
def prepare_results_file(camera_efficiency_lst, mocker, tmp_test_directory):
    """Create a minimal camera-efficiency result table for analysis tests."""
    test_results_file = (
        Path(tmp_test_directory)
        / "telescope_throughput_North_LSTN-02_za0.0deg_azm000deg_shower.ecsv"
    )

    wavelength = np.arange(200.0, 1001.0)
    efficiencies = {
        "cherenkov_at_ground": 2.0,
        "cherenkov_after_mirrors": 2.0 * 0.75 * 1.25,
        "cherenkov_at_photodetector": 2.0 * 0.5,
        "cherenkov_detected": 2.0 * 0.3,
        "cherenkov_detected_without_mirrors": 2.0 * 0.4,
        "nsb_before_correction": 3.0,
        "nsb_after_mirrors": 3.0 * 0.9,
        "nsb_at_photodetector": 3.0 * 0.4,
        "nsb_detected": 3.0 * 0.2,
        "nsb_detected_without_mirrors": 3.0 * 0.25,
    }
    result_table = Table(
        {
            "wl": wavelength,
            **{name: np.full(wavelength.size, value) for name, value in efficiencies.items()},
            "masts": np.full(wavelength.size, 1.25),
        }
    )
    result_table.write(test_results_file, format="ascii.ecsv", overwrite=True)
    mocker.patch.object(camera_efficiency_lst, "_file", {"results": test_results_file})
    return test_results_file


def test_report_and_result_file_name(camera_efficiency_lst):
    assert str(camera_efficiency_lst) == "CameraEfficiency(label=validate_camera_efficiency)\n"
    assert camera_efficiency_lst._file["results"].name.startswith(_TELESCOPE_THROUGHPUT_FILE_PREFIX)


@pytest.mark.parametrize(
    ("supplied", "initialize_called"),
    [
        ({"telescope_model": True, "site_model": True}, False),
        ({"telescope_model": True, "site_model": False}, True),
        ({"telescope_model": False, "site_model": True}, True),
    ],
)
def test_init_reuses_supplied_simulation_models(
    config_data_lst, mocker, run_context, supplied, initialize_called
):
    initialize_models = mocker.patch(
        "simtools.camera.camera_efficiency.initialize_simulation_models",
        return_value=(MagicMock(site="North", name="LSTN-01"), MagicMock(), None),
    )
    kwargs = {name: MagicMock() for name, given in supplied.items() if given}

    camera_efficiency = CameraEfficiency(
        config_data=config_data_lst,
        efficiency_type="shower",
        label="validate_camera_efficiency",
        context=run_context,
        **kwargs,
    )

    for name, model in kwargs.items():
        assert getattr(camera_efficiency, name) is model
    assert initialize_models.called is initialize_called


def test_simulate(camera_efficiency_lst, caplog, mocker):
    export_correction = mocker.patch.object(
        get_model_writer(camera_efficiency_lst.telescope_model),
        "export_nsb_spectrum_to_telescope_altitude_correction_file",
    )
    mock_calculate = mocker.patch.object(
        CameraEfficiencyCalculator, "calculate", return_value=Table()
    )
    with caplog.at_level(logging.INFO):
        camera_efficiency_lst.simulate()
        assert "Simulating CameraEfficiency" in caplog.text
    export_correction.assert_called_once_with(
        model_directory=camera_efficiency_lst.telescope_model.config_file_directory
    )
    mock_calculate.assert_called_once()


def test_read_results_and_derived_efficiencies(camera_efficiency_lst, prepare_results_file):
    camera_efficiency_lst._read_results()
    assert isinstance(camera_efficiency_lst._results, Table)
    assert camera_efficiency_lst._has_results is True
    assert camera_efficiency_lst.calc_camera_efficiency() == pytest.approx(0.32)
    tel_efficiency = camera_efficiency_lst.calc_tel_efficiency()
    assert tel_efficiency == pytest.approx(0.6127171314741036)
    assert camera_efficiency_lst.calc_tot_efficiency(tel_efficiency) == pytest.approx(
        0.958684357445742
    )
    assert camera_efficiency_lst.calc_reflectivity() == pytest.approx(0.75)
    result = camera_efficiency_lst.calc_partial_efficiency(350, 450)
    assert isinstance(result, (float, np.floating))
    assert 0 <= result <= 1


def test_calc_nsb_rate(camera_efficiency_lst, prepare_results_file, mocker):
    camera_efficiency_lst._read_results()
    mocker.patch.object(
        camera_efficiency_lst.telescope_model, "get_parameter_value", return_value=[1.0]
    )
    _, nsb_rate_ref_conditions = camera_efficiency_lst.calc_nsb_rate()
    assert camera_efficiency_lst.nsb_pixel_pe_per_ns == pytest.approx(0.04806)
    assert nsb_rate_ref_conditions.value > 0


def test_export_results(mocker, camera_efficiency_lst, caplog, prepare_results_file):
    with caplog.at_level(logging.ERROR):
        camera_efficiency_lst.export_results()
    assert "Cannot export results because they do not exist" in caplog.text

    mocker.patch.object(camera_efficiency_lst, "results_summary", return_value={"test": "data"})
    camera_efficiency_lst._read_results()
    with caplog.at_level(logging.INFO):
        camera_efficiency_lst.export_results()
    assert "Exporting summary results" in caplog.text


def test_analyze(camera_efficiency_lst, prepare_results_file):
    with pytest.raises(RuntimeError, match="must be simulated before it can be analyzed"):
        camera_efficiency_lst.analyze(export=False, force=True)

    camera_efficiency_lst._read_results()
    camera_efficiency_lst.analyze()
    assert camera_efficiency_lst._has_results is True


def test_analyze_calculates_derived_efficiencies(camera_efficiency_lst, mocker, tmp_test_directory):
    parameter_names = (
        "wl eff eff_atm qe ref masts filt pixel atm_trans cher nsb atm_corr "
        "nsb_site nsb_site_eff nsb_be nsb_be_eff"
    ).split()
    row = dict.fromkeys(parameter_names, 1.0)
    row.update(wl=400.0, qe=0.5, ref=0.5, masts=0.5, filt=0.5, pixel=0.5, atm_trans=2.0, nsb_be=3.0)
    camera_efficiency_lst._calculated_results = [row]
    mocker.patch.object(camera_efficiency_lst, "calc_nsb_rate", return_value=(0.0, None))
    mocker.patch.object(camera_efficiency_lst, "results_summary", return_value={})
    output_file = Path(tmp_test_directory) / "camera_efficiency.ecsv"
    camera_efficiency_lst._file["results"] = output_file

    camera_efficiency_lst.analyze(export=True, force=True)
    exported_results = Table.read(output_file, format="ascii.ecsv")
    assert exported_results.colnames == camera_efficiency_lst._results.colnames
    for column in exported_results.colnames:
        assert exported_results[column].description
        assert exported_results[column].description == (
            camera_efficiency_lst._results[column].description
        )
        assert exported_results[column][0] == pytest.approx(
            camera_efficiency_lst._results[column][0]
        )

    assert exported_results["qe"].description == "Photodetector quantum efficiency."
    assert exported_results["nsb_after_mirrors"].description == (
        "NSB spectrum after mirror reflection and shadowing."
    )
    results = camera_efficiency_lst._results[0]
    assert camera_efficiency_lst._results.colnames == [
        *parameter_names,
        "cherenkov_at_ground",
        "cherenkov_after_mirrors",
        "cherenkov_at_photodetector",
        "cherenkov_detected",
        "cherenkov_detected_without_mirrors",
        "nsb_before_correction",
        "nsb_after_mirrors",
        "nsb_at_photodetector",
        "nsb_detected",
        "nsb_detected_without_mirrors",
    ]
    expected = {
        "cherenkov_at_ground": 2.0,
        "cherenkov_after_mirrors": 0.5,
        "cherenkov_at_photodetector": 0.125,
        "cherenkov_detected": 0.0625,
        "cherenkov_detected_without_mirrors": 0.25,
        "nsb_before_correction": 3.0,
        "nsb_after_mirrors": 0.75,
        "nsb_at_photodetector": 0.1875,
        "nsb_detected": 0.09375,
        "nsb_detected_without_mirrors": 0.375,
    }
    for name, value in expected.items():
        assert results[name] == pytest.approx(value)


def test_results_summary_by_efficiency_type(camera_efficiency_lst, prepare_results_file):
    camera_efficiency_lst._read_results()
    camera_efficiency_lst.calc_nsb_rate()
    summary = camera_efficiency_lst.results_summary()
    assert summary["meta"]["tel"] == "LSTN-01"

    camera_efficiency_lst.efficiency_type = "nsb"
    camera_efficiency_lst.nsb_pixel_pe_per_ns = pytest.approx(0.5)
    camera_efficiency_lst.nsb_rate_ref_conditions = pytest.approx(0.25)
    summary = camera_efficiency_lst.results_summary()
    assert summary["nsb_rate"]["value"] == pytest.approx(0.5)
    assert (
        summary["nsb_rate"]["description"]
        == "Expected NSB pixel rate for the provided NSB spectrum"
    )
    assert summary["nsb_ref"]["value"] == pytest.approx(0.25)
    assert summary["nsb_ref"]["description"] == "Expected NSB pixel rate for the reference NSB"

    camera_efficiency_lst.efficiency_type = "muon"
    summary = camera_efficiency_lst.results_summary()
    assert (
        "Fraction of light (from muons) in the wavelength range 200-290 nm (B-TEL-0095)"
        in summary["muon_frac"]["description"]
    )


def test_results_summary_with_custom_nsb_spectrum(camera_efficiency_lst, prepare_results_file):
    camera_efficiency_lst._read_results()
    camera_efficiency_lst.config["nsb_spectrum"] = "/resource/path/custom_spectrum.fits"
    assert camera_efficiency_lst.results_summary()["meta"]["nsb"] == "custom_spectrum.fits"


@pytest.mark.parametrize("efficiency_type", ["shower", "muon", "nsb"])
def test_plot_efficiency(camera_efficiency_lst, mocker, prepare_results_file, efficiency_type):
    camera_efficiency_lst._read_results()
    camera_efficiency_lst.efficiency_type = efficiency_type
    plot_table_mock = mocker.patch("simtools.visualization.visualize.plot_table")
    camera_efficiency_lst.plot_efficiency()
    plot_table_mock.assert_called_once()


def test_save_plot(camera_efficiency_lst, mocker, caplog):
    with caplog.at_level(logging.INFO):
        camera_efficiency_lst._save_plot(mocker.MagicMock(), "test_plot")
    assert "Saved plot test_plot efficiency to" in caplog.text


@pytest.mark.parametrize(
    ("reference_conditions", "value", "expected"),
    [(False, 5.0, 5.0), (False, 6.0 * u.GHz, 6.0), (True, None, 7.0)],
)
def test_get_nsb_pixel_rate(camera_efficiency_lst, mocker, reference_conditions, value, expected):
    mocker.patch.object(
        camera_efficiency_lst.telescope_model,
        "get_parameter_value",
        return_value=20 if reference_conditions else 10,
    )
    if reference_conditions:
        camera_efficiency_lst.nsb_rate_ref_conditions = 7.0
    else:
        camera_efficiency_lst.nsb_pixel_pe_per_ns = value

    nsb_pixel_rate = camera_efficiency_lst.get_nsb_pixel_rate(
        reference_conditions=reference_conditions
    )

    assert nsb_pixel_rate.unit == u.GHz
    assert len(nsb_pixel_rate) == (20 if reference_conditions else 10)
    assert nsb_pixel_rate[0].value == pytest.approx(expected)


def test_get_x_max_for_efficiency_type_muon(camera_efficiency_lst, mocker, caplog):
    camera_efficiency_lst.efficiency_type = "muon"
    mock_atmo = mocker.MagicMock()
    mock_atmo.interpolate.return_value = 850.5
    mocker.patch("simtools.camera.camera_efficiency.AtmosphereProfile", return_value=mock_atmo)
    mocker.patch.object(
        camera_efficiency_lst.site_model, "get_parameter_value_with_unit", return_value=5 * u.km
    )
    mocker.patch.object(
        camera_efficiency_lst.site_model, "get_parameter_value", return_value="atmosphere.txt"
    )
    with caplog.at_level(logging.INFO):
        x_max = camera_efficiency_lst._get_x_max_for_efficiency_type()
    assert x_max == pytest.approx(850.5)
    assert "Using X-max for muon efficiency" in caplog.text


def test_get_x_max_for_efficiency_type_muon_exports_ecsv_profile(camera_efficiency_lst, mocker):
    camera_efficiency_lst.efficiency_type = "muon"
    mock_atmo = mocker.MagicMock()
    mock_atmo.interpolate.return_value = 850.5
    mocker.patch("simtools.camera.camera_efficiency.AtmosphereProfile", return_value=mock_atmo)
    mocker.patch.object(
        camera_efficiency_lst.site_model, "get_parameter_value_with_unit", return_value=5 * u.km
    )
    mocker.patch.object(
        camera_efficiency_lst.site_model,
        "get_parameter_value",
        return_value="atmospheric_profile-1.0.0.ecsv",
    )
    export_file = mocker.patch.object(
        get_model_writer(camera_efficiency_lst.site_model),
        "export_model_parameter_as_simtel_file",
    )

    x_max = camera_efficiency_lst._get_x_max_for_efficiency_type()

    expected_file = (
        f"atmospheric_profile-"
        f"{Path(camera_efficiency_lst.telescope_model.config_file_path).stem}.dat"
    )
    export_file.assert_called_once_with(
        "atmospheric_profile",
        camera_efficiency_lst.telescope_model.config_file_directory,
        table_format="plain",
        output_name=expected_file,
    )
    assert x_max == pytest.approx(850.5)


def test_dump_nsb_pixel_rate(camera_efficiency_lst, mocker, caplog, run_context):
    camera_efficiency_lst.nsb_pixel_pe_per_ns = 5.0
    mocker.patch.object(
        camera_efficiency_lst,
        "get_nsb_pixel_rate",
        return_value=u.Quantity(np.full(10, 5.0), u.GHz),
    )
    mock_dump = mocker.patch(
        "simtools.data_model.model_data_writer.ModelDataWriter.write_model_parameter"
    )
    camera_efficiency_lst.context = run_context.replace(
        args={"telescope": "LSTN-01", "parameter_version": "1.0.0"}
    )
    with caplog.at_level(logging.INFO):
        camera_efficiency_lst.dump_nsb_pixel_rate()
    mock_dump.assert_called_once()
    call_kwargs = mock_dump.call_args[1]
    assert call_kwargs["parameter_name"] == "nsb_pixel_rate"
    assert call_kwargs["instrument"] == "LSTN-01"
    assert call_kwargs["parameter_version"] == "1.0.0"


def test_results_summary_nsb_type_values(camera_efficiency_lst, prepare_results_file):
    camera_efficiency_lst._read_results()
    camera_efficiency_lst.efficiency_type = "nsb"
    camera_efficiency_lst.config["nsb_spectrum"] = "/resource/path/custom_spectrum.fits"
    camera_efficiency_lst.nsb_pixel_pe_per_ns = 0.5
    camera_efficiency_lst.nsb_rate_ref_conditions = 0.25
    summary = camera_efficiency_lst.results_summary()
    assert summary["meta"]["nsb"] == "custom_spectrum.fits"
    assert summary["nsb_rate"]["value"] == pytest.approx(0.5)
    assert summary["nsb_ref"]["value"] == pytest.approx(0.25)
