#!/usr/bin/python3

import copy
import subprocess
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import numpy as np
import pytest
from astropy.table import Table

from simtools.camera.single_photon_electron_spectrum import SinglePhotonElectronSpectrum


@pytest.fixture
def spe_spectrum():
    args_dict = {
        "output_file": "output_file",
        "simtel_path": "/path/to/simtel",
        "step_size": 0.1,
        "max_amplitude": 1.0,
        "afterpulse_spectrum": None,
        "input_spectrum": "input_spectrum",
        "afterpulse_amplitude_range": [4.0, 42.0],
    }
    return SinglePhotonElectronSpectrum(args_dict)


@pytest.fixture
def spe_data():
    return "0.0,0.4694\n0.02,0.46378\n0.04,0.45267\n0.06,0.44172"


@pytest.fixture
def afterpulse_column():
    return "frequency (afterpulsing)"


@patch("simtools.io.io_handler.IOHandler")
@patch("simtools.camera.single_photon_electron_spectrum.MetadataCollector")
def test_init(mock_metadata_collector, mock_io_handler, spe_spectrum):
    mock_io_handler_instance = mock_io_handler.return_value
    mock_metadata_collector_instance = mock_metadata_collector.return_value

    spe_spectrum.io_handler = mock_io_handler_instance
    spe_spectrum.metadata = mock_metadata_collector_instance
    tmp_spe_spectrum = copy.deepcopy(spe_spectrum)

    assert tmp_spe_spectrum.args_dict["output_file"] == "output_file.ecsv"
    assert tmp_spe_spectrum.io_handler == mock_io_handler_instance
    assert tmp_spe_spectrum.data == ""
    assert tmp_spe_spectrum.metadata == mock_metadata_collector_instance


@patch(
    "simtools.camera.single_photon_electron_spectrum."
    "SinglePhotonElectronSpectrum._derive_spectrum_norm_spe"
)
def test_derive_single_pe_spectrum(mock_derive_spectrum_norm_spe, spe_spectrum):
    spe_spectrum.args_dict["use_norm_spe"] = True
    spe_spectrum.derive_single_pe_spectrum()

    # Check that _derive_spectrum_norm_spe is called with the correct parameters
    mock_derive_spectrum_norm_spe.assert_called_once_with(
        input_spectrum=spe_spectrum.args_dict["input_spectrum"],
        afterpulse_spectrum=spe_spectrum.args_dict.get("afterpulse_spectrum"),
        afterpulse_fitted_spectrum=None,  # Add the missing parameter
    )

    spe_spectrum.args_dict["use_norm_spe"] = False
    with pytest.raises(
        NotImplementedError,
        match=(
            r"Derivation of single photon electron spectrum using a simtool is not yet implemented."
        ),
    ):
        spe_spectrum.derive_single_pe_spectrum()


@patch("simtools.camera.single_photon_electron_spectrum.io_handler.IOHandler.get_output_directory")
@patch("simtools.camera.single_photon_electron_spectrum.writer.ModelDataWriter.dump")
@patch("builtins.open", new_callable=MagicMock)
def test_write_single_pe_spectrum(
    mock_open, mock_dump, mock_get_output_directory, spe_spectrum, tmp_test_directory
):
    mock_get_output_directory.return_value = tmp_test_directory / "output" / "directory"
    mock_open.return_value.__enter__.return_value = MagicMock()

    tmp_spe_spectrum = copy.deepcopy(spe_spectrum)

    tmp_spe_spectrum.data = """
# comment
0.0\t0.4694\t0.4694
0.02\t0.46378\t0.46378
0.04\t0.45267\t0.45267
0.06\t0.44172\t0.44172
"""
    tmp_spe_spectrum.write_single_pe_spectrum()

    mock_open.assert_called_once_with(
        (tmp_test_directory / "output" / "directory" / "output_file.dat"), "w", encoding="utf-8"
    )
    mock_dump.assert_called_once()


@patch("simtools.camera.single_photon_electron_spectrum.subprocess.run")
@patch(
    "simtools.camera.single_photon_electron_spectrum.SinglePhotonElectronSpectrum._get_input_data"
)
def test_derive_spectrum_norm_spe(
    mock_get_input_data, mock_subprocess_run, spe_spectrum, spe_data, tmp_test_directory
):
    tmpfile_path = tmp_test_directory / "test_spe_data.txt"
    tmpfile_path.write_text(spe_data, encoding="utf-8")

    # Create a mock file object that behaves like the NamedTemporaryFile
    class MockTempFile:
        def __init__(self, path):
            self.name = str(path)

    tmpfile = MockTempFile(tmpfile_path)
    # first call to _get_input_data returns tmpfile, second call None
    mock_get_input_data.side_effect = [tmpfile, None]
    mock_subprocess_run.return_value.stdout = spe_data
    mock_subprocess_run.return_value.returncode = 0

    return_code = spe_spectrum._derive_spectrum_norm_spe(
        input_spectrum=spe_spectrum.args_dict["input_spectrum"],
        afterpulse_spectrum=None,
        afterpulse_fitted_spectrum=None,  # Add the missing parameter
    )

    assert mock_get_input_data.call_count == 2
    mock_subprocess_run.assert_called_once_with(
        ["/path/to/simtel/sim_telarray/bin/norm_spe", "-r", "0.1,1.0", ANY],
        capture_output=True,
        text=True,
        check=True,
    )
    assert return_code == 0
    assert spe_spectrum.data == spe_data

    mock_get_input_data.reset_mock()
    mock_subprocess_run.reset_mock()

    tmp_spe_spectrum = copy.deepcopy(spe_spectrum)
    tmp_spe_spectrum.args_dict["afterpulse_spectrum"] = "afterpulse_spectrum"
    tmp_spe_spectrum.args_dict["scale_afterpulse_spectrum"] = 1.0
    mock_get_input_data.side_effect = [tmpfile, tmpfile]

    return_code = tmp_spe_spectrum._derive_spectrum_norm_spe(
        input_spectrum=tmp_spe_spectrum.args_dict["input_spectrum"],
        afterpulse_spectrum="afterpulse_spectrum",
        afterpulse_fitted_spectrum=None,  # Add the missing parameter
    )

    mock_subprocess_run.assert_called_with(
        [
            "/path/to/simtel/sim_telarray/bin/norm_spe",
            "-r",
            "0.1,1.0",
            "-a",
            ANY,
            "-s",
            "1.0",
            "-t",
            "4.0",
            ANY,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert return_code == 0

    # Reset mocks again
    mock_get_input_data.reset_mock()
    mock_subprocess_run.reset_mock()

    # Test error handling
    spe_spectrum = copy.deepcopy(spe_spectrum)
    mock_get_input_data.side_effect = [tmpfile, None]
    mock_subprocess_run.side_effect = subprocess.CalledProcessError(
        returncode=1, cmd="norm_spe", output="Error", stderr="Error message"
    )

    with pytest.raises(subprocess.CalledProcessError):
        spe_spectrum._derive_spectrum_norm_spe(
            input_spectrum=spe_spectrum.args_dict["input_spectrum"],
            afterpulse_spectrum=None,
            afterpulse_fitted_spectrum=None,  # Add the missing parameter
        )


@patch("builtins.open", new_callable=MagicMock)
def test_get_input_data(mock_open, spe_spectrum, spe_data):
    assert (
        spe_spectrum._get_input_data(None, None, spe_spectrum.prompt_column) is None
    )  # Add the missing parameter

    mock_open.return_value.__enter__.return_value.read.return_value = spe_data.replace(" ", ",")
    input_data = spe_spectrum._get_input_data(
        "input_spectrum", None, spe_spectrum.prompt_column
    )  # Add the missing parameter
    mock_open.assert_called_once_with(Path("input_spectrum"), encoding="utf-8")
    assert input_data is not None
    with open(input_data.name, encoding="utf-8") as f:
        assert f.read() == spe_data

    input_data = spe_spectrum._get_input_data(
        "input_spectrum", None, spe_spectrum.afterpulse_column
    )  # Add the missing parameter
    assert input_data is not None
    with open(input_data.name, encoding="utf-8") as f:
        assert f.read() == spe_data.replace(" ", ",")

    with patch(
        "simtools.data_model.validate_data.DataValidator.validate_and_transform"
    ) as mock_validator:
        mock_table = Table()
        mock_table["amplitude"] = [0.0, 0.02, 0.04, 0.06]
        mock_table["frequency (prompt)"] = [0.4694, 0.46378, 0.45267, 0.44172]
        mock_validator.return_value = mock_table

        ecsv_data = spe_spectrum._get_input_data(
            "input_spectrum.ecsv", None, spe_spectrum.prompt_column
        )  # Add the missing parameter
        assert ecsv_data is not None
        with open(ecsv_data.name, encoding="utf-8") as f:
            table_data = f.read()
            assert table_data.splitlines()[0] == "0.0,0.4694"


@patch("simtools.camera.single_photon_electron_spectrum.Table")
def test_read_afterpulse_spectrum_for_fit(mock_table, spe_spectrum, afterpulse_column):
    mock_data = Table()
    mock_data["amplitude"] = [1.0, 2.0, 3.0, 4.0, 5.0]
    mock_data[afterpulse_column] = [0.1, 0.2, 0.0, 0.4, 0.5]
    mock_data["frequency stdev (afterpulsing)"] = [0.01, 0.02, 0.03, 0.04, 0.05]
    mock_table.read.return_value = mock_data

    x, y, y_err = spe_spectrum._read_afterpulse_spectrum_for_fit("dummy.ecsv", 3.0)
    mock_table.read.assert_called_once_with("dummy.ecsv", format="ascii.ecsv")
    assert len(x) == 2
    assert len(y) == 2
    assert len(y_err) == 2
    np.testing.assert_array_equal(x, [4.0, 5.0])
    np.testing.assert_array_equal(y, [0.4, 0.5])
    np.testing.assert_array_equal(y_err, [0.04, 0.05])


def test_afterpulse_fit_statistics(spe_spectrum):
    # Test case 1: without fixed k
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([0.1, 0.05, 0.02])
    y_err = np.array([0.01, 0.01, 0.01])
    params = np.array([0.5, 1.0])
    param_errors = np.array([0.1, 0.2])
    predicted = np.array([0.11, 0.04, 0.03])
    fix_k = None

    result = spe_spectrum._afterpulse_fit_statistics(
        x, y, y_err, params, param_errors, predicted, fix_k
    )

    assert "params" in result
    assert "errors" in result
    assert "chi2_ndf" in result
    np.testing.assert_array_equal(result["params"], [0.5, 1.0])
    np.testing.assert_array_equal(result["errors"], [0.1, 0.2])
    assert isinstance(result["chi2_ndf"], float)

    # Test case 2: with fixed k
    fix_k = 25.0
    result = spe_spectrum._afterpulse_fit_statistics(
        x, y, y_err, params, param_errors, predicted, fix_k
    )

    assert len(result["params"]) == 3
    assert len(result["errors"]) == 3
    np.testing.assert_array_equal(result["params"], [0.5, 1.0, 25.0])
    np.testing.assert_array_equal(result["errors"], [0.1, 0.2, 0.0])

    # Test case 3: zero degrees of freedom
    x = np.array([1.0, 2.0])
    y = np.array([0.1, 0.05])
    y_err = np.array([0.01, 0.01])
    params = np.array([0.5, 1.0])
    param_errors = np.array([0.1, 0.2])
    predicted = np.array([0.11, 0.04])

    result = spe_spectrum._afterpulse_fit_statistics(
        x, y, y_err, params, param_errors, predicted, fix_k=None
    )

    assert np.isnan(result["chi2_ndf"])


def test_afterpulse_fit_function(spe_spectrum):
    # Test case 1: without fixed k parameter
    func, p0, bounds = spe_spectrum.afterpulse_fit_function(fix_k=None)

    # Check return values
    assert callable(func)
    assert len(p0) == 3
    assert len(bounds) == 2

    # Test the returned function
    x = np.array([1.0, 2.0, 3.0])
    test_params = [1e-5, 8.0, 25.0]
    y = func(x, *test_params)
    assert isinstance(y, np.ndarray)
    assert len(y) == len(x)

    # Test case 2: with fixed k parameter
    fixed_k = 15.0
    func_fixed, p0_fixed, bounds_fixed = spe_spectrum.afterpulse_fit_function(fix_k=fixed_k)

    # Check return values
    assert callable(func_fixed)
    assert len(p0_fixed) == 2
    assert len(bounds_fixed) == 2

    # Test the returned function with fixed k
    y_fixed = func_fixed(x, 1e-5, 8.0)
    assert isinstance(y_fixed, np.ndarray)
    assert len(y_fixed) == len(x)

    # Verify that both functions give same results when using same parameters
    y1 = func(x, 1e-5, 8.0, 15.0)
    y2 = func_fixed(x, 1e-5, 8.0)
    np.testing.assert_array_almost_equal(y1, y2)


@patch("simtools.camera.single_photon_electron_spectrum.curve_fit")
def test_fit_afterpulse_spectrum(mock_curve_fit, spe_spectrum, afterpulse_column):
    # Mock input data
    spe_spectrum.args_dict["afterpulse_amplitude_range"] = [4.0, 42.0]
    spe_spectrum.args_dict["step_size"] = 0.1

    # Mock the read_afterpulse_spectrum_for_fit method
    x = np.array([4.0, 5.0, 6.0])
    y = np.array([0.1, 0.05, 0.02])
    y_err = np.array([0.01, 0.01, 0.01])

    with patch.object(
        spe_spectrum, "_read_afterpulse_spectrum_for_fit", return_value=(x, y, y_err)
    ):
        # Test case 1: without fixed k
        mock_params = np.array([1e-5, 8.0, 25.0])
        mock_covariance = np.array([[1e-10, 0, 0], [0, 1, 0], [0, 0, 1]])
        mock_curve_fit.return_value = (mock_params, mock_covariance)

        result = spe_spectrum.fit_afterpulse_spectrum()

        assert isinstance(result, Table)
        assert "amplitude" in result.colnames
        assert afterpulse_column in result.colnames
        assert len(result) > 0

        # Test case 2: with fixed k
        spe_spectrum.args_dict["afterpulse_decay_factor_fixed_value"] = 15.0
        mock_params = np.array([1e-5, 8.0])
        mock_covariance = np.array([[1e-10, 0], [0, 1]])
        mock_curve_fit.return_value = (mock_params, mock_covariance)

        result = spe_spectrum.fit_afterpulse_spectrum()

        assert isinstance(result, Table)
        assert "amplitude" in result.colnames
        assert afterpulse_column in result.colnames
        assert len(result) > 0

        # Test case 3: curve_fit raises RuntimeError
        mock_curve_fit.side_effect = RuntimeError("Optimal parameters not found")

        with pytest.raises(RuntimeError, match="Optimal parameters not found"):
            spe_spectrum.fit_afterpulse_spectrum()
