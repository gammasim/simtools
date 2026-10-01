#!/usr/bin/python3

import copy
from io import BytesIO
from unittest.mock import patch

import numpy as np
import pytest
from astropy.table import Table

from simtools.camera.single_photon_electron_spectrum import SinglePhotonElectronSpectrum


@pytest.fixture
def spe_spectrum():
    args_dict = {
        "output_file": "output_file",
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
    spe_spectrum.derive_single_pe_spectrum()

    mock_derive_spectrum_norm_spe.assert_called_once_with(
        input_spectrum=spe_spectrum.args_dict["input_spectrum"],
        afterpulse_spectrum=spe_spectrum.args_dict.get("afterpulse_spectrum"),
        afterpulse_fitted_spectrum=None,
    )


@patch("simtools.camera.single_photon_electron_spectrum.io_handler.IOHandler.get_output_directory")
@patch("simtools.camera.single_photon_electron_spectrum.writer.ModelDataWriter.write_product_data")
def test_write_single_pe_spectrum(
    mock_dump, mock_get_output_directory, spe_spectrum, tmp_test_directory
):
    mock_get_output_directory.return_value = tmp_test_directory / "output" / "directory"

    tmp_spe_spectrum = copy.deepcopy(spe_spectrum)

    tmp_spe_spectrum.data = """
# comment
0.0\t0.4694\t0.4694
0.02\t0.46378\t0.46378
0.04\t0.45267\t0.45267
0.06\t0.44172\t0.44172
"""
    tmp_spe_spectrum.write_single_pe_spectrum()

    mock_dump.assert_called_once_with(
        output_file="output_file.ecsv",
        output_file_format=None,
        metadata=tmp_spe_spectrum.metadata,
        product_data=mock_dump.call_args.kwargs["product_data"],
        validate_schema_file=tmp_spe_spectrum.output_schema,
        metadata_output_file=(tmp_test_directory / "output" / "directory" / "output_file.ecsv"),
    )
    assert mock_dump.call_args.kwargs["product_data"].colnames == [
        "amplitude",
        "response",
        "response_with_ap",
    ]


def test_get_output_columns_uses_model_parameter_schema(spe_spectrum):
    assert spe_spectrum._get_output_columns() == ["amplitude", "response", "response_with_ap"]


def test_write_single_pe_spectrum_rejects_unexpected_norm_spe_columns(spe_spectrum):
    spe_spectrum.data = "0.0\t0.4694\n"

    with pytest.raises(ValueError, match="expected 3 columns, got 2"):
        spe_spectrum.write_single_pe_spectrum()


def test_derive_spectrum_norm_spe(spe_spectrum, tmp_test_directory):
    input_file = tmp_test_directory / "prompt.csv"
    input_file.write_text("0,0\n1,1\n2,0\n", encoding="utf-8")
    spe_spectrum.args_dict.update(input_spectrum=input_file, max_amplitude=2.0)

    assert spe_spectrum._derive_spectrum_norm_spe(input_file, None, None) == 0

    output = np.loadtxt(BytesIO(spe_spectrum.data.encode("utf-8")))
    np.testing.assert_allclose(output[:, 0], np.arange(0.0, 2.1, 0.1))
    np.testing.assert_allclose(output[:, 1], np.maximum(0.0, 1.0 - np.abs(output[:, 0] - 1.0)))
    np.testing.assert_allclose(output[:, 2], output[:, 1])


def test_normalize_prompt_spectrum_rejects_invalid_input():
    with pytest.raises(ValueError, match="increasing"):
        SinglePhotonElectronSpectrum._normalize_prompt_spectrum(
            np.array([0.0, 0.0]), np.array([1.0, 1.0])
        )
    with pytest.raises(ValueError, match="non-positive"):
        SinglePhotonElectronSpectrum._normalize_prompt_spectrum(
            np.array([0.0, 1.0]), np.array([0.0, 0.0])
        )


def test_normalize_prompt_spectrum_uses_norm_spe_first_moment():
    amplitude, prompt = SinglePhotonElectronSpectrum._normalize_prompt_spectrum(
        np.array([0.0, 1.0, 3.0]), np.array([1.0, 4.0, 2.0])
    )
    scale = 8.5 / 13.25
    np.testing.assert_allclose(amplitude, [0.0, scale, 3 * scale])
    np.testing.assert_allclose(prompt, np.array([1.0, 4.0, 2.0]) / (8.5 * scale))


def test_linear_interpolate_uses_end_values_outside_input_range():
    result = SinglePhotonElectronSpectrum._linear_interpolate(
        np.array([0.0, 1.0]), np.array([0.2, 0.4]), np.array([-1.0, 0.5, 2.0])
    )
    np.testing.assert_allclose(result, [0.2, 0.3, 0.4])


def test_fold_afterpulse_spectrum(spe_spectrum):
    spe_spectrum.args_dict["scale_afterpulse_spectrum"] = 1.0
    amplitude, prompt, combined = spe_spectrum._fold_afterpulse_spectrum(
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 0.2, 0.0]),
    )

    np.testing.assert_allclose(amplitude, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(prompt, [0.0, 1.0, 0.0])
    np.testing.assert_allclose(combined, [0.0, 1.0, 0.2])


def test_fold_afterpulse_spectrum_preserves_legacy_prompt_tail(spe_spectrum):
    spe_spectrum.args_dict["scale_afterpulse_spectrum"] = 1.0
    amplitude = np.array([0.0, 1.0, 2.0])
    prompt = np.array([0.0, 1.0, 0.5])
    afterpulse_amplitude = np.array([0.0, 1.0, 2.0, 3.0])
    afterpulse = np.zeros(4)

    folded_amplitude, folded_prompt, _ = spe_spectrum._fold_afterpulse_spectrum(
        amplitude,
        prompt,
        afterpulse_amplitude,
        afterpulse,
        prompt_maximum=2.5,
    )

    output_amplitude = np.array([2.0, 2.5, 3.0])
    output_prompt = spe_spectrum._linear_interpolate(
        folded_amplitude, folded_prompt, output_amplitude
    )
    np.testing.assert_allclose(output_prompt, [0.5, 0.25, 0.0])


def test_read_input_data(spe_spectrum, tmp_test_directory):
    assert spe_spectrum._read_input_data(None, None, spe_spectrum.prompt_column) is None

    input_file = tmp_test_directory / "input_spectrum"
    input_file.write_text("0,0.4\n1,0.2\n", encoding="utf-8")
    amplitude, frequency = spe_spectrum._read_input_data(
        input_file, None, spe_spectrum.prompt_column
    )
    np.testing.assert_allclose(amplitude, [0.0, 1.0])
    np.testing.assert_allclose(frequency, [0.4, 0.2])

    afterpulse_file = tmp_test_directory / "afterpulse_spectrum"
    afterpulse_file.write_text("0.0,0.4\n1.0,0.2\n", encoding="utf-8")
    amplitude, frequency = spe_spectrum._read_input_data(
        afterpulse_file, None, spe_spectrum.afterpulse_column
    )
    np.testing.assert_allclose(amplitude, [0.0, 1.0])
    np.testing.assert_allclose(frequency, [0.4, 0.2])

    with patch(
        "simtools.data_model.validate_data.DataValidator.validate_and_transform"
    ) as mock_validator:
        mock_table = Table()
        mock_table["amplitude"] = [0.0, 0.02]
        mock_table["frequency (prompt)"] = [0.4694, 0.46378]
        mock_validator.return_value = mock_table
        amplitude, frequency = spe_spectrum._read_input_data(
            tmp_test_directory / "input_spectrum.ecsv", None, spe_spectrum.prompt_column
        )
        np.testing.assert_allclose(amplitude, [0.0, 0.02])
        np.testing.assert_allclose(frequency, [0.4694, 0.46378])


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
