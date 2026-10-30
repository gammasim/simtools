#!/usr/bin/python3

import copy
from io import BytesIO
from unittest.mock import patch

import numpy as np
import pytest
from astropy.table import Table

from simtools.camera.single_photon_electron_spectrum import SinglePhotonElectronSpectrum


@pytest.fixture
def spe_spectrum(run_context):
    args_dict = {
        "output_file": "output_file",
        "step_size": 0.1,
        "max_amplitude": 1.0,
        "afterpulse_spectrum": None,
        "input_spectrum": "input_spectrum",
        "afterpulse_amplitude_range": [4.0, 42.0],
    }
    return SinglePhotonElectronSpectrum(args_dict, context=run_context)


@pytest.fixture
def spe_data():
    return "0.0,0.4694\n0.02,0.46378\n0.04,0.45267\n0.06,0.44172"


@pytest.fixture
def afterpulse_column():
    return "frequency (afterpulsing)"


def test_derive_single_pe_spectrum(spe_spectrum, tmp_test_directory):
    input_file = tmp_test_directory / "prompt.ecsv"
    Table(
        {
            "amplitude": [0.0, 1.0, 2.0],
            "frequency (prompt)": [0.0, 1.0, 0.0],
        }
    ).write(input_file, format="ascii.ecsv")
    spe_spectrum.args_dict["input_spectrum"] = input_file

    assert spe_spectrum.derive_single_pe_spectrum() == 0
    assert spe_spectrum.data


@patch("simtools.camera.single_photon_electron_spectrum.writer.ModelDataWriter.write_product_data")
def test_write_single_pe_spectrum(mock_dump, spe_spectrum, tmp_test_directory):
    spe_spectrum.context.paths.output_path["default"] = tmp_test_directory / "output" / "directory"

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
        context=spe_spectrum.context,
    )
    assert mock_dump.call_args.kwargs["product_data"].colnames == [
        "amplitude",
        "response",
        "response_with_ap",
    ]


def test_write_single_pe_spectrum_rejects_unexpected_spectrum_columns(spe_spectrum):
    spe_spectrum.data = "0.0\t0.4694\n"

    with pytest.raises(ValueError, match="expected 3 columns, got 2"):
        spe_spectrum.write_single_pe_spectrum()


def test_derive_spectrum(spe_spectrum, tmp_test_directory):
    input_file = tmp_test_directory / "prompt.ecsv"
    Table(
        {
            "amplitude": [0.0, 1.0, 2.0],
            "frequency (prompt)": [0.0, 1.0, 0.0],
        }
    ).write(input_file, format="ascii.ecsv")
    spe_spectrum.args_dict.update(input_spectrum=input_file, max_amplitude=2.0)

    assert spe_spectrum.derive_single_pe_spectrum() == 0

    output = np.loadtxt(BytesIO(spe_spectrum.data.encode("utf-8")))
    np.testing.assert_allclose(output[:, 0], np.arange(0.0, 2.1, 0.1))
    np.testing.assert_allclose(output[:, 1], np.maximum(0.0, 1.0 - np.abs(output[:, 0] - 1.0)))
    np.testing.assert_allclose(output[:, 2], output[:, 1])


def test_derive_spectrum_does_not_exceed_maximum(spe_spectrum, tmp_test_directory):
    input_file = tmp_test_directory / "prompt.ecsv"
    Table(
        {
            "amplitude": [0.0, 1.0, 2.0],
            "frequency (prompt)": [0.0, 1.0, 0.0],
        }
    ).write(input_file, format="ascii.ecsv")
    spe_spectrum.args_dict.update(input_spectrum=input_file, max_amplitude=1.0, step_size=0.6)

    spe_spectrum.derive_single_pe_spectrum()

    output = np.loadtxt(BytesIO(spe_spectrum.data.encode("utf-8")))
    np.testing.assert_allclose(output[:, 0], [0.0, 0.6])
    assert np.all(output[:, 0] <= 1.0)


def test_derive_spectrum_uses_normalized_prompt_endpoint(spe_spectrum):
    prompt_data = (np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, 1.0]))
    afterpulse_data = (np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    normalized_prompt = (np.array([0.0, 0.5, 1.0]), np.array([1.0, 2.0, 1.0]))
    folded_data = (
        np.array([0.0, 0.5, 1.0]),
        np.array([1.0, 2.0, 1.0]),
        np.array([1.0, 2.0, 1.0]),
    )

    with (
        patch.object(spe_spectrum, "_read_input_data", side_effect=[prompt_data, afterpulse_data]),
        patch.object(spe_spectrum, "_normalize_prompt_spectrum", return_value=normalized_prompt),
        patch.object(spe_spectrum, "_fold_afterpulse_spectrum", return_value=folded_data) as fold,
    ):
        spe_spectrum._derive_spectrum("prompt", "afterpulse", None)

    assert fold.call_args.kwargs["prompt_maximum"] == pytest.approx(1.0)


def test_normalize_prompt_spectrum_rejects_invalid_input():
    with pytest.raises(ValueError, match="increasing"):
        SinglePhotonElectronSpectrum._normalize_prompt_spectrum(
            np.array([0.0, 0.0]), np.array([1.0, 1.0])
        )
    with pytest.raises(ValueError, match="non-positive"):
        SinglePhotonElectronSpectrum._normalize_prompt_spectrum(
            np.array([0.0, 1.0]), np.array([0.0, 0.0])
        )


def test_normalize_prompt_spectrum_uses_first_moment():
    amplitude, prompt = SinglePhotonElectronSpectrum._normalize_prompt_spectrum(
        np.array([0.0, 1.0, 3.0]), np.array([1.0, 4.0, 2.0])
    )
    scale = 8.5 / 13.25
    np.testing.assert_allclose(amplitude, [0.0, scale, 3 * scale])
    np.testing.assert_allclose(prompt, np.array([1.0, 4.0, 2.0]) / (8.5 * scale))


def test_fold_afterpulse_spectrum(spe_spectrum):
    spe_spectrum.args_dict.update(
        afterpulse_amplitude_range=[0.0, 42.0], scale_afterpulse_spectrum=1.0
    )
    amplitude, prompt, combined = spe_spectrum._fold_afterpulse_spectrum(
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 0.2, 0.0]),
    )

    np.testing.assert_allclose(amplitude, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(prompt, [0.0, 1.0, 0.0])
    np.testing.assert_allclose(combined, [0.0, 1.0, 0.2])


def test_fold_afterpulse_spectrum_preserves_prompt_tail(spe_spectrum):
    spe_spectrum.args_dict.update(
        afterpulse_amplitude_range=[0.0, 42.0], scale_afterpulse_spectrum=1.0
    )
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
