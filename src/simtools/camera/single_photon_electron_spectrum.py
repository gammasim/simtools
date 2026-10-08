"""Single photon electron spectral analysis."""

import logging
from io import BytesIO
from pathlib import Path

import numpy as np
from astropy.table import Table
from scipy.optimize import curve_fit

import simtools.data_model.model_data_writer as writer
from simtools.constants import MODEL_PARAMETER_SCHEMA_URL, SCHEMA_PATH
from simtools.data_model import schema, validate_data
from simtools.data_model.metadata_collector import MetadataCollector
from simtools.data_model.table_asset import get_simtel_serialization
from simtools.io import io_handler

ECSV_SUFFIX = ".ecsv"


class SinglePhotonElectronSpectrum:
    """
    Single photon electron spectral analysis.

    Parameters
    ----------
    args_dict: dict
        Dictionary with input arguments.
    """

    prompt_column = "frequency (prompt)"
    afterpulse_column = "frequency (afterpulsing)"
    afterpulse_error_column = "frequency stdev (afterpulsing)"

    input_schema = SCHEMA_PATH / "input" / "single_pe_spectrum.schema.yml"
    output_parameter = "pm_photoelectron_spectrum"
    output_schema = schema.get_model_parameter_schema_file(output_parameter)

    def __init__(self, args_dict):
        """Initialize SinglePhotonElectronSpectrum class."""
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Initialize SinglePhotonElectronSpectrum class.")

        self.args_dict = args_dict
        # default output is of ecsv format
        self.args_dict["output_file"] = str(
            Path(self.args_dict["output_file"]).with_suffix(ECSV_SUFFIX)
        )
        self.io_handler = io_handler.IOHandler()
        self.data = ""  # Single photon electron spectrum data (as string)
        self.args_dict["metadata_product_data_name"] = self.output_parameter
        self.args_dict["metadata_product_data_url"] = (
            MODEL_PARAMETER_SCHEMA_URL + "/pm_photoelectron_spectrum.schema.yml"
        )
        metadata_args = dict(self.args_dict)
        metadata_args["output_file"] = Path(self.args_dict["output_file"]).name
        self.metadata = MetadataCollector(args_dict=metadata_args)

    def derive_single_pe_spectrum(self):
        """Derive single photon electron spectrum."""
        afterpulse_fitted_spectrum = (
            self.fit_afterpulse_spectrum() if self.args_dict.get("fit_afterpulse") else None
        )

        return self._derive_spectrum(
            input_spectrum=self.args_dict["input_spectrum"],
            afterpulse_spectrum=self.args_dict.get("afterpulse_spectrum"),
            afterpulse_fitted_spectrum=afterpulse_fitted_spectrum,
        )

    def write_single_pe_spectrum(self):
        """
        Write single photon electron spectrum plus metadata to disk.

        Write the generated model-parameter ECSV and its metadata.

        """
        output_file = Path(self.args_dict["output_file"])
        metadata_output_file = Path(self.io_handler.get_output_directory()) / output_file.name

        table = Table.read(
            BytesIO(self.data.encode("utf-8")),
            format="ascii.no_header",
            comment="#",
            delimiter="\t",
        )
        output_columns = self._get_output_columns()
        if len(table.colnames) != len(output_columns):
            raise ValueError(
                "Spectrum output does not match the pm_photoelectron_spectrum "
                f"schema: expected {len(output_columns)} columns, got {len(table.colnames)}"
            )
        table.rename_columns(table.colnames, output_columns)

        writer.ModelDataWriter.write_product_data(
            output_file=self.args_dict["output_file"],
            output_file_format=self.args_dict.get("output_file_format"),
            metadata=self.metadata,
            product_data=table,
            validate_schema_file=self.output_schema,
            metadata_output_file=metadata_output_file.with_suffix(ECSV_SUFFIX),
        )

    @classmethod
    def _get_output_columns(cls):
        """Return the ordered output columns declared by the output schema."""
        output_schema = schema.get_model_parameter_schema(cls.output_parameter)
        serialization = get_simtel_serialization(output_schema)
        return [*serialization["columns"], *serialization.get("optional_columns", [])]

    def _derive_spectrum(self, input_spectrum, afterpulse_spectrum, afterpulse_fitted_spectrum):
        """
        Derive a normalized single photon electron spectrum.

        Parameters
        ----------
        input_spectrum : str
            Input file with amplitude spectrum
            (prompt spectrum only if afterpulse spectrum is given).
        afterpulse_spectrum : str
            Input file with afterpulse spectrum.
        afterpulse_fitted_spectrum : astro.Table
            Fitted afterpulse spectrum data.

        Returns
        -------
        int
            Zero when the spectrum was derived successfully.
        """
        amplitude, prompt = self._read_input_data(
            input_file=input_spectrum,
            input_table=None,
            frequency_column=self.prompt_column,
        )
        normalized_amplitude, normalized_prompt = self._normalize_prompt_spectrum(amplitude, prompt)
        afterpulse_data = self._read_input_data(
            input_file=afterpulse_spectrum,
            input_table=afterpulse_fitted_spectrum,
            frequency_column=self.afterpulse_column,
        )

        if afterpulse_data is None:
            folded_amplitude = normalized_amplitude
            folded_prompt = normalized_prompt
            prompt_plus_afterpulse = normalized_prompt
        else:
            folded_amplitude, folded_prompt, prompt_plus_afterpulse = (
                self._fold_afterpulse_spectrum(
                    normalized_amplitude,
                    normalized_prompt,
                    *afterpulse_data,
                    prompt_maximum=normalized_amplitude[-1],
                )
            )

        output_amplitude = np.arange(
            0.0,
            np.nextafter(self.args_dict["max_amplitude"], np.inf),
            self.args_dict["step_size"],
        )
        output_prompt = self._linear_interpolate(folded_amplitude, folded_prompt, output_amplitude)
        output_combined = self._linear_interpolate(
            folded_amplitude, prompt_plus_afterpulse, output_amplitude
        )
        self.data = self._format_spectrum(output_amplitude, output_prompt, output_combined)
        return 0

    @staticmethod
    def _normalize_prompt_spectrum(amplitude, prompt):
        """Normalize prompt amplitudes to a mean of one photoelectron."""
        if len(amplitude) < 2 or np.any(np.diff(amplitude) <= 0):
            raise ValueError("Amplitude values must contain at least two increasing values.")

        integral = np.trapezoid(prompt, amplitude)
        interval_width = np.diff(amplitude)
        interval_midpoint = (amplitude[1:] + amplitude[:-1]) / 2
        interval_frequency = (prompt[1:] + prompt[:-1]) / 2
        first_moment = np.sum(interval_midpoint * interval_width * interval_frequency)
        if integral <= 0 or first_moment <= 0:
            raise ValueError("Cannot normalize a spectrum with non-positive integral or mean.")

        amplitude_scale = integral / first_moment
        prompt_scale = 1.0 / (integral * amplitude_scale)
        return amplitude * amplitude_scale, prompt * prompt_scale

    def _fold_afterpulse_spectrum(
        self, amplitude, prompt, afterpulse_amplitude, afterpulse, prompt_maximum=None
    ):
        """Fold an afterpulse probability density into a prompt spectrum."""
        if len(afterpulse_amplitude) < 2 or np.any(np.diff(afterpulse_amplitude) <= 0):
            raise ValueError("Afterpulse amplitudes must contain at least two increasing values.")

        if prompt_maximum is None:
            prompt_maximum = amplitude[-1]

        step = (amplitude[-1] - amplitude[0]) / (len(amplitude) - 1)
        maximum = max(amplitude[-1], afterpulse_amplitude[-1], self.args_dict["max_amplitude"])
        extra_samples = int(np.ceil((maximum - amplitude[-1]) / step))
        folded_amplitude = amplitude[0] + step * np.arange(len(amplitude) + extra_samples)
        folded_prompt = self._linear_interpolate(amplitude, prompt, folded_amplitude)
        folded_prompt[folded_amplitude > prompt_maximum] = 0.0
        afterpulse_minimum = self.args_dict["afterpulse_amplitude_range"][0]
        filtered_afterpulse = np.where(afterpulse_amplitude >= afterpulse_minimum, afterpulse, 0.0)
        sampled_afterpulse = np.interp(
            folded_amplitude,
            afterpulse_amplitude,
            filtered_afterpulse,
            left=0.0,
            right=0.0,
        )
        afterpulse_scale = self.args_dict["scale_afterpulse_spectrum"]
        combined = (
            folded_prompt
            + step
            * np.convolve(folded_prompt, afterpulse_scale * sampled_afterpulse)[
                : len(folded_prompt)
            ]
        )
        return folded_amplitude, folded_prompt, combined

    @staticmethod
    def _linear_interpolate(amplitude, frequency, output_amplitude):
        """Linearly interpolate, using the end value outside the input range."""
        return np.interp(output_amplitude, amplitude, frequency)

    @staticmethod
    def _format_spectrum(amplitude, prompt, prompt_plus_afterpulse):
        """Format a spectrum for sim_telarray's three-column table format."""
        rows = zip(amplitude, prompt, prompt_plus_afterpulse, strict=True)
        return "".join(f"{x:8.6f}\t{y:<12.5g}\t{z:<12.5g}\n" for x, y, z in rows)

    def _read_input_data(self, input_file, input_table, frequency_column):
        """
        Read input data for spectrum normalization.

        Input is validated using the single_pe_spectrum schema.

        Parameters
        ----------
        input_file : str
            Input file with amplitude spectrum.
        input_table : astro.Table
            Input table with amplitude spectrum.
        frequency_column : str
            Column name of the frequency data.
        """
        if not input_file:
            return None
        input_file = Path(input_file)

        if input_file.suffix != ECSV_SUFFIX and input_table is None:
            raise ValueError("Input spectrum must be an ECSV file.")

        data_validator = validate_data.DataValidator(
            schema_file=self.input_schema,
            data_table=input_table,
            data_file=input_file if input_table is None else None,
        )
        table = data_validator.validate_and_transform()
        return (
            np.asarray(table["amplitude"], dtype=float),
            np.asarray(table[frequency_column], dtype=float),
        )

    def fit_afterpulse_spectrum(self):
        """
        Fit afterpulse spectrum with a exponential decay function.

        Assume input to be in ecsv format with columns 'amplitude', 'frequency (afterpulsing)',
        and 'frequency stdev (afterpulsing)'.

        Returns
        -------
        astro.Table
            Table with fitted afterpulse spectrum data.
        """
        ap_min = self.args_dict["afterpulse_amplitude_range"][0]
        fix_k = self.args_dict.get("afterpulse_decay_factor_fixed_value")

        x, y, y_err = self._read_afterpulse_spectrum_for_fit(
            self.args_dict.get("afterpulse_spectrum"), ap_min
        )
        fit_func, p0, bounds = self.afterpulse_fit_function(fix_k=fix_k)

        result = curve_fit(fit_func, x, y, sigma=y_err, p0=p0, bounds=bounds, absolute_sigma=True)
        params, covariance = result[0], result[1]
        param_errors = np.sqrt(np.diag(covariance))
        predicted = fit_func(x, *params)
        self._afterpulse_fit_statistics(x, y, y_err, params, param_errors, predicted, fix_k)

        # table with fitted afterpulse spectrum
        x_fit = np.arange(
            ap_min, self.args_dict["afterpulse_amplitude_range"][1], self.args_dict["step_size"]
        )
        y_fit = fit_func(x_fit, *params)
        return Table([x_fit, y_fit], names=["amplitude", self.afterpulse_column])

    def afterpulse_fit_function(self, fix_k):
        """
        Afterpulse fit function: exponential decay with linear term in the exponent.

        Starting values and bounds are set for the other parameters using values typical
        for LSTN-design. Allows to fix the K parameter.

        Parameters
        ----------
        fix_K : float
            Fixed value for K parameter.

        Returns
        -------
        function
            Exponential decay function with linear term in the exponent.
        """

        def exp_decay(x, a, b, k):
            return a * np.exp(-1.0 / (b * (k / (x + k))) * x)

        p0 = [1e-5, 8.0]  # Initial guess for [A, B] typical LSTN values
        bounds_lower = [0, 0]
        bounds_upper = [1.0, 20.0]

        if fix_k is None:
            p0.append(25.0)
            bounds_lower.append(5.0)
            bounds_upper.append(35.0)
            return exp_decay, p0, (bounds_lower, bounds_upper)

        def exp_decay_fixed_k(x, a, b):
            return exp_decay(x, a, b, k=fix_k)

        return exp_decay_fixed_k, p0, (bounds_lower, bounds_upper)

    def _afterpulse_fit_statistics(self, x, y, y_err, params, param_errors, predicted, fix_k):
        """Print and return afterpulse fit statistics."""
        chi2 = np.sum(((y - predicted) / y_err) ** 2)
        ndf = len(x) - len(params)

        result = {
            "params": params.tolist(),
            "errors": param_errors.tolist(),
            "chi2_ndf": chi2 / ndf if ndf > 0 else np.nan,
        }
        if fix_k is not None:
            result["params"].append(fix_k)
            result["errors"].append(0.0)

        self._logger.info(f"Fit results: {result}")
        return result

    def _read_afterpulse_spectrum_for_fit(self, afterpulse_spectrum, fit_min_pe):
        """
        Read afterpulse spectrum data for fitting.

        Parameters
        ----------
        afterpulse_spectrum : str
            Afterpulse spectrum data file.
        fit_min_pe : float
            Minimum amplitude for fitting.

        Returns
        -------
        tuple
            Tuple with x, y, y_err data for fitting.
        """
        table = Table.read(afterpulse_spectrum, format="ascii.ecsv")
        x = table["amplitude"]
        y = table[self.afterpulse_column]
        y_err = table[self.afterpulse_error_column]
        mask = (x >= fit_min_pe) & (y > 0)
        x_fit, y_fit, y_err_fit = x[mask], y[mask], y_err[mask]
        return x_fit, y_fit, y_err_fit
