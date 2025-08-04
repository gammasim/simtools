"""Single photon electron spectral analysis."""

import logging
import re
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path

import numpy as np
from astropy.table import Table
from scipy.optimize import curve_fit

import simtools.data_model.model_data_writer as writer
from simtools.constants import MODEL_PARAMETER_SCHEMA_URL, SCHEMA_PATH
from simtools.data_model import validate_data
from simtools.data_model.metadata_collector import MetadataCollector
from simtools.io import io_handler


class SinglePhotonElectronSpectrum:
    """
    Single photon electron spectral analysis.

    Parameters
    ----------
    args_dict: dict
        Dictionary with input arguments.
    """

    prompt_column = "frequency (prompt)"
    prompt_plus_afterpulse_column = "frequency (prompt+afterpulsing)"
    afterpulse_column = "frequency (afterpulsing)"

    input_schema = SCHEMA_PATH / "input" / "single_pe_spectrum.schema.yml"

    def __init__(self, args_dict):
        """Initialize SinglePhotonElectronSpectrum class."""
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Initialize SinglePhotonElectronSpectrum class.")

        self.args_dict = args_dict
        # default output is of ecsv format
        self.args_dict["output_file"] = str(
            Path(self.args_dict["output_file"]).with_suffix(".ecsv")
        )
        self.io_handler = io_handler.IOHandler()
        self.data = ""  # Single photon electron spectrum data (as string)
        self.args_dict["metadata_product_data_name"] = "single_pe_spectrum"
        self.args_dict["metadata_product_data_url"] = (
            MODEL_PARAMETER_SCHEMA_URL + "/pm_photoelectron_spectrum.schema.yml"
        )
        self.metadata = MetadataCollector(args_dict=self.args_dict)

    def derive_single_pe_spectrum(self):
        """Derive single photon electron spectrum."""
        afterpulse_fitted_spectrum = (
            self.fit_afterpulse_spectrum() if self.args_dict.get("fit_afterpulse") else None
        )

        if self.args_dict.get("use_norm_spe"):
            return self._derive_spectrum_norm_spe(
                input_spectrum=self.args_dict["input_spectrum"],
                afterpulse_spectrum=self.args_dict.get("afterpulse_spectrum"),
                afterpulse_fitted_spectrum=afterpulse_fitted_spectrum,
            )

        raise NotImplementedError(
            "Derivation of single photon electron spectrum using a simtool is not yet implemented."
        )

    def write_single_pe_spectrum(self):
        """
        Write single photon electron spectrum plus metadata to disk.

        Includes writing in simtel and simtools (ecsv) formats.

        """
        simtel_file = self.io_handler.get_output_directory() / Path(
            self.args_dict["output_file"]
        ).with_suffix(".dat")
        self._logger.debug(f"norm_spe output file: {simtel_file}")
        with open(simtel_file, "w", encoding="utf-8") as simtel:
            simtel.write(self.data)

        cleaned_data = re.sub(r"%%%.+", "", self.data)  # remove norm_spe row metadata
        table = Table.read(
            BytesIO(cleaned_data.encode("utf-8")),
            format="ascii.no_header",
            comment="#",
            delimiter="\t",
        )
        table.rename_columns(
            ["col1", "col2", "col3"],
            ["amplitude", self.prompt_column, self.prompt_plus_afterpulse_column],
        )

        writer.ModelDataWriter.dump(
            args_dict=self.args_dict,
            metadata=self.metadata,
            product_data=table,
            validate_schema_file=None,
        )

    def _derive_spectrum_norm_spe(
        self, input_spectrum, afterpulse_spectrum, afterpulse_fitted_spectrum
    ):
        """
        Derive single photon electron spectrum using sim_telarray tool 'norm_spe'.

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
            Return code of the executed command

        Raises
        ------
        subprocess.CalledProcessError
            If the command execution fails.
        """
        tmp_input_file = self._get_input_data(
            input_file=input_spectrum,
            input_table=None,
            frequency_column=self.prompt_column,
        )
        tmp_ap_file = self._get_input_data(
            input_file=afterpulse_spectrum,
            input_table=afterpulse_fitted_spectrum,
            frequency_column=self.afterpulse_column,
        )

        command = [
            f"{self.args_dict['simtel_path']}/sim_telarray/bin/norm_spe",
            "-r",
            f"{self.args_dict['step_size']},{self.args_dict['max_amplitude']}",
        ]
        if tmp_ap_file:
            command.extend(["-a", f"{tmp_ap_file.name}"])
            command.extend(["-s", f"{self.args_dict['scale_afterpulse_spectrum']}"])
            command.extend(["-t", f"{self.args_dict['afterpulse_amplitude_range'][0]}"])
        command.append(tmp_input_file.name)

        self._logger.info(f"Running norm_spe command: {' '.join(command)}")
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as exc:
            self._logger.error(f"Error running norm_spe: {exc}")
            self._logger.error(f"stderr: {exc.stderr}")
            raise exc
        finally:
            for tmp_file in [tmp_input_file, tmp_ap_file]:
                try:
                    Path(tmp_file.name).unlink()
                except (AttributeError, FileNotFoundError):
                    pass

        self.data = result.stdout
        return result.returncode

    def _get_input_data(self, input_file, input_table, frequency_column):
        """
        Return input data in the format required by the norm_spe tool as temporary file.

        The norm_spe tool requires the data to be space separated values of the amplitude spectrum,
        with two columns: amplitude and frequency.
        Input is validated using the single_pe_spectrum schema (legacy input is not validated).

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

        input_data = ""
        if input_file.suffix == ".ecsv" or input_table:
            data_validator = validate_data.DataValidator(
                schema_file=self.input_schema,
                data_table=input_table,
                data_file=input_file if input_table is None else None,
            )
            table = data_validator.validate_and_transform()
            input_data = "\n".join(f"{row['amplitude']} {row[frequency_column]}" for row in table)
        else:  # legacy format
            with open(input_file, encoding="utf-8") as f:
                input_data = (
                    f.read().replace(",", " ")
                    if frequency_column == self.prompt_column
                    else f.read()
                )

        with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8") as tmpfile:
            tmpfile.write(input_data)
        return tmpfile

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
        y_err = table["frequency stdev (afterpulsing)"]
        mask = (x >= fit_min_pe) & (y > 0)
        x_fit, y_fit, y_err_fit = x[mask], y[mask], y_err[mask]
        return x_fit, y_fit, y_err_fit
