import logging
import re
from collections import defaultdict

import astropy.io.ascii
import numpy as np
from astropy.table import Table

import simtools.utils.general as gen
from simtools.io_operations import io_handler
from simtools.model.telescope_model import TelescopeModel
from simtools.simtel.simtel_runner_camera_efficiency import SimtelRunnerCameraEfficiency
from simtools.utils import names
from simtools.visualization import visualize

__all__ = ["CameraEfficiency"]


class CameraEfficiency:
    """
    Class for handling camera efficiency simulations and analysis.

    Parameters
    ----------
    telescope_model: TelescopeModel
        Instance of the TelescopeModel class.
    simtel_source_path: str (or Path)
        Location of sim_telarray installation.
    label: str
        Instance label, optional.
    config_data: dict.
        Dict containing the configurable parameters.
    config_file: str or Path
        Path of the yaml file containing the configurable parameters.
    test: bool
        Is it a test instance (at the moment only affects the location of files).
    """

    def __init__(
        self,
        telescope_model,
        simtel_source_path,
        label=None,
        config_data=None,
        config_file=None,
        test=False,
    ):
        """
        Initiliaze the CameraEfficiency class.
        """

        self._logger = logging.getLogger(__name__)

        self._simtel_source_path = simtel_source_path
        self._telescope_model = self._validate_telescope_model(telescope_model)
        self.label = label if label is not None else self._telescope_model.label
        self.test = test

        self.io_handler = io_handler.IOHandler()
        self._base_directory = self.io_handler.get_output_directory(
            label=self.label,
            sub_dir="camera-efficiency",
            dir_type="test" if self.test else "simtools",
        )

        self._results = None
        self._has_results = False

        _config_data_in = gen.collect_data_from_file_or_dict(
            config_file, config_data, allow_empty=True
        )
        _parameter_file = self.io_handler.get_input_data_file(
            "parameters", "camera-efficiency_parameters.yml"
        )
        _parameters = gen.collect_data_from_file_or_dict(_parameter_file, None)
        self.config = gen.validate_config_data(_config_data_in, _parameters)

        self._load_files()

    @classmethod
    def from_kwargs(cls, **kwargs):
        """
        Builds a CameraEfficiency object from kwargs only.
        The configurable parameters can be given as kwargs, instead of using the
        config_data or config_file arguments.

        Parameters
        ----------
        kwargs
            Containing the arguments and the configurable parameters.

        Returns
        -------
        Instance of this class.
        """
        args, config_data = gen.separate_args_and_config_data(
            expected_args=[
                "telescope_model",
                "label",
                "simtel_source_path",
                "test",
            ],
            **kwargs,
        )
        return cls(**args, config_data=config_data)

    def __repr__(self):
        return f"CameraEfficiency(label={self.label})\n"

    def _validate_telescope_model(self, tel):
        """Validate TelescopeModel

        Parameters
        ----------
        tel: TelescopeModel
            An assumed instance of the TelescopeModel class.
        Raises
        ------
        ValueError
            if tel not of type TelescopeModel

        """
        if isinstance(tel, TelescopeModel):
            self._logger.debug("TelescopeModel OK")
            return tel

        msg = "Invalid TelescopeModel"
        self._logger.error(msg)
        raise ValueError(msg)

    def _load_files(self):
        """Define the variables for the file names, including the results, simtel and log file."""
        # Results file
        file_name_results = names.camera_efficiency_results_file_name(
            site=self._telescope_model.site,
            telescope_model_name=self._telescope_model.name,
            zenith_angle=self.config.zenith_angle,
            azimuth_angle=self.config.azimuth_angle,
            label=self.label,
        )
        self._file_results = self._base_directory.joinpath(file_name_results)
        # sim_telarray output file
        file_name_simtel = names.camera_efficiency_simtel_file_name(
            site=self._telescope_model.site,
            telescope_model_name=self._telescope_model.name,
            zenith_angle=self.config.zenith_angle,
            azimuth_angle=self.config.azimuth_angle,
            label=self.label,
        )
        self._file_simtel = self._base_directory.joinpath(file_name_simtel)
        # Log file
        file_name_log = names.camera_efficiency_log_file_name(
            site=self._telescope_model.site,
            telescope_model_name=self._telescope_model.name,
            zenith_angle=self.config.zenith_angle,
            azimuth_angle=self.config.azimuth_angle,
            label=self.label,
        )
        self._file_log = self._base_directory.joinpath(file_name_log)

    def simulate(self, force=False):
        """
        Simulate camera efficiency using testeff.

        Parameters
        ----------
        force: bool
            Force flag will remove existing files and simulate again.
        """
        self._logger.info("Simulating CameraEfficiency")

        simtel = SimtelRunnerCameraEfficiency(
            simtel_source_path=self._simtel_source_path,
            telescope_model=self._telescope_model,
            zenith_angle=self.config.zenith_angle,
            file_simtel=self._file_simtel,
            file_log=self._file_log,
            label=self.label,
            nsb_spectrum=self.config.nsb_spectrum,
        )
        simtel.run(test=self.test, force=force)

    def analyze(self, export=True, force=False):
        """
        Analyze camera efficiency output file and store the results in _results.

        Parameters
        ----------
        export: bool
            If True, results will be exported to a file automatically. Alternatively, export_results
            function can be used.
        force: bool
            If True, existing results files will be removed and analysis will be done again.
        """
        self._logger.info("Analyzing CameraEfficiency")

        if self._file_results.exists() and not force:
            self._logger.info("Results file exists and force=False - skipping analyze")
            self._read_results()
            return

        # List of parameters to be calculated and stored
        eff_pars = [
            "wl",
            "eff",
            "eff_atm",
            "qe",
            "ref",
            "masts",
            "filt",
            "pixel",
            "atm_trans",
            "cher",
            "nsb",
            "atm_corr",
            "nsb_site",
            "nsb_site_eff",
            "nsb_be",
            "nsb_be_eff",
            "C1",
            "C2",
            "C3",
            "C4",
            "C4x",
            "N1",
            "N2",
            "N3",
            "N4",
            "N4x",
        ]

        _results = defaultdict(list)

        # Search for at least 5 consecutive numbers to see that we are in the table
        re_table = re.compile("{0}{0}{0}{0}{0}".format(r"[-+]?[0-9]*\.?[0-9]+\s+"))
        with open(self._file_simtel, "r", encoding="utf-8") as file:
            for line in file:
                if re_table.match(line):
                    words = line.split()
                    numbers = [float(w) for w in words]
                    for i in range(len(eff_pars) - 10):
                        _results[eff_pars[i]].append(numbers[i])
                    C1 = numbers[8] * (400 / numbers[0]) ** 2
                    C2 = C1 * numbers[4] * numbers[5]
                    C3 = C2 * numbers[6] * numbers[7]
                    C4 = C3 * numbers[3]
                    C4x = C1 * numbers[3] * numbers[6] * numbers[7]
                    _results["C1"].append(C1)
                    _results["C2"].append(C2)
                    _results["C3"].append(C3)
                    _results["C4"].append(C4)
                    _results["C4x"].append(C4x)
                    N1 = numbers[14]
                    N2 = N1 * numbers[4] * numbers[5]
                    N3 = N2 * numbers[6] * numbers[7]
                    N4 = N3 * numbers[3]
                    N4x = N1 * numbers[3] * numbers[6] * numbers[7]
                    _results["N1"].append(N1)
                    _results["N2"].append(N2)
                    _results["N3"].append(N3)
                    _results["N4"].append(N4)
                    _results["N4x"].append(N4x)

        self._results = Table(_results)
        self._has_results = True

        print("\33[40;37;1m")
        self._logger.info(f"\n{self.results_summary()}")
        print("\033[0m")

        if export:
            self.export_results()

    # END of analyze

    def results_summary(self):
        """
        Print a summary of the results.
        Include a header for the zenith/azimuth settings and the NSB spectrum file which was used.
        The summary includes the various CTAO requirements and the final expected NSB pixel rate.
        """
        nsb_pixel_pe_per_ns, nsb_rate_ref_conditions = self.calc_nsb_rate()
        nsb_spectrum_text = (
            f"NSB spectrum file: {self.config.nsb_spectrum}"
            if self.config.nsb_spectrum
            else "default sim_telarray spectrum."
        )
        summary = (
            f"Results summary for {self._telescope_model.name} at "
            f"zenith={self.config.zenith_angle:.1f} deg, "
            f"azimuth={self.config.azimuth_angle:.1f} deg\n"
            f"Using the {nsb_spectrum_text}\n"
            f"\nSpectrum weighted reflectivity: {self.calc_reflectivity():.4f}\n"
            "Camera nominal efficiency with gaps (B-TEL-1170): "
            f"{self.calc_camera_efficiency():.4f}\n"
            "Telescope total efficiency"
            f" with gaps (was A-PERF-2020): {self.calc_tel_efficiency():.4f}\n"
            "Telescope total Cherenkov light efficiency / sqrt(total NSB efficency) "
            "(A-PERF-2025/B-TEL-0090): "
            f"{self.calc_tot_efficiency(self.calc_tel_efficiency()):.4f}\n"
            "Expected NSB pixel rate for the provided NSB spectrum: "
            f"{nsb_pixel_pe_per_ns:.4f} [p.e./ns]\n"
            "Expected NSB pixel rate for the reference NSB: "
            f"{nsb_rate_ref_conditions:.4f} [p.e./ns]\n"
        )

        return summary

    def export_results(self):
        """Export results to a ecsv file."""
        if not self._has_results:
            self._logger.error("Cannot export results because they do not exist")
        else:
            self._logger.info(f"Exporting testeff table to {self._file_results}")
            astropy.io.ascii.write(
                self._results, self._file_results, format="basic", overwrite=True
            )
            _results_summary_file = (
                str(self._file_results).replace(".ecsv", ".txt").replace("-table-", "-summary-")
            )
            self._logger.info(f"Exporting summary results to {_results_summary_file}")
            with open(_results_summary_file, "w", encoding="utf-8") as file:
                file.write(self.results_summary())

    def _read_results(self):
        """Read existing results file and store it in _results."""
        table = astropy.io.ascii.read(self._file_results, format="basic")
        self._results = table
        self._has_results = True

    def calc_tel_efficiency(self):
        """
        Calculate the telescope total efficiency including gaps (as defined in A-PERF-2020).

        Returns
        -------
        tel_efficiency: float
            Telescope efficiency
        """

        # Sum(C1) from 300 - 550 nm:
        c1_reduced_wl = self._results["C1"][[299 < wl_now < 551 for wl_now in self._results["wl"]]]
        c1_sum = np.sum(c1_reduced_wl)
        # Sum(C4) from 200 - 999 nm:
        c4_sum = np.sum(self._results["C4"])
        masts_factor = self._results["masts"][0]
        fill_factor = self._telescope_model.camera.get_camera_fill_factor()

        tel_efficiency = fill_factor * (c4_sum / (masts_factor * c1_sum))

        return tel_efficiency

    def calc_camera_efficiency(self):
        """
        Calculate the camera nominal efficiency including gaps (as defined in B-TEL-1170).

        Returns
        -------
        cam_efficiency: float
            Wavelength-averaged camera efficiency
        """

        # Sum(C1) from 300 - 550 nm:
        c1_reduced_wl = self._results["C1"][[299 < wl_now < 551 for wl_now in self._results["wl"]]]
        c1_sum = np.sum(c1_reduced_wl)
        # Sum(C4x) from 300 - 550 nm:
        c4x_reduced_wl = self._results["C4x"][
            [299 < wl_now < 551 for wl_now in self._results["wl"]]
        ]
        c4x_sum = np.sum(c4x_reduced_wl)
        fill_factor = self._telescope_model.camera.get_camera_fill_factor()

        cam_efficiency_no_gaps = c4x_sum / c1_sum
        cam_efficiency = cam_efficiency_no_gaps * fill_factor

        return cam_efficiency

    def calc_tot_efficiency(self, tel_efficiency):
        """
        Calculate the telescope total efficiency including gaps (as defined in A-PERF-2020).

        Parameters
        ----------
        tel_efficiency: float
            The telescope efficiency as calculated by calc_tel_efficiency()

        Returns
        -------
        Float
            Telescope total efficiency including gaps
        """

        # Sum(N1) from 300 - 550 nm:
        n1_reduced_wl = self._results["N1"][[299 < wl_now < 551 for wl_now in self._results["wl"]]]
        n1_sum = np.sum(n1_reduced_wl)
        # Sum(N4) from 200 - 999 nm:
        n4_sum = np.sum(self._results["N4"])
        masts_factor = self._results["masts"][0]
        fill_factor = self._telescope_model.camera.get_camera_fill_factor()

        tel_efficiency_nsb = fill_factor * (n4_sum / (masts_factor * n1_sum))

        return tel_efficiency / np.sqrt(tel_efficiency_nsb)

    def calc_reflectivity(self):
        """
        Calculate the Cherenkov spectrum weighted reflectivity in the range 300-550 nm.

        Returns
        -------
        Float
            Cherenkov spectrum weighted reflectivity (300-550 nm)
        """

        # Sum(C1) from 300 - 550 nm:
        c1_reduced_wl = self._results["C1"][[299 < wl_now < 551 for wl_now in self._results["wl"]]]
        c1_sum = np.sum(c1_reduced_wl)
        # Sum(C2) from 300 - 550 nm:
        c2_reduced_wl = self._results["C2"][[299 < wl_now < 551 for wl_now in self._results["wl"]]]
        c2_sum = np.sum(c2_reduced_wl)
        cher_spec_weighted_reflectivity = c2_sum / c1_sum / self._results["masts"][0]

        return cher_spec_weighted_reflectivity

    def calc_nsb_rate(self):
        """
        Calculate the NSB rate.

        Returns
        -------
        nsb_rate_provided_spectrum: float
            NSB pixel rate in p.e./ns for the provided NSB spectrum
        nsb_rate_ref_conditions: float
            NSB pixel rate in p.e./ns for reference conditions
            (https://jama.cta-observatory.org/perspective.req#/items/26694?projectId=11)
        """

        nsb_rate_provided_spectrum = (
            np.sum(self._results["N4"])
            * self._telescope_model.camera.get_pixel_active_solid_angle()
            * self._telescope_model.get_on_axis_eff_optical_area().to("m2").value
            / self._telescope_model.get_telescope_transmission_parameters()[0]
        )

        # (integral is in ph./(m^2 ns sr) ) from 300 - 650 nm:
        n1_reduced_wl = self._results["N1"][[299 < wl_now < 651 for wl_now in self._results["wl"]]]
        n1_sum = np.sum(n1_reduced_wl)
        n1_integral_edges = self._results["N1"][
            [wl_now in [300, 650] for wl_now in self._results["wl"]]
        ]
        n1_integral_edges_sum = np.sum(n1_integral_edges)
        nsb_integral = 0.0001 * (n1_sum - 0.5 * n1_integral_edges_sum)
        nsb_rate_ref_conditions = (
            nsb_rate_provided_spectrum
            * self._telescope_model.reference_data["nsb_reference_value"]["Value"]
            / nsb_integral
        )
        return nsb_rate_provided_spectrum, nsb_rate_ref_conditions

    def plot_cherenkov_efficiency(self):
        """
        Plot Cherenkov efficiency vs wavelength.

        Returns
        -------
        fig
            The figure instance of pyplot
        """
        self._logger.info("Plotting Cherenkov efficiency vs wavelength")

        column_titles = {
            "wl": "Wavelength [nm]",
            "C1": r"C1: Cherenkov light on ground",
            "C2": r"C2: C1 $\times$ ref. $\times$ masts",
            "C3": r"C3: C2 $\times$ filter $\times$ lightguide",
            "C4": r"C4: C3 $\times$ q.e.",
            "C4x": r"C4x: C1 $\times$ filter $\times$ lightguide $\times$ q.e.",
        }

        table_to_plot = Table([self._results[col_now] for col_now in column_titles])

        for column_now, column_title in column_titles.items():
            table_to_plot.rename_column(column_now, column_title)

        fig = visualize.plot_table(
            table_to_plot,
            y_title="Cherenkov light efficiency",
            title=f"{self._telescope_model.name} response to Cherenkov light",
            no_markers=True,
        )

        return fig

    def plot_nsb_efficiency(self):
        """
        Plot NSB efficiency vs wavelength.

        Returns
        -------
        fig
            The figure instance of pyplot
        """
        self._logger.info("Plotting NSB efficiency vs wavelength")
        column_titles = {
            "wl": "Wavelength [nm]",
            "N1": r"N1: NSB light on ground (B\&E)",
            "N2": r"N2: N1 $\times$ ref. $\times$ masts",
            "N3": r"N3: N2 $\times$ filter $\times$ lightguide",
            "N4": r"N4: N3 $\times$ q.e.",
            "N4x": r"N4x: N1 $\times$ filter $\times$ lightguide $\times$ q.e.",
        }

        table_to_plot = Table([self._results[col_now] for col_now in column_titles])

        for column_now, column_title in column_titles.items():
            table_to_plot.rename_column(column_now, column_title)

        plot = visualize.plot_table(
            table_to_plot,
            y_title="Nightsky background light efficiency",
            title=f"{self._telescope_model.name} response to nightsky background light",
            no_markers=True,
        )

        plot.gca().set_yscale("log")
        ylim = plot.gca().get_ylim()
        plot.gca().set_ylim(1e-3, ylim[1])

        return plot
