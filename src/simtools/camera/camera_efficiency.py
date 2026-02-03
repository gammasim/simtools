"""Camera efficiency simulations and analysis."""

import logging
import re
from collections import defaultdict
from pathlib import Path

import astropy.io.ascii
import astropy.units as u
import numpy as np
from astropy.table import Table

import simtools.data_model.model_data_writer as writer
from simtools import settings
from simtools.atmosphere import AtmosphereProfile
from simtools.io import ascii_handler, io_handler
from simtools.model.model_utils import initialize_simulation_models
from simtools.simtel.simulator_camera_efficiency import SimulatorCameraEfficiency
from simtools.utils import names
from simtools.visualization import visualize


class CameraEfficiency:
    """
    Camera efficiency simulations and analysis.

    Parameters
    ----------
    label: str
        Instance label, optional.
    config_data: dict.
        Dict containing the configurable parameters.
    efficiency_type: str
        The type of efficiency to simulate (e.g., 'Shower', 'Muon', or 'NSB').
    """

    def __init__(self, label, config_data, efficiency_type):
        """Initialize the CameraEfficiency class."""
        self._logger = logging.getLogger(__name__)

        self.label = label

        self.io_handler = io_handler.IOHandler()
        self.telescope_model, self.site_model, _ = initialize_simulation_models(
            label=self.label,
            model_version=config_data["model_version"],
            site=config_data["site"],
            telescope_name=config_data["telescope"],
        )
        self.output_dir = self.io_handler.get_output_directory()

        self._results = None
        self._has_results = False
        self.efficiency_type = efficiency_type.lower()

        self.config = self._configuration(config_data)
        self._file = self._load_files()

        self.nsb_pixel_pe_per_ns = None
        self.nsb_rate_ref_conditions = None

    def __repr__(self):
        """Return string representation of the CameraEfficiency instance."""
        return f"CameraEfficiency(label={self.label})\n"

    def _configuration(self, config_data):
        """
        Extract configuration data from command line and class parameters.

        Parameters
        ----------
        config_data: dict
            Dict containing the configurable parameters.

        Returns
        -------
        dict
            Configuration data.
        """
        return {
            "zenith_angle": config_data["zenith_angle"].to("deg").value,
            "azimuth_angle": config_data["azimuth_angle"].to("deg").value,
            "nsb_spectrum": config_data.get("nsb_spectrum", None),
            "efficiency_type": self.efficiency_type,
        }

    def _load_files(self):
        """Define variables used for file names, including results, sim_telarray and log files."""
        _file = {}
        for label, suffix in zip(
            ["results", "sim_telarray", "log"],
            [".ecsv", ".dat", ".log"],
        ):
            file_name = names.generate_file_name(
                file_type="camera_efficiency",
                suffix=suffix,
                site=self.telescope_model.site,
                telescope_model_name=self.telescope_model.name,
                zenith_angle=self.config["zenith_angle"],
                azimuth_angle=self.config["azimuth_angle"],
                label=self.efficiency_type,
            )

            _file[label] = self.io_handler.get_output_directory().joinpath(file_name)
        return _file

    def simulate(self):
        """Simulate camera efficiency using testeff."""
        self._logger.info("Simulating CameraEfficiency")

        self.export_model_files()

        simtel = SimulatorCameraEfficiency(
            telescope_model=self.telescope_model,
            site_model=self.site_model,
            zenith_angle=self.config["zenith_angle"],
            file_simtel=self._file["sim_telarray"],
            file_log=self._file["log"],
            label=self.label,
            x_max=self._get_x_max_for_efficiency_type(),
            nsb_spectrum=self.config["nsb_spectrum"],
            skip_correction_to_nsb_spectrum=self.config.get(
                "skip_correction_to_nsb_spectrum", False
            ),
        )
        simtel.run()

    def export_model_files(self):
        """Export model and config files to the output directory."""
        self.telescope_model.write_sim_telarray_config_file()
        if not self.config.get("skip_correction_to_nsb_spectrum", False):
            self.telescope_model.export_nsb_spectrum_to_telescope_altitude_correction_file(
                model_directory=self.telescope_model.config_file_directory
            )

    def get_nsb_pixel_rate(self, reference_conditions=False):
        """
        Return the expected NSB pixel rate for each camera pixel.

        This is an approximation, as testeff calculates the expected NSB pixel rate
        for the on-axis pixel only.

        Returns
        -------
        list
            Expected NSB pixel rate in p.e./ns for the provided NSB spectrum.
        """
        base_rate = (
            self.nsb_rate_ref_conditions if reference_conditions else self.nsb_pixel_pe_per_ns
        )
        # Accept either a plain float (assumed already in GHz) or an astropy Quantity
        if isinstance(base_rate, u.Quantity):
            base_rate_ghz = base_rate.to(u.GHz).value
        else:
            base_rate_ghz = float(base_rate)

        n_pixels = int(self.telescope_model.get_parameter_value("camera_pixels"))
        return u.Quantity(np.full(n_pixels, base_rate_ghz), u.GHz)

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

        if "results" in self._file and not force:
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
        with open(self._file["sim_telarray"], encoding="utf-8") as file:
            for line in file:
                if re_table.match(line):
                    words = line.split()
                    numbers = [float(w) for w in words]
                    for i in range(len(eff_pars) - 10):
                        _results[eff_pars[i]].append(numbers[i])
                    C1 = numbers[8] * (400 / numbers[0]) ** 2  # noqa: N806
                    C2 = C1 * numbers[4] * numbers[5]  # noqa: N806
                    C3 = C2 * numbers[6] * numbers[7]  # noqa: N806
                    C4 = C3 * numbers[3]  # noqa: N806
                    c4x_value = C1 * numbers[3] * numbers[6] * numbers[7]
                    _results["C1"].append(C1)
                    _results["C2"].append(C2)
                    _results["C3"].append(C3)
                    _results["C4"].append(C4)
                    _results["C4x"].append(c4x_value)
                    N1 = numbers[14]  # noqa: N806
                    N2 = N1 * numbers[4] * numbers[5]  # noqa: N806
                    N3 = N2 * numbers[6] * numbers[7]  # noqa: N806
                    N4 = N3 * numbers[3]  # noqa: N806
                    n4x_value = N1 * numbers[3] * numbers[6] * numbers[7]
                    _results["N1"].append(N1)
                    _results["N2"].append(N2)
                    _results["N3"].append(N3)
                    _results["N4"].append(N4)
                    _results["N4x"].append(n4x_value)

        self._results = Table(_results)
        self._has_results = True

        self.nsb_pixel_pe_per_ns, self.nsb_rate_ref_conditions = self.calc_nsb_rate()

        print("\33[40;37;1m")
        self._logger.info(f"\n{self.results_summary()}")
        print("\033[0m")

        if export:
            self.export_results()

    def results_summary(self):
        """
        Fill a dictionary with summary of the results.

        Include a header for the zenith/azimuth settings and the NSB spectrum file which was used.
        The summary includes the various CTAO requirements and the final expected NSB pixel rate.

        Returns
        -------
        dict
            Summary of the results.
        """
        meta = {
            "meta": {
                "tel": self.telescope_model.name,
                "model_version": self.telescope_model.model_version,
                "zen": self.config["zenith_angle"],
                "az": self.config["azimuth_angle"],
                "nsb": (
                    self.config["nsb_spectrum"]
                    if self.config["nsb_spectrum"]
                    else "default sim_telarray spectrum"
                ),
            }
        }

        metrics = {}
        if self.efficiency_type == "shower":
            metrics |= {
                "reflectivity": {
                    "value": self.calc_reflectivity(),
                    "description": "Spectrum weighted reflectivity",
                },
                "cam_eff": {
                    "value": self.calc_camera_efficiency(),
                    "description": "Camera nominal efficiency with gaps (B-TEL-1170)",
                },
                "tel_eff": {
                    "value": self.calc_tel_efficiency(),
                    "description": "Telescope total efficiency with gaps (was A-PERF-2020)",
                },
                "tot_sens": {
                    "value": self.calc_tot_efficiency(self.calc_tel_efficiency()),
                    "description": (
                        "Telescope total Cherenkov light efficiency / sqrt(total NSB efficiency) "
                        "(A-PERF-2025/B-TEL-0090)"
                    ),
                },
            }

        elif self.efficiency_type == "nsb":
            metrics |= {
                "nsb_rate": {
                    "value": self.nsb_pixel_pe_per_ns,
                    "description": "Expected NSB pixel rate for the provided NSB spectrum",
                },
                "nsb_ref": {
                    "value": self.nsb_rate_ref_conditions,
                    "description": "Expected NSB pixel rate for the reference NSB",
                },
            }

        elif self.efficiency_type == "muon":
            metrics |= {
                "muon_frac": {
                    "value": self.calc_partial_efficiency(lambda_min=200.0, lambda_max=290.0),
                    "description": (
                        "Fraction of light (from muons) in the wavelength range 200-290 nm "
                        "(B-TEL-0095)"
                    ),
                },
            }

        return meta | metrics

    def export_results(self):
        """Export results to a ecsv file."""
        if not self._has_results:
            self._logger.error("Cannot export results because they do not exist")
        else:
            self._logger.info(f"Exporting testeff table to {self._file['results']}")
            astropy.io.ascii.write(
                self._results, self._file["results"], format="basic", overwrite=True
            )
            _results_summary_file = str(self._file["results"]).replace(".ecsv", "_summary.yml")
            self._logger.info(f"Exporting summary results to {_results_summary_file}")
            ascii_handler.write_data_to_file(self.results_summary(), Path(_results_summary_file))

    def _read_results(self):
        """Read existing results file and store it in _results."""
        table = astropy.io.ascii.read(self._file["results"], format="basic")
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
        fill_factor = self.telescope_model.camera.get_camera_fill_factor()

        return fill_factor * (c4_sum / (masts_factor * c1_sum))

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
        fill_factor = self.telescope_model.camera.get_camera_fill_factor()

        cam_efficiency_no_gaps = c4x_sum / c1_sum
        return cam_efficiency_no_gaps * fill_factor

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
        fill_factor = self.telescope_model.camera.get_camera_fill_factor()

        tel_efficiency_nsb = fill_factor * (n4_sum / (masts_factor * n1_sum))

        return tel_efficiency / np.sqrt(tel_efficiency_nsb)

    def calc_partial_efficiency(self, lambda_min=200.0, lambda_max=290.0):
        """
        Compare efficiency in a given wavelength range with total efficiency.

        Parameters
        ----------
        lambda_min: float
            Minimum wavelength in nm.
        lambda_max: float
            Maximum wavelength in nm.

        Returns
        -------
        Float
            Fraction of light in the given wavelength range compared to total efficiency.

        """
        # Sum(C4) from lamba_min to lambda_max nm:
        c4_reduced_wl = self._results["C4"][
            [lambda_min < wl_now < lambda_max for wl_now in self._results["wl"]]
        ]
        c4_sum = np.sum(c4_reduced_wl)
        # Sum(C4) from 200 - 999 nm:
        c4_sum_total = np.sum(self._results["C4"])
        # (no need to apply masts or fill factors as in calc_tel_efficiency, they cancel out)

        self._logger.info(
            f"Fraction of light in the wavelength range {lambda_min}-{lambda_max} nm: "
            f"{c4_sum / c4_sum_total:.4f}"
        )

        return c4_sum / c4_sum_total

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
        return c2_sum / c1_sum / self._results["masts"][0]

    def calc_nsb_rate(self, wavelength_range=(300 * u.nm, 650 * u.nm)):
        """
        Calculate the NSB rate.

        CTAO reference wavelength range is 300-650 nm.

        Parameters
        ----------
        wavelength_range: tuple
            Wavelength range used for the NSB rate calculation (default: (300 nm, 650 nm)).

        Returns
        -------
        nsb_rate_provided_spectrum: float
            NSB pixel rate in p.e./ns for the provided NSB spectrum
        nsb_rate_ref_conditions: float
            NSB pixel rate in p.e./ns for reference conditions
            (https://jama.cta-observatory.org/perspective.req#/items/26694?projectId=11)
        """
        self.nsb_pixel_pe_per_ns = (
            np.sum(self._results["N4"])
            * self.telescope_model.camera.get_pixel_active_solid_angle()
            * self.telescope_model.get_on_axis_eff_optical_area().to("m2").value
            / self.telescope_model.get_parameter_value("telescope_transmission")[0]
        )

        wavelength_range = (
            wavelength_range[0].to("nm").value,
            wavelength_range[1].to("nm").value,
        )

        # (integral is in ph./(m^2 ns sr) ) over wavelength_range
        n1_reduced_wl = self._results["N1"][
            [wavelength_range[0] <= wl_now <= wavelength_range[1] for wl_now in self._results["wl"]]
        ]
        n1_sum = np.sum(n1_reduced_wl)
        n1_integral_edges = self._results["N1"][
            [wl_now in [wavelength_range[0], wavelength_range[1]] for wl_now in self._results["wl"]]
        ]
        n1_integral_edges_sum = np.sum(n1_integral_edges)
        nsb_integral = 0.0001 * (n1_sum - 0.5 * n1_integral_edges_sum)
        self.nsb_rate_ref_conditions = (
            self.nsb_pixel_pe_per_ns
            * self.site_model.get_parameter_value("nsb_reference_value")
            / nsb_integral
        )
        return self.nsb_pixel_pe_per_ns * u.GHz, self.nsb_rate_ref_conditions * u.GHz

    def plot_efficiency(self, save_fig=False):
        """
        Plot efficiency vs wavelength.

        Parameters
        ----------
        save_fig: bool
            If True, the figure will be saved to a file.

        Returns
        -------
        fig
            The figure instance of pyplot
        """
        self._logger.info(f"Plotting {self.efficiency_type} efficiency vs wavelength")

        _col_type = "C" if self.efficiency_type in ("shower", "muon") else "N"

        column_titles = {
            "wl": "Wavelength [nm]",
            f"{_col_type}1": rf"{_col_type}1: Cherenkov light on ground",
            f"{_col_type}2": rf"{_col_type}2: {_col_type}1 $\times$ ref. $\times$ masts",
            f"{_col_type}3": rf"{_col_type}3: {_col_type}2 $\times$ filter $\times$ lightguide",
            f"{_col_type}4": rf"{_col_type}4: {_col_type}3 $\times$ q.e.",
            f"{_col_type}4x": (
                rf"{_col_type}4x: {_col_type}1 $\times$ filter $\times$ lightguide $\times$ q.e."
            ),
        }

        table_to_plot = Table([self._results[col_now] for col_now in column_titles])

        for column_now, column_title in column_titles.items():
            table_to_plot.rename_column(column_now, column_title)

        y_title = f"{self.efficiency_type} light efficiency"
        if self.efficiency_type == "nsb":
            y_title = r"Diff. ph. rate [$10^{9} \times $ph/(nm s m$^2$ sr)]"
        plot = visualize.plot_table(
            table_to_plot,
            y_title=y_title,
            title=f"{self.telescope_model.name} response to {self.efficiency_type} light",
            no_markers=True,
        )
        if self.efficiency_type == "nsb":
            plot.gca().set_yscale("log")
            ylim = plot.gca().get_ylim()
            plot.gca().set_ylim(1e-3, ylim[1])
        if save_fig:
            self._save_plot(plot, self.efficiency_type)
        return plot

    def _save_plot(self, fig, plot_title):
        """
        Save plot to pdf and png file.

        Parameters
        ----------
        fig
            The figure instance of pyplot
        plot_title: str
            The title of the plot
        """
        plot_file = self.output_dir.joinpath(
            self.label + "_" + self.telescope_model.name + "_" + plot_title
        )
        visualize.save_figure(fig, plot_file, log_title=f"{plot_title} efficiency")

    def dump_nsb_pixel_rate(self):
        """Write NSB pixel rate parameter file."""
        cfg = settings.config.args

        writer.ModelDataWriter.dump_model_parameter(
            parameter_name="nsb_pixel_rate",
            value=self.get_nsb_pixel_rate(
                reference_conditions=settings.config.args.get(
                    "write_reference_nsb_rate_as_parameter", False
                )
            ),
            instrument=cfg.get("telescope"),
            parameter_version=cfg.get("parameter_version") or "0.0.0",
            output_file=Path(f"nsb_pixel_rate-{cfg.get('parameter_version', '0.0.0')}.json"),
            output_path=self.output_dir / cfg.get("telescope") / "nsb_pixel_rate",
        )

    def _get_x_max_for_efficiency_type(self):
        """
        Get X max value in g/cm2 depending on the efficiency type.

        Returns
        -------
        float
             max value in g/cm2
        """
        # typical value for shower X-max around 10 km (not relevant for NSB type)
        x_max = 300.0
        obs_level = self.site_model.get_parameter_value_with_unit("corsika_observation_level")
        if self.efficiency_type == "muon":
            atmo = AtmosphereProfile(
                self.site_model.config_file_directory
                / self.site_model.get_parameter_value("atmospheric_profile")
            )
            alt = obs_level.to(u.km) + 0.1 * u.km
            x_max = atmo.interpolate(altitude=alt, column="thick")

        self._logger.info(
            f"Using X-max for {self.efficiency_type} efficiency: {x_max:.2f} g/cm2"
            f" (at observation level: {obs_level:.2f})"
        )

        return x_max
