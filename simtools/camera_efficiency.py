import logging
import os
import re
from collections import defaultdict

import astropy.io.ascii
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

import simtools.util.general as gen
from simtools import io_handler, visualize
from simtools.model.telescope_model import TelescopeModel
from simtools.util import names

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

    Attributes
    ----------
    config: namedtuple
        Contains the configurable parameters (zenith_angle).
    io_handler: IOHandler
        Instance of IOHandler
    label: str
        Instance label.
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

        self.io_handler = io_handler.IOHandler()
        self._base_directory = self.io_handler.get_output_directory(
            label=self.label,
            dir_type="camera-efficiency",
            test=test,
        )

        self._has_results = False

        _config_data_in = gen.collect_data_from_yaml_or_dict(
            config_file, config_data, allow_empty=True
        )
        _parameter_file = self.io_handler.get_input_data_file(
            "parameters", "camera-efficiency_parameters.yml"
        )
        _parameters = gen.collect_data_from_yaml_or_dict(_parameter_file, None)
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
        return "CameraEfficiency(label={})\n".format(self.label)

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
        else:
            msg = "Invalid TelescopeModel"
            self._logger.error(msg)
            raise ValueError(msg)

    def _load_files(self):
        """Define the variables for the file names, including the results, simtel and log file."""
        # Results file
        file_name_results = names.camera_efficiency_results_file_name(
            self._telescope_model.site,
            self._telescope_model.name,
            self.config.zenith_angle,
            self.label,
        )
        self._file_results = self._base_directory.joinpath(file_name_results)
        # sim_telarray output file
        file_name_simtel = names.camera_efficiency_simtel_file_name(
            self._telescope_model.site,
            self._telescope_model.name,
            self.config.zenith_angle,
            self.label,
        )
        self._file_simtel = self._base_directory.joinpath(file_name_simtel)
        # Log file
        file_name_log = names.camera_efficiency_log_file_name(
            self._telescope_model.site,
            self._telescope_model.name,
            self.config.zenith_angle,
            self.label,
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

        if self._file_simtel.exists() and not force:
            self._logger.info("Simtel file exists and force=False - skipping simulation")
            return

        # Processing camera pixel features
        pixel_shape = self._telescope_model.camera.get_pixel_shape()
        pixel_shape_cmd = "-hpix" if pixel_shape in [1, 3] else "-spix"
        pixel_diameter = self._telescope_model.camera.get_pixel_diameter()

        # Processing focal length
        focal_length = self._telescope_model.get_parameter_value("effective_focal_length")
        if focal_length == 0.0:
            self._logger.warning("Using focal_length because effective_focal_length is 0")
            focal_length = self._telescope_model.get_parameter_value("focal_length")

        # Processing mirror class
        mirror_class = 1
        if self._telescope_model.has_parameter("mirror_class"):
            mirror_class = self._telescope_model.get_parameter_value("mirror_class")

        # Processing camera transmission
        camera_transmission = 1
        if self._telescope_model.has_parameter("camera_transmission"):
            camera_transmission = self._telescope_model.get_parameter_value("camera_transmission")

        # Processing camera filter
        # A special case is testeff does not support 2D distributions
        camera_filter_file = self._telescope_model.get_parameter_value("camera_filter")
        if self._telescope_model.is_file_2D("camera_filter"):
            camera_filter_file = self._get_one_dim_distribution(
                "camera_filter", "camera_filter_incidence_angle"
            )

        # Processing mirror reflectivity
        # A special case is testeff does not support 2D distributions
        mirror_reflectivity = self._telescope_model.get_parameter_value("mirror_reflectivity")
        if mirror_class == 2:
            mirror_reflectivity_secondary = mirror_reflectivity
        if self._telescope_model.is_file_2D("mirror_reflectivity"):
            mirror_reflectivity = self._get_one_dim_distribution(
                "mirror_reflectivity", "primary_mirror_incidence_angle"
            )
            mirror_reflectivity_secondary = self._get_one_dim_distribution(
                "mirror_reflectivity", "secondary_mirror_incidence_angle"
            )

        # cmd -> Command to be run at the shell
        cmd = str(self._simtel_source_path.joinpath("sim_telarray/bin/testeff"))
        cmd += " -nm -nsb-extra"
        cmd += f" -alt {self._telescope_model.get_parameter_value('altitude')}"
        cmd += f" -fatm {self._telescope_model.get_parameter_value('atmospheric_transmission')}"
        cmd += f" -flen {focal_length * 0.01}"  # focal length in meters
        cmd += f" {pixel_shape_cmd} {pixel_diameter}"
        if mirror_class == 1:
            cmd += f" -fmir {self._telescope_model.get_parameter_value('mirror_list')}"
        cmd += f" -fref {mirror_reflectivity}"
        if mirror_class == 2:
            cmd += " -m2"
            cmd += f" -fref2 {mirror_reflectivity_secondary}"
        cmd += f" -teltrans {self._telescope_model.get_telescope_transmission_parameters()[0]}"
        cmd += f" -camtrans {camera_transmission}"
        cmd += f" -fflt {camera_filter_file}"
        cmd += f" -fang {self._telescope_model.camera.get_lightguide_efficiency_angle_file_name()}"
        cmd += (
            f" -fwl {self._telescope_model.camera.get_lightguide_efficiency_wavelength_file_name()}"
        )
        cmd += f" -fqe {self._telescope_model.get_parameter_value('quantum_efficiency')}"
        cmd += " 200 1000"  # lmin and lmax
        cmd += " 300 26"  # Xmax, ioatm (Konrad always uses 26)
        cmd += f" {self.config.zenith_angle}"
        cmd += f" 2>{self._file_log}"
        cmd += f" >{self._file_simtel}"

        # Moving to sim_telarray directory before running
        cmd = f"cd {self._simtel_source_path.joinpath('sim_telarray')} && {cmd}"

        self._logger.info(f"Running sim_telarray with cmd: {cmd}")
        os.system(cmd)
        return

    # END of simulate

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
        with open(self._file_simtel, "r") as file:
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
        self._logger.info(f"Spectrum weighted reflectivity: {self.calc_reflectivity():.4f}")
        self._logger.info(
            f"Camera nominal efficiency with gaps (B-TEL-1170): {self.calc_camera_efficiency():.4f}"
        )
        self._logger.info(
            "Telescope total efficiency"
            f" with gaps (was A-PERF-2020): {self.calc_tel_efficiency():.4f}"
        )
        self._logger.info(
            (
                "Telescope total Cherenkov light efficiency / sqrt(total NSB efficency) "
                "(A-PERF-2025/B-TEL-0090): "
                f"{self.calc_tot_efficiency(self.calc_tel_efficiency()):.4f}"
            )
        )
        self._logger.info(
            "Expected NSB pixel rate for the reference NSB: "
            f"{self.calc_nsb_rate():.4f} [p.e./ns]"
        )
        print("\033[0m")

        if export:
            self.export_results()

    # END of analyze

    def export_results(self):
        """Export results to a csv file."""
        if not self._has_results:
            self._logger.error("Cannot export results because they do not exist")
        else:
            self._logger.info("Exporting results to {}".format(self._file_results))
            astropy.io.ascii.write(
                self._results, self._file_results, format="basic", overwrite=True
            )

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
        c1_reduced_wl = self._results["C1"][
            [wl_now > 299 and wl_now < 551 for wl_now in self._results["wl"]]
        ]
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
        c1_reduced_wl = self._results["C1"][
            [wl_now > 299 and wl_now < 551 for wl_now in self._results["wl"]]
        ]
        c1_sum = np.sum(c1_reduced_wl)
        # Sum(C4x) from 300 - 550 nm:
        c4x_reduced_wl = self._results["C4x"][
            [wl_now > 299 and wl_now < 551 for wl_now in self._results["wl"]]
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
        tel_total_efficiency
            Telescope total efficiency
        """

        # Sum(N1) from 300 - 550 nm:
        n1_reduced_wl = self._results["N1"][
            [wl_now > 299 and wl_now < 551 for wl_now in self._results["wl"]]
        ]
        n1_sum = np.sum(n1_reduced_wl)
        # Sum(N4) from 200 - 999 nm:
        n4_sum = np.sum(self._results["N4"])
        masts_factor = self._results["masts"][0]
        fill_factor = self._telescope_model.camera.get_camera_fill_factor()

        tel_efficiency_nsb = fill_factor * (n4_sum / (masts_factor * n1_sum))
        tel_total_efficiency = tel_efficiency / np.sqrt(tel_efficiency_nsb)

        return tel_total_efficiency

    def calc_reflectivity(self):
        """
        Calculate the Cherenkov spectrum weighted reflectivity in the range 300-550 nm.

        Returns
        -------
        Float
            Cherenkov spectrum weighted reflectivity (300-550 nm)
        """

        # Sum(C1) from 300 - 550 nm:
        c1_reduced_wl = self._results["C1"][
            [wl_now > 299 and wl_now < 551 for wl_now in self._results["wl"]]
        ]
        c1_sum = np.sum(c1_reduced_wl)
        # Sum(C2) from 300 - 550 nm:
        c2_reduced_wl = self._results["C2"][
            [wl_now > 299 and wl_now < 551 for wl_now in self._results["wl"]]
        ]
        c2_sum = np.sum(c2_reduced_wl)
        cher_spec_weighted_reflectivity = c2_sum / c1_sum / self._results["masts"][0]

        return cher_spec_weighted_reflectivity

    def calc_nsb_rate(self):
        """
        Calculate the NSB rate.

        Returns
        -------
        nsb_rate
            NSB rate in p.e./ns
        """

        nsb_pe_per_ns = (
            np.sum(self._results["N4"])
            * self._telescope_model.camera.get_pixel_active_solid_angle()
            * self._telescope_model.get_on_axis_eff_optical_area().to("m2").value
            / self._telescope_model.get_telescope_transmission_parameters()[0]
        )

        print(self._telescope_model.get_on_axis_eff_optical_area().to("m2").value)

        # NSB input spectrum is from Benn&Ellison
        # (integral is in ph./(cm² ns sr) ) from 300 - 650 nm:
        n1_reduced_wl = self._results["N1"][
            [wl_now > 299 and wl_now < 651 for wl_now in self._results["wl"]]
        ]
        n1_sum = np.sum(n1_reduced_wl)
        n1_integral_edges = self._results["N1"][
            [wl_now == 300 or wl_now == 650 for wl_now in self._results["wl"]]
        ]
        n1_integral_edges_sum = np.sum(n1_integral_edges)
        nsb_integral = 0.0001 * (n1_sum - 0.5 * n1_integral_edges_sum)
        nsb_rate = (
            nsb_pe_per_ns
            * self._telescope_model.reference_data["nsb_reference_value"]["Value"]
            / nsb_integral
        )
        return nsb_rate

    def plot(self, key, **kwargs):  # FIXME - remove this function, probably not needed
        """
        Plot key vs wavelength.

        Parameters
        ----------
        key: str
            cherenkov or nsb
        **kwargs:
            kwargs for plt.plot

        Raises
        ------
        KeyError
            If key is not among the valid options.
        """
        if key not in ["cherenkov", "nsb"]:
            msg = "Invalid key to plot"
            self._logger.error(msg)
            raise KeyError(msg)

        ax = plt.gca()
        first_letter = "C" if key == "cherenkov" else "N"
        for par in ["1", "2", "3", "4", "4x"]:
            ax.plot(
                self._results["wl"],
                self._results[first_letter + par],
                label=first_letter + par,
                **kwargs,
            )

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
            title="{} response to Cherenkov light".format(self._telescope_model.name),
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

        plt = visualize.plot_table(
            table_to_plot,
            y_title="Nightsky background light efficiency",
            title="{} response to nightsky background light".format(self._telescope_model.name),
            no_markers=True,
        )

        plt.gca().set_yscale("log")
        ylim = plt.gca().get_ylim()
        plt.gca().set_ylim(1e-3, ylim[1])

        return plt

    def _get_one_dim_distribution(self, two_dim_parameter, weighting_distribution_parameter):
        """
        Calculate an average one-dimensional curve for testeff from the two-dimensional curve.
        The two-dimensional distribution is provided in two_dim_parameter. The distribution
        of weights to use for averaging the two-dimensional distribution is given in
        weighting_distribution_parameter.

        Returns
        -------
        one_dim_file: Path
            The file path and name with the new one-dimensional distribution
        """
        incidence_angle_distribution_file = self._telescope_model.get_parameter_value(
            weighting_distribution_parameter
        )
        incidence_angle_distribution = self._telescope_model.read_incidence_angle_distribution(
            incidence_angle_distribution_file
        )
        self._logger.warning(
            f"The {' '.join(two_dim_parameter.split('_'))} distribution "
            "is a 2D one which testeff does not support. "
            "Instead of using the 2D distribution, the two dimensional distribution "
            "will be averaged, using the photon incidence angle distribution as weights. "
            "The incidence angle distribution is taken "
            f"from the file - {incidence_angle_distribution_file})."
        )
        two_dim_distribution = self._telescope_model.read_two_dim_wavelength_angle(
            self._telescope_model.get_parameter_value(two_dim_parameter)
        )
        distribution_to_export = self._telescope_model.calc_average_curve(
            two_dim_distribution, incidence_angle_distribution
        )
        new_file_name = (
            f"weighted_average_1D_{self._telescope_model.get_parameter_value(two_dim_parameter)}"
        )
        one_dim_file = self._telescope_model.export_table_to_model_directory(
            new_file_name, distribution_to_export
        )

        return one_dim_file
