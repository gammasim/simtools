"""Simulation runner for camera efficiency calculations."""

import logging
from pathlib import Path

from simtools.io import ascii_handler
from simtools.runners.simtel_runner import SimtelRunner
from simtools.utils import general


class SimulatorCameraEfficiency(SimtelRunner):
    """
    Interface with the testeff tool of sim_telarray to perform camera efficiency simulations.

    Parameters
    ----------
    telescope_model: TelescopeModel
        Instance of TelescopeModel class.
    site_model: SiteModel
        Instance of SiteModel class.
    label: str
        Instance label. Important for output file naming.
    simtel_path: str or Path
        Location of sim_telarray installation.
    file_simtel: str or Path
        Location of the sim_telarray testeff tool output file.
    zenith_angle: float
        Zenith angle given in the config to CameraEfficiency.
    nsb_spectrum: str or Path
        Path to the nsb spectrum file.
    skip_correction_to_nsb_spectrum: bool
        If True, skip the correction to the original altitude where the NSB spectrum was derived.
    """

    def __init__(
        self,
        telescope_model,
        site_model,
        label=None,
        simtel_path=None,
        file_simtel=None,
        file_log=None,
        zenith_angle=None,
        nsb_spectrum=None,
        skip_correction_to_nsb_spectrum=False,
    ):
        """Initialize SimtelRunner."""
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init SimulatorCameraEfficiency")

        super().__init__(label=label, simtel_path=simtel_path)

        self._telescope_model = telescope_model
        self._site_model = site_model
        self.label = label if label is not None else self._telescope_model.label

        self._file_simtel = file_simtel
        self._file_log = file_log
        self.zenith_angle = zenith_angle
        self.nsb_spectrum = nsb_spectrum
        self.skip_correction_to_nsb_spectrum = skip_correction_to_nsb_spectrum

    @property
    def nsb_spectrum(self):
        """nsb_spectrum property."""
        return self._nsb_spectrum

    @nsb_spectrum.setter
    def nsb_spectrum(self, nsb_spectrum):
        """Setter for nsb_spectrum."""
        if nsb_spectrum is not None:
            self._nsb_spectrum = self._validate_or_fix_nsb_spectrum_file_format(nsb_spectrum)
        else:
            self._nsb_spectrum = (
                self._site_model.config_file_directory
                / Path(self._site_model.get_parameter_value("nsb_reference_spectrum")).name
            )

    def _make_run_command(self, run_number=None, input_file=None):  # pylint: disable=unused-argument
        """Prepare the command used to run testeff."""
        self._logger.debug("Preparing the command to run testeff")

        pixel_shape = self._telescope_model.camera.get_pixel_shape()
        pixel_shape_cmd = "-hpix" if pixel_shape in [1, 3] else "-spix"
        pixel_diameter = self._telescope_model.camera.get_pixel_diameter()

        focal_length = self._telescope_model.get_telescope_effective_focal_length("m", True)

        mirror_class = self._telescope_model.get_parameter_value("mirror_class")
        curvature_radius = self._get_curvature_radius(mirror_class)
        camera_transmission = self._telescope_model.get_parameter_value("camera_transmission")

        camera_filter_file = self._telescope_model.get_parameter_value("camera_filter")
        # testeff does not support 2D distributions
        if self._telescope_model.is_file_2d("camera_filter"):
            camera_filter_file = self._get_one_dim_distribution(
                "camera_filter", "camera_filter_incidence_angle"
            )

        mirror_reflectivity = self._telescope_model.get_parameter_value("mirror_reflectivity")
        if mirror_class == 2:
            mirror_reflectivity_secondary = mirror_reflectivity
        # testeff does not support 2D distributions
        if self._telescope_model.is_file_2d("mirror_reflectivity"):
            mirror_reflectivity = self._get_one_dim_distribution(
                "mirror_reflectivity", "primary_mirror_incidence_angle"
            )
            mirror_reflectivity_secondary = self._get_one_dim_distribution(
                "mirror_reflectivity", "secondary_mirror_incidence_angle"
            )

        command = str(self._simtel_path.joinpath("sim_telarray/testeff"))
        if self.skip_correction_to_nsb_spectrum:
            command += " -nc"  # Do not apply correction to original altitude where B&E was derived
        command += " -I"  # Clear the fall-back configuration directories
        command += f" -I{self._telescope_model.config_file_directory}"
        if self.nsb_spectrum is not None:
            command += f" -fnsb {self.nsb_spectrum}"
        command += " -nm -nsb-extra"
        command += f" -alt {self._site_model.get_parameter_value('corsika_observation_level')}"
        command += f" -fatm {self._site_model.get_parameter_value('atmospheric_transmission')}"
        command += f" -flen {focal_length}"
        command += f" -fcur {curvature_radius:.3f}"
        command += f" {pixel_shape_cmd} {pixel_diameter}"
        if mirror_class == 0:
            command += f" -fmir {self._telescope_model.get_parameter_value('mirror_list')}"
        if mirror_class == 2:
            command += f" -fmir {self._telescope_model.get_parameter_value('fake_mirror_list')}"
        command += f" -fref {mirror_reflectivity}"
        if mirror_class == 2:
            command += " -m2"
            command += f" -fref2 {mirror_reflectivity_secondary}"
        command += " -teltrans "
        command += f"{self._telescope_model.get_parameter_value('telescope_transmission')[0]}"
        command += f" -camtrans {camera_transmission}"
        command += f" -fflt {camera_filter_file}"
        command += (
            f" -fang {self._telescope_model.camera.get_lightguide_efficiency_angle_file_name()}"
        )
        command += (
            f" -fwl {self._telescope_model.camera.get_lightguide_efficiency_wavelength_file_name()}"
        )
        command += f" -fqe {self._telescope_model.get_parameter_value('quantum_efficiency')}"
        command += " 200 1000"  # lmin and lmax
        command += " 300"  # Xmax
        command += f" {self._site_model.get_parameter_value('atmospheric_profile')}"
        command += f" {self.zenith_angle}"

        # Remove the default sim_telarray configuration directories
        command = general.clear_default_sim_telarray_cfg_directories(command)

        return (
            f"cd {self._simtel_path.joinpath('sim_telarray')} && {command}",
            self._file_simtel,
            self._file_log,
        )

    def _check_run_result(self, run_number=None):  # pylint: disable=unused-argument
        """Check run results.

        Raises
        ------
        RuntimeError
            if camera efficiency simulation results file does not exist.
        """
        # Checking run
        if not self._file_simtel.exists():
            msg = f"Camera efficiency simulation results file does not exist ({self._file_simtel})."
            self._logger.error(msg)
            raise RuntimeError(msg)

        self._logger.debug("Everything looks fine with output file.")

    def _get_one_dim_distribution(self, two_dim_parameter, weighting_distribution_parameter):
        """
        Calculate an average one-dimensional curve for testeff from the two-dimensional curve.

        The two-dimensional distribution is provided in two_dim_parameter. The distribution
        of weights to use for averaging the two-dimensional distribution is given in
        weighting_distribution_parameter.

        Parameters
        ----------
        two_dim_parameter: str
            The name of the two-dimensional distribution parameter.
        weighting_distribution_parameter: str
            The name of the parameter with the distribution of weights.

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
            f"weighted_average_1D_{weighting_distribution_parameter}"
            f"_{self._telescope_model.get_parameter_value(two_dim_parameter)}"
        )
        return self._telescope_model.export_table_to_model_directory(
            new_file_name, distribution_to_export
        )

    def _validate_or_fix_nsb_spectrum_file_format(self, nsb_spectrum_file):
        """
        Validate or fix the nsb spectrum file format.

        The nsb spectrum file format required by sim_telarray has three columns:
        wavelength (nm), ignored, NSB flux [1e9 * ph/m2/s/sr/nm],
        where the second column is ignored by sim_telarray and the third is used for the NSB flux.
        This function makes sure the file has at least three columns,
        by copying the second column to the third.

        Parameters
        ----------
        nsb_spectrum_file: str or Path
            The path to the nsb spectrum file.

        Returns
        -------
        validated_nsb_spectrum_file: Path
            The path to the validated nsb spectrum file.
        """
        validated_nsb_spectrum_file = (
            self._telescope_model.config_file_directory / Path(nsb_spectrum_file).name
        )

        lines = ascii_handler.read_file_encoded_in_utf_or_latin(nsb_spectrum_file)

        with open(validated_nsb_spectrum_file, "w", encoding="utf-8") as file:
            for line in lines:
                if line.startswith("#"):
                    file.write(line)
                    continue
                split_line = line.split()
                if len(split_line) == 2:
                    split_line.append(split_line[1])
                    file.write(f"{split_line[0]} {split_line[1]} {split_line[2]}\n")
                else:
                    file.write(line)
        return validated_nsb_spectrum_file

    def _get_curvature_radius(self, mirror_class=1):
        """Get radius of curvature of dish."""
        if mirror_class == 2:
            return (
                self._telescope_model.get_parameter_value_with_unit("primary_mirror_diameter")
                .to("m")
                .value
            )

        if self._telescope_model.get_parameter_value("parabolic_dish"):
            return (
                2.0
                * self._telescope_model.get_parameter_value_with_unit("dish_shape_length")
                .to("m")
                .value
            )

        return (
            self._telescope_model.get_parameter_value_with_unit("dish_shape_length").to("m").value
        )
