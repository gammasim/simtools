import logging
from pathlib import Path

from simtools.simtel.simtel_runner import SimtelRunner

__all__ = ["SimtelRunnerCameraEfficiency"]


class SimtelRunnerCameraEfficiency(SimtelRunner):
    """
    SimtelRunnerCameraEfficiency is the interface with the testeff tool of sim_telarray to perform\
    camera efficiency simulations.

    Parameters
    ----------
    telescope_model: str
        Instance of TelescopeModel class.
    label: str
        Instance label. Important for output file naming.
    simtel_source_path: str or Path
        Location of sim_telarray installation.
    file_simtel: str or Path
        location of the sim_telarray testeff tool output file.
    zenith_angle: float
        The zenith angle given in the config to CameraEfficiency.
    """

    def __init__(
        self,
        telescope_model,
        label=None,
        simtel_source_path=None,
        file_simtel=None,
        file_log=None,
        zenith_angle=None,
        nsb_spectrum=None,
    ):
        """
        Initialize SimtelRunner.
        """
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init SimtelRunnerCameraEfficiency")

        super().__init__(label=label, simtel_source_path=simtel_source_path)

        self._telescope_model = telescope_model
        self.label = label if label is not None else self._telescope_model.label

        self._file_simtel = file_simtel
        self._file_log = file_log
        self.zenith_angle = zenith_angle
        self.nsb_spectrum = nsb_spectrum

    @property
    def nsb_spectrum(self):
        """nsb_spectrum property"""
        return self._nsb_spectrum

    @nsb_spectrum.setter
    def nsb_spectrum(self, nsb_spectrum):
        """Setter for nsb_spectrum"""
        if nsb_spectrum is not None:
            self._nsb_spectrum = self._validate_or_fix_nsb_spectrum_file_format(nsb_spectrum)
        else:
            self._nsb_spectrum = None

    def _shall_run(self, **kwargs):  # pylint: disable=unused-argument; applies only to this line
        """Tells if simulations should be run again based on the existence of output files."""
        return not self._file_simtel.exists()

    def _make_run_command(self, **kwargs):  # pylint: disable=unused-argument
        """
        Prepare the command used to run testeff
        """

        self._logger.debug("Preparing the command to run testeff")

        # Processing camera pixel features
        pixel_shape = self._telescope_model.camera.get_pixel_shape()
        pixel_shape_cmd = "-hpix" if pixel_shape in [1, 3] else "-spix"
        pixel_diameter = self._telescope_model.camera.get_pixel_diameter()

        # Processing focal length
        focal_length = self._telescope_model.get_parameter_value_with_unit("effective_focal_length")
        if focal_length == 0.0:
            self._logger.warning("Using focal_length because effective_focal_length is 0")
            focal_length = self._telescope_model.get_parameter_value_with_unit("focal_length")

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
        if self._telescope_model.is_file_2d("camera_filter"):
            camera_filter_file = self._get_one_dim_distribution(
                "camera_filter", "camera_filter_incidence_angle"
            )

        # Processing mirror reflectivity
        # A special case is testeff does not support 2D distributions
        mirror_reflectivity = self._telescope_model.get_parameter_value("mirror_reflectivity")
        if mirror_class == 2:
            mirror_reflectivity_secondary = mirror_reflectivity
        if self._telescope_model.is_file_2d("mirror_reflectivity"):
            mirror_reflectivity = self._get_one_dim_distribution(
                "mirror_reflectivity", "primary_mirror_incidence_angle"
            )
            mirror_reflectivity_secondary = self._get_one_dim_distribution(
                "mirror_reflectivity", "secondary_mirror_incidence_angle"
            )

        command = str(self._simtel_source_path.joinpath("sim_telarray/bin/testeff"))
        if self.nsb_spectrum is not None:
            command += f" -fnsb {self.nsb_spectrum}"
        command += " -nm -nsb-extra"
        command += f" -alt {self._telescope_model.get_parameter_value('altitude')}"
        command += f" -fatm {self._telescope_model.get_parameter_value('atmospheric_transmission')}"
        command += f" -flen {focal_length.to('m').value}"
        command += f" {pixel_shape_cmd} {pixel_diameter}"
        if mirror_class == 1:
            command += f" -fmir {self._telescope_model.get_parameter_value('mirror_list')}"
        command += f" -fref {mirror_reflectivity}"
        if mirror_class == 2:
            command += " -m2"
            command += f" -fref2 {mirror_reflectivity_secondary}"
        command += f" -teltrans {self._telescope_model.get_telescope_transmission_parameters()[0]}"
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
        command += " 300 26"  # Xmax, ioatm (Konrad always uses 26)
        command += f" {self.zenith_angle}"
        command += f" 2>{self._file_log}"
        command += f" >{self._file_simtel}"

        # Moving to sim_telarray directory before running
        command = f"cd {self._simtel_source_path.joinpath('sim_telarray')} && {command}"

        return command

    def _check_run_result(self, **kwargs):  # pylint: disable=unused-argument
        """Checking run results

        Raises
        ------
        RuntimeError
            if camera efficiency simulation results file does not exist.
        """
        # Checking run
        if not self._file_simtel.exists():
            msg = "Camera efficiency simulation results file does not exist"
            self._logger.error(msg)
            raise RuntimeError(msg)

        self._logger.debug("Everything looks fine with output file.")

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

    def _validate_or_fix_nsb_spectrum_file_format(self, nsb_spectrum_file):
        """
        Validate or fix the nsb spectrum file format.
        The nsb spectrum file format required by sim_telarray has three columns:
            wavelength (nm), ignored, NSB flux [1e9 * ph/m2/s/sr/nm],
        where the second column is ignored by sim_telarray and the third is used for the NSB flux.
        This function makes sure the file has at least three columns,
        by copying the second column to the third.
        """

        validated_nsb_spectrum_file = (
            self._telescope_model.get_config_directory() / Path(nsb_spectrum_file).name
        )
        with open(nsb_spectrum_file, "r", encoding="utf-8") as file:
            lines = file.readlines()
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
