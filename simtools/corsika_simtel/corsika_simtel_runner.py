import stat
from pathlib import Path

from simtools.corsika.corsika_runner import CorsikaRunner
from simtools.simtel.simtel_runner_array import SimtelRunnerArray

__all__ = ["CorsikaSimtelRunner"]


class CorsikaSimtelRunner(CorsikaRunner, SimtelRunnerArray):
    """
    CorsikaSimtelRunner is responsible for running CORSIKA and piping it to sim_telarray
    using the multipipe functionality. CORSIKA is set up using corsika_autoinputs program
    provided by the sim_telarray package. It creates the multipipe script and sim_telarray command
    corresponding to the requested configuration.

    It uses CorsikaConfig to manage the CORSIKA configuration and SimtelRunnerArray
    for the sim_telarray configuration. User parameters must be given by the
    common_args, corsika_args and simtel_args arguments.
    The corsika_args and simtel_args are explained in
    CorsikaRunner and SimtelRunnerArray respectively.
    An example of the common_args is given below.

    .. code-block:: python

        common_args = {
            'label': 'test-production',
            'simtel_source_path': '/workdir/sim_telarray/',
        }

    Parameters
    ----------
    common_args: dict
        Arguments common to both CORSIKA and sim_telarray runners
    corsika_args: dict
        Arguments for the CORSIKA runner (see full list in CorsikaRunner documentation).
    simtel_args: dict
        Arguments for the sim_telarray runner (see full list in SimtelRunnerArray documentation).
    """

    def __init__(self, common_args=None, corsika_args=None, simtel_args=None):

        CorsikaRunner.__init__(self, use_multipipe=True, **(common_args | corsika_args))
        SimtelRunnerArray.__init__(self, **(common_args | simtel_args))

    def prepare_run_script(self, use_pfp=False, **kwargs):
        """
        Get the full path of the run script file for a given run number.

        Parameters
        ----------
        use_pfp: bool
            Whether to use the preprocessor in preparing the CORSIKA input file
        kwargs: dict
            The following optional parameters can be provided:
                run_number: int
                    Run number.

        Returns
        -------
        Path:
            Full path of the run script file.
        """
        self.export_multipipe_script(**kwargs)
        return CorsikaRunner.prepare_run_script(self, use_pfp=use_pfp, **kwargs)

    def export_multipipe_script(self, **kwargs):
        """
        Write the multipipe script used in piping CORSIKA to sim_telarray.

        Parameters
        ----------
        kwargs: dict
            The following optional parameters can be provided:
                run_number: int
                    Run number.

        Returns
        -------
        Path:
            Full path of the run script file.
        """

        kwargs = {
            "run_number": None,
            **kwargs,
        }
        run_number = self._validate_run_number(kwargs["run_number"])

        run_command = self._make_run_command(
            run_number=run_number,
            input_file="-",  # Tell sim_telarray to take the input from standard output
        )
        multipipe_file = Path(self.corsika_config._config_file_path.parent).joinpath(
            self.corsika_config.get_file_name("multipipe")
        )
        with open(multipipe_file, "w") as file:
            file.write(f"{run_command}")
        self._export_multipipe_executable(multipipe_file)

    def _export_multipipe_executable(self, multipipe_file):
        """
        Write the multipipe executable used to call the multipipe_corsika command.

        Parameters
        ----------
        multipipe_file: str or Path
            The name of the multipipe file which contains all of the multipipe commands.
        """

        multipipe_executable = Path(self.corsika_config._config_file_path.parent).joinpath(
            "run_cta_multipipe"
        )
        with open(multipipe_executable, "w") as file:
            multipipe_command = Path(self._simtel_source_path).joinpath(
                "sim_telarray/bin/multipipe_corsika "
                f"-c {multipipe_file}"
                " || echo 'Fan-out failed'"
            )
            file.write(f"{multipipe_command}")

        multipipe_executable.chmod(multipipe_executable.stat().st_mode | stat.S_IEXEC)

    def _make_run_command(self, **kwargs):
        """
        Builds and returns the command to run simtel_array.

        Parameters
        ----------
        kwargs: dict
            The dictionary must include the following parameters (unless listed as optional):
                input_file: str
                    Full path of the input CORSIKA file.
                    Use '-' to tell sim_telarray to read from standard output
                run_number: int
                    run number

        """

        # TODO: These definitions of the files can probably be separated from
        # the the run command and put back into the parent class.
        info_for_file_name = SimtelRunnerArray.get_info_for_file_name(self, kwargs["run_number"])
        self._log_file = SimtelRunnerArray.get_file_name(
            self, file_type="log", **info_for_file_name
        )
        histogram_file = SimtelRunnerArray.get_file_name(
            self, file_type="histogram", **info_for_file_name
        )
        output_file = SimtelRunnerArray.get_file_name(
            self, file_type="output", **info_for_file_name
        )

        # TODO: Implement the weak pointing for divergent pointing
        # TODO: Think how to create multiple run commands for various pipes (e.g., NSB levels)
        # Array
        command = str(self._simtel_source_path.joinpath("sim_telarray/bin/sim_telarray"))
        command += f" -c {self.array_model.get_config_file()}"
        command += f" -I{self.array_model.get_config_directory()}"
        command += super()._config_option("telescope_theta", self.config.zenith_angle)
        command += super()._config_option("telescope_phi", self.config.azimuth_angle)
        command += super()._config_option("power_law", "2.5")
        command += super()._config_option("histogram_file", histogram_file)
        command += super()._config_option("output_file", output_file)
        command += super()._config_option("random_state", "auto")
        command += super()._config_option("show", "all")
        command += f" {kwargs['input_file']}"
        command += f" 2>&1 | gzip > {self._log_file} || exit"

        return command

    def get_file_name(self, file_type, run_number=None, **kwargs):
        """
        Get a CORSIKA or sim_telarray style file name for various file types.
        See the implementations in CorsikaRunner and SimtelRunnerArray for details.
        """

        if file_type in ["output", "log", "histogram"]:
            return SimtelRunnerArray.get_file_name(self, file_type=file_type, **kwargs)
        else:
            return CorsikaRunner.get_file_name(
                self, file_type=file_type, run_number=run_number, **kwargs
            )

    def get_info_for_file_name(self, run_number):
        """
        Get a dictionary with the info necessary for building
        a CORSIKA or sim_telarray runner file names.

        Returns
        -------
        dict
            Dictionary with the keys necessary for building
            a CORSIKA or sim_telarray runner file names.
        """
        run_number = self._validate_run_number(run_number)
        return {
            "run": run_number,
            "primary": self.corsika_config.primary,
            "array_name": self.layout_name,
            "site": self.site,
            "label": self.label,
            "zenith": self.config.zenith_angle,
            "azimuth": self.config.azimuth_angle,
        }
