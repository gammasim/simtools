from simtools.corsika.corsika_runner import CorsikaRunner
from simtools.simtel.simtel_runner_array import SimtelRunnerArray

__all__ = ["CorsikaSimtelRunner"]


class MissingRequiredEntryInCorsikaConfig(Exception):
    """Exception for missing required entry in corsika config."""


class CorsikaSimtelRunner(CorsikaRunner, SimtelRunnerArray):
    """
    CorsikaSimtelRunner is responsible for running CORSIKA and piping it to sim_telarray
    using the multipipe functionality. CORSIKA is set up using corsika_autoinputs program
    provided by the sim_telarray package. It provides shell scripts to be run externally or by
    the module simulator. Same instance can be used to generate scripts for any given run number.

    It uses CorsikaConfig to manage the CORSIKA configuration. User parameters must be given by the
    corsika_config_data or corsika_config_file arguments. An example of corsika_config_data follows
    below.

    .. code-block:: python

        corsika_config_data = {
            'data_directory': .
            'primary': 'proton',
            'nshow': 10000,
            'nrun': 1,
            'zenith': 20 * u.deg,
            'viewcone': 5 * u.deg,
            'erange': [10 * u.GeV, 100 * u.TeV],
            'eslope': -2,
            'phi': 0 * u.deg,
            'cscat': [10, 1500 * u.m, 0]
        }

    The remaining CORSIKA parameters can be set as a yaml file, using the argument
    corsika_parameters_file. When not given, corsika_parameters will be loaded from
    data/parameters/corsika_parameters.yml.

    The CORSIKA output directory must be set by the data_directory entry. The following directories
    will be created to store the logs and input file:
    {data_directory}/corsika/$site/$primary/logs
    {data_directory}/corsika/$site/$primary/scripts

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

    def get_run_script(self, **kwargs):
        """
        Get the full path of the run script file for a given run number.

        Parameters
        ----------
        run_number: int
            Run number.
        extra_commands: str
            Additional commands for running simulations.

        Returns
        -------
        Path:
            Full path of the run script file.
        """
        self.export_multipipe_script(**kwargs)
        # SimtelRunnerArray.array_model.export_all_simtel_config_files(self)
        return CorsikaRunner.get_run_script(self, use_pfp=False, **kwargs)

    def export_multipipe_script(self, **kwargs):
        """
        Write the multipipe script used in piping CORSIKA to sim_telarray.

        Parameters
        ----------
        run_number: int
            Run number.
        extra_commands: str
            Additional commands for running simulations.

        Returns
        -------
        Path:
            Full path of the run script file.
        """

        run_command = self._make_run_command(
            run_number=kwargs["run_number"],
            input_file="-",  # Tell sim_telarray to take the input from standard output
        )
        multipipe_file = (
            "/workdir/external/gammasim-tools/simtools-output/TEST/corsika_simtel/"
            f"multi_cta-{self.site}-{self.layout_name}.cfg"
        )
        with open(multipipe_file, "w") as file:
            file.write(f"{run_command}")

    def _make_run_command(self, **kwargs):
        """
        Builds and returns the command to run simtel_array.

        Parameters
        ----------
        kwargs: dict
            The dictionary must include the following parameters (unless listed as optional):
                input_file: str
                    Full path of the input CORSIKA file
                run_number: int (optional)
                    run number

        """

        # TODO: These definitions of the files can probably be separated from
        # the the run command and put back into the parent class.
        run_number = kwargs["run_number"] if "run_number" in kwargs else 1
        info_for_file_name = SimtelRunnerArray.get_info_for_file_name(self, run_number)
        self._log_file = SimtelRunnerArray.get_file_name(
            self, file_type="log", **info_for_file_name
        )
        histogram_file = SimtelRunnerArray.get_file_name(
            self, file_type="histogram", **info_for_file_name
        )
        output_file = SimtelRunnerArray.get_file_name(
            self, file_type="output", **info_for_file_name
        )

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
        command += " " + str(kwargs["input_file"])
        command += " > " + str(self._log_file) + " 2>&1"

        return command
