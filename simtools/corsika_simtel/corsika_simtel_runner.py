from simtools.corsika.corsika_runner import CorsikaRunner

__all__ = ["CorsikaSimtelRunner"]


class MissingRequiredEntryInCorsikaConfig(Exception):
    """Exception for missing required entry in corsika config."""


class CorsikaSimtelRunner(CorsikaRunner):
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
    mongo_db_config: dict
        MongoDB configuration.
    site: str
        South or North.
    layout_name: str
        Name of the layout.
    label: str
        Instance label.
    keep_seeds: bool
        Use seeds based on run number and primary particle.  If False, use sim_telarray seeds.
    simtel_source_path: str or Path
        Location of source of the sim_telarray/CORSIKA package.
    corsika_config_data: dict
        Dict with CORSIKA config data.
    corsika_config_file: str or Path
        Path to yaml file containing CORSIKA config data.
    corsika_parameters_file: str or Path
        Path to yaml file containing CORSIKA parameters.
    """

    def __init__(self, *args, **kwargs):
        super(CorsikaSimtelRunner, self).__init__(use_multipipe=True, *args, **kwargs)

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
        return super().get_run_script(use_pfp=False, **kwargs)

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
        extra_commands = ""
        if kwargs["extra_commands"] is not None and len(kwargs["extra_commands"]) > 0:
            extra_commands = kwargs["extra_commands"]
        multipipe_file = (
            "/workdir/external/gammasim-tools/simtools-output/TEST/corsika_simtel/"
            f"multi_cta-{self.site}-{self.layout_name}.cfg"
        )
        with open(multipipe_file, "w") as file:
            file.write(
                'env offset="0.0" cfg="cta-prod5-lapalma" '
                f'extra_defs="{extra_commands} -DNSB_AUTOSCALE -DNECTARCAM" '
                'extra_suffix="-2100m-LaPalma-dark" extension="zst" ./generic_run.sh'
            )
