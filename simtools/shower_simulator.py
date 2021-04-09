
__all__ = ['ShowerSimulator']


class ShowerSimulator:
    '''
    Class for handling ray tracing simulations and analysis.

    Attributes
    ----------
    label: str
        Instance label.

    Methods
    -------
    simulate(test=False, force=False)
        Simulate RayTracing using SimtelRunner.
    analyse(export=True, force=False, useRX=False, noTelTransmission=False)
        Analyze RayTracing, meaning read simtel files, compute psfs and eff areas and store the
        results in _results.
    exportResults()
        Export results to a csv file.
    plot(key, **kwargs)
        Plot key vs off-axis angle.
    plotHistogram(key, **kwargs)
        Plot histogram of key (d80_cm, d80_deg, eff_area, eff_flen).
    getMean(key)
        Get mean value of key(d80_cm, d80_deg, eff_area, eff_flen).
    getStdDev(key)
        Get std dev of key(d80_cm, d80_deg, eff_area, eff_flen).
    images()
        Get list of PSFImages.
    '''

    def __init__(
        self,
        label=None,
        simtelSourcePath=None,
        filesLocation=None
    ):
        '''
        RayTracing init.

        Parameters
        ----------
        telescopeModel: TelescopeModel
            Instance of the TelescopeModel class.
        label: str
            Instance label.
        simtelSourcePath: str (or Path), optional
            Location of sim_telarray installation. If not given, it will be taken from the
            config.yml file.
        filesLocation: str (or Path), optional
            Parent location of the output files created by this class. If not given, it will be
            taken from the config.yml file.
        singleMirrorMode: bool
        useRandomFocalLength: bool
        **kwargs:
            Physical parameters with units (if applicable). Options: zenithAngle, offAxisAngle,
            sourceDistance, mirrorNumbers
        '''
        self._logger = logging.getLogger(__name__)

        self._simtelSourcePath = Path(cfg.getConfigArg('simtelPath', simtelSourcePath))
        self._filesLocation = cfg.getConfigArg('outputLocation', filesLocation)

    # End of init

# End of ShowerSimulator
