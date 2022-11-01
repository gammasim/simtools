import logging
import shlex
import subprocess
from copy import copy
from math import pi, tan
from pathlib import Path

import astropy.io.ascii
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import QTable

import simtools.util.general as gen
from simtools import io_handler, visualize
from simtools.model.telescope_model import TelescopeModel
from simtools.psf_analysis import PSFImage
from simtools.simtel.simtel_runner_ray_tracing import SimtelRunnerRayTracing
from simtools.util import names
from simtools.util.model import compute_telescope_transmission

__all__ = ["RayTracing"]


class RayTracing:
    """
    Class for handling ray tracing simulations and analysis.

    Configurable parameters:
        zenithAngle:
            len: 1
            unit: deg
            default: 20 deg
        offAxisAngle:
            len: null
            unit: deg
            default: [0 deg]
        sourceDistance:
            len: 1
            unit: km
            default: 10 km
        singleMirrorMode:
            len: 1
            default: False
        useRandomFocalLength:
            len: 1
            default: False
        mirrorNumbers:
            len: null
            default: 'all'

    Attributes
    ----------
    label: str
        Instance label.
    config: namedtuple
        Contains the configurable parameters (zenithAngle).

    Methods
    -------
    simulate(test=False, force=False)
        Simulate RayTracing using SimtelRunnerRayTracing.
    analyse(export=True, force=False, useRX=False, noTelTransmission=False)
        Analyze RayTracing, meaning read simtel files, compute psfs and eff areas and store the
        results in _results.
    export_results()
        Export results to a csv file.
    plot(key, **kwargs)
        Plot key vs off-axis angle.
    plot_histogram(key, **kwargs)
        Plot histogram of key (d80_cm, d80_deg, eff_area, eff_flen).
    get_mean(key)
        Get mean value of key(d80_cm, d80_deg, eff_area, eff_flen).
    get_std_dev(key)
        Get std dev of key(d80_cm, d80_deg, eff_area, eff_flen).
    images()
        Get list of PSFImages.
    """

    YLABEL = {
        "d80_cm": r"$D_{80}$",
        "d80_deg": r"$D_{80}$",
        "eff_area": "Eff. mirror area",
        "eff_flen": "Eff. focal length",
    }

    def __init__(
        self,
        telescopeModel,
        simtelSourcePath,
        label=None,
        configData=None,
        configFile=None,
    ):
        """
        RayTracing init.

        Parameters
        ----------
        telescopeModel: TelescopeModel
            Instance of the TelescopeModel class.
        label: str
            Instance label.
        simtelSourcePath: str (or Path)
            Location of sim_telarray installation.
        configData: dict.
            Dict containing the configurable parameters.
        configFile: str or Path
            Path of the yaml file containing the configurable parameters.
        """
        self._logger = logging.getLogger(__name__)

        self._simtelSourcePath = Path(simtelSourcePath)
        self._io_handler = io_handler.IOHandler()

        self._telescopeModel = self._validate_telescope_model(telescopeModel)

        _configDataIn = gen.collect_data_from_yaml_or_dict(configFile, configData)
        _parameterFile = self._io_handler.get_input_data_file(
            parentDir="parameters", fileName="ray-tracing_parameters.yml"
        )
        _parameters = gen.collect_data_from_yaml_or_dict(_parameterFile, None)
        self.config = gen.validate_config_data(_configDataIn, _parameters)

        # Due to float representation, round the off-axis angles so the values in results table
        # are the same as provided.
        self.config = self.config._replace(offAxisAngle=np.around(self.config.offAxisAngle, 5))

        self.label = label if label is not None else self._telescopeModel.label

        self._outputDirectory = self._io_handler.get_output_directory(self.label, "ray-tracing")

        # Loading relevant attributes in case of single mirror mode.
        if self.config.singleMirrorMode:
            # Recalculating source distance.
            self._logger.debug(
                "Single mirror mode is activate - source distance is being recalculated to 2 * flen"
            )
            mirFlen = self._telescopeModel.get_parameter_value("mirror_focal_length")
            self._sourceDistance = 2 * float(mirFlen) * u.cm.to(u.km)  # km

            # Setting mirror numbers.
            if self.config.mirrorNumbers[0] == "all":
                self._mirrorNumbers = list(range(0, self._telescopeModel.mirrors.numberOfMirrors))
            else:
                self._mirrorNumbers = self.config.mirrorNumbers
        else:
            self._sourceDistance = self.config.sourceDistance

        self._hasResults = False

        # Results file
        fileNameResults = names.ray_tracing_results_file_name(
            self._telescopeModel.site,
            self._telescopeModel.name,
            self._sourceDistance,
            self.config.zenithAngle,
            self.label,
        )
        self._outputDirectory.joinpath("results").mkdir(parents=True, exist_ok=True)
        self._fileResults = self._outputDirectory.joinpath("results").joinpath(fileNameResults)

    @classmethod
    def from_kwargs(cls, **kwargs):
        """
        Builds a RayTracing object from kwargs only.
        The configurable parameters can be given as kwargs, instead of using the
        configData or configFile arguments.

        Parameters
        ----------
        kwargs
            Containing the arguments and the configurable parameters.

        Returns
        -------
        Instance of this class.
        """
        args, configData = gen.separate_args_and_config_data(
            expectedArgs=[
                "telescopeModel",
                "label",
                "simtelSourcePath",
            ],
            **kwargs
        )
        return cls(**args, configData=configData)

    def __repr__(self):
        return "RayTracing(label={})\n".format(self.label)

    def _validate_telescope_model(self, tel):
        """Validate TelescopeModel"""
        if isinstance(tel, TelescopeModel):
            self._logger.debug("RayTracing contains a valid TelescopeModel")
            return tel
        else:
            msg = "Invalid TelescopeModel"
            self._logger.error(msg)
            raise ValueError(msg)

    def simulate(self, test=False, force=False):
        """
        Simulate RayTracing using SimtelRunnerRayTracing.

        Parameters
        ----------
        test: bool
            Test flag will make it faster by simulating much fewer photons.
        force: bool
            Force flag will remove existing files and simulate again.
        """
        allMirrors = self._mirrorNumbers if self.config.singleMirrorMode else [0]
        for thisOffAxis in self.config.offAxisAngle:
            for thisMirror in allMirrors:
                self._logger.info(
                    "Simulating RayTracing for offAxis={}, mirror={}".format(
                        thisOffAxis, thisMirror
                    )
                )
                simtel = SimtelRunnerRayTracing(
                    simtelSourcePath=self._simtelSourcePath,
                    telescopeModel=self._telescopeModel,
                    configData={
                        "zenithAngle": self.config.zenithAngle * u.deg,
                        "sourceDistance": self._sourceDistance * u.km,
                        "offAxisAngle": thisOffAxis * u.deg,
                        "mirrorNumber": thisMirror,
                        "useRandomFocalLength": self.config.useRandomFocalLength,
                    },
                    singleMirrorMode=self.config.singleMirrorMode,
                    forceSimulate=force,
                )
                simtel.run(test=test, force=force)

    def analyze(
        self,
        export=True,
        force=False,
        useRX=False,
        noTelTransmission=False,
        containment_fraction=0.8,
    ):
        """
        Analyze RayTracing, meaning read simtel files, compute psfs and eff areas and store the
        results in _results.

        Parameters
        ----------
        export: bool
            If True, results will be exported to a file automatically. Alternatively,
            export_results function can be used.
        force: bool
            If True, existing results files will be removed and analysis will be done again.
        useRX: bool
            If True, calculations are done using the rx binary provided by sim_telarray. If False,
            calculations are done internally, by the module psf_analysis.
        noTelTransmission: bool
            If True, the telescope transmission is not applied.
        containment_fraction: float
            Containment fraction for PSF containment calculation. Allowed values are in the
            inverval [0,1]
        """

        doAnalyze = not self._fileResults.exists() or force

        focalLength = float(self._telescopeModel.get_parameter_value("focal_length"))
        telTransmissionPars = (
            self._telescopeModel.get_telescope_transmission_parameters()
            if not noTelTransmission
            else [1, 0, 0, 0]
        )

        cmToDeg = 180.0 / pi / focalLength

        self._psfImages = dict()
        if doAnalyze:
            _rows = list()
        else:
            self._read_results()

        allMirrors = self._mirrorNumbers if self.config.singleMirrorMode else [0]
        for thisOffAxis in self.config.offAxisAngle:
            for thisMirror in allMirrors:
                self._logger.debug("Analyzing RayTracing for offAxis={}".format(thisOffAxis))
                if self.config.singleMirrorMode:
                    self._logger.debug("mirrorNumber={}".format(thisMirror))

                photonsFileName = names.ray_tracing_file_name(
                    self._telescopeModel.site,
                    self._telescopeModel.name,
                    self._sourceDistance,
                    self.config.zenithAngle,
                    thisOffAxis,
                    thisMirror if self.config.singleMirrorMode else None,
                    self.label,
                    "photons",
                )

                photonsFile = self._outputDirectory.joinpath(photonsFileName)
                telTransmission = compute_telescope_transmission(telTransmissionPars, thisOffAxis)
                image = PSFImage(focalLength, None)
                image.read_photon_list_from_simtel_file(photonsFile)
                self._psfImages[thisOffAxis] = copy(image)

                if not doAnalyze:
                    continue

                if useRX:
                    containment_diameter_cm, centroidX, centroidY, effArea = self._process_rx(
                        photonsFile
                    )
                    containment_diameter_deg = containment_diameter_cm * cmToDeg
                    image.set_psf(containment_diameter_cm, fraction=containment_fraction, unit="cm")
                    image.centroidX = centroidX
                    image.centroidY = centroidY
                    image.set_effective_area(effArea * telTransmission)
                else:
                    containment_diameter_cm = image.get_psf(containment_fraction, "cm")
                    containment_diameter_deg = image.get_psf(containment_fraction, "deg")
                    centroidX = image.centroidX
                    centroidY = image.centroidY
                    effArea = image.get_effective_area() * telTransmission

                effFlen = np.nan if thisOffAxis == 0 else centroidX / tan(thisOffAxis * pi / 180.0)
                _currentResults = (
                    thisOffAxis * u.deg,
                    containment_diameter_cm * u.cm,
                    containment_diameter_deg * u.deg,
                    effArea * u.m * u.m,
                    effFlen * u.cm,
                )
                if self.config.singleMirrorMode:
                    _currentResults += (thisMirror,)
                _rows.append(_currentResults)

        if doAnalyze:
            _columns = ["Off-axis angle"]
            _columns.extend(list(self.YLABEL.keys()))
            if self.config.singleMirrorMode:
                _columns.append("mirror_number")
            self._results = QTable(rows=_rows, names=_columns)

        self._hasResults = True
        if export:
            self.export_results()

    def _process_rx(self, file, containment_fraction=0.8):
        """
        Process sim_telarray photon list with rx binary and return the results
        (containment_diameter_cm, centroids and eff area).

        Parameters
        ----------
        file: str or Path
            Photon list file.
        containment_fraction: float
            Containment fraction for PSF containment calculation. Allowed values are in the
            inverval [0,1]

        Returns
        -------
        (containment_diameter_cm, xMean, yMean, effArea)

        """

        try:
            with open(file) as _stdin:
                rxOutput = subprocess.Popen(
                    shlex.split(
                        "{}/sim_telarray/bin/rx -f {:.2f} -v".format(
                            self._simtelSourcePath, containment_fraction
                        )
                    ),
                    stdin=_stdin,
                    stdout=subprocess.PIPE,
                ).communicate()[0]
        except FileNotFoundError:
            self._logger.error("Photon list file not found: {}".format(file))
            raise
        try:
            rxOutput = rxOutput.splitlines()[-1:][0].split()
        except IndexError:
            self._logger.error("Invalid output from rx: {}".format(rxOutput))
            raise
        containment_diameter_cm = 2 * float(rxOutput[0])
        xMean = float(rxOutput[1])
        yMean = float(rxOutput[2])
        effArea = float(rxOutput[5])
        return containment_diameter_cm, xMean, yMean, effArea

    def export_results(self):
        """Export results to a csv file."""
        if not self._hasResults:
            self._logger.error("Cannot export results because it does not exist")
        else:
            self._logger.info("Exporting results to {}".format(self._fileResults))
            astropy.io.ascii.write(self._results, self._fileResults, format="ecsv", overwrite=True)

    def _read_results(self):
        """Read existing results file and store it in _results."""
        self._results = astropy.io.ascii.read(self._fileResults, format="ecsv")
        self._hasResults = True

    def plot(self, key, save=False, **kwargs):
        """
        Plot key vs off-axis angle and save the figure in pdf.

        Parameters
        ----------
        key: str
            d80_cm, d80_deg, eff_area or eff_flen
        save: bool
            If True, figure will be saved.
        **kwargs:
            kwargs for plt.plot

        Raises
        ------
        KeyError
            If key is not among the valid options.
        """
        if key not in self.YLABEL:
            msg = "Invalid key to plot"
            self._logger.error(msg)
            raise KeyError(msg)

        self._logger.info("Plotting {} vs off-axis angle".format(key))

        plt = visualize.plot_table(
            self._results["Off-axis angle", key], self.YLABEL[key], noLegend=True, **kwargs
        )

        if save:
            plotFileName = names.ray_tracing_plot_file_name(
                key,
                self._telescopeModel.site,
                self._telescopeModel.name,
                self._sourceDistance,
                self.config.zenithAngle,
                self.label,
            )
            self._outputDirectory.joinpath("figures").mkdir(exist_ok=True)
            plotFile = self._outputDirectory.joinpath("figures").joinpath(plotFileName)
            self._logger.info("Saving fig in {}".format(plotFile))
            plt.savefig(plotFile)

    def plot_histogram(self, key, **kwargs):
        """
        Plot histogram of key.

        Parameters
        ----------
        key: str
            d80_cm, d80_deg, eff_area or eff_flen
        **kwargs:
            kwargs for plt.hist

        Raises
        ------
        KeyError
            If key is not among the valid options.
        """
        if key not in self.YLABEL:
            msg = "Invalid key to plot"
            self._logger.error(msg)
            raise KeyError(msg)

        ax = plt.gca()
        ax.hist([r.value for r in self._results[key]], **kwargs)

    def get_mean(self, key):
        """
        Get mean value of key.

        Parameters
        ----------
        key: str
            d80_cm, d80_deg, eff_area or eff_flen

        Returns
        -------
        float
            Mean value of key.

        Raises
        ------
        KeyError
            If key is not among the valid options.
        """
        if key not in self.YLABEL:
            msg = "Invalid key to plot"
            self._logger.error(msg)
            raise KeyError(msg)
        return np.mean(self._results[key])

    def get_std_dev(self, key):
        """
        Get std dev of key.

        Parameters
        ----------
        key: str
            d80_cm, d80_deg, eff_area or eff_flen

        Returns
        -------
        float
            Std deviation of key.

        Raises
        ------
        KeyError
            If key is not among the valid options.
        """
        if key not in self.YLABEL:
            msg = "Invalid key to plot"
            self._logger.error(msg)
            raise KeyError(msg)
        return np.std(self._results[key])

    def images(self):
        """
        Get list of PSFImages.

        Returns
        -------
        List of PSFImage's
        """
        images = list()
        for thisOffAxis in self.config.offAxisAngle:
            if thisOffAxis in self._psfImages:
                images.append(self._psfImages[thisOffAxis])
        if len(images) == 0:
            self._logger.error("No image found")
            return None
        return images
