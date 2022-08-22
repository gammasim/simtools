import logging

__all__ = [
    'validateModelVersionName',
    'validateSimtelModeName',
    'validateSiteName',
    'validateLayoutArrayName',
    'validateTelescopeModelName',
    'validateCameraName',
    'convertTelescopeModelNameToYaml',
    'splitTelescopeModelName',
    'getSiteFromTelescopeName',
    'rayTracingFileName',
    'simtelTelescopeConfigFileName',
    'simtelArrayConfigFileName',
    'simtelSingleMirrorListFileName',
    'corsikaConfigFileName',
    'corsikaOutputFileName',
    'corsikaSubLogFileName'
]


def validateSubSystemName(name):
    """
    Validate a sub system name (optics structure or camera)

    Raises
    ------
    ValueError
        If name is not valid.

    Parameters
    ----------
    name: str

    Returns
    -------
    str
        Validated name.
    """
    return validateName(name, {**allCameraNames, **allStructureNames})


def validateCameraName(name):
    """
    Validate a camera name.

    Raises
    ------
    ValueError
        If name is not valid.

    Parameters
    ----------
    name: str

    Returns
    -------
    str
        Validated name.
    """
    return validateName(name, allCameraNames)

def validateTelescopeIDName(name):
    """
    Validate a telescope ID name

    Valid names e.g.,
    - D
    - telescope ID

    Raises
    ------
    ValueError
        If name is not valid.

    Parameters
    ----------
    name: str

    Returns
    -------
    str
        Validated name.
    """

    # FIXME: validate telescope id range
    if name == 'D' or name.isdigit():
        return name

    _logger = logging.getLogger(__name__)
    msg = "Invalid telescope ID name {}".format(name)
    _logger.error(msg)
    raise ValueError(msg)


def validateModelVersionName(name):
    """
    Validate a model version name.

    Raises
    ------
    ValueError
        If name is not valid.

    Parameters
    ----------
    name: str

    Returns
    -------
    str
        Validated name.
    """
    return validateName(name, allModelVersionNames)


def validateSimtelModeName(name):
    """
    Validate a sim_telarray mode name.

    Raises
    ------
    ValueError
        If name is not valid.

    Parameters
    ----------
    name: str

    Returns
    -------
    str
        Validated name.
    """
    return validateName(name, allSimtelModeNames)


def validateSiteName(name):
    """
    Validate a site name.

    Raises
    ------
    ValueError
        If name is not valid.

    Parameters
    ----------
    name: str

    Returns
    -------
    str
        Validated name.
    """
    return validateName(name, allSiteNames)


def validateLayoutArrayName(name):
    """
    Validate a layout array name.

    Raises
    ------
    ValueError
        If name is not valid.

    Parameters
    ----------
    name: str

    Returns
    -------
    str
        Validated name.
    """
    return validateName(name, allLayoutArrayNames)


def validateName(name, allNames):
    """
    Validate a name given the allNames options. For each key in allNames, a list of options is
    given. If name is in this list, the key name is returned.

    Raises
    ------
    ValueError
        If name is not valid.

    Parameters
    ----------
    name: str
    allNames: dict

    Returns
    -------
    str
        Validated name.
    """
    _logger = logging.getLogger(__name__)

    if not isValidName(name, allNames):
        msg = "Invalid name {}".format(name)
        _logger.error(msg)
        raise ValueError(msg)
    for mainName, listOfNames in allNames.items():
        if name.lower() in listOfNames + [mainName.lower()]:
            if name != mainName:
                _logger.debug("Correcting name {} -> {}".format(name, mainName))
            return mainName
    return None


def isValidName(name, allNames):
    """
    Parameters
    ----------
    name: str
    allNames: dict

    Returns
    -------
    bool
    """
    if not isinstance(name, str):
        return False
    for mainName in allNames.keys():
        if name.lower() in allNames[mainName] + [mainName.lower()]:
            return True
    return False


def validateTelescopeModelName(name):
    """
    Validate a telescope model name.

    Raises
    ------
    ValueError
        If name is not valid.

    Parameters
    ----------
    name: str

    Returns
    -------
    str
        Validated name.
    """
    telClass, telType = splitTelescopeModelName(name)
    telClass = validateName(telClass, allTelescopeClassNames)
    if "flashcam" in telType:
        telType = telType.replace("flashcam", "FlashCam")
    if "nectarcam" in telType:
        telType = telType.replace("nectarcam", "NectarCam")
    if "1m" in telType:
        telType = telType.replace("1m", "1M")
    if "gct" in telType:
        telType = telType.replace("gct", "GCT")
    if "astri" in telType:
        telType = telType.replace("astri", "ASTRI")
    if "-d" in "-" + telType:
        telType = telType.replace("d", "D")

    return telClass + "-" + telType


def splitTelescopeModelName(name):
    """
    Split a telescope name into class and type.

    Raises
    ------
    ValueError
        If name is not valid.

    Parameters
    ----------
    name: str
        Telescope name.

    Returns
    -------
    str, str
       class (LST, MST, SST ...) and type (any complement).
    """
    nameParts = name.split("-")
    telClass = nameParts[0]
    telType = "-".join(nameParts[1:])
    return telClass, telType


def getSiteFromTelescopeName(name):
    """
    Get site name (South or North) from the (validated) telescope name.

    Parameters
    ----------
    name: str
        Telescope name.

    Returns
    -------
    str
        Site name (South or North).
    """
    nameParts = name.split("-")
    thisSite = validateSiteName(nameParts[0])
    return thisSite


def convertTelescopeModelNameToYaml(name):
    """
    Get telescope name following the old convention (yaml files) from the current telescope name.

    Parameters
    ----------
    name: str
        Telescope model name.

    Returns
    -------
    str
        Telescope name (old convention).
    """
    telClass, telType = splitTelescopeModelName(name)
    newName = telClass + "-" + telType
    oldNames = {
        "SST-D": "SST",
        "SST-1M": "SST-1M",
        "SST-ASTRI": "SST-2M-ASTRI",
        "SST-GCT": "SST-2M-GCT-S",
        "MST-FlashCam-D": "MST-FlashCam",
        "MST-NectarCam-D": "MST-NectarCam",
        "SCT-D": "SCT",
        "LST-D234": "LST",
        "LST-1": "LST",
    }

    if newName not in oldNames.keys():
        raise ValueError(
            "Telescope name {} could not be converted to yml names".format(name)
        )
    else:
        return oldNames[newName]


allTelescopeClassNames = {
    "SST": ["sst"],
    "MST": ["mst"],
    "SCT": ["sct"],
    "LST": ["lst"],
}

allCameraNames = {
    "SST": ["sst"],
    "ASTRI": ["astri"],
    "GCT": ["gct", "gct-s"],
    "1M": ["1m"],
    "FlashCam": ["flashcam", "flash-cam"],
    "NectarCam": ["nectarcam", "nectar-cam"],
    "SCT": ["sct"],
    "LST": ["lst"],
}

allStructureNames = {
    "Structure": ["Structure", "structure"]
}

allSiteNames = {"South": ["paranal", "south"], "North": ["lapalma", "north"]}

allModelVersionNames = {
    "prod3_compatible": ["p3", "prod3", "prod3b"],
    "prod4": ["p4"],
    "post_prod3_updates": [""],
    "2018-11-07": [""],
    "2019-02-22": [""],
    "2019-05-13": [""],
    "2019-11-20": [""],
    "2019-12-30": [""],
    "2020-02-26": [""],
    "2020-06-28": ["prod5"],
    "prod4-prototype": [""],
    "default": [],
    "Current": [],
    "Latest": [],
}

allSimtelModeNames = {
    "RayTracing": ["raytracing", "ray-tracing"],
    "RayTracingSingleMirror": [
        "raytracing-singlemirror",
        "ray-tracing-singlemirror",
        "ray-tracing-single-mirror",
    ],
    "Trigger": ["trigger"],
}

allLayoutArrayNames = {
    "4LST": ["4-lst", "4lst"],
    "1LST": ["1-lst", "1lst"],
    "4MST": ["4-mst", "4mst"],
    "1MST": ["1-mst", "mst"],
    "4SST": ["4-sst", "4sst"],
    "1SST": ["1-sst", "sst"],
    "Prod5": ["prod5", "p5"],
}

def simtoolsInstrumentName(site, telescopeClassName, subSystemName, telescopeIDName):
    """
    Instrument name following gammasim-tools naming convention

    Parameters
    ----------
    site: str
        South or North.
    telescopeClassName: str
        LST, MST, ...
    subSystemName: str
        FlashCam, NectarCam
    telescopeIDName: str
        telescope ID (e.g., D, numerial value)


    Returns
    -------
    instrumentname str
        instrument name

    """

    return validateSiteName(site) \
        + "-" + \
        validateName(telescopeClassName, allTelescopeClassNames) \
        + "-" + \
        validateSubSystemName(subSystemName) \
        + "-" + \
        validateTelescopeIDName(telescopeIDName)


def simtelTelescopeConfigFileName(
    site, telescopeModelName, modelVersion, label, extraLabel
):
    """
    sim_telarray config file name for a telescope.

    Parameters
    ----------
    site: str
        South or North.
    telescopeModelName: str
        LST-1, MST-FlashCam, ...
    modelVersion: str
        Version of the model.
    label: str
        Instance label.
    extraLabel: str
        Extra label in case of multiple telescope config files.

    Returns
    -------
    str
        File name.
    """
    name = "CTA-{}-{}-{}".format(site, telescopeModelName, modelVersion)
    name += "_{}".format(label) if label is not None else ""
    name += "_{}".format(extraLabel) if extraLabel is not None else ""
    name += ".cfg"
    return name


def simtelArrayConfigFileName(arrayName, site, version, label):
    """
    sim_telarray config file name for an array.

    Parameters
    ----------
    arrayName: str
        Prod5, ...
    site: str
        South or North.
    version: str
        Version of the model.
    label: str
        Instance label.

    Returns
    -------
    str
        File name.
    """
    name = "CTA-{}-{}-{}".format(arrayName, site, version)
    name += "_{}".format(label) if label is not None else ""
    name += ".cfg"
    return name


def simtelSingleMirrorListFileName(
    site, telescopeModelName, modelVersion, mirrorNumber, label
):
    """
    sim_telarray mirror list file with a single mirror.

    Parameters
    ----------
    site: str
        South or North.
    telescopeModelName: str
        North-LST-1, South-MST-FlashCam, ...
    modelVersion: str
        Version of the model.
    mirrorNumber: int
        Mirror number.
    label: str
        Instance label.

    Returns
    -------
    str
        File name.
    """
    name = "CTA-single-mirror-list-{}-{}-{}".format(
        site, telescopeModelName, modelVersion
    )
    name += "-mirror{}".format(mirrorNumber)
    name += "_{}".format(label) if label is not None else ""
    name += ".dat"
    return name


def layoutTelescopeListFileName(name, label):
    """
    File name for files required at the RayTracing class.

    Parameters
    ----------
    name: str
        Name of the array.
    label: str
        Instance label.

    Returns
    -------
    str
        File name.
    """
    fileName = "telescope_positions-{}".format(name)
    fileName += "_{}".format(label) if label is not None else ""
    fileName += ".ecsv"
    return fileName


def rayTracingFileName(
    site,
    telescopeModelName,
    sourceDistance,
    zenithAngle,
    offAxisAngle,
    mirrorNumber,
    label,
    base,
):
    """
    File name for files required at the RayTracing class.

    Parameters
    ----------
    site: str
        South or North.
    telescopeModelName: str
        LST-1, MST-FlashCam, ...
    sourceDistance: float
        Source distance (km).
    zenithAngle: float
        Zenith angle (deg).
    offAxisAngle: float
        Off-axis angle (deg).
    mirrorNumber: int
        Mirror number. None if not single mirror case.
    label: str
        Instance label.
    base: str
        Photons, stars or log.

    Returns
    -------
    str
        File name.
    """
    name = "{}-{}-{}-d{:.1f}-za{:.1f}-off{:.3f}".format(
        base, site, telescopeModelName, sourceDistance, zenithAngle, offAxisAngle
    )
    name += "_mirror{}".format(mirrorNumber) if mirrorNumber is not None else ""
    name += "_{}".format(label) if label is not None else ""
    name += ".log" if base == "log" else ".lis"
    return name


def rayTracingResultsFileName(
    site, telescopeModelName, sourceDistance, zenithAngle, label
):
    """
    Ray tracing results file name.

    Parameters
    ----------
    site: str
        South or North.
    telescopeModelName: str
        LST-1, MST-FlashCam, ...
    sourceDistance: float
        Source distance (km).
    zenithAngle: float
        Zenith angle (deg).
    label: str
        Instance label.

    Returns
    -------
    str
        File name.
    """
    name = "ray-tracing-{}-{}-d{:.1f}-za{:.1f}".format(
        site, telescopeModelName, sourceDistance, zenithAngle
    )
    name += "_{}".format(label) if label is not None else ""
    name += ".ecsv"
    return name


def rayTracingPlotFileName(
    key, site, telescopeModelName, sourceDistance, zenithAngle, label
):
    """
    Ray tracing plot file name.

    Parameters
    ----------
    key: str
        Quantity to be plotted (d80_cm, d80_deg, eff_area or eff_flen)
    site: str
        South or North.
    telescopeModelName: str
        LST-1, MST-FlashCam, ...
    sourceDistance: float
        Source distance (km).
    zenithAngle: float
        Zenith angle (deg).
    label: str
        Instance label.

    Returns
    -------
    str
        File name.
    """
    name = "ray-tracing-{}-{}-{}-d{:.1f}-za{:.1f}".format(
        site, telescopeModelName, key, sourceDistance, zenithAngle
    )
    name += "_{}".format(label) if label is not None else ""
    name += ".pdf"
    return name


def cameraEfficiencyResultsFileName(site, telescopeModelName, zenithAngle, label):
    """
    Camera efficiency results file name.

    Parameters
    ----------
    site: str
        South or North.
    telescopeModelName: str
        LST-1, MST-FlashCam, ...
    zenithAngle: float
        Zenith angle (deg).
    label: str
        Instance label.

    Returns
    -------
    str
        File name.
    """
    name = "camera-efficiency-{}-{}-za{:.1f}".format(
        site, telescopeModelName, zenithAngle
    )
    name += "_{}".format(label) if label is not None else ""
    name += ".ecsv"
    return name


def cameraEfficiencySimtelFileName(site, telescopeModelName, zenithAngle, label):
    """
    Camera efficiency simtel output file name.

    Parameters
    ----------
    site: str
        South or North.
    telescopeModelName: str
        LST-1, MST-FlashCam-D, ...
    zenithAngle: float
        Zenith angle (deg).
    label: str
        Instance label.

    Returns
    -------
    str
        File name.
    """
    name = "camera-efficiency-{}-{}-za{:.1f}".format(
        site, telescopeModelName, zenithAngle
    )
    name += "_{}".format(label) if label is not None else ""
    name += ".dat"
    return name


def cameraEfficiencyLogFileName(site, telescopeModelName, zenithAngle, label):
    """
    Camera efficiency log file name.

    Parameters
    ----------
    site: str
        South or North.
    telescopeModelName: str
        LST-1, MST-FlashCam-D, ...
    zenithAngle: float
        Zenith angle (deg).
    label: str
        Instance label.

    Returns
    -------
    str
        File name.
    """
    name = "camera-efficiency-{}-{}-za{:.1f}".format(
        site, telescopeModelName, zenithAngle
    )
    name += "_{}".format(label) if label is not None else ""
    name += ".log"
    return name


def corsikaConfigFileName(arrayName, site, primary, zenith, viewCone, label=None):
    """
    Corsika config file name.

    Parameters
    ----------
    arrayName: str
        Array name.
    site: str
        Paranal or LaPalma.
    primary: str
        Primary particle (e.g gamma, proton etc).
    zenith: float
        Zenith angle (deg).
    viewCone: list of float
        View cone limits (len = 2).
    label: str
        Instance label.

    Returns
    -------
    str
        File name.
    """
    isDiffuse = viewCone[0] != 0 or viewCone[1] != 0

    name = "corsika-config_{}_{}_{}".format(site, arrayName, primary)
    name += "_za{:d}-{:d}".format(int(zenith[0]), int(zenith[1]))
    name += (
        "_cone{:d}-{:d}".format(int(viewCone[0]), int(viewCone[1])) if isDiffuse else ""
    )
    name += "_{}".format(label) if label is not None else ""
    name += ".input"
    return name


def corsikaConfigTmpFileName(
    arrayName, site, primary, zenith, viewCone, run, label=None
):
    """
    Corsika config file name.

    Parameters
    ----------
    arrayName: str
        Array name.
    site: str
        South or North.
    primary: str
        Primary particle (e.g gamma, proton etc).
    zenith: float
        Zenith angle (deg).
    viewCone: list of float
        View cone limits (len = 2).
    run: int
        Run number.
    label: str
        Instance label.

    Returns
    -------
    str
        File name.
    """
    isDiffuse = viewCone[0] != 0 or viewCone[1] != 0

    name = "corsika-config-run{}".format(run)
    name += "_{}_{}_{}".format(arrayName, site, primary)
    name += "_za{:d}-{:d}".format(int(zenith[0]), int(zenith[1]))
    name += (
        "_cone{:d}-{:d}".format(int(viewCone[0]), int(viewCone[1])) if isDiffuse else ""
    )
    name += "_{}".format(label) if label is not None else ""
    name += ".txt"
    return name


def corsikaOutputFileName(run, primary, arrayName, site, zenith, azimuth, label=None):
    """
    Corsika output file name.

    Warnings
    --------
        zst extension is hardcoded here.

    Parameters
    ----------
    arrayName: str
        Array name.
    site: str
        Paranal or LaPalma.
    zenith: float
        Zenith angle (deg).
    viewCone: list of float
        View cone limits (len = 2).
    run: int
        Run number.
    label: str
        Instance label.

    Returns
    -------
    str
        File name.
    """
    name = "run{}_{}_za{:d}deg_azm{:d}deg-{}-{}".format(
        run, primary, int(zenith), int(azimuth), site, arrayName
    )
    name += "_{}".format(label) if label is not None else ""
    name += ".corsika.zst"
    return name


def corsikaOutputGenericFileName(arrayName, site, label=None):
    name = "run${RUNNR}_${PRMNAME}_za${ZA}deg_azm${AZM}deg"
    name += "-{}-{}".format(site, arrayName)
    name += "_{}".format(label) if label is not None else ""
    name += ".corsika.zst"
    return name


def corsikaRunScriptFileName(arrayName, site, primary, run, label=None):
    """
    Corsika script file path.

    Parameters
    ----------
    arrayName: str
        Array name.
    site: str
        Paranal or LaPalma.
    run: int
        RUn number.
    label: str
        Instance label.

    Returns
    -------
    str
        File path.
    """
    name = "run{}-corsika-{}-{}-{}".format(run, arrayName, site, primary)
    name += "_{}".format(label) if label is not None else ""
    name += ".sh"
    return name


def corsikaRunLogFileName(arrayName, site, primary, run, label=None):
    """
    Corsika script file name.

    Parameters
    ----------
    arrayName: str
        Array name.
    site: str
        Paranal or LaPalma.
    primary: str
        Primary particle name.
    run: int
        RUn number.
    label: str
        Instance label.

    Returns
    -------
    str
        File path.
    """
    name = 'log-corsika-run{}-{}-{}-{}'.format(run, arrayName, site, primary)
    name += '_{}'.format(label) if label is not None else ''
    name += '.log'
    return name


def corsikaSubLogFileName(arrayName, site, primary, run, mode, label=None):
    """
    Corsika submission file name.

    Parameters
    ----------
    arrayName: str
        Array name.
    site: str
        Paranal or LaPalma.
    primary: str
        Primary particle name.
    run: int
        RUn number.
    mode: str
        out or err.
    label: str
        Instance label.

    Returns
    -------
    str
        File path.
    """
    name = 'log-sub-corsika-run{}-{}-{}-{}'.format(run, arrayName, site, primary)
    name += '_{}'.format(label) if label is not None else ''
    name += '.log'
    if len(mode)>0:
        name += "." + mode
    return name


def simtelOutputFileName(run, primary, arrayName, site, zenith, azimuth, label=None):
    """
    sim_telarray output file name.

    Warning
    -------
        zst extension is hardcoded here.

    Parameters
    ----------
    arrayName: str
        Array name.
    site: str
        Paranal or LaPalma.
    zenith: float
        Zenith angle (deg).
    viewCone: list of float
        View cone limits (len = 2).
    run: int
        Run number.
    label: str
        Instance label.

    Returns
    -------
    str
        File name.
    """
    name = "run{}_{}_za{:d}deg_azm{:d}deg-{}-{}".format(
        run, primary, int(zenith), int(azimuth), site, arrayName
    )
    name += "_{}".format(label) if label is not None else ""
    name += ".simtel.zst"
    return name


def simtelHistogramFileName(run, primary, arrayName, site, zenith, azimuth, label=None):
    """
    sim_telarray histogram file name.

    Warning
    -------
        zst extension is hardcoded here.

    Parameters
    ----------
    arrayName: str
        Array name.
    site: str
        Paranal or LaPalma.
    zenith: float
        Zenith angle (deg).
    viewCone: list of float
        View cone limits (len = 2).
    run: int
        Run number.
    label: str
        Instance label.

    Returns
    -------
    str
        File name.
    """
    name = "run{}_{}_za{:d}deg_azm{:d}deg-{}-{}".format(
        run, primary, int(zenith), int(azimuth), site, arrayName
    )
    name += "_{}".format(label) if label is not None else ""
    name += ".hdata.zst"
    return name


def simtelLogFileName(run, primary, arrayName, site, zenith, azimuth, label=None):
    """
    sim_telarray histogram file name.

    Warning
    -------
        zst extension is hardcoded here.

    Parameters
    ----------
    arrayName: str
        Array name.
    site: str
        Paranal or LaPalma.
    zenith: float
        Zenith angle (deg).
    viewCone: list of float
        View cone limits (len = 2).
    run: int
        Run number.
    label: str
        Instance label.

    Returns
    -------
    str
        File name.
    """
    name = "run{}_{}_za{:d}deg_azm{:d}deg-{}-{}".format(
        run, primary, int(zenith), int(azimuth), site, arrayName
    )
    name += "_{}".format(label) if label is not None else ""
    name += ".log"
    return name

def simtelSubLogFileName(run, primary, arrayName, site, zenith, azimuth, mode, label=None):
    """
    sim_telarray submission log file name.

    Parameters
    ----------
    arrayName: str
        Array name.
    site: str
        Paranal or LaPalma.
    zenith: float
        Zenith angle (deg).
    viewCone: list of float
        View cone limits (len = 2).
    run: int
        Run number.
    mode: str
        out or err
    label: str
        Instance label.

    Returns
    -------
    str
        File name.
    """
    name = 'log-sub-run{}_{}_za{:d}deg_azm{:d}deg-{}-{}'.format(
        run,
        primary,
        int(zenith),
        int(azimuth),
        site,
        arrayName
    )
    name += '_{}'.format(label) if label is not None else ''
    name += '.log'
    if len(mode)>0:
        name += "." + mode
    return name
