import logging

__all__ = [
    'validateModelVersionName',
    'validateSimtelModeName',
    'validateSiteName',
    'validateLayoutArrayName',
    'validateTelescopeName',
    'validateCameraName',
    'convertTelescopeNameToYaml',
    'splitTelescopeName',
    'getSiteFromTelescopeName',
    'rayTracingFileName',
    'simtelTelescopeConfigFileName',
    'simtelArrayConfigFileName',
    'simtelSingleMirrorListFileName',
    'corsikaConfigFileName',
    'corsikaOutputFileName'
]


def validateCameraName(name):
    '''
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
    '''
    return validateName(name, allCameraNames)


def validateModelVersionName(name):
    '''
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
    '''
    return validateName(name, allModelVersionNames)


def validateSimtelModeName(name):
    '''
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
    '''
    return validateName(name, allSimtelModeNames)


def validateSiteName(name):
    '''
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
    '''
    return validateName(name, allSiteNames)


def validateLayoutArrayName(name):
    '''
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
    '''
    return validateName(name, allLayoutArrayNames)


def validateName(name, allNames):
    '''
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
    '''
    _logger = logging.getLogger(__name__)

    if not isValidName(name, allNames):
        msg = 'Invalid name {}'.format(name)
        _logger.error(msg)
        raise ValueError(msg)
    for mainName, listOfNames in allNames.items():
        if name.lower() in listOfNames + [mainName.lower()]:
            if name != mainName:
                _logger.debug('Correcting name {} -> {}'.format(name, mainName))
            return mainName
    return None


def isValidName(name, allNames):
    '''
    Parameters
    ----------
    name: str
    allNames: dict

    Returns
    -------
    bool
    '''
    if not isinstance(name, str):
        return False
    for mainName in allNames.keys():
        if name.lower() in allNames[mainName] + [mainName.lower()]:
            return True
    return False


def validateTelescopeName(name):
    '''
    Validate a telescope name.

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
    '''
    telSite, telClass, telType = splitTelescopeName(name)
    telSite = validateSiteName(telSite)
    telClass = validateName(telClass, allTelescopeClassNames)
    if 'flashcam' in telType:
        telType = telType.replace('flashcam', 'FlashCam')
    if 'nectarcam' in telType:
        telType = telType.replace('nectarcam', 'NectarCam')
    if '1m' in telType:
        telType = telType.replace('1m', '1M')
    if 'gct' in telType:
        telType = telType.replace('gct', 'GCT')
    if 'astri' in telType:
        telType = telType.replace('astri', 'ASTRI')
    if '-d' in '-' + telType:
        telType = telType.replace('d', 'D')

    return telSite + '-' + telClass + '-' + telType


def splitTelescopeName(name):
    '''
    Split a telescope name into site, class and type.

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
    str, str, str
        Site (South or North), class (LST, MST, SST ...) and type (any complement).
    '''
    nameParts = name.split('-')
    thisSite = nameParts[0]
    telClass = nameParts[1]
    telType = '-'.join(nameParts[2:])
    return thisSite, telClass, telType


def getSiteFromTelescopeName(name):
    '''
    Get site name (South or North) from the (validated) telescope name.

    Parameters
    ----------
    name: str
        Telescope name.

    Returns
    -------
    str
        Site name (South or North).
    '''
    nameParts = name.split('-')
    thisSite = validateSiteName(nameParts[0])
    return thisSite


def convertTelescopeNameToYaml(name):
    '''
    Get telescope name following the old convention (yaml files) from the current telescope name.

    Parameters
    ----------
    name: str
        Telescope name.

    Returns
    -------
    str
        Telescope name (old convention).
    '''
    telSite, telClass, telType = splitTelescopeName(name)
    newName = telClass + '-' + telType
    oldNames = {
        'SST-D': 'SST',
        'SST-1M': 'SST-1M',
        'SST-ASTRI': 'SST-2M-ASTRI',
        'SST-GCT': 'SST-2M-GCT-S',
        'MST-FlashCam-D': 'MST-FlashCam',
        'MST-NectarCam-D': 'MST-NectarCam',
        'SCT-D': 'SCT',
        'LST-D234': 'LST',
        'LST-1': 'LST'
    }

    if newName not in oldNames.keys():
        raise ValueError('Telescope name {} could not be converted to yml names'.format(name))
    else:
        return oldNames[newName]


allTelescopeClassNames = {
    'SST': ['sst'],
    'MST': ['mst'],
    'SCT': ['sct'],
    'LST': ['lst']
}

allCameraNames = {
    'SST': ['sst'],
    'ASTRI': ['astri'],
    'GCT': ['gct', 'gct-s'],
    '1M': ['1m'],
    'FlashCam': ['flashcam', 'flash-cam'],
    'NectarCam': ['nectarcam', 'nectar-cam'],
    'SCT': ['sct'],
    'LST': ['lst']
}


allSiteNames = {
    'South': ['paranal', 'south'],
    'North': ['lapalma', 'north']
}

allModelVersionNames = {
    'prod3_compatible': ['p3', 'prod3', 'prod3b'],
    'prod4': ['p4'],
    'post_prod3_updates': [''],
    '2018-11-07': [''],
    '2019-02-22': [''],
    '2019-05-13': [''],
    '2019-11-20': [''],
    '2019-12-30': [''],
    '2020-02-26': [''],
    '2020-06-28': ['prod5'],
    'prod4-prototype': [''],
    'default': [],
    'Current': [],
    'Latest': []
}

allSimtelModeNames = {
    'RayTracing': ['raytracing', 'ray-tracing'],
    'RayTracingSingleMirror': [
        'raytracing-singlemirror',
        'ray-tracing-singlemirror',
        'ray-tracing-single-mirror'
    ],
    'Trigger': ['trigger']
}

allLayoutArrayNames = {
    '4LST': ['4-lst', '4lst'],
    '1LST': ['1-lst', '1lst'],
    '4MST': ['4-mst', '4mst'],
    '1MST': ['1-mst', 'mst'],
    '4SST': ['4-sst', '4sst'],
    '1SST': ['1-sst', 'sst'],
    'Prod5': ['prod5', 'p5']
}


def simtelTelescopeConfigFileName(version, telescopeName, label):
    '''
    sim_telarray config file name for a telescope.

    Parameters
    ----------
    version: str
        Version of the model.
    telescopeName: str
        North-LST-1, South-MST-FlashCam, ...
    label: str
        Instance label.

    Returns
    -------
    str
        File name.
    '''
    name = 'CTA-{}-{}'.format(version, telescopeName)
    name += '_{}'.format(label) if label is not None else ''
    name += '.cfg'
    return name


def simtelArrayConfigFileName(arrayName, site, version, label):
    '''
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
    '''
    name = 'CTA-{}-{}-{}'.format(arrayName, site, version)
    name += '_{}'.format(label) if label is not None else ''
    name += '.cfg'
    return name


def simtelSingleMirrorListFileName(version, telescopeName, mirrorNumber, label):
    '''
    sim_telarray mirror list file with a single mirror.

    Parameters
    ----------
    version: str
        Version of the model.
    telescopeName: str
        North-LST-1, South-MST-FlashCam, ...
    mirrorNumber: int
        Mirror number.
    label: str
        Instance label.

    Returns
    -------
    str
        File name.
    '''
    name = 'CTA-single-mirror-list-{}-{}'.format(version, telescopeName)
    name += '-mirror{}'.format(mirrorNumber)
    name += '_{}'.format(label) if label is not None else ''
    name += '.dat'
    return name


def layoutTelescopeListFileName(name, label):
    '''
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
    '''
    fileName = 'telescope_positions-{}'.format(name)
    fileName += '_{}'.format(label) if label is not None else ''
    fileName += '.ecsv'
    return fileName


def rayTracingFileName(
    telescopeName,
    sourceDistance,
    zenithAngle,
    offAxisAngle,
    mirrorNumber,
    label,
    base
):
    '''
    File name for files required at the RayTracing class.

    Parameters
    ----------
    telescopeName: str
        North-LST-1, South-MST-FlashCam, ...
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
    '''
    name = '{}-{}-d{:.1f}-za{:.1f}-off{:.3f}'.format(
        base,
        telescopeName,
        sourceDistance,
        zenithAngle,
        offAxisAngle
    )
    name += '_mirror{}'.format(mirrorNumber) if mirrorNumber is not None else ''
    name += '_{}'.format(label) if label is not None else ''
    name += '.log' if base == 'log' else '.lis'
    return name


def rayTracingResultsFileName(telescopeName, sourceDistance, zenithAngle, label):
    '''
    Ray tracing results file name.

    Parameters
    ----------
    telescopeName: str
        North-LST-1, South-MST-FlashCam, ...
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
    '''
    name = 'ray-tracing-{}-d{:.1f}-za{:.1f}'.format(telescopeName, sourceDistance, zenithAngle)
    name += '_{}'.format(label) if label is not None else ''
    name += '.cvs'
    return name


def rayTracingPlotFileName(key, telescopeName, sourceDistance, zenithAngle, label):
    '''
    Ray tracing plot file name.

    Parameters
    ----------
    key: str
        Quantity to be plotted (d80_cm, d80_deg, eff_area or eff_flen)
    telescopeName: str
        South-LST-1, North-MST-FlashCam, ...
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
    '''
    name = 'ray-tracing-{}-{}-d{:.1f}-za{:.1f}'.format(
        telescopeName,
        key,
        sourceDistance,
        zenithAngle
    )
    name += '_{}'.format(label) if label is not None else ''
    name += '.pdf'
    return name


def cameraEfficiencyResultsFileName(telescopeName, zenithAngle, label):
    '''
    Camera efficiency results file name.

    Parameters
    ----------
    telescopeName: str
        South-LST-1, North-MST-FlashCam, ...
    zenithAngle: float
        Zenith angle (deg).
    label: str
        Instance label.

    Returns
    -------
    str
        File name.
    '''
    name = 'camera-efficiency-{}-za{:.1f}'.format(telescopeName, zenithAngle)
    name += '_{}'.format(label) if label is not None else ''
    name += '.csv'
    return name


def cameraEfficiencySimtelFileName(telescopeName, zenithAngle, label):
    '''
    Camera efficiency simtel output file name.

    Parameters
    ----------
    telescopeName: str
        North-LST-1, South-MST-FlashCam, ...
    zenithAngle: float
        Zenith angle (deg).
    label: str
        Instance label.

    Returns
    -------
    str
        File name.
    '''
    name = 'camera-efficiency-{}-za{:.1f}'.format(telescopeName, zenithAngle)
    name += '_{}'.format(label) if label is not None else ''
    name += '.dat'
    return name


def cameraEfficiencyLogFileName(telescopeName, zenithAngle, label):
    '''
    Camera efficiency log file name.

    Parameters
    ----------
    telescopeName: str
        South-LST-1, North-MST-FlashCam, ...
    zenithAngle: float
        Zenith angle (deg).
    label: str
        Instance label.

    Returns
    -------
    str
        File name.
    '''
    name = 'camera-efficiency-{}-za{:.1f}'.format(telescopeName, zenithAngle)
    name += '_{}'.format(label) if label is not None else ''
    name += '.log'
    return name


def corsikaConfigFileName(arrayName, site, zenith, viewCone, label=None):
    '''
    Corsika config file name.

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
    label: str
        Instance label.

    Returns
    -------
    str
        File name.
    '''
    isDiffuse = (viewCone[0] != 0 or viewCone[1] != 0)

    name = 'corsika-config_{}_{}'.format(site, arrayName)
    name += '_za{:d}-{:d}'.format(int(zenith[0]), int(zenith[1]))
    name += '_cone{:d}-{:d}'.format(int(viewCone[0]), int(viewCone[1])) if isDiffuse else ''
    name += '_{}'.format(label) if label is not None else ''
    name += '.input'
    return name


def corsikaConfigTmpFileName(arrayName, site, zenith, viewCone, run, label=None):
    '''
    Corsika config file name.

    Parameters
    ----------
    arrayName: str
        Array name.
    site: str
        South or North.
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
    '''
    isDiffuse = (viewCone[0] != 0 or viewCone[1] != 0)

    name = 'corsika-config-run{}'.format(run)
    name += '-{}-{}'.format(arrayName, site)
    name += '-za{:d}-{:d}'.format(int(zenith[0]), int(zenith[1]))
    name += '-cone{:d}-{:d}'.format(int(viewCone[0]), int(viewCone[1])) if isDiffuse else ''
    name += '_{}'.format(label) if label is not None else ''
    name += '.txt'
    return name


def corsikaOutputFileName(run, primary, arrayName, site, zenith, azimuth, label=None):
    '''
    Corsika output file name.

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
    '''
    name = 'run{}_{}_za{:d}deg_azm{:d}deg-{}-{}'.format(
        run,
        primary,
        int(zenith),
        int(azimuth),
        site,
        arrayName
    )
    name += '_{}'.format(label) if label is not None else ''
    name += '.corsika.zst'
    return name


def corsikaOutputGenericFileName(arrayName, site, label=None):
    name = 'run${RUNNR}_${PRMNAME}_za${ZA}deg_azm${AZM}deg'
    name += '-{}-{}'.format(site, arrayName)
    name += '_{}'.format(label) if label is not None else ''
    name += '.corsika.zst'
    return name


def corsikaRunScriptFileName(arrayName, site, run, label=None):
    '''
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
    '''
    name = 'run-corsika-run{}-{}-{}'.format(run, arrayName, site)
    name += '_{}'.format(label) if label is not None else ''
    name += '.sh'
    return name


def corsikaRunLogFileName(arrayName, site, run, label=None):
    '''
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
    '''
    name = 'log-corsika-run{}-{}-{}'.format(run, arrayName, site)
    name += '_{}'.format(label) if label is not None else ''
    name += '.log'
    return name
