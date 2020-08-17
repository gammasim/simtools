''' Utilities to deal with names validation and file names. '''

import logging

__all__ = [
    'validateModelVersionName',
    'validateSimtelModeName',
    'validateSiteName',
    'validateArrayName',
    'validateTelescopeName',
    'validateCameraName',
    'convertTelescopeNameToYaml'
    'splitTelescopeName',
    'getSiteFromTelescopeName',
    'rayTracingFileName',
    'simtelConfigFileName',
    'simtelSingleMirrorListFileName',
    'corsikaConfigFileName',
    'corsikaOutputFileName'
]

logger = logging.getLogger(__name__)


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


def validateArrayName(name):
    '''
    Validate a array name.

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
    return validateName(name, allArrayNames)


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
    if not isValidName(name, allNames):
        msg = 'Invalid name {}'.format(name)
        logger.error(msg)
        raise ValueError(msg)
    for mainName, listOfNames in allNames.items():
        if name.lower() in listOfNames + [mainName.lower()]:
            if name != mainName:
                logger.debug('Correcting name {} -> {}'.format(name, mainName))
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
    'default': []
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

allArrayNames = {
    '4LST': ['4-lst', '4lst'],
    '1LST': ['1-lst', '1lst'],
    '4MST': ['4-mst', '4mst'],
    '1MST': ['1-mst', 'mst'],
    '4SST': ['4-sst', '4sst'],
    '1SST': ['1-sst', 'sst']
}


def simtelConfigFileName(version, telescopeName, label):
    '''
    sim_telarray config file name.

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

    name = 'corsika-config-{}-{}'.format(arrayName, site)
    name += '-za{:d}-{:d}'.format(int(zenith[0]), int(zenith[1]))
    name += '-cone{:d}-{:d}'.format(int(viewCone[0]), int(viewCone[1])) if isDiffuse else ''
    name += '_{}'.format(label) if label is not None else ''
    name += '.txt'
    return name


def corsikaOutputFileName(arrayName, site, zenith, viewCone, run, label=None):
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
    isDiffuse = (viewCone[0] != 0 or viewCone[1] != 0)

    name = 'corsika-run{}-{}-{}-za{:d}-{:d}'.format(
        run,
        arrayName,
        site,
        int(zenith[0]),
        int(zenith[1])
    )
    name += '-cone{:d}-{:d}'.format(int(viewCone[0]), int(viewCone[1])) if isDiffuse else ''
    name += '_{}'.format(label) if label is not None else ''
    name += '.corsika.zst'
    return name
