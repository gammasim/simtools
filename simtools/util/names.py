''' Utilities to deal with names validation and file names. '''

import logging

__all__ = [
    'validateName',
    'isValidName',
    'rayTracingFileName',
    'simtelConfigFileName',
    'simtelSingleMirrorListFileName',
    'corsikaConfigFileName',
    'corsikaOutputFileName'
]

logger = logging.getLogger(__name__)


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
    nameParts = name.split('-')
    thisSite = validateName(nameParts[0], allSiteNames)
    telClass = validateName(nameParts[1], allTelescopeClassNames)
    telType = ''
    for ii in range(2, len(nameParts)):
        telType += nameParts[ii] + ('' if ii == len(nameParts) - 1 else '-') 
    return thisSite + '-' + telClass + '-' + telType


allTelescopeTypeNames = {
    'SST': ['sst'],
    'SST-1M': ['1m'],
    'SST-2M-ASTRI': ['sst-astri', 'astri'],
    'SST-2M-GCT-S': ['sst-gct', 'gct', 'sst-gct-s'],
    'MST-FlashCam': ['flashcam', 'mst-fc'],
    'MST-NectarCam': ['nectarcam', 'mst-nc'],
    'SCT': ['mst-sct', 'sct'],
    'LST': [],
    'North-LST-1': ['north-lst-1'],
    'North-LST-D234': ['north-lst-d234'],
    'North-MST-FlashCam-D': ['north-flashcam-d', 'north-mst-fc-d'],
    'North-MST-NectarCam-D': ['north-nectarcam-d', 'north-mst-nc-d'],
    'North-SCT-D': ['north-mst-sct-d', 'north-sct-d'],
    'South-SST-D': ['south-sst-d'],
    'North-LST-Test': ['north-lst-test']
}

allTelescopeClassNames = {
    'SST': ['sst'],
    'MST-FlashCam': ['mst'],
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


def simtelConfigFileName(version, site, telescopeType, label):
    '''
    sim_telarray config file name.

    Parameters
    ----------
    version: str
        Version of the model.
    site: str
        Paranal or LaPalma
    telescopeType: str
        LST, MST-FlashCam, ...
    label: str
        Instance label.

    Returns
    -------
    str
        File name.
    '''
    name = 'CTA-{}-{}-{}'.format(version, site, telescopeType)
    name += '_{}'.format(label) if label is not None else ''
    name += '.cfg'
    return name


def simtelSingleMirrorListFileName(version, site, telescopeType, mirrorNumber, label):
    '''
    sim_telarray mirror list file with a single mirror.

    Parameters
    ----------
    version: str
        Version of the model.
    site: str
        Paranal or LaPalma
    telescopeType: str
        LST, MST-FlashCam, ...
    mirrorNumber: int
        Mirror number.
    label: str
        Instance label.

    Returns
    -------
    str
        File name.
    '''
    name = 'CTA-single-mirror-list-{}-{}-{}'.format(version, site, telescopeType)
    name += '-mirror{}'.format(mirrorNumber)
    name += '_{}'.format(label) if label is not None else ''
    name += '.dat'
    return name


def rayTracingFileName(
    telescopeType,
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
    telescopeType: str
        LST, MST-FlashCam, ...
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
        telescopeType,
        sourceDistance,
        zenithAngle,
        offAxisAngle
    )
    name += '_mirror{}'.format(mirrorNumber) if mirrorNumber is not None else ''
    name += '_{}'.format(label) if label is not None else ''
    name += '.log' if base == 'log' else '.lis'
    return name


def rayTracingResultsFileName(telescopeType, sourceDistance, zenithAngle, label):
    '''
    Ray tracing results file name.

    Parameters
    ----------
    telescopeType: str
        LST, MST-FlashCam, ...
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
    name = 'ray-tracing-{}-d{:.1f}-za{:.1f}'.format(telescopeType, sourceDistance, zenithAngle)
    name += '_{}'.format(label) if label is not None else ''
    name += '.cvs'
    return name

def rayTracingPlotFileName(key, telescopeType, sourceDistance, zenithAngle, label):
    '''
    Ray tracing plot file name.

    Parameters
    ----------
    key: str
        Quantity to be plotted (d80_cm, d80_deg, eff_area or eff_flen)
    telescopeType: str
        LST, MST-FlashCam, ...
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
    name = 'ray-tracing-{}-{}-d{:.1f}-za{:.1f}'.format(telescopeType, key, sourceDistance, zenithAngle)
    name += '_{}'.format(label) if label is not None else ''
    name += '.pdf'
    return name

def cameraEfficiencyResultsFileName(telescopeType, zenithAngle, label):
    '''
    Camera efficiency results file name.

    Parameters
    ----------
    telescopeType: str
        LST, MST-FlashCam, ...
    zenithAngle: float
        Zenith angle (deg).
    label: str
        Instance label.

    Returns
    -------
    str
        File name.
    '''
    name = 'camera-efficiency-{}-za{:.1f}'.format(telescopeType, zenithAngle)
    name += '_{}'.format(label) if label is not None else ''
    name += '.csv'
    return name


def cameraEfficiencySimtelFileName(telescopeType, zenithAngle, label):
    '''
    Camera efficiency simtel output file name.

    Parameters
    ----------
    telescopeType: str
        LST, MST-FlashCam, ...
    zenithAngle: float
        Zenith angle (deg).
    label: str
        Instance label.

    Returns
    -------
    str
        File name.
    '''
    name = 'camera-efficiency-{}-za{:.1f}'.format(telescopeType, zenithAngle)
    name += '_{}'.format(label) if label is not None else ''
    name += '.dat'
    return name


def cameraEfficiencyLogFileName(telescopeType, zenithAngle, label):
    '''
    Camera efficiency log file name.

    Parameters
    ----------
    telescopeType: str
        LST, MST-FlashCam, ...
    zenithAngle: float
        Zenith angle (deg).
    label: str
        Instance label.

    Returns
    -------
    str
        File name.
    '''
    name = 'camera-efficiency-{}-za{:.1f}'.format(telescopeType, zenithAngle)
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
