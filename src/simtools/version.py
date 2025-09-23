"""Software version setting."""

# this is adapted from https://github.com/cta-observatory/ctapipe/blob/main/ctapipe/version.py
# which is adapted from https://github.com/astropy/astropy/blob/master/astropy/version.py
# see https://github.com/astropy/astropy/pull/10774 for a discussion on why this needed.

from packaging.version import InvalidVersion, Version

try:
    try:
        from ._dev_version import version
    except ImportError:
        from ._version import version
except Exception:  # pylint: disable=broad-except
    import warnings

    warnings.warn("Could not determine simtools version; this indicates a broken installation.")
    del warnings
    version = "0.0.0"  # pylint: disable=invalid-name

__version__ = version


def resolve_version_to_latest_patch(partial_version, available_versions):
    """
    Resolve a partial version (major.minor) to the latest patch version.

    Given a partial version string (e.g., "6.0") and a list of available versions,
    finds the latest patch version that matches the major.minor pattern.

    Parameters
    ----------
    partial_version : str
        Partial version string in format "major.minor" (e.g., "6.0", "5.2")
    available_versions : list of str
        List of available semantic versions (e.g., ["5.0.0", "5.0.1", "6.0.0", "6.0.2"])

    Returns
    -------
    str
        Latest patch version matching the partial version pattern

    Raises
    ------
    ValueError
        If partial_version is not in major.minor format
    ValueError
        If no matching versions are found

    Examples
    --------
    >>> versions = ["5.0.0", "5.0.1", "6.0.0", "6.0.2", "6.1.0"]
    >>> resolve_version_to_latest_patch("6.0", versions)
    '6.0.2'
    >>> resolve_version_to_latest_patch("5.0", versions)
    '5.0.1'
    >>> resolve_version_to_latest_patch("5.0.1", versions)
    '5.0.1'
    """
    try:
        pv = Version(partial_version)
    except InvalidVersion as exc:
        raise ValueError(f"Invalid version string: {partial_version}") from exc

    if pv.release and len(pv.release) >= 3:
        return str(pv)

    if len(pv.release) != 2:
        raise ValueError(f"Partial version must be major.minor, got: {partial_version}")

    major, minor = pv.release

    candidates = [
        v for v in available_versions if Version(v).major == major and Version(v).minor == minor
    ]

    if not candidates:
        raise ValueError(
            f"No versions found matching '{partial_version}.x' "
            f"in available versions: {sorted(available_versions)}"
        )

    return str(max(map(Version, candidates)))


def semver_to_int(version_string):
    """
    Convert a semantic version string to an integer.

    Parameters
    ----------
    version_string : str
        Semantic version string (e.g., "6.0.2")

    Returns
    -------
    int
        Integer representation of the version (e.g., 60002 for "6.0.2")

    """
    try:
        v = Version(version_string)
    except InvalidVersion as exc:
        raise ValueError(f"Invalid version: {version_string}") from exc

    release = v.release + (0,) * (3 - len(v.release))
    major, minor, patch = release[:3]
    return major * 10000 + minor * 100 + patch


def sort_versions(version_list, reverse=False):
    """
    Sort a list of semantic version strings.

    Parameters
    ----------
    version_list : list of str
        List of semantic version strings (e.g., ["5.0.0", "6.0.2", "5.1.0"])
    reverse : bool, optional
        Sort in descending order if True (default False)

    Returns
    -------
    list of str
        Sorted list of version strings.
    """
    try:
        return [str(v) for v in sorted(map(Version, version_list), reverse=reverse)]
    except InvalidVersion as exc:
        raise ValueError(f"Invalid version in list: {version_list}") from exc
