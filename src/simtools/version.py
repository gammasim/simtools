"""Software version setting."""

# this is adapted from https://github.com/cta-observatory/ctapipe/blob/main/ctapipe/version.py
# which is adapted from https://github.com/astropy/astropy/blob/master/astropy/version.py
# see https://github.com/astropy/astropy/pull/10774 for a discussion on why this needed.

import re

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

MAJOR_MINOR_PATCH = "major.minor.patch"
MAJOR_MINOR = "major.minor"

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


def version_kind(version_string):
    """
    Determine the kind of version string.

    Parameters
    ----------
    version_string : str
        The version string to analyze.

    Returns
    -------
    str
        The kind of version string ("major.minor", "major.minor.patch", or "major").
    """
    try:
        ver = Version(version_string)
    except InvalidVersion as exc:
        raise ValueError(f"Invalid version string: {version_string}") from exc
    if ver.release and len(ver.release) >= 3:
        return MAJOR_MINOR_PATCH
    if len(ver.release) == 2:
        return MAJOR_MINOR
    return "major"


def compare_versions(version_string_1, version_string_2, level=MAJOR_MINOR_PATCH):
    """
    Compare two versions at the given level: "major", "major.minor", "major.minor.patch".

    Parameters
    ----------
    version_string_1 : str
        First version string to compare.
    version_string_2 : str
        Second version string to compare.
    level : str, optional
        Level of comparison: "major", "major.minor", or "major.minor.patch"

    Returns
    -------
    int
        -1 if version_string_1 < version_string_2
         0 if version_string_1 == version_string_2
         1 if version_string_1 > version_string_2
    """
    ver1 = Version(version_string_1).release
    ver2 = Version(version_string_2).release

    if level == "major":
        ver1, ver2 = ver1[:1], ver2[:1]
    elif level == MAJOR_MINOR:
        ver1, ver2 = ver1[:2], ver2[:2]
    elif level != MAJOR_MINOR_PATCH:
        raise ValueError(f"Unknown level: {level}")

    return (ver1 > ver2) - (ver1 < ver2)


def base_version_for_patch_delta(version_string):
    """Return major.minor.0 version for patch releases (x.y.z -> x.y.0) or None."""
    if not version_string:
        return None

    try:
        v = Version(str(version_string).strip().removeprefix("v"))
    except InvalidVersion:
        return None

    if len(v.release) == 3 and v.release[2] > 0:
        major, minor, _ = v.release
        return f"{major}.{minor}.0"

    return None


def is_valid_semantic_version(version_string, strict=True):
    """
    Check if a string is a valid semantic version.

    Parameters
    ----------
    version_string : str
        The version string to validate (e.g., "6.0.2", "1.0.0-alpha").
    strict : bool, optional
        If True, use PEP 440 validation (packaging.version.Version).
        If False, use SemVer 2.0.0 regex pattern (allows more flexible pre-release identifiers).

    Returns
    -------
    bool
        True if the version string is valid, False otherwise.
    """
    if not version_string:
        return False

    if strict:
        try:
            Version(version_string)
            return True
        except InvalidVersion:
            return False
    else:
        semver_regex = (
            r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"  # major.minor.patch
            r"(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?"  # pre-release
            r"(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$"  # build metadata
        )
        return bool(re.match(semver_regex, version_string))


def check_version_constraint(version_string, constraint):
    """
    Check if a version satisfies a constraint.

    Parameters
    ----------
    version_string : str
        The version string to check (e.g., "6.0.2").
    constraint : str
        The version constraint to check against (e.g., ">=6.0.0").

    Returns
    -------
    bool
        True if the version satisfies the constraint, False otherwise.
    """
    spec = SpecifierSet(constraint.strip(), prereleases=True)
    ver = Version(version_string)
    if ver in spec:
        return True
    return False


def resolve_by_version(config, model_version):
    """
    Resolve version-dependent values in a configuration dictionary.

    Fields whose value is a dict with a single ``by_version`` key are replaced
    by the first matching constraint value. All other fields are left unchanged.

    Parameters
    ----------
    config : dict
        Configuration dictionary (typically loaded from a YAML file).
    model_version : str or list
        Semantic version string or list of strings used to evaluate constraints
        (e.g. "6.0.2" or ["6.0.2", "6.1.1"]).

    Returns
    -------
    dict
        New dictionary with all ``by_version`` fields resolved to their
        matching value, or ``None`` when no constraint matches.
    """
    if not model_version:
        return config

    versions = model_version if isinstance(model_version, list) else [model_version]
    parsed_versions = [Version(str(version_item)) for version_item in versions]

    def resolve_by_version_field(by_version_dict, parsed_versions, key):
        """Resolve a single by_version field for all versions, enforcing consistency."""
        matched_values = [resolve_single_version(by_version_dict, pv) for pv in parsed_versions]
        first_match = matched_values[0]
        if not all(match == first_match for match in matched_values):
            raise ValueError(
                f"Inconsistent by_version resolution for key '{key}' and model versions "
                f"{versions}: {matched_values}"
            )
        return first_match

    def resolve_single_version(by_version_dict, parsed_version):
        for constraint, result in by_version_dict.items():
            if check_version_constraint(str(parsed_version), constraint):
                return result
        return None

    resolved = {}
    for key, value in config.items():
        if isinstance(value, dict) and list(value) == ["by_version"]:
            resolved[key] = resolve_by_version_field(value["by_version"], parsed_versions, key)
        else:
            resolved[key] = value
    return resolved
