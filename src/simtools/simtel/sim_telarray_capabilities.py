"""Bypass_Optics levels supported by the installed sim_telarray.

Bypass_Optics levels 0, 1 and 2 exist in all sim_telarray releases known to simtools. Level 3
(photons given at the camera entrance, sim_telarray applies only the quantum efficiency) is not
in any known release. ``BYPASS_OPTICS_MODE_3_MIN_VERSION`` is the first release that supports
it; None means that no known release does, so level 3 is refused unless it is overridden.

A patched local sim_telarray build declares its highest supported level in the environment
variable ``SIMTOOLS_SIM_TELARRAY_BYPASS_OPTICS_MAX`` (an integer). The override takes precedence
over the version table and no ``sim_telarray --version`` call is made when it is set.

The version of the installed sim_telarray is read with
``simtools.runtime.dependencies.get_sim_telarray_version``, using the container run-time prefix
of the context.
"""

import os

from packaging.version import InvalidVersion, Version

from simtools.runtime import context as run_context
from simtools.runtime import dependencies

#: First sim_telarray release supporting Bypass_Optics=3 (None: no known release).
BYPASS_OPTICS_MODE_3_MIN_VERSION = None
#: Environment variable with the highest Bypass_Optics level of a patched sim_telarray build.
BYPASS_OPTICS_MAX_ENVIRONMENT_VARIABLE = "SIMTOOLS_SIM_TELARRAY_BYPASS_OPTICS_MAX"
ALWAYS_SUPPORTED_LEVELS = (0, 1, 2)
KNOWN_LEVELS = (0, 1, 2, 3)


def supports_bypass_optics(level, version, max_level_override=None):
    """
    Return whether a sim_telarray build supports a Bypass_Optics level.

    Parameters
    ----------
    level : int
        Bypass_Optics level.
    version : str or None
        sim_telarray release (e.g. "2024.271.0"); None if unknown.
    max_level_override : int or None
        Highest supported level of a patched build; takes precedence over the version table.

    Returns
    -------
    bool
        True if the level is supported.
    """
    if level not in KNOWN_LEVELS:
        return False
    if max_level_override is not None:
        return level <= max_level_override
    if level in ALWAYS_SUPPORTED_LEVELS:
        return True
    return _release_supports_mode_3(version)


def _release_supports_mode_3(version):
    """Return whether a release is at or after the first release supporting Bypass_Optics=3."""
    if version is None or BYPASS_OPTICS_MODE_3_MIN_VERSION is None:
        return False
    try:
        return Version(version) >= Version(BYPASS_OPTICS_MODE_3_MIN_VERSION)
    except InvalidVersion:
        return False


def bypass_optics_override():
    """
    Return the highest Bypass_Optics level set by the environment, or None if not set.

    Raises
    ------
    ValueError
        If the environment variable is not an integer.
    """
    value = os.environ.get(BYPASS_OPTICS_MAX_ENVIRONMENT_VARIABLE, "").strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            f"{BYPASS_OPTICS_MAX_ENVIRONMENT_VARIABLE} must be an integer, got {value!r}."
        ) from exc


def installed_sim_telarray_version(context=None):
    """
    Return the release of the installed sim_telarray, or None if it cannot be determined.

    The version is read with ``sim_telarray --version`` (with the container run-time prefix of
    the context).
    """
    context = context or run_context.RunContext.from_environment()
    try:
        return dependencies.get_sim_telarray_version(run_time=context.run_time, context=context)
    except ValueError:
        return None


def check_bypass_optics(level, context=None):
    """
    Raise ValueError if the installed sim_telarray does not support a Bypass_Optics level.

    Levels 0, 1 and 2 are accepted without calling sim_telarray. For other levels, the
    override variable is used if set; otherwise the installed release is checked.

    Parameters
    ----------
    level : int
        Bypass_Optics level requested.
    context : RunContext, optional
        Runtime context (sim_telarray executable and container run-time prefix).

    Raises
    ------
    ValueError
        If the level is not supported; the message names the version found, the level and the
        override variable.
    """
    override = bypass_optics_override()
    if override is None and level in ALWAYS_SUPPORTED_LEVELS:
        return
    if override is not None:
        if supports_bypass_optics(level, None, override):
            return
        raise ValueError(
            f"Bypass_Optics={level} is not supported: {BYPASS_OPTICS_MAX_ENVIRONMENT_VARIABLE}="
            f"{override} declares the highest supported level of the patched sim_telarray build."
        )
    version = installed_sim_telarray_version(context)
    if supports_bypass_optics(level, version):
        return
    raise ValueError(
        f"Bypass_Optics={level} is not supported by the installed sim_telarray "
        f"(release {version or 'unknown'}). Use a release that supports it, or, for a patched "
        f"build, set {BYPASS_OPTICS_MAX_ENVIRONMENT_VARIABLE} to its highest supported level."
    )
