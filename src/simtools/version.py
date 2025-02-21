"""Software version setting."""

# this is adapted from https://github.com/cta-observatory/ctapipe/blob/main/ctapipe/version.py
# which is adapted from https://github.com/astropy/astropy/blob/master/astropy/version.py
# see https://github.com/astropy/astropy/pull/10774 for a discussion on why this needed.

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
