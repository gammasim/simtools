# Try to use setuptools_scm to get the current version; this is only used
# in development installations from the git repository.
# Copied from astropy, see simtools/version.py for details
import os.path as pth

try:
    from setuptools_scm import get_version

    version = get_version(root=pth.join("..", ".."), relative_to=__file__)
except Exception as e:
    raise ImportError(f"setuptools_scm broken or not installed: {e}")
