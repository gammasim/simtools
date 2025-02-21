"""Helper functions for integration testing."""

import os
from pathlib import Path


def skip_camera_efficiency(config):
    """Skip camera efficiency tests if the old version of testeff is used."""
    if "camera-efficiency" in config["APPLICATION"] and not _new_testeff_version():
        return (
            "Any applications calling the old version of testeff are skipped "
            "due to a limitation of the old testeff not allowing to specify "
            "the include directory. Please update your sim_telarray tarball."
        )
    return None


def _new_testeff_version():
    """
    Testeff has been updated to allow to specify the include directory.

    This test checks if the new version is used.
    """
    testeff_path = Path(os.getenv("SIMTOOLS_SIMTEL_PATH")) / "sim_telarray/testeff.c"
    try:
        with open(testeff_path, encoding="utf-8") as file:
            file_content = file.read()
            if (
                "/* Combine the include paths such that those from '-I...' options */"
                in file_content
            ):
                return True
            return False
    except FileNotFoundError as exc:
        raise FileNotFoundError("The testeff executable could not be found.") from exc
