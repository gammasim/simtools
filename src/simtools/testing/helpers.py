"""Helper functions for integration testing."""

from simtools import settings


def skip_camera_efficiency(config):
    """Skip camera efficiency tests if the old version of testeff is used."""
    if "camera-efficiency" in config["application"] and not _new_testeff_version():
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
    testeff_path = settings.config.sim_telarray_path / "testeff.c"
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


def skip_multiple_version_test(config, model_version):
    """Skip a test which is not meant for multiple versions if multiple versions are given."""
    message = "Skipping test not meant for multiple model versions."

    if not isinstance(model_version, list):
        return None

    config_model_version = config.get("configuration", {}).get("model_version", [])

    if not isinstance(config_model_version, list):
        config_model_version = [config_model_version]

    if 1 < len(model_version) != len(config_model_version):
        return message

    return None
