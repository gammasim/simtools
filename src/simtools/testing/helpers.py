"""Helper functions for integration testing."""


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
