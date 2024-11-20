"""Helper functions for the production configuration tools."""

import os

import simtools.utils.general as gen


def load_metrics(file_path: str) -> dict:
    """
    Load metrics from a YAML file or dict.

    This function reads the metrics defined in the given YAML file and returns
    them as a dictionary.

    Parameters
    ----------
    file_path : str
        Path to the metrics YAML file.

    Returns
    -------
    dict
        Dictionary of metrics loaded from the YAML file.
    """
    if file_path and os.path.exists(file_path):
        return gen.collect_data_from_file_or_dict(file_name=file_path, in_dict=None)
    return {}
