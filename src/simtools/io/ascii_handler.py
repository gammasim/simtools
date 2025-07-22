"""Helper module for ASCII file operations."""

import json
import logging
import tempfile
import urllib.request
from pathlib import Path

import astropy.units as u
import numpy as np
import yaml

from simtools.utils.general import is_url

_logger = logging.getLogger(__name__)


def collect_data_from_file(file_name, yaml_document=None):
    """
    Collect data from file based on its extension.

    Parameters
    ----------
    file_name: str
        Name of the yaml/json/ascii file.
    yaml_document: None, int
        Return list of yaml documents or a single document (for yaml files with several documents).

    Returns
    -------
    data: dict or list
        Data as dict or list.
    """
    if is_url(file_name):
        return collect_data_from_http(file_name)

    suffix = Path(file_name).suffix.lower()
    try:
        with open(file_name, encoding="utf-8") as file:
            return _collect_data_from_different_file_types(file, file_name, suffix, yaml_document)
    # broad exception to catch all possible errors in reading the file
    except Exception as exc:  # pylint: disable=broad-except
        raise type(exc)(f"Failed to read file {file_name}: {exc}") from exc


def _collect_data_from_different_file_types(file, file_name, suffix, yaml_document):
    """Collect data from different file types."""
    if suffix == ".json":
        return json.load(file)
    if suffix in (".list", ".txt"):
        return [line.strip() for line in file.readlines()]
    if suffix in [".yml", ".yaml"]:
        return _collect_data_from_yaml_file(file, file_name, yaml_document)
    raise TypeError(f"File type {suffix} not supported.")


def _collect_data_from_yaml_file(file, file_name, yaml_document):
    """Collect data from a yaml file (allow for multi-document yaml files)."""
    try:
        return yaml.safe_load(file)
    except yaml.constructor.ConstructorError:
        return _load_yaml_using_astropy(file)
    except yaml.composer.ComposerError:
        pass
    file.seek(0)
    if yaml_document is None:
        return list(yaml.safe_load_all(file))
    try:
        return list(yaml.safe_load_all(file))[yaml_document]
    except IndexError as exc:
        raise IndexError(
            f"YAML file {file_name} does not contain {yaml_document} documents."
        ) from exc


def _load_yaml_using_astropy(file):
    """
    Load a yaml file using astropy's yaml loader.

    Parameters
    ----------
    file: file
        File to be loaded.

    Returns
    -------
    dict
        Dictionary containing the file content.
    """
    # pylint: disable=import-outside-toplevel
    import astropy.io.misc.yaml as astropy_yaml

    file.seek(0)
    return astropy_yaml.load(file)


def collect_data_from_http(url):
    """
    Download yaml or json file from url and return it contents as dict.

    File is downloaded as a temporary file and deleted afterwards.

    Parameters
    ----------
    url: str
        URL of the yaml/json file.

    Returns
    -------
    dict
        Dictionary containing the file content.

    Raises
    ------
    TypeError
        If url is not a valid URL.
    FileNotFoundError
        If downloading the yaml file fails.

    """
    try:
        with tempfile.NamedTemporaryFile(mode="w+t") as tmp_file:
            urllib.request.urlretrieve(url, tmp_file.name)
            data = _collect_data_from_different_file_types(
                tmp_file, url, Path(url).suffix.lower(), None
            )
    except TypeError as exc:
        raise TypeError(f"Invalid url {url}") from exc
    except urllib.error.HTTPError as exc:
        raise FileNotFoundError(f"Failed to download file from {url}") from exc

    _logger.debug(f"Downloaded file from {url}")
    return data


def read_file_encoded_in_utf_or_latin(file_name):
    """
    Read a file encoded in UTF-8 or Latin-1.

    Parameters
    ----------
    file_name: str
        Name of the file to be read.

    Returns
    -------
    list
        List of lines read from the file.

    Raises
    ------
    UnicodeDecodeError
        If the file cannot be decoded using UTF-8 or Latin-1.
    """
    try:
        with open(file_name, encoding="utf-8") as file:
            lines = file.readlines()
    except UnicodeDecodeError:
        _logger.debug("Unable to decode file using UTF-8. Trying Latin-1.")
        try:
            with open(file_name, encoding="latin-1") as file:
                lines = file.readlines()
        except UnicodeDecodeError as exc:
            msg = f"Unable to decode file {file_name} using UTF-8 or Latin-1."
            raise UnicodeDecodeError(exc.encoding, exc.object, exc.start, exc.end, msg) from exc

    return lines


def is_utf8_file(file_name):
    """
    Check if a file is encoded in UTF-8.

    Parameters
    ----------
    file_name: str, Path
        Name of the file to be checked.

    Returns
    -------
    bool
        True if the file is encoded in UTF-8, False otherwise.
    """
    try:
        with open(file_name, encoding="utf-8") as file:
            file.read()
        return True
    except UnicodeDecodeError:
        return False


def write_data_to_file(data, output_file, sort_keys=False, numpy_types=False):
    """
    Write structured data to JSON or YAML file.

    The file type is determined by the file extension.

    Parameters
    ----------
    data: dict or list
        Data to be written to the file.
    output_file: str or Path
        Name of the file to be written.
    sort_keys: bool, optional
        If True, sort the keys.
    numpy_types: bool, optional
        If True, convert numpy types to native Python types.
    """
    output_file = Path(output_file)
    if output_file.suffix.lower() == ".json":
        return _write_to_json(data, output_file, sort_keys, numpy_types)
    if output_file.suffix.lower() in [".yml", ".yaml"]:
        return _write_to_yaml(data, output_file, sort_keys)

    raise ValueError(
        f"Unsupported file type {output_file.suffix}. Only .json, .yml, and .yaml are supported."
    )


def _write_to_json(data, output_file, sort_keys, numpy_types):
    """
    Write data to a JSON file.

    Parameters
    ----------
    data: dict or list
        Data to be written to the file.
    output_file: Path
        Name of the file to be written.
    sort_keys: bool
        If True, sort the keys.
    numpy_types: bool
        If True, convert numpy types to native Python types.
    """
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=4,
            sort_keys=sort_keys,
            cls=JsonNumpyEncoder if numpy_types else None,
        )
        file.write("\n")


def _write_to_yaml(data, output_file, sort_keys):
    """
    Write data to a YAML file.

    Parameters
    ----------
    data: dict or list
        Data to be written to the file.
    output_file: Path
        Name of the file to be written.
    sort_keys: bool
        If True, sort the keys.

    """
    with open(output_file, "w", encoding="utf-8") as file:
        yaml.dump(data, file, indent=4, sort_keys=sort_keys, explicit_start=True)


class JsonNumpyEncoder(json.JSONEncoder):
    """Convert numpy to python types as accepted by json.dump."""

    def default(self, o):
        """Return default encoder."""
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, u.core.CompositeUnit | u.core.IrreducibleUnit | u.core.Unit):
            return str(o) if o != u.dimensionless_unscaled else None
        if np.issubdtype(type(o), np.bool_):
            return bool(o)
        return super().default(o)
