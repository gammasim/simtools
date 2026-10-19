"""Generic helpers for matching and validating supported file types."""

from functools import lru_cache
from importlib.resources import files
from pathlib import Path

import jsonschema
import yaml

_FORMATS = "formats"
_PRODUCTS = "products"
_FILENAME = "filename"
_UNKNOWN_PRODUCT = "Unknown file product '{product_id}'."


@lru_cache(maxsize=1)
def _file_catalog():
    """Load and validate the packaged file-format and product declarations."""
    catalog_path = files("simtools").joinpath("resources/file_types.yml")
    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    _validate_file_catalog_schema(catalog)
    _validate_file_catalog(catalog)
    return catalog


def _validate_file_catalog_schema(catalog):
    """Validate catalog fields against the packaged JSON Schema."""
    schema_path = files("simtools").joinpath("schemas/file_types.schema.yml")
    schema = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft7Validator(schema).validate(catalog)


def _validate_file_catalog(catalog):
    """Check references and defaults after schema validation."""
    formats = catalog[_FORMATS]
    products = catalog[_PRODUCTS]
    for product_id, product in products.items():
        suffixes = _product_suffixes(product_id, products, formats)
        generated_suffix = product.get("default_suffix")
        if generated_suffix is not None and generated_suffix not in suffixes:
            raise ValueError(
                f"Generated suffix for file product '{product_id}' is not an accepted suffix."
            )
        if _FILENAME not in product and not suffixes:
            raise ValueError(f"File product '{product_id}' must define accepted suffixes.")


def _product_suffixes(product_id, products, formats):
    """Resolve suffixes for one catalog product."""
    product = products.get(product_id)
    if product is None:
        raise ValueError(_UNKNOWN_PRODUCT.format(product_id=product_id))
    if "suffixes_from" in product:
        format_id = product["suffixes_from"]
        if format_id in formats:
            return tuple(formats[format_id])
        return _product_suffixes(format_id, products, formats)
    return tuple(product.get("suffixes", ()))


FILE_TYPE_SUFFIXES = {
    file_type: tuple(suffixes) for file_type, suffixes in _file_catalog()[_FORMATS].items()
}


def _get_product_value(product_id, key):
    """Return one value from a registered file product."""
    try:
        return _file_catalog()[_PRODUCTS][product_id][key]
    except KeyError as exc:
        raise ValueError(_UNKNOWN_PRODUCT.format(product_id=product_id)) from exc


def _get_product_suffixes(product_id):
    """Return accepted suffixes for one registered file product."""
    catalog = _file_catalog()
    return _product_suffixes(product_id, catalog[_PRODUCTS], catalog[_FORMATS])


def _get_product_filename(product_id):
    """Return a fixed filename for one registered file product."""
    catalog = _file_catalog()
    if product_id not in catalog[_PRODUCTS]:
        raise ValueError(_UNKNOWN_PRODUCT.format(product_id=product_id))
    filename = catalog[_PRODUCTS][product_id].get(_FILENAME)
    if filename is None:
        raise ValueError(f"File product '{product_id}' does not define a fixed filename.")
    return filename


def validate_file_type(file_path, file_type):
    """
    Validate that a file has one of the expected suffixes.

    Parameters
    ----------
    file_path : str or Path
        Path to the file to validate.
    file_type : str
        Registered file type name, e.g., "hdf5", "json_or_yaml",
    """
    path = Path(file_path)
    expected_suffixes = _suffixes_for_file_type(file_type)
    if not matches_suffix(path, expected_suffixes):
        raise ValueError(
            f"File '{file_path}' has unsupported suffix, expected one of {expected_suffixes}"
        )
    return path


def _suffixes_for_file_type(file_type):
    """Return registered suffixes for a named file type."""
    try:
        return FILE_TYPE_SUFFIXES[file_type]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported file type '{file_type}', expected one of {sorted(FILE_TYPE_SUFFIXES)}"
        ) from exc


def matches_suffix(file_path, expected_suffixes):
    """Return whether the path ends with one of the expected suffix patterns."""
    name = Path(file_path).name.lower()
    return any(name.endswith(suffix) for suffix in expected_suffixes)


def is_file_type(file_path, file_type):
    """Return whether the path matches one registered file type."""
    return matches_suffix(file_path, _suffixes_for_file_type(file_type))


def looks_like_text_file(file_path, sample_size=4096):
    """Return whether the file appears to be UTF-8 text."""
    try:
        sample = Path(file_path).read_bytes()[:sample_size]
    except OSError:
        return False
    if b"\x00" in sample:
        return False
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True
