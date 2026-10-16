"""Generic helpers for matching and validating supported file types."""

from functools import lru_cache
from importlib.resources import files
from pathlib import Path

import jsonschema
import yaml


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
    """Check catalog structure and product references."""
    if not isinstance(catalog, dict) or set(catalog) != {"formats", "products"}:
        raise ValueError("File catalog must contain formats and products mappings.")

    formats = catalog["formats"]
    products = catalog["products"]
    if not isinstance(formats, dict) or not isinstance(products, dict):
        raise ValueError("File catalog formats and products must be mappings.")
    if any(not isinstance(suffixes, list) or not suffixes for suffixes in formats.values()):
        raise ValueError("Each file format must declare a non-empty suffix list.")

    for product_id, product in products.items():
        _validate_file_product(product_id, product, products, formats)


def _validate_file_product(product_id, product, products, formats):
    """Validate one product declaration and its format reference."""
    if not isinstance(product, dict):
        raise ValueError(f"File product '{product_id}' must be a mapping.")
    suffixes = _product_suffixes(product_id, product, products, formats)
    _validate_product_naming(product_id, product, suffixes)
    if product.get("role") not in {"input", "intermediate", "result", "diagnostic", "metadata"}:
        raise ValueError(f"File product '{product_id}' has an invalid role.")
    if product.get("retention") not in {"persistent", "temporary", "workflow_defined"}:
        raise ValueError(f"File product '{product_id}' has an invalid retention value.")


def _validate_product_naming(product_id, product, suffixes):
    """Validate naming fields and the generated suffix for one product."""
    generated_suffix = product.get("default_suffix")
    if generated_suffix is not None and generated_suffix not in suffixes:
        raise ValueError(
            f"Generated suffix for file product '{product_id}' is not an accepted suffix."
        )
    has_suffixes = "suffixes" in product
    has_suffix_source = "suffixes_from" in product
    has_filename = "filename" in product
    if has_suffixes and has_suffix_source:
        raise ValueError(f"File product '{product_id}' has conflicting suffix fields.")
    if has_filename and (has_suffixes or has_suffix_source):
        raise ValueError(f"File product '{product_id}' has conflicting naming fields.")
    if not has_filename and not (has_suffixes or has_suffix_source):
        raise ValueError(f"File product '{product_id}' has no naming or suffix fields.")
    if has_filename and not product["filename"]:
        raise ValueError(f"File product '{product_id}' has an empty filename.")
    if not has_filename and not suffixes:
        raise ValueError(f"File product '{product_id}' must define accepted suffixes.")


def _product_suffixes(product_id, product=None, products=None, formats=None):
    """Resolve suffixes for one catalog product."""
    catalog = _file_catalog() if products is None else None
    products = catalog["products"] if products is None else products
    formats = catalog["formats"] if formats is None else formats
    product = products.get(product_id) if product is None else product
    if product is None:
        raise ValueError(f"Unknown file product '{product_id}'.")
    if "suffixes_from" in product:
        format_id = product["suffixes_from"]
        if format_id in formats:
            return tuple(formats[format_id])
        return _product_suffixes(format_id, products=products, formats=formats)
    return tuple(product.get("suffixes", ()))


FILE_TYPE_SUFFIXES = {
    file_type: tuple(suffixes) for file_type, suffixes in _file_catalog()["formats"].items()
}


def _get_product_value(product_id, key):
    """Return one value from a registered file product."""
    try:
        return _file_catalog()["products"][product_id][key]
    except KeyError as exc:
        raise ValueError(f"Unknown file product '{product_id}'.") from exc


def _get_product_suffixes(product_id):
    """Return accepted suffixes for one registered file product."""
    return _product_suffixes(product_id)


def _get_product_filename(product_id):
    """Return a fixed filename for one registered file product."""
    filename = _get_product_value(product_id, "filename")
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
