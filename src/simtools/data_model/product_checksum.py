"""
SHA-256 checksums of data product files.

Checksum rule (the same for every format):

- The checksum is the SHA-256 digest of the file bytes. For files that embed their
  metadata (ecsv ``# meta:`` header), the single header line ``sha256: <64 hex digits>``
  (optionally quoted and prefixed by ``#``) is removed, including its line ending, before
  hashing. The checksum therefore does not depend on its own value and can be written into
  the file it describes.
- Files without an embedded checksum line (json, yml, hdf5, ...) are hashed in full. Their
  checksum is stored in the sidecar metadata (``cta.product.sha256``).
- Plots and markdown reports carry no checksum.
"""

import hashlib
import re
from pathlib import Path

SHA256_KEY = "sha256"
CHECKSUM_PLACEHOLDER = "0" * 64

# Embedded checksum line (ecsv header). Group 1 is the digest value.
_EMBEDDED_CHECKSUM_LINE = re.compile(
    rb"^[ \t#]*sha256:[ \t]*['\"]?([0-9a-fA-F]{64})['\"]?[ \t]*\r?\n",
    re.MULTILINE,
)


def _remove_embedded_checksum(data):
    """Return file bytes without the embedded checksum line (at most one allowed)."""
    matches = list(_EMBEDDED_CHECKSUM_LINE.finditer(data))
    if len(matches) > 1:
        raise ValueError("More than one embedded sha256 line found in data product.")
    if not matches:
        return data
    return data[: matches[0].start()] + data[matches[0].end() :]


def sha256_of_bytes(data):
    """
    Return the checksum of file content, excluding an embedded checksum line.

    Parameters
    ----------
    data: bytes
        File content.

    Returns
    -------
    str
        Hexadecimal SHA-256 digest.
    """
    return hashlib.sha256(_remove_embedded_checksum(data)).hexdigest()


def sha256_of_file(file_name):
    """
    Return the checksum of a data product file (see module documentation for the rule).

    Parameters
    ----------
    file_name: str or Path
        Path of the data product.

    Returns
    -------
    str
        Hexadecimal SHA-256 digest.
    """
    return sha256_of_bytes(Path(file_name).read_bytes())


def embedded_sha256(file_name):
    """
    Return the checksum embedded in a data product file, or None if there is none.

    Parameters
    ----------
    file_name: str or Path
        Path of the data product.

    Returns
    -------
    str or None
        Hexadecimal SHA-256 digest stored in the file.

    Raises
    ------
    ValueError
        If more than one checksum line is found.
    """
    matches = list(_EMBEDDED_CHECKSUM_LINE.finditer(Path(file_name).read_bytes()))
    if len(matches) > 1:
        raise ValueError(f"More than one embedded sha256 line found in {file_name}.")
    return matches[0].group(1).decode("ascii") if matches else None


def write_embedded_sha256(file_name):
    """
    Replace the checksum placeholder in an ecsv file with the file checksum.

    The file must contain exactly one embedded checksum line (written with the placeholder
    value). The digest replaces the placeholder value in place, so the file size does not
    change and the hashed content (without the line) is unchanged.

    Parameters
    ----------
    file_name: str or Path
        Path of the data product.

    Returns
    -------
    str
        Hexadecimal SHA-256 digest written to the file.

    Raises
    ------
    ValueError
        If the file contains no unique embedded checksum line.
    """
    file_name = Path(file_name)
    data = file_name.read_bytes()
    matches = list(_EMBEDDED_CHECKSUM_LINE.finditer(data))
    if len(matches) != 1:
        raise ValueError(f"Expected one embedded sha256 line in {file_name}, found {len(matches)}.")
    digest = sha256_of_bytes(data)
    start, end = matches[0].span(1)
    file_name.write_bytes(data[:start] + digest.encode("ascii") + data[end:])
    return digest


def verify_sha256(file_name, expected=None):
    """
    Verify a data product against its checksum.

    Parameters
    ----------
    file_name: str or Path
        Path of the data product.
    expected: str, optional
        Expected hexadecimal SHA-256 digest (e.g., from the sidecar metadata). If None,
        the checksum embedded in the file is used.

    Returns
    -------
    bool
        True if the file checksum matches the expected digest.

    Raises
    ------
    ValueError
        If no expected digest is given and the file embeds none.
    """
    if expected is None:
        expected = embedded_sha256(file_name)
        if expected is None:
            raise ValueError(f"No expected sha256 given and none embedded in {file_name}.")
    return sha256_of_file(file_name) == expected
