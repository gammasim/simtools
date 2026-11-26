"""
SHA-256 checksums of data product files.

The checksum is the SHA-256 digest of the file bytes. For files that embed their metadata
(ecsv ``# meta:`` header), the single header line ``sha256: <64 hex digits>`` (optionally
quoted and prefixed by ``#``) is removed, including its line ending, before hashing, so the
checksum can be written into the file it describes. Other files (json, yml, hdf5, ...) are
hashed in full and their checksum is stored in the sidecar metadata
(``ctao.product.sha256``). Plots and markdown reports carry no checksum.
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


def _checksum_match(data, file_name):
    """Return the embedded checksum line match (None if absent; at most one allowed)."""
    matches = list(_EMBEDDED_CHECKSUM_LINE.finditer(data))
    if len(matches) > 1:
        raise ValueError(f"More than one embedded sha256 line found in {file_name}.")
    return matches[0] if matches else None


def _sha256_of_bytes(data, file_name):
    """Return the checksum of file content, excluding an embedded checksum line."""
    match = _checksum_match(data, file_name)
    if match is not None:
        data = data[: match.start()] + data[match.end() :]
    return hashlib.sha256(data).hexdigest()


def sha256_of_file(file_name):
    """Return the SHA-256 checksum of a data product file (see module documentation)."""
    return _sha256_of_bytes(Path(file_name).read_bytes(), file_name)


def embedded_sha256(file_name):
    """Return the checksum embedded in a data product file, or None if there is none."""
    match = _checksum_match(Path(file_name).read_bytes(), file_name)
    return match.group(1).decode("ascii") if match else None


def write_embedded_sha256(file_name):
    """
    Replace the checksum placeholder in an ecsv file by the file checksum and return it.

    The file must contain exactly one embedded checksum line. The digest replaces the
    placeholder in place, so the hashed content (without the line) is unchanged.
    """
    file_name = Path(file_name)
    data = file_name.read_bytes()
    match = _checksum_match(data, file_name)
    if match is None:
        raise ValueError(f"Expected one embedded sha256 line in {file_name}, found 0.")
    digest = _sha256_of_bytes(data, file_name)
    start, end = match.span(1)
    file_name.write_bytes(data[:start] + digest.encode("ascii") + data[end:])
    return digest


def verify_sha256(file_name, expected=None):
    """
    Return True if a data product matches its checksum.

    The expected digest defaults to the checksum embedded in the file; a ValueError is
    raised if none is given and none is embedded.
    """
    if expected is None:
        expected = embedded_sha256(file_name)
        if expected is None:
            raise ValueError(f"No expected sha256 given and none embedded in {file_name}.")
    return sha256_of_file(file_name) == expected
