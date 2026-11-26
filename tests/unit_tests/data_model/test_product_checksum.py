import hashlib
from pathlib import Path

import pytest

from simtools.data_model import product_checksum as pc

ECSV_HEADER = (
    b"# %ECSV 1.0\n"
    b"# meta: !!omap\n"
    b"#       sha256: " + pc.CHECKSUM_PLACEHOLDER.encode() + b"\n"
    b"# schema: astropy-2.0\n"
    b"pixel\n"
    b"25\n"
)
_WITHOUT_LINE = b"# %ECSV 1.0\n# meta: !!omap\n# schema: astropy-2.0\npixel\n25\n"


@pytest.mark.parametrize(
    "line",
    [
        b"#   sha256: 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'\n",
        b"sha256: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n",
        b'#   sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"\n',
        b"",
    ],
)
def test_sha256_of_file_excludes_embedded_line(tmp_test_directory, line):
    file_name = Path(tmp_test_directory) / "variant.ecsv"
    file_name.write_bytes(b"# header\n" + line + b"data\n")

    assert pc.embedded_sha256(file_name) == ("a" * 64 if line else None)
    assert pc.sha256_of_file(file_name) == hashlib.sha256(b"# header\ndata\n").hexdigest()


def test_write_embedded_sha256_replaces_placeholder_in_place(tmp_test_directory):
    file_name = Path(tmp_test_directory) / "table.ecsv"
    file_name.write_bytes(ECSV_HEADER)
    size_before = file_name.stat().st_size

    digest = pc.write_embedded_sha256(file_name)

    assert digest == hashlib.sha256(_WITHOUT_LINE).hexdigest()
    assert pc.embedded_sha256(file_name) == digest
    assert file_name.stat().st_size == size_before
    assert pc.verify_sha256(file_name)
    file_name.write_bytes(file_name.read_bytes().replace(b"25\n", b"26\n"))
    assert not pc.verify_sha256(file_name)


@pytest.mark.parametrize(
    ("content", "match"),
    [
        (b"# header\ndata\n", "Expected one embedded sha256 line"),
        (ECSV_HEADER + ECSV_HEADER, "More than one embedded sha256 line"),
    ],
)
def test_write_embedded_sha256_requires_exactly_one_line(tmp_test_directory, content, match):
    file_name = Path(tmp_test_directory) / "lines.ecsv"
    file_name.write_bytes(content)

    with pytest.raises(ValueError, match=match):
        pc.write_embedded_sha256(file_name)


def test_verify_sha256_with_expected_value(tmp_test_directory):
    file_name = Path(tmp_test_directory) / "data.json"
    file_name.write_text('{"value": 1}\n', encoding="utf-8")

    assert pc.verify_sha256(file_name, expected=pc.sha256_of_file(file_name))
    assert not pc.verify_sha256(file_name, expected="0" * 64)
    with pytest.raises(ValueError, match=r"No expected sha256 given"):
        pc.verify_sha256(file_name)
