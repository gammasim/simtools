import hashlib
from pathlib import Path

import pytest

from simtools.data_model import product_checksum as pc

ECSV_HEADER = (
    b"# %ECSV 1.0\n"
    b"# ---\n"
    b"# meta: !!omap\n"
    b"# - cta:\n"
    b"#     product:\n"
    b"#       format: ecsv\n"
    b"#       sha256: " + pc.CHECKSUM_PLACEHOLDER.encode() + b"\n"
    b"# - title: test\n"
    b"# schema: astropy-2.0\n"
    b"pixel\n"
    b"25\n"
    b"30\n"
)


def _without_line(data):
    lines = data.splitlines(keepends=True)
    return b"".join(line for line in lines if b"sha256:" not in line)


def test_sha256_of_bytes_matches_plain_digest_without_checksum_line():
    data = b'{\n  "value": 1\n}\n'
    assert pc.sha256_of_bytes(data) == hashlib.sha256(data).hexdigest()


def test_sha256_of_bytes_excludes_embedded_checksum_line():
    with_line = ECSV_HEADER
    assert pc.sha256_of_bytes(with_line) == hashlib.sha256(_without_line(with_line)).hexdigest()
    other_value = with_line.replace(pc.CHECKSUM_PLACEHOLDER.encode(), b"a" * 64)
    assert pc.sha256_of_bytes(other_value) == pc.sha256_of_bytes(with_line)


@pytest.mark.parametrize(
    "line",
    [
        b"#   sha256: 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'\n",
        b"sha256: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n",
        b'#   sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"\n',
    ],
)
def test_embedded_line_variants_are_recognised(tmp_test_directory, line):
    file_name = Path(tmp_test_directory) / "variant.ecsv"
    file_name.write_bytes(b"# header\n" + line + b"data\n")
    assert pc.embedded_sha256(file_name) == "a" * 64
    assert pc.sha256_of_file(file_name) == hashlib.sha256(b"# header\ndata\n").hexdigest()


def test_write_embedded_sha256_replaces_placeholder_in_place(tmp_test_directory):
    file_name = Path(tmp_test_directory) / "table.ecsv"
    file_name.write_bytes(ECSV_HEADER)
    size_before = file_name.stat().st_size

    digest = pc.write_embedded_sha256(file_name)

    assert pc.embedded_sha256(file_name) == digest
    assert file_name.stat().st_size == size_before
    assert digest == hashlib.sha256(_without_line(ECSV_HEADER)).hexdigest()
    assert pc.verify_sha256(file_name)


def test_write_embedded_sha256_requires_exactly_one_line(tmp_test_directory):
    file_name = Path(tmp_test_directory) / "no_line.ecsv"
    file_name.write_bytes(b"# header\ndata\n")
    with pytest.raises(ValueError, match=r"Expected one embedded sha256 line"):
        pc.write_embedded_sha256(file_name)

    file_name.write_bytes(ECSV_HEADER + ECSV_HEADER)
    with pytest.raises(ValueError, match=r"found 2"):
        pc.write_embedded_sha256(file_name)


def test_embedded_sha256_rejects_multiple_lines(tmp_test_directory):
    file_name = Path(tmp_test_directory) / "two_lines.ecsv"
    file_name.write_bytes(ECSV_HEADER + ECSV_HEADER)
    with pytest.raises(ValueError, match=r"More than one embedded sha256 line"):
        pc.embedded_sha256(file_name)
    with pytest.raises(ValueError, match=r"More than one embedded sha256 line"):
        pc.sha256_of_bytes(ECSV_HEADER + ECSV_HEADER)


def test_embedded_sha256_is_none_without_line(tmp_test_directory):
    file_name = Path(tmp_test_directory) / "plain.json"
    file_name.write_text('{"value": 1}\n', encoding="utf-8")
    assert pc.embedded_sha256(file_name) is None


def test_verify_sha256_with_expected_value(tmp_test_directory):
    file_name = Path(tmp_test_directory) / "data.json"
    file_name.write_text('{"value": 1}\n', encoding="utf-8")
    digest = pc.sha256_of_file(file_name)
    assert pc.verify_sha256(file_name, expected=digest)
    assert not pc.verify_sha256(file_name, expected="0" * 64)


def test_verify_sha256_without_expected_value_requires_embedded_line(tmp_test_directory):
    file_name = Path(tmp_test_directory) / "data.json"
    file_name.write_text('{"value": 1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"No expected sha256 given"):
        pc.verify_sha256(file_name)


def test_verify_sha256_detects_modified_content(tmp_test_directory):
    file_name = Path(tmp_test_directory) / "table.ecsv"
    file_name.write_bytes(ECSV_HEADER)
    digest = pc.write_embedded_sha256(file_name)
    file_name.write_bytes(file_name.read_bytes().replace(b"25\n", b"26\n"))
    assert not pc.verify_sha256(file_name, expected=digest)
