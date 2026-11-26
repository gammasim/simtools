import pytest

from simtools.data_model import metadata_keys

_BLOCK = {"product": {"id": "abc"}}


@pytest.mark.parametrize("key", ["ctao", "CTAO", "cta", "CTA"])
def test_get_observatory_metadata_reads_ctao_and_legacy_cta(key):
    assert metadata_keys.get_observatory_metadata({key: _BLOCK}) == _BLOCK


def test_get_observatory_metadata_prefers_ctao():
    metadata = {"cta": {"product": {"id": "old"}}, "ctao": _BLOCK}

    assert metadata_keys.get_observatory_metadata(metadata) == _BLOCK


def test_get_observatory_metadata_without_block():
    assert metadata_keys.get_observatory_metadata({"site": "North"}) == {}


@pytest.mark.parametrize(("key", "expected"), [("cta", "ctao"), ("CTA", "CTAO")])
def test_normalize_observatory_key_renames_legacy_key(key, expected):
    normalized = metadata_keys.normalize_observatory_key({key: _BLOCK, "site": "North"})

    assert normalized == {expected: _BLOCK, "site": "North"}


def test_normalize_observatory_key_keeps_existing_ctao_block():
    metadata = {"ctao": _BLOCK, "cta": {"product": {"id": "old"}}}

    assert metadata_keys.normalize_observatory_key(metadata) == metadata


def test_normalize_observatory_key_does_not_modify_input():
    metadata = {"cta": _BLOCK}

    metadata_keys.normalize_observatory_key(metadata)

    assert metadata == {"cta": _BLOCK}
