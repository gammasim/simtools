import pytest

from simtools.data_model import metadata_keys

_BLOCK = {"product": {"id": "abc"}}
_OLD = {"product": {"id": "old"}}


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ({"ctao": _BLOCK}, _BLOCK),
        ({"CTAO": _BLOCK}, _BLOCK),
        ({"cta": _BLOCK}, _BLOCK),
        ({"CTA": _BLOCK}, _BLOCK),
        ({"cta": _OLD, "ctao": _BLOCK}, _BLOCK),
        ({"site": "North"}, {}),
    ],
)
def test_get_observatory_metadata(metadata, expected):
    assert metadata_keys.get_observatory_metadata(metadata) == expected


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ({"cta": _BLOCK, "site": "North"}, {"ctao": _BLOCK, "site": "North"}),
        ({"ctao": _BLOCK, "cta": _OLD}, {"ctao": _BLOCK, "cta": _OLD}),
        ({"site": "North"}, {"site": "North"}),
    ],
)
def test_normalize_observatory_key(metadata, expected):
    original = dict(metadata)

    assert metadata_keys.normalize_observatory_key(metadata) == expected
    assert metadata == original
