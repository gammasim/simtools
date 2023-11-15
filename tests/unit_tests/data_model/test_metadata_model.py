from simtools.data_model import metadata_model


def test_top_level_reference_schema():
    _top_meta = metadata_model.top_level_reference_schema()

    assert isinstance(_top_meta, dict)
    assert len(_top_meta) > 0

    assert "VERSION" in _top_meta["CTA"]["REFERENCE"]
