from simtools.util import data_model


def test_top_level_reference_schema():
    """(very hard to test this)"""

    _top_meta = data_model.top_level_reference_schema()

    assert isinstance(_top_meta, dict)
    assert len(_top_meta) > 0


def test_user_input_reference_schema():
    """(very hard to test this)"""

    _top_ref = data_model.user_input_reference_schema()

    assert isinstance(_top_ref, dict)
    assert len(_top_ref) > 0


def test_workflow_configuration_schema():
    """(very hard to test this)"""

    _config = data_model.workflow_configuration_schema()

    assert isinstance(_config, dict)
    assert len(_config) > 0
