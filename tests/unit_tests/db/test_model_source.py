"""Tests for the MongoDB model-source adapter."""

from unittest.mock import Mock

from simtools.db.model_source import MongoDBModelSource


def test_mongodb_source_delegates_model_operations():
    """The adapter delegates source operations to the database handler."""
    handler = Mock(model_source_name="simulation-model-db")
    handler.get_model_versions.return_value = ["1.0.0"]
    handler.read_production_table_from_db.return_value = {"collection": "telescopes"}
    source = MongoDBModelSource(handler)

    assert source.source_name == "simulation-model-db"
    assert source.get_model_versions("telescopes") == ["1.0.0"]
    assert source.read_production_table("telescopes", "1.0.0") == {"collection": "telescopes"}
    handler.get_model_versions.assert_called_once_with("telescopes")
    handler.read_production_table_from_db.assert_called_once_with("telescopes", "1.0.0")


def test_mongodb_source_builds_parameter_query_and_returns_documents():
    """Parameter lookup translates source-neutral filters into a DB query."""
    handler = Mock(model_source_name="simulation-model-db")
    handler._read_db.return_value = {  # pylint: disable=protected-access
        "diameter": {"parameter": "diameter", "parameter_version": "1.0.0"}
    }
    source = MongoDBModelSource(handler)

    result = source.read_parameters(
        {"diameter": "1.0.0"}, "telescopes", instrument="LSTN-01", site="North"
    )

    assert result == [{"parameter": "diameter", "parameter_version": "1.0.0"}]
    handler._read_db.assert_called_once_with(  # pylint: disable=protected-access
        {
            "$or": [{"parameter": "diameter", "parameter_version": "1.0.0"}],
            "instrument": "LSTN-01",
            "site": "North",
        },
        "telescopes",
    )
