"""Tests for the MongoDB model-source adapter."""

from unittest.mock import Mock

from simtools.db.model_source import MongoDBModelSource


def test_mongodb_source_delegates_model_operations():
    """The adapter delegates source operations to the database handler."""
    handler = Mock(model_source_name="simulation-model-db")
    handler.is_configured.return_value = True
    handler.get_model_versions.return_value = ["1.0.0"]
    handler.read_production_table_from_db.return_value = {"collection": "telescopes"}
    source = MongoDBModelSource(handler)

    assert source.source_name == "simulation-model-db"
    assert source.is_configured() is True
    assert source.get_model_versions("telescopes") == ["1.0.0"]
    assert source.read_production_table("telescopes", "1.0.0") == {"collection": "telescopes"}
    handler.get_model_versions.assert_called_once_with("telescopes")
    handler.read_production_table_from_db.assert_called_once_with("telescopes", "1.0.0")
    handler.export_model_files.return_value = {"model.dat": "copied"}
    handler.get_ecsv_file_as_astropy_table.return_value = "table"
    assert source.export_model_files(file_names="model.dat") == {"model.dat": "copied"}
    assert source.get_ecsv_file_as_astropy_table("model.ecsv") == "table"


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
    handler._read_db.reset_mock()  # pylint: disable=protected-access
    handler._read_db.return_value = {}  # pylint: disable=protected-access
    assert source.read_parameters({}, "configuration_corsika", instrument="xSTx-design") == []


def test_mongodb_source_caches_and_copies_reads():
    """Repeated source reads use per-instance caches and defensive copies."""
    handler = Mock(model_source_name="db")
    handler.get_model_versions.return_value = ["1.0.0"]
    production = {"collection": "telescopes", "parameters": {"LSTN-01": {}}}
    handler.read_production_table_from_db.return_value = production
    handler._read_db.return_value = {"p": {"parameter": "p", "value": 1}}
    source = MongoDBModelSource(handler)

    versions = source.get_model_versions()
    versions.append("2.0.0")
    assert source.get_model_versions() == ["1.0.0"]
    table = source.read_production_table("telescopes", "1.0.0")
    table["parameters"]["LSTN-01"]["changed"] = True
    assert (
        "changed"
        not in source.read_production_table("telescopes", "1.0.0")["parameters"]["LSTN-01"]
    )
    parameters = source.read_parameters({"p": "1.0.0"}, "telescopes", instrument="LSTN-01")
    parameters[0]["changed"] = True
    assert (
        "changed"
        not in source.read_parameters({"p": "1.0.0"}, "telescopes", instrument="LSTN-01")[0]
    )
    handler.get_model_versions.assert_called_once_with("telescopes")
    handler.read_production_table_from_db.assert_called_once_with("telescopes", "1.0.0")
    assert handler._read_db.call_count == 1  # pylint: disable=protected-access
