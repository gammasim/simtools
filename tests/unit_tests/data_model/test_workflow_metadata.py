#!/usr/bin/python3

from pathlib import Path
from unittest import mock

import yaml

from simtools.data_model import workflow_metadata


def test_build_workflow_activity_metadata_uses_uncleaned_metadata(monkeypatch):
    mock_collector = mock.Mock()
    mock_collector.get_top_level_metadata.return_value = {"cta": {"activity": {"id": "wf-id"}}}
    metadata_collector_cls = mock.Mock(return_value=mock_collector)
    monkeypatch.setattr(
        "simtools.data_model.workflow_metadata.MetadataCollector", metadata_collector_cls
    )

    activity = workflow_metadata.build_workflow_activity_metadata(
        args_dict={"config_file": "dummy.yml"},
        workflow_activity_id="wf-id",
        workflow_start=mock.Mock(isoformat=mock.Mock(return_value="2026-01-01T00:00:00+00:00")),
        workflow_end=mock.Mock(isoformat=mock.Mock(return_value="2026-01-01T00:00:01+00:00")),
        runtime_environment={"image": "test-image"},
        workflow_site="North",
        workflow_instrument="LSTN-design",
    )

    metadata_collector_cls.assert_called_once()
    assert metadata_collector_cls.call_args.args[0]["activity_id"] == "wf-id"
    assert metadata_collector_cls.call_args.args[0]["site"] == "North"
    assert metadata_collector_cls.call_args.args[0]["instrument"] == "LSTN-design"
    assert metadata_collector_cls.call_args.kwargs["clean_meta"] is False
    assert activity == {"id": "wf-id"}


def test_update_model_parameter_metadata_file(tmp_test_directory):
    metadata_file = tmp_test_directory / "pm.meta.yml"
    metadata_dict = {
        "cta": {
            "product": {"id": "prod-id"},
            "activity": {"id": "old-id"},
            "context": {"associated_activities": [{"name": "old", "activity_id": "old-id"}]},
        }
    }
    metadata_file.write_text(yaml.safe_dump(metadata_dict), encoding="utf-8")

    workflow_activity = {"id": "workflow-id", "name": "setting_workflow"}
    associated_activities = [
        {"name": "app1", "activity_id": "a1"},
        {"name": "app2", "activity_id": "a2"},
    ]

    workflow_metadata.update_model_parameter_metadata_file(
        metadata_file=metadata_file,
        workflow_activity=workflow_activity,
        associated_activities=associated_activities,
        logger=mock.Mock(),
    )

    updated = yaml.safe_load(metadata_file.read_text(encoding="utf-8"))
    assert updated["cta"]["product"]["id"] == "prod-id"
    assert updated["cta"]["activity"]["id"] == "workflow-id"
    assert updated["cta"]["context"]["associated_activities"] == [
        {"name": "old", "activity_id": "old-id"},
        {"name": "app1", "activity_id": "a1"},
        {"name": "app2", "activity_id": "a2"},
    ]


def test_update_model_parameter_metadata_file_missing_file():
    logger = mock.Mock()
    workflow_metadata.update_model_parameter_metadata_file(
        metadata_file=Path("missing.meta.yml"),
        workflow_activity={"id": "workflow-id"},
        associated_activities=[],
        logger=logger,
    )
    logger.debug.assert_called_once()
