import json
import logging

import numpy as np
import pytest

from simtools.production_configuration.scale_events_manager import ScaleEventsManager


@pytest.fixture
def args_dict(tmp_path):
    """Fixture to provide a mock args_dict for testing."""
    return {
        "base_path": tmp_path,
        "zeniths": [20, 40],
        "offsets": [0.5, 1.0],
        "query_point": [1.0, 180.0, 20.0, 4.0, 0.5],
        "output_file": "scaled_events.json",
        "metrics_file": str(tmp_path / "metrics.yml"),
        "file_name_template": "mock_file_zenith_{zenith}.fits",
    }


@pytest.fixture
def mock_metrics_file(tmp_path):
    """Fixture to create a mock metrics file."""
    metrics_file = tmp_path / "metrics.yml"
    metrics_file.write_text("mock_metrics: true\n")
    return metrics_file


@pytest.fixture
def mock_event_files(tmp_path):
    """Fixture to create mock event files."""
    for zenith in [20, 40]:
        file_path = tmp_path / f"mock_file_zenith_{zenith}.fits"
        file_path.write_text("Mock event data")
    return tmp_path


def test_initialize_evaluators(args_dict, mock_metrics_file, mock_event_files):
    """Test the initialization of evaluators."""
    args_dict["metrics_file"] = str(mock_metrics_file)
    manager = ScaleEventsManager(args_dict)

    manager.initialize_evaluators()

    # Check that evaluators are initialized
    assert len(manager.evaluator_instances) == 4  # 2 zeniths x 2 offsets
    for evaluator in manager.evaluator_instances:
        assert evaluator.file_path.exists()
        assert evaluator.file_path.suffix == ".fits"


def test_perform_interpolation(args_dict, mock_metrics_file, mock_event_files):
    """Test the interpolation functionality."""
    args_dict["metrics_file"] = str(mock_metrics_file)
    manager = ScaleEventsManager(args_dict)

    manager.initialize_evaluators()
    scaled_events = manager.perform_interpolation()

    # Check that interpolation results are returned
    assert scaled_events is not None
    assert isinstance(scaled_events, np.ndarray)
    assert scaled_events.shape == (1,)  # Single query point


def test_write_output(args_dict, mock_metrics_file, mock_event_files, tmp_path):
    """Test writing the interpolation results to a file."""
    args_dict["metrics_file"] = str(mock_metrics_file)
    args_dict["output_file"] = "test_output.json"
    manager = ScaleEventsManager(args_dict)

    manager.initialize_evaluators()
    scaled_events = np.array([123.45])  # Mock interpolation result
    manager.write_output(scaled_events)

    # Check that the output file is created
    output_file = tmp_path / "test_output.json"
    assert output_file.exists()

    # Check the contents of the output file
    with open(output_file, encoding="utf-8") as f:
        output_data = json.load(f)
        assert "query_point" in output_data
        assert "scaled_events" in output_data
        assert output_data["scaled_events"] == [123.45]


def test_run_workflow(args_dict, mock_metrics_file, mock_event_files, tmp_path, caplog):
    """Test the full workflow of the ScaleEventsManager."""
    args_dict["metrics_file"] = str(mock_metrics_file)
    args_dict["output_file"] = "workflow_output.json"
    manager = ScaleEventsManager(args_dict)

    with caplog.at_level(logging.INFO):
        manager.run()

    # Check that the output file is created
    output_file = tmp_path / "workflow_output.json"
    assert output_file.exists()

    # Check the logs for workflow execution
    assert "args dict" in caplog.text
    assert "Output saved to" in caplog.text
