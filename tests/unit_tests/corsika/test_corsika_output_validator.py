import pytest

from simtools.corsika.corsika_output_validator import (
    validate_corsika_output,
    validate_event_numbers,
    validate_log_files,
)


def test_validate_corsika_output_with_valid_files(tmp_path, mocker):
    """Test validation with valid data and log files."""
    data_file = tmp_path / "corsika.data"
    log_file = tmp_path / "corsika.log"

    mock_validate_events = mocker.patch(
        "simtools.corsika.corsika_output_validator.validate_event_numbers"
    )
    mock_validate_logs = mocker.patch(
        "simtools.corsika.corsika_output_validator.validate_log_files"
    )

    validate_corsika_output([data_file], [log_file], expected_shower_events=1000)

    mock_validate_events.assert_called_once()
    mock_validate_logs.assert_called_once()


def test_validate_corsika_output_with_none_data_files(tmp_path, mocker):
    """Test validation when data_files is None."""
    log_file = tmp_path / "corsika.log"

    mock_validate_events = mocker.patch(
        "simtools.corsika.corsika_output_validator.validate_event_numbers"
    )
    mock_validate_logs = mocker.patch(
        "simtools.corsika.corsika_output_validator.validate_log_files"
    )

    validate_corsika_output(None, [log_file], expected_shower_events=1000)

    mock_validate_events.assert_called_once_with([], 1000)
    mock_validate_logs.assert_called_once()


def test_validate_corsika_output_curved_atmosphere(tmp_path, mocker):
    """Test validation with curved atmosphere option enabled."""
    data_file = tmp_path / "corsika.data"
    log_file = tmp_path / "corsika.log"

    mocker.patch("simtools.corsika.corsika_output_validator.validate_event_numbers")
    mock_validate_log_files = mocker.patch(
        "simtools.corsika.corsika_output_validator.validate_log_files"
    )

    validate_corsika_output([data_file], [log_file], expected_shower_events=500, curved_atmo=True)

    mock_validate_log_files.assert_called_once_with(
        [log_file], expected_shower_events=500, curved_atmo=True
    )


def test_validate_event_numbers_with_matching_events(tmp_path, mocker):
    """Test event validation when counts match and tolerates small mismatches."""
    data_file = tmp_path / "corsika.data"

    mock_get_events = mocker.patch(
        "simtools.corsika.corsika_output_validator.file_info.get_simulated_events",
        return_value=(1001, 100),
    )

    validate_event_numbers([data_file], expected_shower_events=1000, tolerance=0.01)

    mock_get_events.assert_called_once_with(data_file)


def test_validate_event_numbers_with_mismatch_raises_error(tmp_path, mocker):
    """Test event validation raises ValueError on significant mismatch."""
    data_file = tmp_path / "corsika.data"

    mocker.patch(
        "simtools.corsika.corsika_output_validator.file_info.get_simulated_events",
        return_value=(900, 100),
    )

    with pytest.raises(ValueError, match="Inconsistent event counts"):
        validate_event_numbers([data_file], expected_shower_events=1000)


def test_validate_log_files_without_expected_patterns_raises_error(tmp_path, mocker):
    """Test log file validation raises ValueError when patterns are missing."""
    log_file = tmp_path / "corsika.log"

    mocker.patch("simtools.corsika.corsika_output_validator.check_plain_logs", return_value=False)

    with pytest.raises(ValueError, match="do not contain expected patterns"):
        validate_log_files([log_file], expected_shower_events=1000)


def test_validate_event_numbers_empty_list():
    """Test event validation with empty data file list."""
    validate_event_numbers([], expected_shower_events=1000)


def test_validate_event_numbers_multiple_files(tmp_path, mocker):
    """Test event validation with multiple data files."""
    data_file1 = tmp_path / "corsika1.data"
    data_file2 = tmp_path / "corsika2.data"

    mock_get_events = mocker.patch(
        "simtools.corsika.corsika_output_validator.file_info.get_simulated_events",
        side_effect=[(500, 50), (500, 50)],
    )

    validate_event_numbers([data_file1, data_file2], expected_shower_events=500)

    assert mock_get_events.call_count == 2


def test_validate_event_numbers_multiple_files_one_mismatches(tmp_path, mocker):
    """Test event validation raises error when one of multiple files mismatches."""
    data_file1 = tmp_path / "corsika1.data"
    data_file2 = tmp_path / "corsika2.data"

    mocker.patch(
        "simtools.corsika.corsika_output_validator.file_info.get_simulated_events",
        side_effect=[(500, 50), (400, 50)],
    )

    with pytest.raises(ValueError, match="Inconsistent event counts"):
        validate_event_numbers([data_file1, data_file2], expected_shower_events=500)


def test_validate_event_numbers_custom_tolerance(tmp_path, mocker):
    """Test event validation with custom tolerance threshold."""
    data_file = tmp_path / "corsika.data"

    mocker.patch(
        "simtools.corsika.corsika_output_validator.file_info.get_simulated_events",
        return_value=(950, 100),
    )

    with pytest.raises(ValueError, match="Inconsistent event counts"):
        validate_event_numbers([data_file], expected_shower_events=1000, tolerance=0.01)


def test_validate_log_files_with_curved_atmosphere(tmp_path, mocker):
    """Test log file validation with curved atmosphere enabled."""
    log_file = tmp_path / "corsika.log"

    mock_check = mocker.patch(
        "simtools.corsika.corsika_output_validator.check_plain_logs", return_value=True
    )

    validate_log_files([log_file], expected_shower_events=500, curved_atmo=True)

    call_args = mock_check.call_args
    assert "CURVED VERSION WITH SLIDING PLANAR ATMOSPHERE" in call_args[0][1]["pattern"]


def test_validate_log_files_without_curved_atmosphere(tmp_path, mocker):
    """Test log file validation includes curved bad pattern when not using curved atmo."""
    log_file = tmp_path / "corsika.log"

    mock_check = mocker.patch(
        "simtools.corsika.corsika_output_validator.check_plain_logs", return_value=True
    )

    validate_log_files([log_file], expected_shower_events=500, curved_atmo=False)

    call_args = mock_check.call_args
    assert "CORSIKA was compiled without CURVED option." in call_args[0][1]["forbidden_pattern"]


def test_validate_log_files_without_expected_events_specified(tmp_path, mocker):
    """Test log file validation when expected_shower_events is not specified."""
    log_file = tmp_path / "corsika.log"

    mock_check = mocker.patch(
        "simtools.corsika.corsika_output_validator.check_plain_logs", return_value=True
    )

    validate_log_files([log_file], expected_shower_events=None, curved_atmo=False)

    call_args = mock_check.call_args
    patterns = call_args[0][1]["pattern"]
    assert not any("NUMBER OF GENERATED EVENTS" in str(p) for p in patterns)


def test_validate_log_files_multiple_files(tmp_path, mocker):
    """Test log file validation with multiple log files."""
    log_file1 = tmp_path / "corsika1.log"
    log_file2 = tmp_path / "corsika2.log"

    mock_check = mocker.patch(
        "simtools.corsika.corsika_output_validator.check_plain_logs", return_value=True
    )

    validate_log_files([log_file1, log_file2], expected_shower_events=1000)

    mock_check.assert_called_once()
    call_args = mock_check.call_args
    assert call_args[0][0] == [log_file1, log_file2]


def test_validate_log_files_event_string_includes_number(tmp_path, mocker):
    """Test that event string includes the expected shower event number."""
    log_file = tmp_path / "corsika.log"
    expected_events = 2500

    mock_check = mocker.patch(
        "simtools.corsika.corsika_output_validator.check_plain_logs", return_value=True
    )

    validate_log_files([log_file], expected_shower_events=expected_events)

    call_args = mock_check.call_args
    patterns = call_args[0][1]["pattern"]
    assert f"NUMBER OF GENERATED EVENTS =          {expected_events}" in patterns
