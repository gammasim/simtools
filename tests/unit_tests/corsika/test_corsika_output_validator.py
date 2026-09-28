import pytest

from simtools.corsika.corsika_output_validator import (
    validate_corsika_output,
    validate_event_numbers,
    validate_log_files,
)


def test_validate_corsika_output_with_valid_files(tmp_test_directory, mocker):
    data_file = tmp_test_directory / "corsika.data"
    log_file = tmp_test_directory / "corsika.log"

    mock_validate_events = mocker.patch(
        "simtools.corsika.corsika_output_validator.validate_event_numbers"
    )
    mock_validate_logs = mocker.patch(
        "simtools.corsika.corsika_output_validator.validate_log_files"
    )

    validate_corsika_output([data_file], [log_file], expected_shower_events=1000)

    mock_validate_events.assert_called_once_with([data_file], 1000)
    mock_validate_logs.assert_called_once_with(
        [log_file], expected_shower_events=1000, curved_atmo=False
    )


@pytest.mark.parametrize(
    ("shower_events", "tolerance", "should_warn"),
    [(1000, 0.001, False), (999, 0.001, True)],
)
def test_validate_event_numbers_accepts_counts_within_tolerance(
    tmp_test_directory, mocker, caplog, shower_events, tolerance, should_warn
):
    data_file = tmp_test_directory / "corsika.data"

    mock_get_events = mocker.patch(
        "simtools.corsika.corsika_output_validator.file_info.get_simulated_events",
        return_value=(shower_events, 100),
    )

    validate_event_numbers([data_file], expected_shower_events=1000, tolerance=tolerance)

    mock_get_events.assert_called_once_with(data_file)
    assert ("Small mismatch" in caplog.text) is should_warn


def test_validate_event_numbers_with_mismatch_raises_error(tmp_test_directory, mocker):
    data_file = tmp_test_directory / "corsika.data"

    mocker.patch(
        "simtools.corsika.corsika_output_validator.file_info.get_simulated_events",
        return_value=(900, 100),
    )

    with pytest.raises(ValueError, match="Inconsistent event counts"):
        validate_event_numbers([data_file], expected_shower_events=1000)


def test_validate_log_files_without_expected_patterns_raises_error(tmp_path, mocker):
    log_file = tmp_path / "corsika.log"

    mocker.patch("simtools.corsika.corsika_output_validator.check_plain_logs", return_value=False)

    with pytest.raises(ValueError, match="do not contain expected patterns"):
        validate_log_files([log_file], expected_shower_events=1000)


def test_validate_event_numbers_multiple_files(tmp_path, mocker):
    data_file1 = tmp_path / "corsika1.data"
    data_file2 = tmp_path / "corsika2.data"

    mock_get_events = mocker.patch(
        "simtools.corsika.corsika_output_validator.file_info.get_simulated_events",
        side_effect=[(500, 50), (500, 50)],
    )

    validate_event_numbers([data_file1, data_file2], expected_shower_events=500)

    assert mock_get_events.call_count == 2


def test_validate_log_files_with_curved_atmosphere(tmp_path, mocker):
    log_file = tmp_path / "corsika.log"

    mock_check = mocker.patch(
        "simtools.corsika.corsika_output_validator.check_plain_logs", return_value=True
    )

    validate_log_files([log_file], expected_shower_events=500, curved_atmo=True)

    call_args = mock_check.call_args
    assert "CURVED VERSION WITH SLIDING PLANAR ATMOSPHERE" in call_args[0][1]["pattern"]


def test_validate_log_files_without_expected_events_specified(tmp_path, mocker):
    log_file = tmp_path / "corsika.log"

    mock_check = mocker.patch(
        "simtools.corsika.corsika_output_validator.check_plain_logs", return_value=True
    )

    validate_log_files([log_file], expected_shower_events=None, curved_atmo=False)

    call_args = mock_check.call_args
    patterns = call_args[0][1]["pattern"]
    assert not any("NUMBER OF GENERATED EVENTS" in str(p) for p in patterns)
