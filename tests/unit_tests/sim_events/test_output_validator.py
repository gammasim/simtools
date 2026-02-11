import pytest

from simtools.sim_events.output_validator import valid_event_numbers, validate_sim_events


def test_validate_sim_events_calls_valid_event_numbers(tmp_path, monkeypatch):
    """Test validate_sim_events properly wraps valid_event_numbers."""
    test_files = [tmp_path / f"test_{i}.fits" for i in range(2)]
    for f in test_files:
        f.touch()

    mock_tables = {"SHOWERS": [1, 2, 3, 4]}
    monkeypatch.setattr(
        "simtools.sim_events.output_validator.table_handler.read_tables",
        lambda *args, **kwargs: mock_tables,
    )

    validate_sim_events(test_files, 4)


def test_validate_sim_events_with_string_input(tmp_path, monkeypatch):
    """Test validate_sim_events handles single file as string."""
    test_file = tmp_path / "test_events.fits"
    test_file.touch()

    mock_tables = {"SHOWERS": [1, 2, 3]}
    monkeypatch.setattr(
        "simtools.sim_events.output_validator.table_handler.read_tables",
        lambda *args, **kwargs: mock_tables,
    )

    validate_sim_events(str(test_file), 3)


def test_valid_event_numbers_single_file_match(tmp_path, monkeypatch):
    """Test valid_event_numbers with matching event count."""
    test_file = tmp_path / "test_events.fits"
    test_file.touch()

    mock_tables = {"SHOWERS": [1, 2, 3]}
    monkeypatch.setattr(
        "simtools.sim_events.output_validator.table_handler.read_tables",
        lambda *args, **kwargs: mock_tables,
    )

    valid_event_numbers(str(test_file), 3)


def test_valid_event_numbers_single_file_mismatch(tmp_path, monkeypatch):
    """Test valid_event_numbers raises ValueError when event count mismatches."""
    test_file = tmp_path / "test_events.fits"
    test_file.touch()

    mock_tables = {"SHOWERS": [1, 2, 3]}
    monkeypatch.setattr(
        "simtools.sim_events.output_validator.table_handler.read_tables",
        lambda *args, **kwargs: mock_tables,
    )

    with pytest.raises(ValueError, match="Inconsistent event counts found in reduced event lists"):
        valid_event_numbers(str(test_file), 5)


def test_valid_event_numbers_multiple_files_all_match(tmp_path, monkeypatch):
    """Test valid_event_numbers with multiple files all matching."""
    test_files = [tmp_path / f"test_{i}.fits" for i in range(2)]
    for f in test_files:
        f.touch()

    mock_tables = {"SHOWERS": [1, 2, 3, 4]}
    monkeypatch.setattr(
        "simtools.sim_events.output_validator.table_handler.read_tables",
        lambda *args, **kwargs: mock_tables,
    )

    valid_event_numbers(test_files, 4)


def test_valid_event_numbers_multiple_files_one_mismatch(tmp_path, monkeypatch):
    """Test valid_event_numbers with multiple files where one mismatches."""
    test_files = [tmp_path / f"test_{i}.fits" for i in range(2)]
    for f in test_files:
        f.touch()

    call_count = 0

    def mock_read_tables(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {"SHOWERS": [1, 2, 3, 4]}
        return {"SHOWERS": [1, 2]}

    monkeypatch.setattr(
        "simtools.sim_events.output_validator.table_handler.read_tables", mock_read_tables
    )

    with pytest.raises(ValueError, match="Inconsistent event counts found in reduced event lists"):
        valid_event_numbers(test_files, 4)

    assert call_count == 2


def test_valid_event_numbers_missing_showers_table(tmp_path, monkeypatch):
    """Test valid_event_numbers raises ValueError when SHOWERS table is missing."""
    test_file = tmp_path / "test_events.fits"
    test_file.touch()

    mock_tables = {"OTHER_TABLE": [1, 2, 3]}
    monkeypatch.setattr(
        "simtools.sim_events.output_validator.table_handler.read_tables",
        lambda *args, **kwargs: mock_tables,
    )

    with pytest.raises(ValueError, match="SHOWERS table not found in reduced event list"):
        valid_event_numbers(str(test_file), 3)
