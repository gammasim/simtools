import gzip
import logging
from pathlib import Path

import pytest

from simtools.constants import MODEL_PARAMETER_SCHEMA_PATH
from simtools.testing import assertions
from simtools.testing.assertions import check_plain_log

logging.getLogger().setLevel(logging.DEBUG)


@pytest.fixture
def test_json_file():
    return Path("tests/resources/reference_point_altitude.json")


@pytest.fixture
def test_yaml_file():
    return MODEL_PARAMETER_SCHEMA_PATH / "num_gains.schema.yml"


@pytest.fixture
def tar_with_log(tmp_test_directory, safe_tar_open):
    """Helper fixture to create tar.gz with log content."""

    def _create_tar(log_content: bytes, filename: str = "test_logs.tar.gz"):
        tmp_path = Path(tmp_test_directory)
        tar_path = tmp_path / filename
        with safe_tar_open(tar_path, "w:gz") as tar:
            log_gz = tmp_path / "test.log.gz"
            with gzip.open(log_gz, "wb") as gz:
                gz.write(log_content)
            tar.add(log_gz, arcname="test.log.gz")
        return tar_path

    return _create_tar


def test_assert_file_type_json(test_json_file, test_yaml_file):
    assert assertions.assert_file_type("json", test_json_file)
    assert not assertions.assert_file_type("json", "tests/resources/does_not_exist.json")
    assert not assertions.assert_file_type("json", test_yaml_file)

    assert assertions.assert_file_type("json", Path(test_json_file))


def test_assert_file_type_yaml(test_json_file, test_yaml_file, caplog):
    assert assertions.assert_file_type("yaml", test_yaml_file)
    assert assertions.assert_file_type("yml", test_yaml_file)
    assert not assertions.assert_file_type("yml", "tests/resources/does_not_exit.schema.yml")

    assert not assertions.assert_file_type(
        "yaml", "tests/resources/telescope_positions-South-ground.ecsv"
    )


def test_assert_file_type_others(caplog):
    with caplog.at_level(logging.INFO):
        assert assertions.assert_file_type(
            "ecsv", "tests/resources/telescope_positions-South-ground.ecsv"
        )
    assert (
        "File type test is checking suffix only for tests/resources/"
        "telescope_positions-South-ground.ecsv (suffix: ecsv)" in caplog.text
    )


def test_assert_no_suffix():
    assert not assertions.assert_file_type("yml", "tests/resources/does_not_exit_yml")


def test_check_simulation_logs_no_patterns():
    file_test = {}
    result = assertions.check_simulation_logs(Path("dummy.tar.gz"), file_test)
    assert result


def test_check_simulation_logs_not_tar_file(tmp_test_directory):
    not_tar = Path(tmp_test_directory) / "not_a_tar.txt"
    not_tar.write_text("not a tar file", encoding="utf-8")
    file_test = {"expected_log_output": {"pattern": ["test"]}}

    with pytest.raises(ValueError, match=r"is not a tar file"):
        assertions.check_simulation_logs(not_tar, file_test)


@pytest.mark.parametrize(
    ("log_content", "patterns", "should_pass"),
    [
        (
            b"Log line with pattern_A\nAnother line\nLine with pattern_B\n",
            ["pattern_A", "pattern_B"],
            True,
        ),
        (b"Log line with pattern_A\nAnother line\n", ["pattern_A", "missing_pattern"], False),
    ],
)
def test_check_simulation_logs_patterns(tar_with_log, log_content, patterns, should_pass):
    tar_path = tar_with_log(log_content)
    file_test = {"expected_log_output": {"pattern": patterns}}
    assert assertions.check_simulation_logs(tar_path, file_test) is should_pass


def test_check_simulation_logs_skip_non_log_files(tmp_test_directory, safe_tar_open):
    tmp_path = Path(tmp_test_directory)
    tar_path = tmp_path / "test_logs.tar.gz"

    with safe_tar_open(tar_path, "w:gz") as tar:
        not_log = tmp_path / "readme.txt"
        not_log.write_text("This is not a log file", encoding="utf-8")
        tar.add(not_log, arcname="readme.txt")

    file_test = {"expected_log_output": {"pattern": ["pattern"]}}
    assert assertions.check_simulation_logs(tar_path, file_test) is False


def test_read_log(tmp_test_directory, safe_tar_open):
    tmp_path = Path(tmp_test_directory)
    tar_path = tmp_path / "test.tar.gz"
    log_content = b"Test log content\nSecond line\n"

    with safe_tar_open(tar_path, "w:gz") as tar:
        log_gz = tmp_path / "test.log.gz"
        with gzip.open(log_gz, "wb") as gz:
            gz.write(log_content)
        tar.add(log_gz, arcname="test.log.gz")

    with safe_tar_open(tar_path, "r:gz") as tar:
        member = tar.getmembers()[0]
        result = assertions._read_log(member, tar)

    assert result == "Test log content\nSecond line\n"


def test_find_patterns():
    text = "This is a test line with pattern1 and pattern2"
    patterns = ["pattern1", "pattern2", "pattern3"]

    found = assertions._find_patterns(text, patterns)

    assert "pattern1" in found
    assert "pattern2" in found
    assert "pattern3" not in found
    assert len(found) == 2


@pytest.mark.parametrize(
    ("log_content", "expected_log_output", "should_pass"),
    [
        (
            b"Log line with CURVED VERSION WITH SLIDING PLANAR ATMOSPHERE\nAnother line\n",
            {"forbidden_pattern": ["CURVED VERSION WITH SLIDING PLANAR ATMOSPHERE"]},
            False,
        ),
        (
            b"Log line with normal content\nAnother line\n",
            {"forbidden_pattern": ["CURVED VERSION", "FATAL ERROR"]},
            True,
        ),
        (
            b"Log line with expected_pattern\nAnother line with good content\n",
            {"pattern": ["expected_pattern"], "forbidden_pattern": ["CURVED VERSION", "ERROR"]},
            True,
        ),
        (
            b"Log line with expected_pattern\nAnother line with ERROR\n",
            {"pattern": ["expected_pattern"], "forbidden_pattern": ["ERROR", "FATAL"]},
            False,
        ),
        (
            b"Log with ERROR\nAnother line with FATAL\nThird line with WARNING\n",
            {"forbidden_pattern": ["ERROR", "FATAL", "CRITICAL"]},
            False,
        ),
        (
            b"Log line with any content\n",
            {"forbidden_pattern": []},
            True,
        ),
    ],
)
def test_check_simulation_logs_forbidden_patterns(
    tar_with_log, log_content, expected_log_output, should_pass
):
    tar_path = tar_with_log(log_content)
    file_test = {"expected_log_output": expected_log_output}
    assert assertions.check_simulation_logs(tar_path, file_test) is should_pass


@pytest.mark.parametrize(
    ("content", "file_test", "should_pass"),
    [
        (
            "start\nAll good\nOK done\n",
            {"expected_log_output": {"pattern": ["OK"], "forbidden_pattern": []}},
            True,
        ),
        (
            "ERROR: failure happened\n",
            {"expected_log_output": {"pattern": [], "forbidden_pattern": ["ERROR"]}},
            False,
        ),
        (
            "Error: something went wrong\n",
            {"expected_log_output": {"forbidden_pattern": ["error"]}},
            False,
        ),
        ("Success: all good\n", {"expected_log_output": {"pattern": ["success"]}}, True),
    ],
)
def test_check_plain_log(tmp_path: Path, content, file_test, should_pass):
    log_file = tmp_path / "run.log"
    log_file.write_text(content, encoding="utf-8")
    assert check_plain_log(log_file, file_test) is should_pass


def test_check_plain_log_missing_file_returns_false(tmp_path: Path):
    log_file = tmp_path / "missing.log"
    file_test = {"expected_log_output": {"pattern": ["hello"], "forbidden_pattern": []}}
    assert check_plain_log(log_file, file_test) is False


def test_check_plain_log_top_level_keys_fallback(tmp_path: Path):
    log_file = tmp_path / "run.log"
    log_file.write_text("pipeline finished successfully\n", encoding="utf-8")
    file_test = {"expected_log_output": None, "pattern": ["finished"], "forbidden_pattern": []}
    assert check_plain_log(log_file, file_test) is True


def test_check_plain_log_no_patterns_returns_true(tmp_path: Path, caplog):
    log_file = tmp_path / "run.log"
    log_file.write_text("some content\n", encoding="utf-8")
    file_test = {"expected_log_output": {}}
    with caplog.at_level(logging.DEBUG):
        assert check_plain_log(log_file, file_test) is True
        assert "No expected log output provided" in caplog.text


def test_check_plain_log_missing_expected_patterns(tmp_path: Path, caplog):
    log_file = tmp_path / "run.log"
    log_file.write_text("only info lines here\n", encoding="utf-8")
    file_test = {"expected_log_output": {"pattern": ["NEEDED"], "forbidden_pattern": []}}
    with caplog.at_level(logging.ERROR):
        assert check_plain_log(log_file, file_test) is False
        assert "Missing expected patterns" in caplog.text


def test_check_output_from_sim_telarray_no_expected_output(tmp_path: Path):
    sim_file = tmp_path / "test.simtel.zst"
    sim_file.write_bytes(b"dummy")
    file_test = {}

    result = assertions.check_output_from_sim_telarray(sim_file, file_test)
    assert result is True


@pytest.mark.parametrize(
    ("file_test_key", "expected_result"),
    [
        ("expected_output", True),
        ("expected_simtel_metadata", True),
    ],
)
def test_check_output_from_sim_telarray_with_expected(
    tmp_path: Path, mocker, file_test_key, expected_result
):
    import simtools.testing.sim_telarray_output as sim_telarray_output

    sim_file = tmp_path / "test.simtel.zst"
    sim_file.write_bytes(b"dummy")
    file_test = {file_test_key: {"key": "value"}}

    mocker.patch.object(
        sim_telarray_output, "assert_expected_sim_telarray_output", return_value=True
    )
    mocker.patch.object(
        sim_telarray_output, "assert_expected_sim_telarray_metadata", return_value=True
    )
    mocker.patch.object(sim_telarray_output, "assert_n_showers_and_energy_range", return_value=True)

    assert assertions.check_output_from_sim_telarray(sim_file, file_test) is expected_result


def test_check_output_from_sim_telarray_assertion_fails(tmp_path: Path, mocker):
    import simtools.testing.sim_telarray_output as sim_telarray_output

    sim_file = tmp_path / "test.simtel.zst"
    sim_file.write_bytes(b"dummy")
    file_test = {"expected_output": {"key": "value"}}

    mocker.patch.object(
        sim_telarray_output, "assert_expected_sim_telarray_output", return_value=False
    )
    mocker.patch.object(
        sim_telarray_output, "assert_expected_sim_telarray_metadata", return_value=True
    )
    mocker.patch.object(sim_telarray_output, "assert_n_showers_and_energy_range", return_value=True)

    assert assertions.check_output_from_sim_telarray(sim_file, file_test) is False
