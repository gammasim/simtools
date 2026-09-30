import json
from pathlib import Path

import pytest

from simtools.applications import maintain_simulation_model_compare_productions as compare


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_compare_json_dirs_counts_changed_and_one_sided_files(tmp_test_directory):
    root = Path(tmp_test_directory)
    first = root / "first"
    second = root / "second"
    _write_json(first / "changed.json", {"value": 1})
    _write_json(second / "changed.json", {"value": 2})
    _write_json(first / "first_only.json", {"value": 1})
    _write_json(second / "second_only.json", {"value": 1})

    assert compare._compare_json_dirs(first, second) == 3


def test_compare_json_dirs_ignores_model_version(tmp_test_directory):
    root = Path(tmp_test_directory)
    first = root / "first"
    second = root / "second"
    _write_json(first / "same.json", {"model_version": "1.0.0", "value": 1})
    _write_json(second / "same.json", {"model_version": "2.0.0", "value": 1})

    assert compare._compare_json_dirs(first, second) == 0


def test_main_exits_nonzero_for_differences(mocker, tmp_test_directory):
    context = mocker.Mock()
    context.args = {
        "directory_1": Path(tmp_test_directory) / "first",
        "directory_2": Path(tmp_test_directory) / "second",
    }
    application = mocker.Mock()
    application.start.return_value = context
    mocker.patch.object(compare, "APPLICATION", application)
    mocker.patch.object(compare, "_compare_json_dirs", return_value=1)

    with pytest.raises(SystemExit) as error:
        compare.main()

    assert error.value.code == 1
