import json
import math
from pathlib import Path

import pytest
import yaml

from simtools.science_tests import telescope_throughput as throughput

MSTN_01 = "MSTN-01"
ADVISORY = "advisory"
CHANGES = "changes"
EMPTY = "empty"
METRICS = "metrics"
RELATIVE = "relative"
RESULTS = "results"
STATUS = "status"
UTF_8 = "utf-8"
VALUE = "value"

TELESCOPE = "LSTN-01"
CAM_EFF = "cam_eff"
NSB_RATE = "nsb_rate"
SUMMARY_NAME = "telescope_throughput_summary_North_LSTN-01_za0.0deg_azm000deg.yml"
DIAGNOSTICS = "diagnostics"


def write_result(directory, cam_eff=0.5, nsb_rate=0.25, meta=None):
    directory.mkdir(parents=True, exist_ok=True)
    summary = {
        "meta": meta or {"tel": directory.name, "zen": 0.0},
        CAM_EFF: {VALUE: cam_eff, "description": "camera efficiency"},
        NSB_RATE: {VALUE: {VALUE: nsb_rate, "unit": "GHz"}},
    }
    (directory / SUMMARY_NAME).write_text(yaml.safe_dump(summary), encoding=UTF_8)
    (directory / "x_shower_summary.yml").write_text("ignored: 1\n", encoding=UTF_8)
    (directory / "x_za0.0deg_shower.ecsv").write_text("wl eff\n300 0.5\n", encoding=UTF_8)
    return directory


def test_read_result_returns_case_metrics_and_tables(tmp_test_directory):
    result = throughput.read_result(write_result(Path(tmp_test_directory) / TELESCOPE))

    assert result["telescope"] == TELESCOPE
    assert result["case"] == {"tel": TELESCOPE, "zen": 0.0}
    assert result[METRICS][NSB_RATE].unit == "GHz"
    assert math.isclose(result[METRICS][CAM_EFF].value, 0.5)
    assert list(result["tables"]) == ["shower"]


def test_read_result_requires_one_summary_and_finite_metrics(tmp_test_directory):
    directory = Path(tmp_test_directory) / RESULTS
    directory.mkdir()
    with pytest.raises(ValueError, match="Expected one summary"):
        throughput.read_result(directory)
    write_result(directory / TELESCOPE, cam_eff=float("nan"))
    with pytest.raises(ValueError, match="finite scalar"):
        throughput.read_result(directory / TELESCOPE)
    (directory / TELESCOPE / SUMMARY_NAME).write_text(
        yaml.safe_dump({CAM_EFF: {VALUE: "text"}}), encoding=UTF_8
    )
    with pytest.raises(ValueError, match="Invalid metric value"):
        throughput.read_result(directory / TELESCOPE)
    (directory / TELESCOPE / SUMMARY_NAME).write_text("- list\n", encoding=UTF_8)
    with pytest.raises(ValueError, match="must be a mapping"):
        throughput.read_result(directory / TELESCOPE / SUMMARY_NAME)


def test_load_results_accepts_file_directory_or_parent(tmp_test_directory):
    root = Path(tmp_test_directory) / RESULTS
    write_result(root / TELESCOPE)
    write_result(root / MSTN_01)

    assert list(throughput.load_results(root)) == [TELESCOPE, MSTN_01]
    assert list(throughput.load_results(root / TELESCOPE)) == [TELESCOPE]
    assert list(throughput.load_results(root / TELESCOPE / SUMMARY_NAME)) == [TELESCOPE]
    (root / EMPTY).mkdir()
    with pytest.raises(ValueError, match="No telescope-throughput results"):
        throughput.load_results(root / EMPTY)


def test_compare_results_converts_units_and_flags_tolerances(tmp_test_directory):
    root = Path(tmp_test_directory)
    baseline = throughput.read_result(write_result(root / "a" / TELESCOPE))
    candidate = throughput.read_result(write_result(root / "b" / TELESCOPE, cam_eff=0.6))
    candidate[METRICS][NSB_RATE] = candidate[METRICS][NSB_RATE].to("MHz")

    comparison = throughput.compare_results(baseline, candidate, {CAM_EFF: 0.1, NSB_RATE: 0.1})

    assert comparison[STATUS] == throughput.COMPLETE
    assert comparison[CHANGES][CAM_EFF][RELATIVE] == pytest.approx(0.2)
    assert comparison[CHANGES][CAM_EFF][ADVISORY] is True
    assert comparison[CHANGES][NSB_RATE]["unit"] == "GHz"
    assert comparison[CHANGES][NSB_RATE]["candidate"] == pytest.approx(0.25)
    assert comparison[CHANGES][NSB_RATE][ADVISORY] is False
    json.dumps(comparison)


def test_compare_results_reports_case_metric_and_tolerance_problems(tmp_test_directory):
    root = Path(tmp_test_directory)
    baseline = throughput.read_result(write_result(root / "a" / TELESCOPE, cam_eff=0.0))
    candidate = throughput.read_result(
        write_result(root / "b" / TELESCOPE, meta={"tel": TELESCOPE, "zen": 20.0})
    )
    del candidate[METRICS][NSB_RATE]

    comparison = throughput.compare_results(baseline, candidate, {"unknown": 0.1})

    assert comparison[STATUS] == throughput.INCOMPLETE
    assert comparison[CHANGES][CAM_EFF][RELATIVE] is None
    assert any("zen differs" in item for item in comparison[DIAGNOSTICS])
    assert any("unknown metric unknown" in item for item in comparison[DIAGNOSTICS])
    assert any("nsb_rate: missing" in item for item in comparison[DIAGNOSTICS])
    allowed = throughput.compare_results(baseline, baseline, compare_by=["zen"])
    assert allowed[STATUS] == throughput.COMPLETE


def test_parse_tolerances_accepts_nested_values_and_rejects_invalid():
    assert throughput.parse_tolerances([["a=0.1", "b=0"], "c=1"]) == {"a": 0.1, "b": 0, "c": 1}
    for value in ("a", "a=-1", "a=nan", "=0.1"):
        with pytest.raises(ValueError, match="Invalid tolerance"):
            throughput.parse_tolerances([value])


def test_write_json_creates_directory(tmp_test_directory):
    path = Path(tmp_test_directory) / "new" / "result.json"
    throughput.write_json(path, {"path": Path("x")})
    assert json.loads(path.read_text(encoding=UTF_8)) == {"path": "x"}
