import json
from pathlib import Path

import numpy as np
import yaml
from astropy.table import Table

from simtools.applications import compare_science_results as application

CANNOT_COMPARE_EFFICIENCY_COMPONENTS_REF = "Cannot compare efficiency components: 'ref'"
CAM_EFF = "cam_eff"
COMPARE = "compare"
INCOMPLETE = "incomplete"
SHOWER = "shower"
UTF_8 = "utf-8"

TELESCOPE = "LSTN-01"
SUMMARY_NAME = "telescope_throughput_summary_North_za0.0deg.yml"
RESULT_FILE = "science-comparison.json"
STATUS = "status"
DIAGNOSTICS = "diagnostics"
COMPARISONS = "comparisons"


def write_result(directory, cam_eff=0.5, channels=(SHOWER, "nsb")):
    directory.mkdir(parents=True, exist_ok=True)
    summary = {"meta": {"tel": directory.name}, CAM_EFF: {"value": cam_eff}}
    (directory / SUMMARY_NAME).write_text(yaml.safe_dump(summary), encoding=UTF_8)
    for channel in channels:
        table = Table({"wl": [300.0, 400.0], "cherenkov_detected": [1.0, 2.0]})
        table["nsb_detected"] = [1.0, 1.0]
        table.write(directory / f"x_{channel}.ecsv", format="ascii.ecsv")
    return directory


def run_main(mocker, output, **overrides):
    args = {
        "baseline_path": None,
        "candidate_path": None,
        "baseline_label": "v1",
        "candidate_label": "v2",
        "telescopes": [],
        "compare_by": [],
        "tolerance": [["cam_eff=0.1"]],
        "require_wavelength_tables": False,
    } | overrides
    context = mocker.Mock(args=args)
    context.paths.get_output_directory.return_value = output
    mocker.patch(
        "simtools.application.definition.ApplicationDefinition.start", return_value=context
    )
    application.main()
    return json.loads((output / RESULT_FILE).read_text(encoding=UTF_8))


def test_main_compares_all_telescopes(tmp_test_directory, mocker):
    root = Path(tmp_test_directory) / COMPARE
    write_result(root / "a" / TELESCOPE)
    write_result(root / "b" / TELESCOPE, cam_eff=0.6)

    result = run_main(mocker, root, baseline_path=root / "a", candidate_path=root / "b")

    assert result[STATUS] == "complete"
    assert result[COMPARISONS][TELESCOPE]["changes"][CAM_EFF]["advisory"] is True
    assert result[COMPARISONS][TELESCOPE]["table_diagnostics"] == [
        CANNOT_COMPARE_EFFICIENCY_COMPONENTS_REF,
        CANNOT_COMPARE_EFFICIENCY_COMPONENTS_REF,
    ]
    assert (root / "report.pdf").stat().st_size > 0


def test_main_requires_tables_when_requested(tmp_test_directory, mocker):
    root = Path(tmp_test_directory) / COMPARE
    write_result(root / "a" / TELESCOPE, channels=(SHOWER,))
    write_result(root / "b" / TELESCOPE, channels=("nsb",))

    result = run_main(
        mocker,
        root,
        baseline_path=root / "a",
        candidate_path=root / "b",
        require_wavelength_tables=True,
    )

    assert result[STATUS] == INCOMPLETE
    assert any("nsb is missing" in item for item in result[DIAGNOSTICS])


def test_main_reports_missing_telescopes_and_input_errors(tmp_test_directory, mocker):
    root = Path(tmp_test_directory) / COMPARE
    write_result(root / "a" / TELESCOPE, channels=())
    write_result(root / "b" / "MSTN-01", channels=())

    result = run_main(mocker, root, baseline_path=root / "a", candidate_path=root / "b")
    assert result[STATUS] == INCOMPLETE
    assert result[COMPARISONS][TELESCOPE][DIAGNOSTICS] == ["Missing baseline or candidate."]

    selected = run_main(
        mocker,
        root,
        baseline_path=root / "a",
        candidate_path=root / "a",
        telescopes=[TELESCOPE],
        require_wavelength_tables=True,
    )
    assert selected[DIAGNOSTICS] == [f"{TELESCOPE}: No wavelength tables found."]

    failed = run_main(mocker, root, baseline_path=root / "missing", candidate_path=root / "b")
    assert failed[STATUS] == INCOMPLETE
    assert failed[COMPARISONS] == {}
    assert np.size(failed[DIAGNOSTICS]) == 1
