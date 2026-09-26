"""Integration checks for the published-style obdeect wheel."""

import subprocess

import pytest


def test_published_obdeect_wheel_runs_reference_trace(tmp_path):
    obdeect = pytest.importorskip("obdeect")
    from obdeect.result_contract import read_arrivals

    result_file = tmp_path / "arrivals.csv"
    subprocess.run(
        [
            str(obdeect.executable_path()),
            "--telescope",
            "MST",
            "--photons",
            "8",
            "--output",
            str(result_file),
        ],
        check=True,
    )

    arrivals = read_arrivals(result_file)
    assert len(arrivals) == 8
    assert {arrival.source_kind for arrival in arrivals} == {"star"}
