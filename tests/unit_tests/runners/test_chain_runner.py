"""Tests for the run script of simulation chains."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from simtools.runners.chain_runner import ChainRunner
from simtools.runners.stages import STDIN, StageInputs, StreamTo


def _fake_stage(name, streams=True, copies=1):
    """Return a fake stage class recording its calls."""

    class FakeStage:
        calls = []

        def __init__(self, index):
            self.index = index
            self.name = name
            self.streams = streams
            self.file_list = {f"{name}_files": f"{name}-{index}"}

        @classmethod
        def instances(cls, inputs, label=None):
            return [cls(index) for index in range(copies)]

        def prepare(self, run_number, input_file=None, output=None):
            FakeStage.calls.append((self.index, input_file, output))
            return [f"run {name} {self.index} {run_number}"]

        def output_file(self, run_number):
            return Path(f"{name}-{run_number}.out")

        def environment(self):
            return {f"{name.upper()}_ENV": "1"}

        def get_resources(self, runtime=None):
            return {"runtime": runtime}

    return FakeStage


def _runner(*stage_classes):
    inputs = StageInputs(
        array_models=(), simulation_parameters=(), shower_configurations=(), args={}
    )
    return ChainRunner(SimpleNamespace(stage_classes=list(stage_classes)), inputs, label="test")


def test_single_stage_reads_input_file_and_writes_extras(tmp_test_directory):
    stage = _fake_stage("telescope")
    runner = _runner(stage)
    script = Path(tmp_test_directory) / "run.sh"

    runner.prepare_run(3, script, input_file="events.in", extra_commands=["export A=1"])

    text = script.read_text(encoding="utf-8")
    assert text.startswith("#!/usr/bin/env bash\nset -e\nset -o pipefail\n")
    assert "export A=1\n# End of extras" in text
    assert text.endswith("run telescope 0 3\n")
    assert stage.calls == [(0, "events.in", None)]
    assert runner.file_list == {"telescope_files": "telescope-0"}


def test_streaming_stages_fan_out_per_array_model(tmp_test_directory):
    shower = _fake_stage("shower")
    telescope = _fake_stage("telescope", copies=2)
    runner = _runner(shower, telescope)
    script = Path(tmp_test_directory) / "run.sh"

    runner.prepare_run(1, script, extra_commands="echo extra")

    assert telescope.calls == [(0, STDIN, None), (1, STDIN, None)]
    assert shower.calls == [(0, None, StreamTo(("run telescope 0 1", "run telescope 1 1")))]
    text = script.read_text(encoding="utf-8")
    assert "echo extra" in text
    assert "run shower 0 1" in text
    assert "run telescope" not in text
    assert runner.file_list == {
        "shower_files": "shower-0",
        "telescope_files": ["telescope-0", "telescope-1"],
    }


def test_non_streaming_stages_hand_over_files(tmp_test_directory):
    first = _fake_stage("first", streams=False)
    second = _fake_stage("second")
    runner = _runner(first, second)
    script = Path(tmp_test_directory) / "run.sh"

    runner.prepare_run(2, script)

    assert second.calls == [(0, Path("first-2.out"), None)]
    text = script.read_text(encoding="utf-8")
    assert text.index("run first 0 2") < text.index("run second 0 2")


def test_environment_and_resources():
    runner = _runner(_fake_stage("shower"), _fake_stage("telescope"))

    assert runner.environment() == {"SHOWER_ENV": "1", "TELESCOPE_ENV": "1"}
    assert runner.get_resources(runtime=4.0)["runtime"] == pytest.approx(4.0)
