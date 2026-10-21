import json
from pathlib import Path
from types import SimpleNamespace

import astropy.units as u

from simtools.camera import camera_efficiency_workflow as workflow

_SUMMARY_FILE_TYPE = "telescope_throughput_summary"
_MANIFEST_FILE_NAME = "telescope_throughput_manifest.json"


def test_run_camera_efficiency_reuses_models_and_exports_products(tmp_test_directory, monkeypatch):
    output = Path(tmp_test_directory)
    model = object()
    calls = []
    file_types = []

    class FakeEfficiency:
        def __init__(self, **kwargs):
            self.efficiency_type = kwargs["efficiency_type"].lower()
            self.output_dir = output
            self.telescope_model = SimpleNamespace(site="North", name="MSTN-01")
            self.config = {"zenith_angle": 20, "azimuth_angle": 0}
            calls.append(("init", self.efficiency_type, kwargs["telescope_model"]))

        def simulate(self):
            calls.append(("simulate", self.efficiency_type))

        def analyze(self, force):
            calls.append(("analyze", self.efficiency_type, force))
            (output / f"{self.efficiency_type}.ecsv").touch()

        def results_summary(self):
            return {self.efficiency_type: {"value": 1}}

        def plot_efficiency(self, save_fig):
            calls.append(("plot", self.efficiency_type, save_fig))

        def dump_nsb_pixel_rate(self):
            calls.append(("nsb",))

        def calc_partial_efficiency(self):
            calls.append(("muon",))

    monkeypatch.setattr(workflow, "CameraEfficiency", FakeEfficiency)
    monkeypatch.setattr(
        workflow,
        "initialize_simulation_models",
        lambda **kwargs: (model, object(), None),
    )

    def generate_file_name(**kwargs):
        file_types.append(kwargs["file_type"])
        if kwargs["file_type"] == _SUMMARY_FILE_TYPE:
            return "telescope_throughput_summary.yml"
        return "telescope_throughput.ecsv"

    monkeypatch.setattr(workflow.names, "generate_file_name", generate_file_name)
    monkeypatch.setattr(workflow, "write_data_to_file", lambda data, path, **kwargs: path.touch())

    result = workflow.run_camera_efficiency(
        {
            "site": "North",
            "telescope": "MSTN-01",
            "model_version": "test",
            "zenith_angle": 20 * u.deg,
            "azimuth_angle": 0 * u.deg,
        },
        output,
        label="test",
    )

    assert result["summary_path"] == output / "telescope_throughput_summary.yml"
    assert result["manifest_path"].name == _MANIFEST_FILE_NAME
    assert file_types == ["telescope_throughput"] * 3 + [_SUMMARY_FILE_TYPE]
    assert result["results"] == {"shower": {"value": 1}, "nsb": {"value": 1}, "muon": {"value": 1}}
    manifest = json.loads(result["manifest_path"].read_text(encoding="utf-8"))
    assert manifest["software"]["simtools_version"] == workflow.__version__
    assert len([call for call in calls if call[0] == "simulate"]) == 3
    assert ("nsb",) in calls
    assert ("muon",) in calls
    assert all(call[-1] is model for call in calls if call[0] == "init")
