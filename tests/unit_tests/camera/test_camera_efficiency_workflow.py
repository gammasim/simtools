from pathlib import Path
from types import SimpleNamespace

import astropy.units as u

from simtools.camera import camera_efficiency_workflow as workflow


def test_run_camera_efficiency_reuses_models_and_exports_products(tmp_test_directory, monkeypatch):
    output = Path(tmp_test_directory)
    model = object()
    calls = []

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
    monkeypatch.setattr(
        workflow.names,
        "generate_file_name",
        lambda **kwargs: "summary.yml" if kwargs["file_type"].endswith("summary") else "table.ecsv",
    )
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

    assert result["summary_path"] == output / "summary.yml"
    assert result["results"] == {"shower": {"value": 1}, "nsb": {"value": 1}, "muon": {"value": 1}}
    assert len([call for call in calls if call[0] == "simulate"]) == 3
    assert ("nsb",) in calls
    assert ("muon",) in calls
    assert all(call[-1] is model for call in calls if call[0] == "init")
