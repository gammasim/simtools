"""Tests for the telescope-structure plotting application."""

import sys
from pathlib import Path
from types import ModuleType

import astropy.units as u
import pytest
from astropy.table import Table

from simtools.applications import plot_telescope_structure as app


@pytest.mark.parametrize("shadowing", [False, True])
def test_application_plot_custom_structure(
    monkeypatch, tmp_test_directory, mock_model_reader, shadowing
):
    get_parameters = mock_model_reader.get_model_parameters.side_effect

    def model_parameters(*args, **kwargs):
        parameters = get_parameters(*args, **kwargs)
        parameters["mirror_list"]["value"] = "mirrors.ecsv"
        return parameters

    def export_files(*args, **kwargs):
        destination = kwargs["dest"]
        for record in kwargs["parameters"].values():
            assert record.get("value") != str(structure_file)
        table = Table(
            {
                "mirror_x": [0.0] * u.cm,
                "mirror_y": [0.0] * u.cm,
                "mirror_diameter": [120.0] * u.cm,
                "focal_length": [1600.0] * u.cm,
                "shape_type": [1],
            }
        )
        table.write(destination / "mirrors.ecsv", format="ascii.ecsv", overwrite=True)

    mock_model_reader.get_model_parameters.side_effect = model_parameters
    mock_model_reader.export_model_files.side_effect = export_files
    structure_file = tmp_test_directory / "custom.ecsv"
    structure_file.write_text(
        "# %ECSV 1.0\n# ---\n# datatype:\n"
        "# - {name: id, datatype: string}\n"
        "# - {name: group, datatype: string}\n"
        "# - {name: shape, datatype: string}\n"
        "# - {name: diameter, unit: m, datatype: float64}\n"
        "# - {name: x1, unit: m, datatype: float64}\n"
        "# - {name: y1, unit: m, datatype: float64}\n"
        "# - {name: z1, unit: m, datatype: float64}\n"
        "# - {name: x2, unit: m, datatype: float64}\n"
        "# - {name: y2, unit: m, datatype: float64}\n"
        "# - {name: z2, unit: m, datatype: float64}\n"
        "# meta: !!omap\n# - {coordinate_frame: primary_mirror_frame}\n"
        "id group shape diameter x1 y1 z1 x2 y2 z2\n"
        "mast mast cylinder 0.1 1 0 0 1 0 1\n",
        encoding="utf-8",
    )
    if shadowing:

        class OpticalModel:
            @classmethod
            def from_records(cls, records, read_file, **kwargs):
                assert "mast mast cylinder" in read_file("telescope_structure")
                return cls()

            def star_photons(self, photons, seed):
                return photons

            def obscurer_hits(self, photons, seed):
                return {
                    "name": ["mast"],
                    "group": ["mast"],
                    "sole": [1],
                    "crossed": [1],
                    "unobscured": 10,
                }

        monkeypatch.delitem(sys.modules, "simtools.ray_tracing.backends.obdeect", raising=False)
        obdeect = ModuleType("obdeect")
        model = ModuleType("obdeect.model")
        model.OpticalModel = OpticalModel
        model.DETECTED = 1
        monkeypatch.setitem(sys.modules, "obdeect", obdeect)
        monkeypatch.setitem(sys.modules, "obdeect.model", model)
    monkeypatch.setattr(
        "sys.argv",
        [
            "plot_telescope_structure",
            "--site",
            "South",
            "--telescope",
            "SSTS-design",
            "--model_version",
            "6.0.2",
            "--output_path",
            str(tmp_test_directory),
            "--telescope_structure_file",
            str(structure_file),
        ]
        + (["--shadowing"] if shadowing else []),
    )
    try:
        app.main()
    finally:
        if shadowing:
            sys.modules.pop("simtools.ray_tracing.backends.obdeect", None)
    assert list(Path(tmp_test_directory).rglob("plot_telescope_structure*.png"))
