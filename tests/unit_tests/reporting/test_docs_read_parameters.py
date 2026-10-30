from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from simtools.data_model import parameter_catalog as catalog
from simtools.reporting.docs_read_parameters import ReadParameters
from simtools.reporting.model_figures import (
    write_flasher_pulse_shape_plots,
    write_telescope_structure_figure,
)
from simtools.simtel import simtel_file_writer
from simtools.utils import names

QE_FILE_NAME = "qe_lst1_20200318_high+low.dat"
DESCRIPTION = "Test parameter"
SHORT_DESC = "Short"


class DummyTelescope:
    def __init__(self, site, name, model_version, param_versions=None, design_model=None):
        self.site = site
        self.name = name
        self.model_version = model_version
        self._param_versions = param_versions or {}
        self.design_model = design_model or name

    def get_parameter_version(self, parameter):
        return self._param_versions.get(parameter)


def test_produce_array_element_report(telescope_model_lst, tmp_test_directory, run_context):
    site = telescope_model_lst.site
    # Observatory branch
    rp = ReadParameters(
        args={"site": site, "model_version": "6.0.0", "observatory": True},
        output_path=tmp_test_directory,
        context=run_context,
    )
    rp.context.paths.output_path["default"] = Path(tmp_test_directory)
    with patch.object(
        rp.model_reader,
        "get_model_parameters",
        return_value={"site_elevation": {"value": 2200, "unit": "m"}},
    ):
        rp.produce_array_element_report()
    assert (tmp_test_directory / f"OBS-{site}.md").exists()

    # Telescope branch
    rp2 = ReadParameters(
        args={"site": site, "model_version": "6.0.0", "telescope": telescope_model_lst.name},
        output_path=tmp_test_directory,
        context=run_context,
    )
    with (
        patch.object(
            rp2.model_reader,
            "get_model_parameters",
            return_value={
                "focal_length": {
                    "value": 2800.0,
                    "unit": "cm",
                    "parameter_version": "1.0.0",
                    "instrument": telescope_model_lst.name,
                }
            },
        ),
        patch.object(rp2.model_reader, "export_model_files"),
        patch("simtools.reporting.docs_read_parameters.write_telescope_structure_figure"),
        patch(
            "simtools.reporting.docs_read_parameters.TelescopeModel",
            return_value=DummyTelescope(
                site=site,
                name=telescope_model_lst.name,
                model_version="6.0.0",
                param_versions={"focal_length": "1.0.0"},
                design_model=telescope_model_lst.name,
            ),
        ),
    ):
        rp2.produce_array_element_report()
    assert (tmp_test_directory / f"{telescope_model_lst.name}.md").exists()


def test_produce_model_parameter_reports(tmp_test_directory, mocker, run_context):
    args = {"site": "North", "telescope": "LSTN-01"}
    read_parameters = ReadParameters(args=args, output_path=tmp_test_directory, context=run_context)
    mock_data = {
        "6.0.2": {
            "quantum_efficiency": {
                "value": "qe_file.dat",
                "parameter_version": "1.0.0",
                "file": True,
                "type": "str",
                "instrument": "LSTN-01",
            }
        }
    }
    mocker.patch.object(
        read_parameters.model_reader,
        "get_model_parameters_for_all_model_versions",
        return_value=mock_data,
    )
    mocker.patch.object(
        read_parameters.model_reader, "get_model_parameters", return_value=mock_data["6.0.2"]
    )
    mocker.patch.object(read_parameters.model_reader, "export_model_files")
    mocker.patch.object(read_parameters, "_generate_plots", return_value=[])

    read_parameters.produce_model_parameter_reports()

    report = (tmp_test_directory / "LSTN-01.md").read_text(encoding="utf-8")
    assert report.startswith("# LSTN-01\n")
    assert "## quantum_efficiency" in report
    assert "qe_file.dat" in report


def test_produce_model_parameter_reports_no_parameters(tmp_test_directory, mocker, run_context):
    read_parameters = ReadParameters(
        args={"site": "North", "telescope": "LSTN-01"},
        output_path=tmp_test_directory,
        context=run_context,
    )
    mocker.patch.object(
        read_parameters.model_reader,
        "get_model_parameters_for_all_model_versions",
        return_value={},
    )
    read_parameters.produce_model_parameter_reports()
    assert not (tmp_test_directory / "LSTN-01.md").exists()


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (
            [[24.74, 9.0, 350.0, 1066.0], ["ns", "ns", "V", "V"], False, "1.0.0"],
            "24.74 ns, 9.0 ns, 350.0 V, 1066.0 V",
        ),
        ([4.0, " ", False, "1.0.0"], "4.0"),
        ([[0.2] * 7, "GHz", False, "1.0.0"], "all: 0.2 GHz"),
        ([[1, 2, 3, 4], "m", False, "1.0.0"], "1 m, 2 m, 3 m, 4 m"),
        ([{"a": 1, "b": 2}, "m", False, "1.0.0"], "{'a': 1, 'b': 2} m"),
        ([[{"a": 1}, {"b": 2}, {"c": 3}], "m", False, "1.0.0"], "[View Test](#test)"),
    ],
)
def test__format_parameter_value(data, expected, tmp_path, run_context):
    read_parameters = ReadParameters(args={}, output_path=tmp_path, context=run_context)
    assert read_parameters._format_parameter_value("test", *data) == expected


def test_produce_observatory_report(mocker, tmp_path, run_context):
    rp = ReadParameters({"site": "North", "model_version": "6.0.0"}, tmp_path, context=run_context)

    # Empty data -> warning, no file
    mock_logger = mocker.patch("logging.Logger.warning")
    with patch.object(rp.model_reader, "get_model_parameters", return_value={}):
        rp.produce_observatory_report()
    mock_logger.assert_called_once()
    assert not (tmp_path / "OBS-North.md").exists()

    # Full data -> file with layouts, triggers, skips None-valued param
    mock_params = {
        "site_elevation": {"value": 2200, "unit": "m", "parameter_version": "1.0"},
        "array_layouts": {
            "value": [{"name": "L1", "elements": ["LST1"]}],
            "unit": None,
            "parameter_version": "1.0",
        },
        "array_triggers": {
            "value": [
                {
                    "name": "T1",
                    "multiplicity": {"value": 2, "unit": None},
                    "width": {"value": 120, "unit": "ns"},
                    "hard_stereo": {"value": True, "unit": None},
                    "min_separation": {"value": None, "unit": "m"},
                }
            ],
            "unit": None,
            "parameter_version": "1.0",
        },
        "none_valued_param": {"value": None, "unit": None, "parameter_version": "1.0"},
    }
    with patch.object(rp.model_reader, "get_model_parameters", return_value=mock_params):
        rp.produce_observatory_report()
    content = (tmp_path / "OBS-North.md").read_text()
    assert "# Observatory Parameters" in content
    assert "| site_elevation | 2200 m |" in content
    assert "none_valued_param" not in content
    assert "## Array Layouts" in content
    assert "## Array Trigger Configurations" in content


def test_produce_simulation_configuration_report(tmp_path, run_context):
    args = {
        "telescope": "LSTN-01",
        "site": "North",
        "model_version": "6.0.0",
        "simulation_software": "sim_telarray",
    }
    read_parameters = ReadParameters(args=args, output_path=tmp_path, context=run_context)
    mock_data = [
        (
            "LSTN-01",
            "iobuf maximum",
            "1.0.0",
            "100 byte",
            "Description",
            "Buffer limits for input and output of eventio data.",
        ),
        ("LSTN-01", "random generator", "1.0.0", "mt19937", "Random generator used.", None),
    ]
    with patch.object(
        read_parameters, "get_simulation_configuration_data", return_value=(mock_data, [])
    ):
        read_parameters.produce_simulation_configuration_report()

    content = (tmp_path / f"configuration_{read_parameters.software}.md").read_text()
    assert f"# configuration_{read_parameters.software}" in content
    assert "| Parameter Name" in content
    assert "Buffer limits for input and output of eventio data." in content
    assert "mt19937" in content

    # corsika: dict table is written as a separate section
    args["simulation_software"] = "corsika"
    read_parameters_corsika = ReadParameters(args=args, output_path=tmp_path, context=run_context)
    mock_data_corsika = [
        (
            "LSTN-01",
            "corsika_starting_grammage",
            "1.0.0",
            "[View Corsika Starting Grammage](#corsika-starting-grammage)",
            "Description",
            "Short",
        )
    ]
    mock_dict_tables = [
        (
            "LSTN-01",
            "corsika_starting_grammage",
            [{"instrument": "LSTN-design", "primary_particle": "muon+", "value": 580.0}],
            "g/cm2",
        )
    ]
    with patch.object(
        read_parameters_corsika,
        "get_simulation_configuration_data",
        return_value=(mock_data_corsika, mock_dict_tables),
    ):
        read_parameters_corsika.produce_simulation_configuration_report()

    content = (tmp_path / "configuration_corsika.md").read_text()
    assert "[View Corsika Starting Grammage](#corsika-starting-grammage)" in content
    assert "## Corsika Starting Grammage" in content
    assert "| instrument | primary_particle | value |" in content
    assert "| LSTN-design | muon+ | 580.0 g/cm2 |" in content


def test_produce_calibration_reports(mocker, tmp_path, run_context):
    rp = ReadParameters(args={"model_version": "6.0.0"}, output_path=tmp_path, context=run_context)
    mocker.patch.object(rp.model_reader, "get_array_elements", return_value=["ILLN-01"])
    mocker.patch.object(rp.model_reader, "get_design_model", return_value="ILLN-design")
    mocker.patch.object(
        rp.model_reader,
        "get_model_parameters",
        return_value={
            "laser_events": {
                "value": 10,
                "unit": None,
                "parameter_version": "1.0.0",
                "instrument": "ILLN-design",
            },
            "array_element_position_ground": {
                "value": [0.0, 0.0, 0.0],
                "unit": "m",
                "parameter_version": "1.0.0",
                "instrument": "ILLN-01",
            },
        },
    )
    desc = {
        "laser_events": {
            "description": "Laser",
            "short_description": None,
            "inst_class": "Calibration",
        },
        "array_element_position_ground": {
            "description": "Pos",
            "short_description": "P",
            "inst_class": "Structure",
        },
    }
    with patch.object(rp, "get_all_parameter_descriptions", return_value=desc):
        rp.produce_calibration_reports()
    content = (tmp_path / "ILLN-01.md").read_text()
    assert "# ILLN-01" in content
    assert "laser events" in content


def test_get_calibration_data(tmp_path, run_context):
    rp = ReadParameters(args={"model_version": "6.0.0"}, output_path=tmp_path, context=run_context)

    def param(value, instrument, unit=None, **extra):
        return {
            "value": value,
            "unit": unit,
            "parameter_version": "1.0.0",
            "instrument": instrument,
            **extra,
        }

    mock_data = {
        "dark_events": param(1, "ILLN-design"),
        "laser_events": param(10, "ILLN-01"),
        "none_value": param(None, "ILLN-design"),
        "dict_parameter": param([{"channel": 1, "value": 12.0}], "ILLN-design", unit="p.e."),
        "array_element_position_ground": param(
            [0.0, 0.0, 0.0], "ILLN-design", unit="m", file=False
        ),
    }

    def desc(short, inst_class):
        return {"description": "d", "short_description": short, "inst_class": inst_class}

    calib_desc = {
        "dark_events": desc("D", "Calibration"),
        "laser_events": desc(None, "Calibration"),
        "none_value": desc("N", "Calibration"),
        "dict_parameter": desc("D", "Calibration"),
    }
    tel_desc = {"array_element_position_ground": desc("pos", "Structure")}
    with patch.object(
        rp,
        "get_all_parameter_descriptions",
        side_effect=[calib_desc, tel_desc, calib_desc, tel_desc],
    ):
        result, dict_tables = rp.get_calibration_data(mock_data, "ILLN-01")
    assert result[0][0] == "Structure"  # reverse-sorted: Structure before Calibration
    assert len(result) == 4
    assert dict_tables == [("dict_parameter", [{"channel": 1, "value": 12.0}], "p.e.")]
    structure_row = next(r for r in result if r[0] == "Structure")
    assert "array_element_position_ground" in structure_row[1]


def test_get_array_element_parameter_data_simple(tmp_test_directory, monkeypatch, run_context):
    rp = ReadParameters(
        args={"telescope": "LSTN-01", "site": "North", "model_version": "1.0.0"},
        output_path=tmp_test_directory,
        context=run_context,
    )
    rp.model_reader = Mock()
    rp.model_reader.export_model_files.return_value = None
    rp.get_all_parameter_descriptions = Mock(
        return_value={
            "test_param": {
                "description": DESCRIPTION,
                "short_description": SHORT_DESC,
                "inst_class": "Telescope",
            },
            "none_param": {
                "description": "none",
                "short_description": "n",
                "inst_class": "Telescope",
            },
        }
    )
    monkeypatch.setattr(names, "is_design_type", lambda _name: False)

    # None-valued param is skipped; OTHER instrument is not emphasised
    rp.model_reader.get_model_parameters.return_value = {
        "test_param": {
            "unit": "m",
            "value": 42,
            "parameter_version": "1.0.0",
            "file": False,
            "instrument": "OTHER",
        },
        "none_param": {
            "unit": "m",
            "value": None,
            "parameter_version": "1.0.0",
            "file": False,
            "instrument": "OTHER",
        },
    }
    monkeypatch.setattr(
        catalog, "model_parameters", lambda *a, **k: {"test_param": {}, "none_param": {}}
    )
    tel = DummyTelescope(
        "North", "LSTN-01", "1.0.0", {"test_param": "1.0.0", "none_param": "1.0.0"}
    )
    data, _dt = rp.get_array_element_parameter_data(telescope_model=tel)
    assert data == [["Telescope", "test_param", "1.0.0", "42 m", DESCRIPTION, SHORT_DESC]]

    # Instrument match -> bold emphasis
    rp.model_reader.get_model_parameters.return_value = {
        "test_param": {
            "unit": "m",
            "value": 42,
            "parameter_version": "1.0.0",
            "file": False,
            "instrument": "LSTN-01",
        },
    }
    monkeypatch.setattr(catalog, "model_parameters", lambda *a, **k: {"test_param": {}})
    data, _dt = rp.get_array_element_parameter_data(telescope_model=tel)
    assert data[0][1] == "***test_param***"


def test_get_array_element_parameter_data_file_parameter(
    tmp_test_directory, monkeypatch, run_context
):
    rp = ReadParameters(
        args={"telescope": "LSTN-01", "site": "North", "model_version": "1.0.0"},
        output_path=tmp_test_directory,
        context=run_context,
    )
    rp.model_reader = Mock()
    rp.model_reader.get_model_parameters.return_value = {
        "file_param": {
            "unit": None,
            "value": "myfile.dat",
            "parameter_version": "1.0.0",
            "file": True,
            "instrument": "OTHER",
        }
    }
    rp.get_all_parameter_descriptions = Mock(
        return_value={
            "file_param": {
                "description": DESCRIPTION,
                "short_description": SHORT_DESC,
                "inst_class": "Telescope",
            }
        }
    )
    monkeypatch.setattr(
        ReadParameters,
        "_convert_to_md",
        lambda self, p, pv, f, **_kwargs: "_data_files/myfile.md",
    )
    monkeypatch.setattr(catalog, "model_parameters", lambda *a, **k: {"file_param": {}})
    monkeypatch.setattr(names, "is_design_type", lambda _name: False)

    tel = DummyTelescope("North", "LSTN-01", "1.0.0", {"file_param": "1.0.0"})
    data, _dt = rp.get_array_element_parameter_data(telescope_model=tel)
    assert data[0][3] == "[myfile.dat](_data_files/myfile.md)"


def test_is_markdown_link(run_context):
    read_parameters = ReadParameters(args={}, output_path=Path(), context=run_context)
    assert read_parameters.is_markdown_link("[example](http://example.com)") is True
    assert read_parameters.is_markdown_link("[text](target)") is True
    assert read_parameters.is_markdown_link("not a link") is False
    assert read_parameters.is_markdown_link("[missing target]") is False
    assert read_parameters.is_markdown_link("(missing text)") is False
    assert read_parameters.is_markdown_link("[text](target") is False
    assert read_parameters.is_markdown_link("[text]target)") is False
    assert read_parameters.is_markdown_link("") is False


def test_flasher_gauss_exponential_uses_simtel_table_writer(tmp_test_directory, run_context):
    rp = ReadParameters(
        args={"telescope": "MSFx-NectarCam", "site": "North", "model_version": "8.0.0"},
        output_path=tmp_test_directory,
        context=run_context,
    )
    rp.context.paths.output_path["default"] = Path(tmp_test_directory)
    selected_parameters = {
        "flasher_pulse_shape": {
            "parameter_version": "1.0.0",
            "value": ["Gauss-Exponential", 1.2879, 2.0961],
        }
    }
    telescope_parameters = {
        "calibration_devices": {"value": {"flat_fielding": "MSFx-NectarCam"}},
        "fadc_sum_bins": {"value": 10},
    }
    comparison_items = [{"parameter_version": "1.0.0", "model_version": "8.0.0"}]
    buf = StringIO()

    with (
        patch.object(
            rp.model_reader,
            "get_model_parameters",
            side_effect=lambda **kwargs: (
                selected_parameters
                if kwargs["array_element_name"] == "MSFx-NectarCam"
                else telescope_parameters
            ),
        ),
        patch.object(
            rp.model_reader,
            "read_production_table",
            return_value={"parameters": {"MSTx-NectarCam": {}}},
        ),
        patch(
            "simtools.reporting.model_figures.simtel_file_writer"
            ".write_light_pulse_table_gauss_exp_conv",
            wraps=simtel_file_writer.write_light_pulse_table_gauss_exp_conv,
        ) as write_table,
    ):
        write_flasher_pulse_shape_plots(
            buf,
            comparison_items,
            "calibration_devices",
            rp.model_reader,
            "North",
            "MSFx-NectarCam",
            context=rp.context,
        )

    write_table.assert_called_once_with(
        file_path=write_table.call_args.kwargs["file_path"],
        width_ns=1.2879,
        exp_decay_ns=2.0961,
        fadc_sum_bins=10,
        time_margin_ns=5.0,
    )
    assert "current sim_telarray configuration generator" in buf.getvalue()
    assert list((Path(tmp_test_directory) / "_images").glob("flasher_pulse_shape_*.png"))


def test_produce_array_element_report_writes_design_note_and_dict_table(
    tmp_path, mocker, run_context
):
    rp = ReadParameters(
        args={"site": "North", "telescope": "LSTN-01", "model_version": "6.0.0"},
        output_path=tmp_path,
        context=run_context,
    )
    mocker.patch(
        "simtools.reporting.docs_read_parameters.TelescopeModel",
        return_value=DummyTelescope(
            site="North", name="LSTN-01", model_version="6.0.0", design_model="LSTN-design"
        ),
    )
    mocker.patch.object(
        rp,
        "get_array_element_parameter_data",
        return_value=(
            [
                ["Structure", "inherited_param", "1.0.0", "1 m", "desc", "short"],
                ["Structure", "***param***", "***1.0.0***", "***2 m***", "desc", "short"],
            ],
            [
                ("dict_param", [{"a": 1, "value": 2}], "m"),
                ("param", [{"a": 1, "value": 2}], "m"),
            ],
        ),
    )
    write_dict_table = mocker.patch.object(rp, "_write_dict_table")
    mocker.patch("simtools.reporting.docs_read_parameters.write_telescope_structure_figure")

    rp.produce_array_element_report()

    content = (tmp_path / "LSTN-01.md").read_text(encoding="utf-8")
    assert "Parameters specific to LSTN-01" in content
    assert "inherited from the design model [LSTN-design](LSTN-design.md)" in content
    assert "***param***" in content
    assert "inherited_param" not in content
    assert write_dict_table.call_count == 1
    assert write_dict_table.call_args[0][0] == "param"


def test_produce_array_element_report_design_model_lists_all_parameters(
    tmp_path, mocker, run_context
):
    rp = ReadParameters(
        args={"site": "North", "telescope": "LSTN-design", "model_version": "6.0.0"},
        output_path=tmp_path,
        context=run_context,
    )
    mocker.patch(
        "simtools.reporting.docs_read_parameters.TelescopeModel",
        return_value=DummyTelescope(site="North", name="LSTN-design", model_version="6.0.0"),
    )
    mocker.patch.object(
        rp,
        "get_array_element_parameter_data",
        return_value=(
            [["Structure", "param", "1.0.0", "1 m", "desc", "short"]],
            [("dict_param", [{"a": 1, "value": 2}], "m")],
        ),
    )
    write_dict_table = mocker.patch.object(rp, "_write_dict_table")
    mocker.patch("simtools.reporting.docs_read_parameters.write_telescope_structure_figure")

    rp.produce_array_element_report()

    content = (tmp_path / "LSTN-design.md").read_text(encoding="utf-8")
    assert "inherited from the design model" not in content
    assert "| param" in content
    assert write_dict_table.call_count == 1


def test_write_telescope_structure_figure_reports_missing_inputs(
    tmp_test_directory, mocker, run_context
):
    rp = ReadParameters(
        args={"site": "North", "telescope": "LSTN-design", "model_version": "6.0.0"},
        output_path=tmp_test_directory,
        context=run_context,
    )
    rp.context.paths.output_path["default"] = Path(tmp_test_directory)
    telescope_model = Mock()
    telescope_model.name = "LSTN-design"
    telescope_model.model_version = "6.0.0"
    mocker.patch(
        "simtools.reporting.model_figures.telescope_geometry.telescope_geometry",
        side_effect=FileNotFoundError("mirror input missing"),
    )
    buf = StringIO()
    write_telescope_structure_figure(buf, telescope_model, rp._logger, context=rp.context)

    assert "Telescope structure plot unavailable" in buf.getvalue()
    assert "mirror input missing" in buf.getvalue()


def test_produce_model_parameter_reports_writes_latest_dict_table(tmp_path, mocker, run_context):
    rp = ReadParameters(
        args={"site": "North", "telescope": "LSTN-01", "model_version": "6.0.0"},
        output_path=tmp_path,
        context=run_context,
    )
    mocker.patch(
        "simtools.data_model.parameter_catalog.model_parameters", return_value={"dict_param": {}}
    )
    mocker.patch.object(
        rp.model_reader, "get_model_parameters_for_all_model_versions", return_value={}
    )
    mocker.patch.object(
        rp,
        "_compare_parameter_across_versions",
        return_value={
            "dict_param": [
                {
                    "parameter_version": "1.0.0",
                    "model_version": "5.0.0",
                    "value": "v1",
                    "file_flag": False,
                    "dict_table": [{"a": 1, "value": 1}],
                    "dict_unit": "m",
                },
                {
                    "parameter_version": "2.0.0",
                    "model_version": "6.0.0",
                    "value": "v2",
                    "file_flag": False,
                    "dict_table": [{"a": 2, "value": 2}],
                    "dict_unit": "m",
                },
            ]
        },
    )
    mocker.patch.object(
        rp, "get_all_parameter_descriptions", return_value={"dict_param": {"description": "desc"}}
    )
    write_dict_table = mocker.patch.object(rp, "_write_dict_table")

    rp.produce_model_parameter_reports()

    assert write_dict_table.call_count == 1
    assert write_dict_table.call_args[0][0] == "dict_param"
    assert write_dict_table.call_args[0][2] == [{"a": 2, "value": 2}]
    assert write_dict_table.call_args[1] == {"heading_level": 3}
    content = (tmp_path / "LSTN-01.md").read_text(encoding="utf-8")
    assert "## dict_param\n\n**Description**: desc" in content
