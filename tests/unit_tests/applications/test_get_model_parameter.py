from pathlib import Path
from types import SimpleNamespace

import pytest

import simtools.applications.get_model_parameter as application


def test_parser_accepts_repository_source_and_file_export():
    args = application.APPLICATION.build_parser().parse_args(
        [
            "--parameter",
            "mirror_list",
            "--model_version",
            "6.0.2",
            "--simulation_models_path",
            "models",
            "--export_model_file",
        ]
    )

    assert args.simulation_models_path.name == "models"
    assert args.export_model_file is True
    assert args.output_file is None


def test_export_file_backed_parameter_uses_reader_and_override(tmp_test_directory):
    output_path = Path(str(tmp_test_directory)) / "output"
    source_path = output_path / "source.dat"
    source_path.parent.mkdir(exist_ok=True)
    source_path.write_text("data", encoding="utf-8")
    context = SimpleNamespace(
        args={
            "parameter": "file_parameter",
            "site": "North",
            "telescope": "LSTN-01",
            "model_version": "6.0.2",
            "parameter_version": None,
            "output_file": "renamed.dat",
            "export_model_file": True,
            "export_model_file_as_table": False,
        },
        io_handler=SimpleNamespace(
            get_output_directory=lambda: output_path,
            get_output_file=lambda name: output_path / name,
        ),
    )
    reader = SimpleNamespace(
        export_model_file=lambda **kwargs: None,
    )
    parameters = {"file_parameter": {"file": True, "value": "source.dat"}}

    result = application._export_parameter_file(context, reader, parameters)

    assert result == [output_path / "renamed.dat"]
    assert (output_path / "renamed.dat").read_text(encoding="utf-8") == "data"


def test_export_non_file_parameter_is_rejected(tmp_test_directory):
    context = SimpleNamespace(
        args={
            "parameter": "scalar_parameter",
            "site": "North",
            "telescope": "LSTN-01",
            "model_version": "6.0.2",
            "parameter_version": None,
            "output_file": None,
            "export_model_file": True,
            "export_model_file_as_table": False,
        },
        io_handler=SimpleNamespace(get_output_directory=lambda: tmp_test_directory),
    )

    with pytest.raises(ValueError, match="does not reference a model file"):
        application._export_parameter_file(
            context,
            SimpleNamespace(),
            {"scalar_parameter": {"file": False, "value": 42}},
        )


def test_run_reads_parameter_from_configured_model_reader(capsys):
    context = SimpleNamespace(
        args={
            "parameter": "scalar_parameter",
            "site": "North",
            "telescope": "LSTN-01",
            "model_version": "6.0.2",
            "parameter_version": None,
            "output_file": None,
            "export_model_file": False,
            "export_model_file_as_table": False,
        },
        model_reader=SimpleNamespace(
            get_model_parameter=lambda **kwargs: {
                "scalar_parameter": {"parameter": "scalar_parameter", "value": 42}
            }
        ),
        logger=SimpleNamespace(info=lambda *args: None),
    )

    application.run(context)

    assert "scalar_parameter" in capsys.readouterr().out
