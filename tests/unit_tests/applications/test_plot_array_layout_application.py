import pytest

import simtools.applications.plot_array_layout as plot_array_layout_app


def test_parser_accepts_parameter_file_layout_selector():
    args = plot_array_layout_app.APPLICATION.build_parser().parse_args(
        [
            "--array_layout_parameter_file",
            "array_layouts.json",
            "--array_layout_name_from_parameter_file",
            "CTAO-South-2-MSTs-5-SSTs",
            "--model_version",
            "7.0.0",
        ]
    )

    assert args.array_layout_parameter_file == "array_layouts.json"
    assert args.array_layout_name_from_parameter_file == ["CTAO-South-2-MSTs-5-SSTs"]


def test_parser_rejects_parameter_file_selector_without_parameter_file(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        [
            "plot_array_layout.py",
            "--array_layout_name",
            "alpha",
            "--array_layout_name_from_parameter_file",
            "beta",
            "--model_version",
            "7.0.0",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        plot_array_layout_app.APPLICATION._parse()

    assert exc.value.code == 2
    assert "requires --array_layout_parameter_file" in capsys.readouterr().err
