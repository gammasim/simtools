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
