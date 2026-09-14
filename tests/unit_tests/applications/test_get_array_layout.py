import simtools.applications.get_array_layout as application


def test_parser_accepts_repository_source():
    args = application.APPLICATION.build_parser().parse_args(
        [
            "--list_available_layouts",
            "--site",
            "North",
            "--model_version",
            "6.0.2",
            "--simulation_models_git_path",
            "models.git",
            "--simulation_models_git_revision",
            "main",
        ]
    )

    assert args.simulation_models_git_path.name == "models.git"
    assert args.simulation_models_git_revision == "main"
