from types import SimpleNamespace

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


def test_run_lists_layouts_from_supplied_reader(mocker, capsys):
    site_model = mocker.patch.object(application, "SiteModel")
    site_model.return_value.get_list_of_array_layouts.return_value = ["layout-a", "layout-b"]
    reader = object()
    context = SimpleNamespace(
        args={"list_available_layouts": True, "site": "North", "model_version": "6.0.2"},
        model_reader=reader,
    )

    application.run(context)

    assert capsys.readouterr().out.strip() == "['layout-a', 'layout-b']"
    site_model.assert_called_once_with(model_version="6.0.2", site="North", model_reader=reader)


def test_run_exports_layout_from_supplied_reader(mocker):
    layout = mocker.Mock()
    reader = object()
    layout_from_source = mocker.patch.object(
        application, "_layout_from_source", return_value=layout
    )
    context = SimpleNamespace(
        args={
            "list_available_layouts": False,
            "array_layout_name": "layout-a",
            "output_file_from_default": True,
        },
        model_reader=reader,
        logger=SimpleNamespace(info=mocker.Mock()),
    )

    application.run(context)

    layout_from_source.assert_called_once_with(context.args, reader)
    layout.pprint.assert_called_once_with()
