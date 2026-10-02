"""CLI wiring for workflow preparation and optional execution."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from simtools.applications import create_setting_workflow


@pytest.mark.parametrize("execute", [False, True])
def test_main(mocker, tmp_test_directory, execute):
    context = SimpleNamespace(
        args={
            "instrument": "LSTN-design",
            "parameter": "min_photons",
            "parameter_version": "2.0.1",
            "run": execute,
        },
        model_reader=mocker.Mock(),
        logger=mocker.Mock(),
    )
    mocker.patch(
        "simtools.application.definition.ApplicationDefinition.start", return_value=context
    )
    config = Path(tmp_test_directory) / "activity" / "config.yml"
    prepare = mocker.patch.object(
        create_setting_workflow, "create_setting_workflow", return_value=config
    )
    run = mocker.patch.object(create_setting_workflow, "run_setting_workflow")
    create_setting_workflow.main()
    prepare.assert_called_once_with(context.args, context.model_reader)
    context.logger.info.assert_any_call(
        "Add the following to the corresponding production-info file:\n%s",
        'LSTN-design:\n  min_photons:\n    version: "2.0.1"\n    activity_id: activity',
    )
    if execute:
        run.assert_called_once_with(config, context.args)
    else:
        run.assert_not_called()


def test_runtime_alias_is_not_prepared_during_startup():
    parser = create_setting_workflow.APPLICATION.build_parser()
    args = parser.parse_args(
        [
            "--instrument",
            "LSTN-design",
            "--parameter",
            "min_photons",
            "--value",
            "120",
            "--parameter_version",
            "2.0.1",
            "--description",
            "Updated threshold",
            "--runtime_environment_file",
            "runtime.yml",
        ]
    )
    assert str(args.workflow_runtime_file) == "runtime.yml"
    assert not hasattr(args, "runtime_environment_file")
