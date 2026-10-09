from pathlib import Path

from simtools.applications import run_science_tests


def test_main_uses_release_context_by_default(mocker):
    args = {
        "release_dir": "release/science_tests",
        "context_file": None,
    }
    mocker.patch(
        "simtools.application.definition.ApplicationDefinition.start",
        return_value=mocker.Mock(args=args),
    )
    run_release = mocker.patch.object(run_science_tests, "run_release")

    run_science_tests.main()

    assert run_release.call_args.kwargs["context_file"] == str(
        Path(args["release_dir"]) / "context.yml"
    )
