"""Test the science CLI forwarding without executing simulations."""

from simtools.applications import run_science_tests


def test_parser_supports_repeatable_selection():
    args = run_science_tests.APPLICATION.build_parser().parse_args(
        [
            "--release_dir",
            "release",
            "--context_file",
            "context.yml",
            "--site",
            "north",
            "--site",
            "south",
            "--test",
            "compare.events",
            "--dry_run",
            "--overwrite",
        ]
    )
    assert args.site == ["north", "south"]
    assert args.test == ["compare.events"]
    assert args.overwrite
    assert args.dry_run


def test_main_forwards_runner_options(mocker):
    args = {
        "release_dir": "release",
        "context_file": "context.yml",
        "dry_run": True,
        "site": ["north"],
        "test": ["compare.events"],
        "env_file": "environment.txt",
    }
    mocker.patch.object(
        run_science_tests.APPLICATION.__class__, "start", return_value=mocker.Mock(args=args)
    )
    execute = mocker.patch.object(run_science_tests, "run_release")
    run_science_tests.main()
    execute.assert_called_once_with(
        "release",
        context_file="context.yml",
        template_dir=None,
        sites=["north"],
        tests=["compare.events"],
        dry_run=True,
        allow_production=False,
        overwrite=False,
        application_args=args,
    )


def test_dry_run_disables_log_file():
    args = {"dry_run": True}
    run_science_tests._post_parse(args, {}, None)
    assert args["disable_log_file"]
