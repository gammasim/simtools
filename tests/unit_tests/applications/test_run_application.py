#!/usr/bin/env python3

import argparse
from types import SimpleNamespace

from simtools.applications import run_application


def test_add_arguments():
    """Test that _add_arguments only adds config_file and steps arguments."""
    parser = argparse.ArgumentParser()

    run_application._add_arguments(parser)
    args = parser.parse_args(
        [
            "--config_file",
            "config.yml",
            "--steps",
            "1",
            "2",
        ]
    )

    assert args.config_file == "config.yml"
    assert args.steps == [1, 2]


def test_main_pass_args_to_run_applications(monkeypatch):
    """Test that main passes arguments and run_time to run_applications."""
    args = {
        "config_file": "config.yml",
        "steps": [1, 2],
    }
    calls = {}

    monkeypatch.setattr(
        run_application,
        "build_application",
        lambda **_: SimpleNamespace(args=args, run_time=["mock", "runtime"]),
    )

    def _fake_run_applications(app_args, run_time=None):
        calls["app_args"] = app_args.copy()
        calls["run_time"] = run_time

    monkeypatch.setattr(run_application, "run_applications", _fake_run_applications)

    run_application.main()

    assert calls["app_args"] == args
    assert calls["run_time"] == ["mock", "runtime"]


def test_main_handles_none_run_time(monkeypatch):
    """Test that main handles the case where run_time is None."""
    args = {
        "config_file": "config.yml",
    }
    calls = {}

    monkeypatch.setattr(
        run_application,
        "build_application",
        lambda **_: SimpleNamespace(args=args, run_time=None),
    )

    def _fake_run_applications(app_args, run_time=None):
        calls["app_args"] = app_args.copy()
        calls["run_time"] = run_time

    monkeypatch.setattr(run_application, "run_applications", _fake_run_applications)

    run_application.main()

    assert calls["app_args"] == args
    assert calls["run_time"] is None
