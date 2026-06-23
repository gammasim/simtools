#!/usr/bin/env python3

import argparse
from pathlib import Path
from types import SimpleNamespace

from simtools.applications import run_application


def test_add_arguments_runtime_environment_file():
    parser = argparse.ArgumentParser()

    run_application._add_arguments(parser)
    args = parser.parse_args(
        [
            "--config_file",
            "config.yml",
            "--runtime_environment_file",
            "run_time.yml",
        ]
    )

    assert args.runtime_environment_file == Path("run_time.yml")


def test_main_uses_runtime_environment_file(monkeypatch):
    args = {
        "config_file": "config.yml",
        "ignore_runtime_environment": False,
        "runtime_environment_file": Path("run_time.yml"),
    }
    calls = {}

    monkeypatch.setattr(
        run_application,
        "build_application",
        lambda **_: SimpleNamespace(args=args),
    )
    monkeypatch.setattr(
        run_application.simtools_runner,
        "prepare_runtime_environment",
        lambda runtime_file: ({"image": f"image-from-{runtime_file}"}, ["runtime", "image"]),
    )

    def _fake_run_applications(app_args, run_time=None):
        calls["app_args"] = app_args.copy()
        calls["run_time"] = run_time

    monkeypatch.setattr(run_application.simtools_runner, "run_applications", _fake_run_applications)

    run_application.main()

    assert calls["app_args"]["runtime_environment"] == {"image": "image-from-run_time.yml"}
    assert calls["run_time"] == ["runtime", "image"]


def test_main_ignores_runtime_environment_file_when_requested(monkeypatch):
    args = {
        "config_file": "config.yml",
        "ignore_runtime_environment": True,
        "runtime_environment_file": Path("run_time.yml"),
    }
    calls = {}

    monkeypatch.setattr(
        run_application,
        "build_application",
        lambda **_: SimpleNamespace(args=args),
    )
    monkeypatch.setattr(
        run_application.simtools_runner,
        "prepare_runtime_environment",
        lambda _: (_ for _ in ()).throw(AssertionError("should not prepare runtime")),
    )

    def _fake_run_applications(app_args, run_time=None):
        calls["app_args"] = app_args.copy()
        calls["run_time"] = run_time

    monkeypatch.setattr(run_application.simtools_runner, "run_applications", _fake_run_applications)

    run_application.main()

    assert "runtime_environment" not in calls["app_args"]
    assert calls["run_time"] is None
