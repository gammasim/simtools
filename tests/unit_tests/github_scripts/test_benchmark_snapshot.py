"""Tests for the benchmark snapshot management script."""

import importlib.util
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

SCRIPT_DIRECTORY = Path(__file__).resolve().parents[3] / ".github/scripts"
SPEC = importlib.util.spec_from_file_location(
    "benchmark_snapshot", SCRIPT_DIRECTORY / "benchmark_snapshot.py"
)
BENCHMARK_SNAPSHOT = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(SCRIPT_DIRECTORY))
try:
    SPEC.loader.exec_module(BENCHMARK_SNAPSHOT)
finally:
    sys.path.remove(str(SCRIPT_DIRECTORY))


def _asset(name, asset_id, state="uploaded"):
    return {"name": name, "id": asset_id, "state": state, "created_at": f"2026-01-0{asset_id}"}


def _releases(base, assets=None, tag="benchmark-data", draft=True, count=1):
    release = {"draft": draft, "tag_name": tag, "assets": assets or []}
    path = Path(base) / "releases.json"
    path.write_text(json.dumps([[release] * count]), encoding="utf-8")
    return path


def _snapshots(*ids):
    return [_asset(f"resource-benchmark-snapshot-{i}-1.tar.gz", i) for i in ids]


def test_select_asset_without_release(tmp_test_directory):
    path = _releases(tmp_test_directory, tag="other")
    result = BENCHMARK_SNAPSHOT.select_asset(path)
    assert result == {"release_exists": False, "asset_name": "", "asset_count": 0}


def test_select_asset_newest(tmp_test_directory):
    path = _releases(tmp_test_directory, _snapshots(3, 1, 2))
    result = BENCHMARK_SNAPSHOT.select_asset(path)
    assert result["asset_name"] == "resource-benchmark-snapshot-3-1.tar.gz"
    assert result["asset_count"] == 3


def test_select_asset_errors(tmp_test_directory):
    with pytest.raises(ValueError, match="multiple draft"):
        BENCHMARK_SNAPSHOT.select_asset(_releases(tmp_test_directory, count=2))
    with pytest.raises(ValueError, match="unknown asset"):
        BENCHMARK_SNAPSHOT.select_asset(_releases(tmp_test_directory, [_asset("x.txt", 1)]))
    incomplete = _asset("resource-benchmark-snapshot-1-1.tar.gz", 1, state="starter")
    with pytest.raises(ValueError, match="incomplete"):
        BENCHMARK_SNAPSHOT.select_asset(_releases(tmp_test_directory, [incomplete]))


def test_old_assets(tmp_test_directory):
    path = _releases(tmp_test_directory, _snapshots(1, 2, 3, 4))
    assert BENCHMARK_SNAPSHOT.old_assets(path) == [
        "resource-benchmark-snapshot-1-1.tar.gz",
        "resource-benchmark-snapshot-2-1.tar.gz",
    ]
    assert BENCHMARK_SNAPSHOT.old_assets(_releases(tmp_test_directory, _snapshots(1))) == []
    with pytest.raises(ValueError, match="exactly one"):
        BENCHMARK_SNAPSHOT.old_assets(_releases(tmp_test_directory, tag="other"))


def test_load_state_errors(tmp_test_directory):
    base = Path(tmp_test_directory)
    with pytest.raises(ValueError, match=r"missing state\.json"):
        BENCHMARK_SNAPSHOT.load_state(base / "state.json")
    (base / "state.json").write_text('{"entries": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid"):
        BENCHMARK_SNAPSHOT.load_state(base / "state.json")


def test_build_and_verify_snapshot(tmp_test_directory):
    base = Path(tmp_test_directory)
    state = base / "state.json"
    state.write_text(json.dumps(BENCHMARK_SNAPSHOT.empty_state()), encoding="utf-8")
    page = base / "index.html"
    page.write_text("<html></html>", encoding="utf-8")
    snapshot = base / "snapshot.tar.gz"

    BENCHMARK_SNAPSHOT.build_snapshot(state, page, snapshot, base / "work")
    BENCHMARK_SNAPSHOT.verify_snapshot(snapshot, base / "verified")

    data = (base / "verified/dev/test-resources/data.js").read_text(encoding="utf-8")
    assert data.startswith("window.BENCHMARK_DATA = ")
    assert json.loads(data.removeprefix("window.BENCHMARK_DATA = ").rstrip(";\n"))["entries"] == {}


def test_scripts_run_without_installed_simtools(tmp_test_directory):
    base = Path(tmp_test_directory)
    state = base / "state.json"
    subprocess.run(
        [sys.executable, "-S", SCRIPT_DIRECTORY / "benchmark_snapshot.py", "init-state", state],
        check=True,
        capture_output=True,
        text=True,
    )
    archive_path = base / "snapshot.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(state, arcname="state.json")
    subprocess.run(
        [
            sys.executable,
            "-S",
            SCRIPT_DIRECTORY / "publish_docs.py",
            "extract",
            archive_path,
            base / "out",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads((base / "out/state.json").read_text(encoding="utf-8"))["entries"] == {}


def test_verify_snapshot_incomplete(tmp_test_directory):
    base = Path(tmp_test_directory)
    (base / "src").mkdir()
    (base / "src/state.json").write_text("{}", encoding="utf-8")
    with tarfile.open(base / "bad.tar.gz", "w:gz") as archive:
        archive.add(base / "src/state.json", arcname="state.json")
    with pytest.raises(ValueError, match="incomplete"):
        BENCHMARK_SNAPSHOT.verify_snapshot(base / "bad.tar.gz", base / "out")


def test_main_commands(tmp_test_directory, monkeypatch, capsys):
    base = Path(tmp_test_directory)
    state = base / "state.json"

    def run(*arguments):
        monkeypatch.setattr(sys, "argv", ["benchmark_snapshot.py", *map(str, arguments)])
        BENCHMARK_SNAPSHOT.main()

    run("init-state", state)
    run("validate-state", state)
    run("select", _releases(base, _snapshots(1)))
    assert json.loads(capsys.readouterr().out)["asset_count"] == 1
    run("old-assets", _releases(base, _snapshots(1)))
    (base / "index.html").write_text("x", encoding="utf-8")
    run(
        "build",
        "--state",
        state,
        "--dashboard",
        base / "index.html",
        "--snapshot",
        base / "s.tar.gz",
        "--work-directory",
        base / "work",
    )
    run("verify", base / "s.tar.gz", base / "out")
    state.write_text("[]", encoding="utf-8")
    with pytest.raises(SystemExit, match="invalid"):
        run("validate-state", state)
