from pathlib import Path

import pytest

from simtools.testing import resource_sync


@pytest.fixture
def resource_roots(tmp_test_directory, monkeypatch):
    root = Path(tmp_test_directory)
    source_root = root / "simtools-tests" / "v0.35.0" / "integration_tests"
    destination_root = root / "simtools" / "tests" / "resources"
    for resource_dir in ("static", "generated"):
        (source_root / resource_dir).mkdir(parents=True, exist_ok=True)
        (destination_root / resource_dir).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        resource_sync.constants, "TEST_RESOURCES_STATIC", str(destination_root / "static")
    )
    monkeypatch.setattr(
        resource_sync.constants,
        "TEST_RESOURCES_GENERATED",
        str(destination_root / "generated"),
    )
    return root, source_root, destination_root


def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_build_sync_report_classifies_files(resource_roots):
    root, source_root, destination_root = resource_roots
    _write(source_root / "static" / "only_source.txt", "new")
    _write(source_root / "static" / "same.txt", "same")
    _write(destination_root / "static" / "same.txt", "same")
    _write(source_root / "generated" / "nested" / "changed.txt", "source")
    _write(destination_root / "generated" / "nested" / "changed.txt", "dest")
    _write(destination_root / "generated" / "obsolete.txt", "old")
    _write(source_root / "log_files" / "ignored.log", "ignore")

    report = resource_sync.build_sync_report(
        root / "simtools-tests", "v0.35.0", ("static", "generated")
    )

    assert report["directories"]["static"]["new"] == ["only_source.txt"]
    assert report["directories"]["static"]["unchanged"] == ["same.txt"]
    assert report["directories"]["generated"]["changed"] == ["nested/changed.txt"]
    assert report["directories"]["generated"]["obsolete"] == ["obsolete.txt"]
    assert report["summary"] == {"new": 1, "changed": 1, "unchanged": 1, "obsolete": 1}


def test_apply_sync_actions_copies_new_and_changed(resource_roots):
    root, source_root, destination_root = resource_roots
    _write(source_root / "static" / "new.txt", "new")
    _write(source_root / "generated" / "changed.txt", "source")
    _write(destination_root / "generated" / "changed.txt", "destination")

    report = resource_sync.build_sync_report(
        root / "simtools-tests", "v0.35.0", ("static", "generated")
    )
    actions = resource_sync.apply_sync_actions(report, sync=True, delete_missing=False)

    assert (destination_root / "static" / "new.txt").read_text(encoding="utf-8") == "new"
    assert (destination_root / "generated" / "changed.txt").read_text(encoding="utf-8") == "source"
    assert sorted(actions["copied"]) == ["generated/changed.txt", "static/new.txt"]
    assert actions["deleted"] == []


def test_apply_sync_actions_deletes_obsolete_and_empty_directories(resource_roots):
    root, _, destination_root = resource_roots
    _write(destination_root / "generated" / "nested" / "obsolete.txt", "obsolete")

    report = resource_sync.build_sync_report(root / "simtools-tests", "v0.35.0", ("generated",))
    actions = resource_sync.apply_sync_actions(report, sync=False, delete_missing=True)

    assert not (destination_root / "generated" / "nested" / "obsolete.txt").exists()
    assert not (destination_root / "generated" / "nested").exists()
    assert (destination_root / "generated").exists()
    assert actions["copied"] == []
    assert actions["deleted"] == ["generated/nested/obsolete.txt"]


def test_apply_sync_actions_preflights_delete_targets(resource_roots):
    root, _, destination_root = resource_roots
    safe_file = destination_root / "static" / "obsolete.txt"
    outside_file = root / "outside.txt"
    _write(safe_file, "obsolete")
    _write(outside_file, "outside")
    report = {
        "destination_directories": {"static": destination_root / "static"},
        "directories": {
            "static": {
                "new": [],
                "changed": [],
                "obsolete": ["obsolete.txt", "outside.txt"],
                "source_files": {},
                "destination_files": {
                    "obsolete.txt": safe_file,
                    "outside.txt": outside_file,
                },
            }
        },
    }

    with pytest.raises(ValueError, match="Refusing to delete file outside test resources"):
        resource_sync.apply_sync_actions(report, sync=False, delete_missing=True)

    assert safe_file.exists()
    assert outside_file.exists()


def test_apply_sync_actions_refuses_symlink_to_outside_resource_root(resource_roots):
    root, _, destination_root = resource_roots
    outside_file = root / "outside.txt"
    symlink = destination_root / "static" / "outside-link.txt"
    _write(outside_file, "outside")
    symlink.symlink_to(outside_file)
    report = {
        "destination_directories": {"static": destination_root / "static"},
        "directories": {
            "static": {
                "new": [],
                "changed": [],
                "obsolete": ["outside-link.txt"],
                "source_files": {},
                "destination_files": {"outside-link.txt": symlink},
            }
        },
    }

    with pytest.raises(ValueError, match="Refusing to delete file outside test resources"):
        resource_sync.apply_sync_actions(report, sync=False, delete_missing=True)

    assert symlink.exists()
    assert outside_file.exists()


def test_build_sync_report_uses_configured_resources_path(resource_roots):
    root, source_root, _ = resource_roots
    custom_destination = root / "custom-resources"
    _write(source_root / "static" / "input.txt", "new")

    report = resource_sync.build_sync_report(
        root / "simtools-tests",
        "v0.35.0",
        ("static",),
        resources_path=custom_destination,
    )

    assert report["destination_root"] == custom_destination.resolve()
    assert report["destination_directories"]["static"] == custom_destination.resolve() / "static"
    assert report["directories"]["static"]["new"] == ["input.txt"]


def test_selected_resource_directories_requires_one_choice():
    with pytest.raises(ValueError, match="Select at least one"):
        resource_sync._selected_resource_directories(
            {"include_static": False, "include_generated": False}
        )


def test_build_sync_report_requires_existing_source_integration_directory(resource_roots):
    root, _, _ = resource_roots

    with pytest.raises(FileNotFoundError, match="Source integration-test directory does not exist"):
        resource_sync.build_sync_report(root / "simtools-tests", "v0.36.0", ("static",))


def test_build_sync_report_requires_existing_source_resource_directory(resource_roots):
    root, source_root, _ = resource_roots
    (source_root / "static").rmdir()

    with pytest.raises(FileNotFoundError, match="Source resource directory does not exist"):
        resource_sync.build_sync_report(root / "simtools-tests", "v0.35.0", ("static",))


def test_render_sync_report_includes_obsolete(resource_roots):
    root, source_root, destination_root = resource_roots
    _write(source_root / "static" / "new.txt", "new")
    _write(destination_root / "generated" / "old.txt", "old")

    report = resource_sync.build_sync_report(
        root / "simtools-tests", "v0.35.0", ("static", "generated")
    )
    report_text = resource_sync.render_sync_report(report)

    assert "Summary: new=1, changed=0, unchanged=0, obsolete=1" in report_text
    assert "new (static):" in report_text
    assert "obsolete (generated):" in report_text


def test_sync_test_resources_report_only(resource_roots, caplog):
    root, source_root, destination_root = resource_roots
    _write(source_root / "static" / "only_source.txt", "new")
    _write(destination_root / "generated" / "obsolete.txt", "old")

    caplog.set_level("INFO", logger="simtools.testing.resource_sync")
    report, actions = resource_sync.sync_test_resources(
        {
            "test_directory": root / "simtools-tests",
            "simtools_version": "v0.35.0",
            "include_static": True,
            "include_generated": True,
            "sync": False,
            "delete_missing": False,
        }
    )

    assert report["summary"] == {"new": 1, "changed": 0, "unchanged": 0, "obsolete": 1}
    assert actions == {"copied": [], "deleted": []}
    assert "obsolete (generated):" in caplog.text
