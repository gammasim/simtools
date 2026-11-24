"""Tests for deriving flasher analysis and plots from the simulation levels."""

import pytest

from simtools.science_tests import flasher_scan


def _workflow(levels):
    return [
        {
            "application": "simtools-simulate-flasher",
            "configuration": {
                "flasher_photons": levels,
                "number_of_events": 20,
                "run_number": 100,
                "run_mode": "full_simulation",
                "max_workers": 4,
            },
        },
        {
            "application": "simtools-analyze-flasher-response",
            "configuration": {
                "simtel_files": ["/simulation/run__FLASHER_RUN__/event__FLASHER_RUN__.zst"],
            },
        },
        {
            "application": "simtools-plot-simtel-events",
            "configuration": {
                "output_path": "/report/__FLASHER_PHOTONS__",
                "flat_fielding": True,
            },
        },
    ]


def test_expand_flasher_scan_uses_simulation_levels():
    apps = flasher_scan.expand(_workflow([30, 100, 300]))
    assert apps[1]["configuration"]["illumination"] == [30, 100, 300]
    assert apps[1]["configuration"]["events_per_level"] == [20]
    assert apps[1]["configuration"]["simtel_files"] == [
        f"/simulation/run{run}/event{run}.zst" for run in ("000100", "000101", "000102")
    ]
    assert [app["configuration"]["output_path"] for app in apps[2:]] == [
        "/report/30",
        "/report/100",
        "/report/300",
    ]
    assert all(app["configuration"]["flat_fielding"] for app in apps[2:])


def test_expand_flasher_scan_handles_one_level():
    apps = flasher_scan.expand(_workflow([30]))
    assert apps[1]["configuration"]["simtel_files"] == ["/simulation/event000100.zst"]
    assert apps[2]["configuration"]["simtel_file"] == "/simulation/event000100.zst"


@pytest.mark.parametrize("levels", [[], [0, 10], [10, 10]])
def test_expand_flasher_scan_rejects_invalid_levels(levels):
    with pytest.raises(ValueError, match="flasher_photons"):
        flasher_scan.expand(_workflow(levels))


def test_expand_flasher_scan_rejects_serial_multiple_levels():
    apps = _workflow([10, 20])
    apps[0]["configuration"]["max_workers"] = 1
    with pytest.raises(ValueError, match="max_workers"):
        flasher_scan.expand(apps)


def test_expand_flasher_scan_rejects_missing_plot():
    with pytest.raises(ValueError, match="exactly one simtools-plot-simtel-events"):
        flasher_scan.expand(_workflow([30])[:2])


@pytest.mark.parametrize("events", [[], [20, 30, 40], [20, 0]])
def test_expand_flasher_scan_rejects_invalid_event_counts(events):
    apps = _workflow([10, 20])
    apps[0]["configuration"]["number_of_events"] = events
    with pytest.raises(ValueError, match="number_of_events"):
        flasher_scan.expand(apps)


def test_expand_flasher_scan_rejects_direct_injection():
    apps = _workflow([10])
    apps[0]["configuration"]["run_mode"] = "direct_injection"
    with pytest.raises(ValueError, match="full_simulation"):
        flasher_scan.expand(apps)


@pytest.mark.parametrize("path", [[], ["/wrong.zst"], ["/first", "/second"]])
def test_expand_flasher_scan_requires_one_file_template(path):
    apps = _workflow([10])
    apps[1]["configuration"]["simtel_files"] = path
    with pytest.raises(ValueError, match="simtel_files template"):
        flasher_scan.expand(apps)


@pytest.mark.parametrize("field", ["simtel_file", "output_path"])
def test_expand_flasher_scan_rejects_invalid_plot_template(field):
    apps = _workflow([10])
    apps[2]["configuration"][field] = "/wrong"
    with pytest.raises(ValueError, match="plot must omit simtel_file"):
        flasher_scan.expand(apps)


def test_expand_requires_ordered_steps():
    apps = _workflow([10])
    with pytest.raises(ValueError, match="simulation, analysis, and plot in that order"):
        flasher_scan.expand([apps[1], apps[0], apps[2]])


def test_expand_requires_per_run_directory():
    apps = _workflow([10])
    apps[1]["configuration"]["simtel_files"] = ["/file__FLASHER_RUN__.zst"]
    with pytest.raises(ValueError, match="per-run directory"):
        flasher_scan.expand(apps)


def test_expand_rejects_files_outside_simulation_output():
    apps = _workflow([10])
    apps[0]["configuration"]["output_path"] = "/other"
    with pytest.raises(ValueError, match="simulation output_path"):
        flasher_scan.expand(apps)


def test_expand_rejects_run_number_overflow():
    apps = _workflow([10, 20])
    apps[0]["configuration"]["run_number"] = 999999
    with pytest.raises(ValueError, match="run numbers"):
        flasher_scan.expand(apps)


def test_expand_multiple_telescopes_uses_each_simulation_levels():
    import copy

    first = _workflow([30, 100])
    second = copy.deepcopy(_workflow([300, 1000, 3000]))
    for telescope, apps in (("LSTS-01", first), ("SSTS-01", second)):
        apps[0]["configuration"]["telescopes"] = [telescope]
        for app in apps[1:]:
            app["configuration"]["telescope"] = telescope
        apps[1]["configuration"]["simtel_files"][0] = apps[1]["configuration"]["simtel_files"][
            0
        ].replace("/simulation/", f"/{telescope}/simulation/")
        apps[2]["configuration"]["output_path"] = f"/report/{telescope}/__FLASHER_PHOTONS__"
    applications = [first[0], second[0], first[1], second[1], first[2], second[2]]
    result = flasher_scan.expand(applications)
    assert [a["configuration"]["illumination"] for a in result[2:4]] == [
        [30, 100],
        [300, 1000, 3000],
    ]
    assert [a["configuration"]["output_path"] for a in result[4:]] == [
        "/report/LSTS-01/30",
        "/report/LSTS-01/100",
        "/report/SSTS-01/300",
        "/report/SSTS-01/1000",
        "/report/SSTS-01/3000",
    ]
    assert all("LSTS-01" in app["configuration"]["simtel_file"] for app in result[4:6])
    assert all("SSTS-01" in app["configuration"]["simtel_file"] for app in result[6:])


def test_expand_multiple_rejects_duplicate_simulation_telescope():
    import copy

    apps = _workflow([10])
    apps[0]["configuration"]["telescopes"] = ["SSTS-01"]
    duplicate = copy.deepcopy(apps[0])
    apps.insert(1, duplicate)
    for app in apps[2:]:
        app["configuration"]["telescope"] = "SSTS-01"
    with pytest.raises(ValueError, match="one distinct telescope"):
        flasher_scan.expand(apps)


def test_expand_multiple_rejects_unmatched_plot():
    import copy

    a = _workflow([10])
    b = copy.deepcopy(a)
    for telescope, apps in (("LSTS-01", a), ("SSTS-01", b)):
        apps[0]["configuration"]["telescopes"] = [telescope]
        apps[1]["configuration"]["telescope"] = telescope
        apps[2]["configuration"]["telescope"] = telescope
    b[2]["configuration"]["telescope"] = "MSTS-01"
    with pytest.raises(ValueError, match="simtools-plot-simtel-events for SSTS-01"):
        flasher_scan.expand([a[0], b[0], a[1], b[1], a[2], b[2]])


def _two_telescope_templates():
    import copy

    first = _workflow([10])
    second = copy.deepcopy(first)
    for telescope, apps in (("LSTS-01", first), ("SSTS-01", second)):
        apps[0]["configuration"]["telescopes"] = [telescope]
        for app in apps[1:]:
            app["configuration"]["telescope"] = telescope
    return first, second


def test_expand_multiple_rejects_unordered_steps():
    first, second = _two_telescope_templates()
    with pytest.raises(ValueError, match="simulation, analysis, and plot in that order"):
        flasher_scan.expand([first[1], first[0], second[0], second[1], first[2], second[2]])


def test_expand_multiple_rejects_extra_unmatched_step():
    first, second = _two_telescope_templates()
    extra = {
        "application": "simtools-plot-simtel-events",
        "configuration": {"telescope": "MSTS-01"},
    }
    with pytest.raises(ValueError, match="must match a simulation telescope"):
        flasher_scan.expand([first[0], second[0], first[1], second[1], first[2], second[2], extra])
