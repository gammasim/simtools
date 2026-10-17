"""Tests for trigger-patch rendering and visualization import boundaries."""

import ast
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

import simtools
from simtools.simtel import trigger_patch_validation as validation
from simtools.visualization import plot_trigger_patches as plots

REPORT_PDF = "report.pdf"
SAVEFIG = "savefig"
SAVE_FAILURE = "Unable to save report page"
PIXEL_SHAPE = "pixel_shape"
PIXEL_DIAMETER = "pixel_diameter"

VISUALIZATION_DIRECTORY = "visualization"
MATPLOTLIB_MODULE = "matplotlib"
MATPLOTLIB_BACKEND_MODULE = "simtools.visualization.matplotlib_backend"
MATPLOTLIB_IMPORT_FUNCTIONS = {"import_module", "lazy_module"}


def test_matplotlib_imports_are_confined_to_visualization():
    """Reject direct or lazy Matplotlib imports from other simtools modules."""
    source_directory = Path(simtools.__file__).parent
    violations = []
    for path in source_directory.rglob("*.py"):
        if VISUALIZATION_DIRECTORY in path.relative_to(source_directory).parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if _imports_matplotlib(node):
                violations.append(str(path.relative_to(source_directory)))
                break

    assert violations == []


def _imports_matplotlib(node):
    if isinstance(node, ast.Import):
        return any(alias.name.split(".")[0] == MATPLOTLIB_MODULE for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        if node.module and (
            node.module.split(".")[0] == MATPLOTLIB_MODULE
            or node.module == MATPLOTLIB_BACKEND_MODULE
        ):
            return True
        if node.module == "simtools.visualization" and any(
            alias.name == "matplotlib_backend" for alias in node.names
        ):
            return True
    if isinstance(node, ast.Call):
        function = node.func
        function_name = getattr(function, "id", getattr(function, "attr", None))
        return (
            function_name in MATPLOTLIB_IMPORT_FUNCTIONS
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and node.args[0].value.startswith(MATPLOTLIB_MODULE)
        )
    return False


def test_fired_patch_outlines_enclose_whole_patches_without_pixel_edges():
    camera = Mock(
        pixels={
            plots.PIXEL_IDS: [0, 1, 2],
            "x": [0, 1, 2],
            "y": [0, 0, 0],
            PIXEL_SHAPE: 2,
            PIXEL_DIAMETER: 1,
        }
    )
    figure, axis = plots.plt.subplots()
    try:
        plots._overlay_trigger_groups(axis, camera, [{0, 1}, {1, 2}], [0, 1, 99], [1])
        assert len(axis.patches) == 2
        assert axis.patches[0].get_edgecolor() != axis.patches[1].get_edgecolor()
        assert [patch.get_label() for patch in axis.patches] == ["Patch 0", "Patch 1"]
        for patch, left in zip(axis.patches, [-0.5, 0.5]):
            vertices = patch.get_xy()
            assert len(vertices) == 5
            assert np.isclose(vertices[:, 0].min(), left, rtol=1e-12, atol=1e-12)
            assert np.isclose(vertices[:, 0].max(), left + 2, rtol=1e-12, atol=1e-12)
            assert np.isclose(vertices[:, 1].min(), -0.5, rtol=1e-12, atol=1e-12)
            assert np.isclose(vertices[:, 1].max(), 0.5, rtol=1e-12, atol=1e-12)
        assert len(axis.collections) == 1
        assert np.asarray(axis.collections[0].get_offsets()) == pytest.approx(np.array([[1, 0]]))
    finally:
        plots.plt.close(figure)


def test_patch_outline_skips_empty_or_unsupported_pixel_shapes():
    camera = Mock(pixels={plots.PIXEL_IDS: [0], "x": [0], "y": [0], PIXEL_SHAPE: 0})
    figure, axis = plots.plt.subplots()
    try:
        plots._outline_patch(axis, camera, {0})
        plots._outline_patch(axis, camera, set())
        assert not axis.patches
    finally:
        plots.plt.close(figure)


def test_patch_palette_cycles_with_distinct_line_styles():
    camera = Mock(
        pixels={
            plots.PIXEL_IDS: [0],
            "x": [0],
            "y": [0],
            PIXEL_SHAPE: 2,
            PIXEL_DIAMETER: 1,
        }
    )
    figure, axis = plots.plt.subplots()
    try:
        plots._overlay_trigger_groups(axis, camera, [{0}] * 6, range(6), [])
        assert len({patch.get_edgecolor() for patch in axis.patches}) == 3
        assert axis.patches[0].get_linestyle() != axis.patches[3].get_linestyle()
        assert axis.get_legend() is not None
    finally:
        plots.plt.close(figure)


def test_observed_patch_efficiency_excludes_untested_groups():
    groups = [
        {validation.GROUP_ID: 0, validation.OBSERVED: 4, validation.INTENDED_SECTOR_FIRED: 3},
        {validation.GROUP_ID: 1, validation.OBSERVED: 0, validation.INTENDED_SECTOR_FIRED: 0},
        {validation.GROUP_ID: 2, validation.OBSERVED: 2, validation.INTENDED_SECTOR_FIRED: 0},
    ]
    assert plots._observed_patch_efficiency(groups) == pytest.approx([0.75, np.nan, 0], nan_ok=True)


@pytest.fixture
def report_data():
    camera = Mock(
        telescope_name="LSTN-01",
        pixels={
            plots.PIXEL_IDS: [0, 1],
            "x": [0, 1],
            "y": [0, 1],
            PIXEL_SHAPE: 2,
            PIXEL_DIAMETER: 1,
        },
    )
    groups = [
        {
            validation.GROUP_ID: 0,
            validation.MEMBERS: "[0, 1]",
            validation.PLANNED: 1,
            validation.OBSERVED: 1,
            validation.TRIGGERED: 1,
            validation.INTENDED_SECTOR_FIRED: 1,
        }
    ]
    rows = [{plots.PHOTOELECTRON_PIXELS: "[0]"}]
    summary = dict.fromkeys(
        (
            validation.PLANNED,
            validation.OBSERVED,
            validation.TRIGGERED,
            validation.NON_TRIGGERED,
            validation.MISSING,
            plots.GROUP_COUNT,
            plots.GROUPS_SAMPLED,
            plots.MAPPING_MISMATCHES,
        ),
        1,
    )
    return camera, groups, rows, summary


@pytest.mark.parametrize("event_count", [0, 1, 3])
def test_write_report_renders_summary_and_event_pages(
    tmp_test_directory, mocker, report_data, event_count
):
    output = Path(tmp_test_directory) / REPORT_PDF
    pages = mocker.spy(plots.PdfPages, SAVEFIG)
    figures = plots.plt.get_fignums()
    events = ({validation.EVENT_ID: index} for index in range(event_count))
    plots.write_report(output, *report_data, events, 1, [{0, 1}], [])
    assert output.read_bytes().startswith(b"%PDF-")
    assert pages.call_count == 3 + event_count
    assert plots.plt.get_fignums() == figures


@pytest.mark.parametrize("failed_page", [1, 2, 3, 4])
def test_write_report_closes_figures_on_failure(
    tmp_test_directory, mocker, report_data, failed_page
):
    failure = ValueError(SAVE_FAILURE)
    mocker.patch.object(plots.PdfPages, SAVEFIG, side_effect=[None] * (failed_page - 1) + [failure])
    figures = plots.plt.get_fignums()
    with pytest.raises(ValueError, match=SAVE_FAILURE):
        plots.write_report(
            tmp_test_directory / REPORT_PDF,
            *report_data,
            [{validation.EVENT_ID: 1}],
            1,
            [{0, 1}],
            [],
        )
    assert plots.plt.get_fignums() == figures


@pytest.mark.parametrize("shower_number", [None, 0, 1, 2])
def test_trigger_event_context_resolves_input_and_fired_groups(shower_number):
    event = {
        validation.MC_EVENT: {validation.SHOWER_NUMBER: shower_number},
        plots.MC_PHOTOELECTRONS: {5: {plots.MC_PHOTOELECTRONS: np.array([0, 3])}},
    }
    context = plots._trigger_event_context(
        event, {validation.HEADER: {validation.LIST_TRGSECT: [1]}}, 1, [{0, 1}, {1}], [[1]]
    )
    expected_pixels = [1] if shower_number == 1 else []
    expected_groups = [0, 1] if expected_pixels else []
    assert context == (expected_pixels, [1], [1], expected_groups)


def test_trigger_event_context_handles_missing_metadata():
    assert plots._trigger_event_context({}, {}, 1, [{0}], []) == ([], [], [], [])


def test_event_image_uses_photoelectrons_without_waveforms(report_data):
    image, label = plots._event_image(
        {plots.MC_PHOTOELECTRONS: {0: {plots.MC_PHOTOELECTRONS: np.array([3, 0])}}},
        {},
        report_data[0],
    )
    assert image == pytest.approx([3, 0])
    assert label == "MC photoelectrons (no recorded waveform)"


def test_event_image_integrates_waveforms(report_data):
    samples = np.full((1, 2, 40), 10.0)
    samples[0, 0, 8:12] = 30.0
    image, label = plots._event_image({}, {plots.ADC_SAMPLES: samples}, report_data[0])
    assert image == pytest.approx([80, 0])
    assert label == "Integrated signal [d.c.]"


def test_pixel_coverage_uses_model_ids():
    camera = Mock(pixels={plots.PIXEL_IDS: [4, 16, 7]})
    assert plots._pixel_coverage(camera, [[16, 4], [16]]) == pytest.approx([1, 2, 0])


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import matplotlib", True),
        ("import matplotlib.pyplot as plt", True),
        ("from matplotlib import pyplot", True),
        ("from simtools.visualization import matplotlib_backend", True),
        ("from simtools.visualization.matplotlib_backend import pyplot", True),
        ('lazy_module("matplotlib.pyplot")', True),
        ('importlib.import_module("matplotlib.pyplot")', True),
        ("import numpy", False),
        ('lazy_module("scipy.spatial")', False),
    ],
)
def test_matplotlib_import_detection(source, expected):
    assert any(_imports_matplotlib(node) for node in ast.walk(ast.parse(source))) is expected
