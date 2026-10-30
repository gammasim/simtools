from dataclasses import dataclass
from pathlib import Path

import numpy as np

from simtools.reporting import figure_cache


@dataclass
class PlotData:
    x: np.ndarray


def test_figure_cache_key_tracks_data_configuration_and_rendering_inputs(
    tmp_test_directory, monkeypatch, run_context
):
    renderer_one = Path(tmp_test_directory / "renderer_one.py")
    renderer_two = Path(tmp_test_directory / "renderer_two.py")
    renderer_one.write_text("def render(): return 1\n", encoding="utf-8")
    renderer_two.write_text("def render(): return 2\n", encoding="utf-8")
    context = run_context.replace(args={"figure_dpi": 120})
    kwargs = {
        "renderer": "test.renderer",
        "renderer_modules": {"renderer": renderer_one},
        "plot_config": {"title": "Test", "x_label": "Time [ns]"},
        "rendered_data": {"samples": np.array([1.0, 2.0])},
    }

    kwargs["context"] = context
    initial_key = figure_cache.figure_cache_key(**kwargs)
    assert figure_cache.figure_cache_key(**kwargs) == initial_key

    changed_data = {**kwargs, "rendered_data": {"samples": np.array([1.0, 3.0])}}
    assert figure_cache.figure_cache_key(**changed_data) != initial_key
    changed_config = {**kwargs, "plot_config": {"title": "Other"}}
    assert figure_cache.figure_cache_key(**changed_config) != initial_key
    changed_renderer = {**kwargs, "renderer_modules": {"renderer": renderer_two}}
    assert figure_cache.figure_cache_key(**changed_renderer) != initial_key

    kwargs["context"] = run_context.replace(args={"figure_dpi": 240})
    assert figure_cache.figure_cache_key(**kwargs) != initial_key

    monkeypatch.setattr(figure_cache, "BACKEND_NAME", "svg")
    assert figure_cache.figure_cache_key(**kwargs) != initial_key
