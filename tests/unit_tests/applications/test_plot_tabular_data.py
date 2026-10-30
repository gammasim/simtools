"""Tests for the plot_tabular_data application (data file and model parameter modes)."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

import simtools.applications.plot_tabular_data as app


def _context(**args):
    paths = Mock()
    paths.get_output_directory.return_value = Path("out")
    paths.get_output_file.side_effect = lambda name: Path("out") / name
    return SimpleNamespace(args=args, paths=paths, model_reader=Mock(), logger=Mock())


@pytest.fixture(name="calls")
def calls_fixture(monkeypatch):
    """Replace plotting and metadata output by recorders (no files are written)."""
    calls = {"plot": [], "dump": [], "generate": []}

    def _generate(**kwargs):
        calls["generate"].append(kwargs)
        return [{"id": 1}, {"id": 2}], [Path("out/a.pdf"), Path("out/b.pdf")]

    monkeypatch.setattr(
        app,
        "plot_tables",
        SimpleNamespace(
            plot=lambda **kwargs: calls["plot"].append(kwargs),
            generate_plot_configurations=_generate,
        ),
    )
    monkeypatch.setattr(
        app,
        "MetadataCollector",
        SimpleNamespace(dump=lambda *args, **kwargs: calls["dump"].append((args, kwargs))),
    )
    return calls


def test_parser_exposes_both_plot_sources():
    actions = {action.dest: action for action in app.APPLICATION.build_parser()._actions}

    assert {"plot_config", "parameter", "plot_type", "output_file"} <= set(actions)


def test_lower_dict_keys_is_recursive():
    assert app._lower_dict_keys({"A": [{"B": 1}], "C": {"D": 2}}) == {
        "a": [{"b": 1}],
        "c": {"d": 2},
    }


def test_data_file_mode_requires_output_file():
    with pytest.raises(ValueError, match="output_file"):
        app._plot_data_file(_context(plot_config="config.yml", output_file=None))


def test_model_parameter_mode_requires_plot_type():
    with pytest.raises(ValueError, match="plot_type"):
        app._plot_model_parameter(_context(parameter="fadc_pulse_shape", plot_type=None))


def test_data_file_mode_plots_lowercased_configuration(tmp_test_directory, calls, monkeypatch):
    config_file = Path(str(tmp_test_directory)) / "plot.yml"
    config_file.write_text(yaml.safe_dump({"PLOT": {"Type": "line"}}), encoding="utf-8")
    monkeypatch.setattr(app.schema, "validate_dict_using_schema", lambda data, _: data)

    app._plot_data_file(_context(plot_config=str(config_file), output_file="figure.pdf"))

    assert calls["plot"][0]["config"] == {"type": "line"}
    assert calls["plot"][0]["output_file"] == Path("out/figure.pdf")
    assert len(calls["dump"]) == 1


def test_model_parameter_mode_plots_every_generated_configuration(calls):
    app._plot_model_parameter(
        _context(
            parameter="fadc_pulse_shape",
            parameter_version="1.0.0",
            site="North",
            telescope=None,
            plot_type="all",
        )
    )

    assert [call["config"] for call in calls["plot"]] == [{"id": 1}, {"id": 2}]
    assert len(calls["dump"]) == 2
