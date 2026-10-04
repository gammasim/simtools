"""Test independent selection of shower, telescope, and source configuration writers."""

from types import SimpleNamespace

import pytest

from simtools.model.array_model import ArrayModel
from simtools.model.model_parameter import ModelParameter
from simtools.simtel.light_emission_config_writer import LightEmissionConfigWriter
from simtools.simtel.model_writer import SimtelModelWriter
from simtools.simulation import configuration


@pytest.fixture
def isolated_writers(monkeypatch):
    """Keep test-only software selections local."""
    for name in ("_WRITERS", "_SHOWER_WRITERS", "_LIGHT_SOURCE_WRITERS"):
        monkeypatch.setattr(configuration, name, getattr(configuration, name).copy())


def test_default_model_writer():
    model = SimpleNamespace(configuration_writers={})
    writer = configuration.get_model_writer(model)
    assert isinstance(writer, SimtelModelWriter)
    assert configuration.get_model_writer(model) is writer
    assert configuration.available_model_writers() == ("sim_telarray",)


def test_model_selection_uses_same_resolved_model(isolated_writers, mocker):
    writer = mocker.Mock()
    factory = mocker.Mock(return_value=writer)
    configuration.register_model_writer("example", factory)
    model = SimpleNamespace(configuration_writers={}, parameters={"focal_length": 28})
    ModelParameter.write_config_file(model, "example", additional_models="calibration", label="run")
    writer.write_config_file.assert_called_once_with(additional_models="calibration", label="run")
    ArrayModel.export_config_files(model, "example")
    writer.export_config_files.assert_called_once()
    factory.assert_called_once_with(model)
    assert model.parameters == {"focal_length": 28}
    assert "example" in configuration.available_model_writers()


def test_shower_selection(isolated_writers, mocker):
    expected = SimpleNamespace(simulation_parameters="physical settings")
    factory = mocker.Mock(return_value=expected)
    configuration.register_shower_writer("example", factory)
    assert configuration.get_shower_configuration("array", 7, "test", "example") is expected
    factory.assert_called_once_with(array_model="array", run_number=7, label="test")


def test_default_shower_writer(mocker):
    expected = object()
    factory = mocker.patch("simtools.corsika.corsika_config.CorsikaConfig", return_value=expected)
    assert configuration.get_shower_configuration("array", 1) is expected
    factory.assert_called_once_with(array_model="array", run_number=1, label=None)


def test_light_source_selection_and_cache(isolated_writers, mocker):
    setup = SimpleNamespace(light_emission_config={}, configuration_writers={})
    default = configuration.get_light_source_writer(setup)
    assert isinstance(default, LightEmissionConfigWriter)
    assert configuration.get_light_source_writer(setup) is default
    other = mocker.Mock()
    factory = mocker.Mock(return_value=other)
    configuration.register_light_source_writer("example", factory)
    setup.light_emission_config["light_source_software"] = "example"
    assert configuration.get_light_source_writer(setup) is other
    factory.assert_called_once_with(setup)
    assert default.simulation is setup


@pytest.mark.parametrize(
    "register",
    [
        configuration.register_model_writer,
        configuration.register_shower_writer,
        configuration.register_light_source_writer,
    ],
)
@pytest.mark.parametrize(("name", "factory"), [("", lambda model: None), ("example", None)])
def test_invalid_registration(register, name, factory):
    with pytest.raises(ValueError, match="software name and callable"):
        register(name, factory)


@pytest.mark.parametrize(
    ("register", "name"),
    [
        (configuration.register_model_writer, "sim_telarray"),
        (configuration.register_shower_writer, "corsika"),
        (configuration.register_light_source_writer, "light_emission"),
    ],
)
def test_duplicate_registration(register, name):
    with pytest.raises(ValueError, match="already registered"):
        register(name, lambda model: None)


@pytest.mark.parametrize(
    "getter",
    [
        lambda: configuration.get_model_writer(
            SimpleNamespace(configuration_writers={}), "missing"
        ),
        lambda: configuration.get_shower_configuration("array", 1, simulation_software="missing"),
        lambda: configuration.get_light_source_writer(
            SimpleNamespace(light_emission_config={"light_source_software": "missing"})
        ),
    ],
)
def test_unknown_writer(getter):
    with pytest.raises(ValueError, match=r"Unknown .*configuration writer"):
        getter()
