"""Test independent selection of shower, telescope, and source configuration writers."""

from types import SimpleNamespace

import pytest

from simtools.simtel.light_emission_config_writer import LightEmissionConfigWriter
from simtools.simtel.model_writer import SimtelModelWriter
from simtools.simulation import configuration


def test_default_model_writer():
    model = SimpleNamespace(configuration_writers={})
    writer = configuration.get_model_writer(model)
    assert isinstance(writer, SimtelModelWriter)
    assert configuration.get_model_writer(model) is writer


def test_default_shower_writer(mocker):
    expected = object()
    factory = mocker.patch("simtools.corsika.corsika_config.CorsikaConfig", return_value=expected)
    assert configuration.get_shower_configuration("array", 1) is expected
    factory.assert_called_once_with(array_model="array", run_number=1, label=None)


def test_default_light_source_writer_and_cache():
    setup = SimpleNamespace(light_emission_config={}, configuration_writers={})
    writer = configuration.get_light_source_writer(setup)
    assert isinstance(writer, LightEmissionConfigWriter)
    assert configuration.get_light_source_writer(setup) is writer
    assert writer.simulation is setup


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
