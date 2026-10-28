from unittest.mock import MagicMock

import pytest

from simtools.constants import SIMTEL_MAX_SEED
from simtools.simtel.simtel_seeds import SimtelSeeds, instance_seed_pairs
from simtools.utils import random


@pytest.fixture
def mock_config(mocker):
    """Fixture providing a mocked configuration."""
    mock = MagicMock()
    mock.args = {}
    mocker.patch("simtools.simtel.simtel_seeds.settings.config", mock)
    return mock


@pytest.fixture
def seeds_instance(mock_config):
    """Fixture providing a SimtelSeeds instance."""
    return SimtelSeeds()


@pytest.mark.parametrize(
    ("simulation_seed", "instruments", "expected_method", "return_value"),
    [
        ([123, 456], None, "_set_fixed_seeds", "123,456"),
        ([123], None, "_generate_seed_pair", "100,123"),
        (None, None, "_generate_seed_pair", "100,789"),
        (300, 1, "_generate_seed_pair", "200,300"),
        (400, 10, "_generate_seeds_with_file", "file-by-run:/path/to/seeds,400"),
    ],
)
def test_initialize_seeds_routes_correctly(
    mocker, mock_config, simulation_seed, instruments, expected_method, return_value
):
    if expected_method == "_generate_seed_pair":
        mocker.patch("simtools.simtel.simtel_seeds.random.seeds", return_value=789)
    mocker.patch.object(SimtelSeeds, expected_method, return_value=return_value)

    seeds = SimtelSeeds()
    seeds.simulation_seed = simulation_seed
    seeds.instruments = instruments
    result = seeds.initialize_seeds("north", "1.0.0", 20.0, 180.0)

    assert result == return_value
    if simulation_seed is None:
        assert seeds.simulation_seed == 789
    if simulation_seed == [123]:
        assert seeds.simulation_seed == 123


def test_initialize_seeds_raises_for_too_many_fixed_seeds(mock_config):
    seeds = SimtelSeeds()
    seeds.simulation_seed = [1, 2, 3]

    with pytest.raises(ValueError, match="simulation_seed must contain one value"):
        seeds.initialize_seeds("north", "1.0.0", 20.0, 180.0)


def test_initialize_seeds_single_value_uses_configured_instrument_seed(mocker, mock_config):
    mock_config.args = {"instrument_seed": 1745, "simulation_seed": [290]}
    random_seed_mock = mocker.patch("simtools.simtel.simtel_seeds.random.seeds", return_value=999)

    seeds = SimtelSeeds()

    result = seeds.seed_string

    assert result == "1745,290"
    assert seeds.instrument_seed == 1745
    assert seeds.simulation_seed == 290
    random_seed_mock.assert_not_called()


def test_set_fixed_seeds(mocker, mock_config):
    mocker.patch.object(SimtelSeeds, "initialize_seeds", return_value="mock")
    mock_logger = mocker.patch("simtools.simtel.simtel_seeds.logging.getLogger")

    seeds = SimtelSeeds()
    seeds.simulation_seed = [123, 456]
    result = seeds._set_fixed_seeds()
    assert result == "123,456"
    mock_logger.return_value.warning.assert_called_once()
    assert "123,456" in mock_logger.return_value.warning.call_args[0][0]

    seeds.simulation_seed = [789]
    with pytest.raises(IndexError, match="Two seeds must be provided"):
        seeds._set_fixed_seeds()


@pytest.mark.parametrize(
    ("instrument_seed", "expected_generates"),
    [
        (None, True),
        (700, False),
    ],
)
def test_generate_seed_pair(mocker, mock_config, instrument_seed, expected_generates):
    """Test _generate_seed_pair generates or uses existing instrument seed."""
    if expected_generates:
        mocker.patch("simtools.simtel.simtel_seeds.random.seeds", return_value=500)
    mocker.patch.object(SimtelSeeds, "initialize_seeds", return_value="mock")
    mock_logger = mocker.patch("simtools.simtel.simtel_seeds.logging.getLogger")

    seeds = SimtelSeeds()
    seeds.instrument_seed = instrument_seed
    seeds.simulation_seed = 600
    result = seeds._generate_seed_pair()

    if expected_generates:
        assert seeds.instrument_seed == 500
        assert result == "500,600"
    else:
        assert result == "700,600"

    mock_logger.return_value.info.assert_called_once()
    assert str(seeds.instrument_seed) in mock_logger.return_value.info.call_args[0][0]


def test_generate_seeds_with_file(mocker, mock_config, tmp_path):
    mocker.patch.object(SimtelSeeds, "initialize_seeds", return_value="mock")
    mocker.patch("simtools.simtel.simtel_seeds.random.seeds", return_value=[101, 102, 103])
    mock_config.args = {"sim_telarray_seed_file": "seeds.txt"}
    mock_logger = mocker.patch("simtools.simtel.simtel_seeds.logging.getLogger")

    seeds = SimtelSeeds()
    seeds.seed_file = tmp_path / "seeds.txt"
    seeds.instruments = 3
    seeds.simulation_seed = 200
    seeds.instrument_seed = None

    result = seeds._generate_seeds_with_file("north", "1.0.0", 20.0, 180.0)

    assert seeds.seed_file.exists()
    content = seeds.seed_file.read_text()
    assert all(str(n) in content for n in [101, 102, 103])
    assert "model version 1.0.0" in content
    assert f"file-by-run:{seeds.seed_file},200" == result
    mock_logger.return_value.info.assert_called_once()

    seeds.instruments = 1025
    with pytest.raises(
        ValueError, match="Number of random instances of instrument must be less than 1024"
    ):
        seeds._generate_seeds_with_file("north", "1.0.0", 20.0, 180.0)


def test_get_instrument_seed_deterministic(mocker, mock_config):
    mocker.patch(
        "simtools.simtel.simtel_seeds.names.site_names",
        return_value={"North": ["north"], "South": ["south"]},
    )

    seeds = SimtelSeeds()
    seeds.instrument_seed = None

    result_base = seeds._get_instrument_seed("north", "1.0.0", 20.0, 180.0)
    assert result_base == 100001020180
    result_site = seeds._get_instrument_seed("south", "1.0.0", 20.0, 180.0)
    assert result_site == 100002020180
    result_zenith = seeds._get_instrument_seed("north", "1.0.0", 30.0, 180.0)
    assert result_zenith == 100001030180
    result_azimuth = seeds._get_instrument_seed("north", "1.0.0", 20.0, 270.0)
    assert result_azimuth == 100001020270

    assert len({result_base, result_site, result_zenith, result_azimuth}) == 4

    with pytest.raises(ValueError, match="Unknown site"):
        seeds._get_instrument_seed("unknown_site", "1.0.0", 20.0, 180.0)


@pytest.mark.parametrize(
    ("args", "instances", "expected"),
    [
        ({"instrument_seed": 5, "simulation_seed": [7]}, 1, [(5, 7)]),
        ({"simulation_seed": [5, 7]}, 1, [(5, 7)]),  # Fixed test seeds
        (
            {"instrument_seed": 5, "simulation_seed": [7]},
            3,
            list(
                zip(
                    random.instance_seeds(5, 3, max_seed=SIMTEL_MAX_SEED),
                    random.instance_seeds(7, 3, max_seed=SIMTEL_MAX_SEED),
                    strict=True,
                )
            ),
        ),
    ],
)
def test_instance_seed_pairs(mock_config, args, instances, expected):
    mock_config.args = args
    assert instance_seed_pairs(instances) == expected


def test_instance_seed_pairs_draws_missing_seeds(mock_config):
    mock_config.args = {}
    (instrument, simulation), *others = instance_seed_pairs(4)
    assert len(others) == 3
    assert 1 <= instrument < SIMTEL_MAX_SEED
    assert 1 <= simulation < SIMTEL_MAX_SEED
    mock_config.args = {"simulation_seed": [1, 2, 3]}
    with pytest.raises(ValueError, match="simulation_seed must contain one value"):
        instance_seed_pairs()


def test_seed_file_holds_the_instance_seeds(mock_config, tmp_test_directory):
    """The sim_telarray seed file and the ray-tracing instances derive seeds alike."""
    mock_config.args = {
        "instrument_seed": 5,
        "random_instrument_instances": 3,
        "sim_telarray_seed_file": "seeds.txt",
    }
    seeds = SimtelSeeds(output_path=tmp_test_directory)
    lines = seeds.seed_file.read_text(encoding="utf-8").splitlines()
    assert [int(line) for line in lines if not line.startswith("#")] == random.instance_seeds(
        5, 3, max_seed=SIMTEL_MAX_SEED
    )
