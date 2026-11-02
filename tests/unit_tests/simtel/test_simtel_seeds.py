from types import SimpleNamespace

import pytest

from simtools.constants import SIMTEL_MAX_SEED
from simtools.simtel.simtel_seeds import SimtelSeeds, instance_seed_pairs
from simtools.utils import random


@pytest.fixture
def mock_config():
    """Fixture providing explicit context arguments."""
    return SimpleNamespace(args={})


@pytest.fixture
def seeds_instance(mock_config):
    """Fixture providing a SimtelSeeds instance."""
    return SimtelSeeds(context=mock_config)


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

    seeds = SimtelSeeds(context=mock_config)
    seeds.simulation_seed = simulation_seed
    seeds.instruments = instruments
    result = seeds.initialize_seeds("north", "1.0.0", 20.0, 180.0)

    assert result == return_value
    if simulation_seed is None:
        assert seeds.simulation_seed == 789
    if simulation_seed == [123]:
        assert seeds.simulation_seed == 123


def test_initialize_seeds_raises_for_too_many_fixed_seeds(mock_config):
    seeds = SimtelSeeds(context=mock_config)
    seeds.simulation_seed = [1, 2, 3]

    with pytest.raises(ValueError, match="simulation_seed must contain one value"):
        seeds.initialize_seeds("north", "1.0.0", 20.0, 180.0)


def test_initialize_seeds_single_value_uses_configured_instrument_seed(mocker, mock_config):
    mock_config.args = {"instrument_seed": 1745, "simulation_seed": [290]}
    random_seed_mock = mocker.patch("simtools.simtel.simtel_seeds.random.seeds", return_value=999)

    seeds = SimtelSeeds(context=mock_config)

    result = seeds.seed_string

    assert result == "1745,290"
    assert seeds.instrument_seed == 1745
    assert seeds.simulation_seed == 290
    random_seed_mock.assert_not_called()


def test_generate_seeds_with_file(mocker, mock_config, tmp_path):
    mocker.patch.object(SimtelSeeds, "initialize_seeds", return_value="mock")
    mocker.patch("simtools.simtel.simtel_seeds.random.seeds", return_value=[101, 102, 103])
    mock_config.args = {"sim_telarray_seed_file": "seeds.txt"}
    mock_logger = mocker.patch("simtools.simtel.simtel_seeds.logging.getLogger")

    seeds = SimtelSeeds(context=mock_config)
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
    assert instance_seed_pairs(instances, context=mock_config) == expected


def test_instance_seed_pairs_draws_missing_seeds(mock_config):
    mock_config.args = {}
    (instrument, simulation), *others = instance_seed_pairs(4, context=mock_config)
    assert len(others) == 3
    assert 1 <= instrument < SIMTEL_MAX_SEED
    assert 1 <= simulation < SIMTEL_MAX_SEED
    mock_config.args = {"simulation_seed": [1, 2, 3]}
    with pytest.raises(ValueError, match="simulation_seed must contain one value"):
        instance_seed_pairs(context=mock_config)


def test_seed_file_holds_the_instance_seeds(mock_config, tmp_test_directory):
    """The sim_telarray seed file and the ray-tracing instances derive seeds alike."""
    mock_config.args = {
        "instrument_seed": 5,
        "random_instrument_instances": 3,
        "sim_telarray_seed_file": "seeds.txt",
    }
    seeds = SimtelSeeds(output_path=tmp_test_directory, context=mock_config)
    lines = seeds.seed_file.read_text(encoding="utf-8").splitlines()
    assert [int(line) for line in lines if not line.startswith("#")] == random.instance_seeds(
        5, 3, max_seed=SIMTEL_MAX_SEED
    )
