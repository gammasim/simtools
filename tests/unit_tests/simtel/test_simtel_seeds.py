from unittest.mock import MagicMock

import pytest

from simtools.simtel.simtel_seeds import SimtelSeeds


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
        (None, None, "_generate_seed_pair", "100,789"),
        (300, 1, "_generate_seed_pair", "200,300"),
        (400, 10, "_generate_seeds_with_file", "file-by-run:/path/to/seeds,400"),
    ],
)
def test_initialize_seeds_routes_correctly(
    mocker, mock_config, simulation_seed, instruments, expected_method, return_value
):
    """Test initialize_seeds routes to correct method based on inputs."""
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


def test_set_fixed_seeds(mocker, mock_config):
    """Test _set_fixed_seeds with valid and invalid inputs."""
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
    """Test _generate_seeds_with_file creates file, handles errors, and logs."""
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


def test_get_instrument_seed_configured(mock_config, seeds_instance):
    """Test _get_instrument_seed returns configured seed when available."""
    seeds_instance.instrument_seed = 999
    result = seeds_instance._get_instrument_seed("north", "1.0.0", 20.0, 180.0)
    assert result == 999


def test_get_instrument_seed_deterministic(mocker, mock_config):
    """Test _get_instrument_seed generates different seeds for different parameters."""
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
    ("site", "model_version", "zenith", "azimuth", "expected_value"),
    [
        ("north", "1.0.0", 20.0, 180.0, 100001020180),
        ("north", None, 20.0, 180.0, 777),
    ],
)
def test_get_instrument_seed_various_params(
    mocker, mock_config, seeds_instance, site, model_version, zenith, azimuth, expected_value
):
    """Test _get_instrument_seed handles various parameter combinations."""
    mocker.patch(
        "simtools.simtel.simtel_seeds.names.site_names",
        return_value={"North": ["north"], "South": ["south"]},
    )
    mocker.patch("simtools.simtel.simtel_seeds.random.seeds", return_value=777)

    seeds_instance.instrument_seed = None
    result = seeds_instance._get_instrument_seed(site, model_version, zenith, azimuth)

    assert isinstance(result, int)
    assert result == expected_value


@pytest.mark.parametrize(
    ("path_type", "instrument", "simulation"),
    [
        ("string", 123, 456),
        ("pathlib", 789, 101),
        ("pathlib", None, None),
    ],
)
def test_save_seeds(
    mocker, mock_config, tmp_path, seeds_instance, path_type, instrument, simulation
):
    """Test save_seeds writes correct data with various path types and seed values."""
    mock_write = mocker.patch("simtools.simtel.simtel_seeds.ascii_handler.write_data_to_file")

    seeds_instance.instrument_seed = instrument
    seeds_instance.simulation_seed = simulation

    path = tmp_path / "seeds.txt" if path_type == "pathlib" else "/path/to/seeds.txt"
    seeds_instance.save_seeds(path)

    mock_write.assert_called_once()
    call_args = mock_write.call_args[0]
    assert call_args[0] == path
    assert call_args[1] == {"instrument_seed": instrument, "simulation_seed": simulation}
