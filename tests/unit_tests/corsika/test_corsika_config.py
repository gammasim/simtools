#!/usr/bin/python3

import copy
import logging
import pathlib
from unittest.mock import MagicMock, Mock, patch

import pytest
from astropy import units as u

from simtools.corsika.corsika_config import CorsikaConfig

logger = logging.getLogger()

CORSIKA_CONFIG_MODE_PARAMETER = "simtools.corsika.corsika_config.ModelParameter"
MUON_PLUS_PARTICLE = "muon+"
PROTON_PARTICLE = "proton"
G_CM2 = "g/cm2"


def create_mock_array_model(mocker, geomag_rotation=0):
    """Helper function to create a mock array model with site model."""
    mock_array_model = mocker.MagicMock()
    mock_site_model = mocker.MagicMock()
    mock_site_model.get_parameter_value.side_effect = lambda x: {
        "geomag_rotation": geomag_rotation,
        "corsika_observation_level": 2200.0,
    }.get(x, geomag_rotation)
    mock_array_model.site_model = mock_site_model
    return mock_array_model


@pytest.fixture
def get_standard_corsika_parameters():
    """Helper function to return standard CORSIKA configuration parameters."""
    return {
        "corsika_iact_max_bunches": {"value": 1000000, "unit": None},
        "corsika_cherenkov_photon_bunch_size": {"value": 5.0, "unit": None},
        "corsika_cherenkov_photon_wavelength_range": {"value": [240.0, 1000.0], "unit": "nm"},
        "corsika_first_interaction_height": {"value": 0.0, "unit": "cm"},
        "corsika_particle_kinetic_energy_cutoff": {
            "value": [0.3, 0.1, 0.020, 0.020],
            "unit": "GeV",
        },
        "corsika_longitudinal_shower_development": {"value": 20.0, "unit": G_CM2},
        "corsika_iact_split_auto": {"value": 15000000, "unit": None},
        "corsika_starting_grammage": {"value": 0.0, "unit": G_CM2},
        "corsika_iact_io_buffer": {"value": 800, "unit": "MB"},
    }


@pytest.fixture
def corsika_config_no_array_model(corsika_config_data, mocker):
    """Fixture for corsika config with no array model."""
    mock_array_model = create_mock_array_model(mocker)

    modified_data = {
        "correct_for_b_field_alignment": False,
        "azimuth_angle": 0 * u.deg,
        "zenith_angle": 20 * u.deg,
        "event_number_first_shower": 1,
        "nshow": 100,
        "energy_range": (10 * u.GeV, 10 * u.TeV),
        "view_cone": (0 * u.deg, 0 * u.deg),
        "core_scatter": (10, 1400 * u.m),
        "primary": "proton",
        "primary_id_type": "common_name",
        "eslope": -2,
    }
    return CorsikaConfig(
        array_model=mock_array_model,
        label="test-corsika-config",
        args_dict=modified_data,
        db_config=None,
    )


@pytest.fixture
def get_standard_corsika_parameters_muon_grammage(get_standard_corsika_parameters):
    """Fixture for CORSIKA configuration parameters with muon grammage."""
    params = get_standard_corsika_parameters.copy()
    params["corsika_starting_grammage"] = {
        "value": [
            {
                "primary_particle": MUON_PLUS_PARTICLE,
                "value": 10.0,
            },
            {
                "primary_particle": "default",
                "value": 0.0,
            },
        ],
        "unit": G_CM2,
    }
    return params


@pytest.fixture
def get_standard_corsika_parameters_teltype_grammage(get_standard_corsika_parameters):
    """Fixture for CORSIKA configuration parameters with muon grammage."""
    params = get_standard_corsika_parameters.copy()
    params["corsika_starting_grammage"] = {
        "value": [
            {
                "instrument": "LSTS-design",
                "primary_particle": MUON_PLUS_PARTICLE,
                "value": 10.0,
            },
            {
                "instrument": "LSTS-design",
                "primary_particle": "default",
                "value": 3.0,  # default values are set to non-standard values for testing
            },
            {
                "instrument": "SSTS-design",
                "primary_particle": MUON_PLUS_PARTICLE,
                "value": 50.0,
            },
            {
                "instrument": "SSTS-design",
                "primary_particle": "default",
                "value": 2.0,  # default values are set to non-standard values for testing
            },
        ],
        "unit": G_CM2,
    }
    return params


def test_fill_corsika_configuration(corsika_config_mock_array_model, mocker):
    """Test CORSIKA configuration with and without args_dict."""
    mock_array_model = create_mock_array_model(mocker)

    args_dict = {
        "azimuth_angle": 0 * u.deg,
        "zenith_angle": 20 * u.deg,
        "event_number_first_shower": 1,
        "nshow": 100,
        "energy_range": (10 * u.GeV, 10 * u.TeV),
        "view_cone": (0 * u.deg, 0 * u.deg),
        "core_scatter": (10, 1400 * u.m),
        "primary": "proton",
        "primary_id_type": "common_name",
        "eslope": -2,
        "correct_for_b_field_alignment": False,
    }

    config_none_args = CorsikaConfig(
        array_model=mock_array_model,
        label="test-corsika-config",
        args_dict=args_dict,
        db_config=None,
    )
    assert isinstance(config_none_args.config, dict)
    assert "USER_INPUT" in config_none_args.config

    assert corsika_config_mock_array_model.get_config_parameter("NSHOW") == 100
    assert corsika_config_mock_array_model.get_config_parameter("THETAP") == [20, 20]
    assert corsika_config_mock_array_model.get_config_parameter("ERANGE") == [10.0, 10000.0]
    # Testing conversion between AZM (sim_telarray) and PHIP (corsika)
    assert corsika_config_mock_array_model.get_config_parameter("PHIP") == [175.467, 175.467]
    assert corsika_config_mock_array_model.get_config_parameter("CSCAT") == [10, 140000.0, 0]

    # db_config is not None
    assert corsika_config_mock_array_model.get_config_parameter("CERSIZ") == pytest.approx(5.0)
    assert corsika_config_mock_array_model.get_config_parameter("MAX_BUNCHES") == 1000000
    assert corsika_config_mock_array_model.get_config_parameter("ECUTS") == "0.3 0.1 0.02 0.02"

    for key in [
        "USER_INPUT",
        "INTERACTION_FLAGS",
        "CHERENKOV_EMISSION_PARAMETERS",
        "DEBUGGING_OUTPUT_PARAMETERS",
        "IACT_PARAMETERS",
    ]:
        assert key in corsika_config_mock_array_model.config


def test_fill_corsika_configuration_model_version(
    corsika_config_mock_array_model, get_standard_corsika_parameters
):
    """Test handling a list of model versions as input, taking the first one only."""

    with patch(CORSIKA_CONFIG_MODE_PARAMETER) as mock_model_parameter:
        mock_params = Mock()
        mock_params.get_simulation_software_parameters.return_value = (
            get_standard_corsika_parameters
        )
        mock_model_parameter.return_value = mock_params

        args_dict = {
            "model_version": ["5.0.0", "6.0.0"],
            "azimuth_angle": 0 * u.deg,
            "zenith_angle": 20 * u.deg,
            "event_number_first_shower": 1,
            "nshow": 100,
            "eslope": 2.0,
            "energy_range": [10 * u.GeV, 10000 * u.GeV],
            "view_cone": [0 * u.deg, 0 * u.deg],
            "core_scatter": [10, 140000 * u.cm],
            "correct_for_b_field_alignment": True,
        }
        config = corsika_config_mock_array_model._fill_corsika_configuration(
            args_dict, db_config={}
        )

        # Verify ModelParameter was instantiated with the correct model (the first one)
        mock_model_parameter.assert_called_with(db_config={}, model_version="5.0.0")
        mock_params.get_simulation_software_parameters.assert_called_with("corsika")

        assert isinstance(config, dict)
        assert "USER_INPUT" in config


def test_corsika_configuration_from_user_input(
    corsika_config_mock_array_model, corsika_config_data
):
    user_dict = corsika_config_mock_array_model._corsika_configuration_from_user_input(
        corsika_config_data
    )
    assert isinstance(user_dict, dict)
    assert "CSCAT" in user_dict
    assert isinstance(user_dict["CSCAT"], list)


def test_corsika_configuration_interaction_flags(
    corsika_config_mock_array_model, get_standard_corsika_parameters
):
    with pytest.raises(KeyError):
        corsika_config_mock_array_model._corsika_configuration_interaction_flags({})

    parameters = corsika_config_mock_array_model._corsika_configuration_interaction_flags(
        get_standard_corsika_parameters
    )
    assert isinstance(parameters, dict)
    assert "ECUTS" in parameters
    assert parameters["MAXPRT"] == ["10"]
    assert len(parameters) == 9


def test_input_config_first_interaction_height(corsika_config_mock_array_model):
    assert corsika_config_mock_array_model._input_config_first_interaction_height(
        {"value": 0.0, "unit": "cm"}
    ) == ["0.00", "0"]

    assert corsika_config_mock_array_model._input_config_first_interaction_height(
        {"value": 10.0, "unit": "m"}
    ) == ["1000.00", "0"]


def test_primary_particle(corsika_config_mock_array_model):
    assert corsika_config_mock_array_model.primary == "proton"


def test_input_config_corsika_starting_grammage(corsika_config_mock_array_model):
    assert (
        corsika_config_mock_array_model._input_config_corsika_starting_grammage(
            {"value": 0.0, "unit": G_CM2}
        )
        == "0.0"
    )
    assert (
        corsika_config_mock_array_model._input_config_corsika_starting_grammage(
            {"value": 10.0, "unit": "kg/cm2"}
        )
        == "10000.0"
    )


def test_input_config_corsika_starting_grammage_muon_grammage(
    corsika_config_mock_array_model,
    get_standard_corsika_parameters_muon_grammage,
    get_standard_corsika_parameters_teltype_grammage,
):
    # default behavior with proton as primary
    assert (
        corsika_config_mock_array_model._input_config_corsika_starting_grammage(
            get_standard_corsika_parameters_muon_grammage["corsika_starting_grammage"]
        )
        == "0.0"
    )
    # change primary to muon+
    corsika_config_mock_array_model_muon = copy.deepcopy(corsika_config_mock_array_model)
    corsika_config_mock_array_model_muon.primary_particle = {
        "primary": MUON_PLUS_PARTICLE,
        "primary_id_type": "common_name",
    }
    assert (
        corsika_config_mock_array_model_muon._input_config_corsika_starting_grammage(
            get_standard_corsika_parameters_muon_grammage["corsika_starting_grammage"]
        )
        == "10.0"
    )
    corsika_config_mock_array_model_muon.primary_particle = {
        "primary": "gamma",
        "primary_id_type": "common_name",
    }
    assert (
        corsika_config_mock_array_model_muon._input_config_corsika_starting_grammage(
            get_standard_corsika_parameters_muon_grammage["corsika_starting_grammage"]
        )
        == "0.0"
    )

    # Telescope-type dependent starting grammage
    tel1 = MagicMock()
    tel1.design_model = "LSTS-design"
    tel1.__str__.return_value = "LSTS-01"

    tel2 = MagicMock()
    tel2.design_model = "SSTS-design"
    tel2.__str__.return_value = "SSTS-14"

    corsika_config_mock_array_model_teltype = copy.deepcopy(corsika_config_mock_array_model)
    corsika_config_mock_array_model_teltype.array_model = MagicMock()

    # Create telescope model mock that behaves like a dictionary
    telescope_model_mock = MagicMock()
    telescope_model_mock.values = MagicMock(return_value=[tel1, tel2])
    corsika_config_mock_array_model_teltype.array_model.telescope_model = telescope_model_mock

    corsika_config_mock_array_model_teltype.primary_particle = {
        "primary": MUON_PLUS_PARTICLE,
        "primary_id_type": "common_name",
    }

    assert (
        corsika_config_mock_array_model_teltype._input_config_corsika_starting_grammage(
            get_standard_corsika_parameters_teltype_grammage["corsika_starting_grammage"]
        )
        == "10.0"
    )

    corsika_config_mock_array_model_teltype.primary_particle = {
        "primary": PROTON_PARTICLE,
        "primary_id_type": "common_name",
    }
    assert (
        corsika_config_mock_array_model_teltype._input_config_corsika_starting_grammage(
            get_standard_corsika_parameters_teltype_grammage["corsika_starting_grammage"]
        )
        == "2.0"
    )


def test_input_config_corsika_particle_kinetic_energy_cutoff(corsika_config_mock_array_model):
    assert corsika_config_mock_array_model._input_config_corsika_particle_kinetic_energy_cutoff(
        {"value": [0.3, 0.1, 0.020, 0.020], "unit": "GeV"}
    ) == ["0.3 0.1 0.02 0.02"]
    assert corsika_config_mock_array_model._input_config_corsika_particle_kinetic_energy_cutoff(
        {"value": [0.3, 0.1, 0.020, 0.020], "unit": "TeV"}
    ) == ["300.0 100.0 20.0 20.0"]


def test_input_config_corsika_longitudinal_parameters(corsika_config_mock_array_model):
    assert corsika_config_mock_array_model._input_config_corsika_longitudinal_parameters(
        {"value": 20.0, "unit": G_CM2}
    ) == ["T", "20.0", "F", "F"]
    assert corsika_config_mock_array_model._input_config_corsika_longitudinal_parameters(
        {"value": 10.0, "unit": "kg/cm2"}
    ) == ["T", "10000.0", "F", "F"]


def test_corsika_configuration_cherenkov_parameters(
    corsika_config_mock_array_model, get_standard_corsika_parameters
):
    cherenk_dict = corsika_config_mock_array_model._corsika_configuration_cherenkov_parameters(
        get_standard_corsika_parameters
    )
    assert isinstance(cherenk_dict, dict)
    assert "CERSIZ" in cherenk_dict
    assert isinstance(cherenk_dict["CERSIZ"], list)


def test_input_config_corsika_cherenkov_wavelength(corsika_config_mock_array_model):
    assert corsika_config_mock_array_model._input_config_corsika_cherenkov_wavelength(
        {"value": [240.0, 1000.0], "unit": "nm"}
    ) == ["240.0", "1000.0"]


def test_corsika_configuration_iact_parameters(
    corsika_config_mock_array_model, get_standard_corsika_parameters
):
    iact_dict = corsika_config_mock_array_model._corsika_configuration_iact_parameters(
        get_standard_corsika_parameters
    )
    assert isinstance(iact_dict, dict)
    assert "MAX_BUNCHES" in iact_dict
    assert isinstance(iact_dict["MAX_BUNCHES"], list)


def test_input_config_io_buff(corsika_config_mock_array_model):
    assert (
        corsika_config_mock_array_model._input_config_io_buff({"value": 800, "unit": "MB"})
        == "800MB"
    )
    assert (
        corsika_config_mock_array_model._input_config_io_buff({"value": 8.5, "unit": "MB"})
        == "8500000"
    )
    assert (
        corsika_config_mock_array_model._input_config_io_buff({"value": 800, "unit": "kB"})
        == "800000"
    )


def test_corsika_configuration_debugging_parameters(corsika_config_mock_array_model):
    assert isinstance(
        corsika_config_mock_array_model._corsika_configuration_debugging_parameters(), dict
    )
    assert len(corsika_config_mock_array_model._corsika_configuration_debugging_parameters()) == 4


def test_rotate_azimuth_by_180deg(corsika_config_mock_array_model):
    """Test azimuth angle rotation with and without geomagnetic field alignment."""
    test_cases = [
        # (input_angle, with_correction, expected_result)
        (0.0, False, 180.0),
        (360.0, False, 180.0),
        (450.0, False, 270.0),
        (180.0, False, 0.0),
        (-180.0, False, 0.0),
        (0.0, True, 175.467),
        (360.0, True, 175.467),
        (450.0, True, 265.467),
        (180.0, True, 355.467),
        (-180.0, True, 355.467),
    ]

    for input_angle, with_correction, expected_result in test_cases:
        assert corsika_config_mock_array_model._rotate_azimuth_by_180deg(
            input_angle, correct_for_geomagnetic_field_alignment=with_correction
        ) == pytest.approx(expected_result)

    for input_angle, with_correction, expected_result in test_cases:
        assert corsika_config_mock_array_model._rotate_azimuth_by_180deg(
            expected_result,
            correct_for_geomagnetic_field_alignment=with_correction,
            invert_operation=True,
        ) == pytest.approx(input_angle % 360)


def test_get_config_parameter(corsika_config_mock_array_model):
    cc = corsika_config_mock_array_model
    assert isinstance(cc.get_config_parameter("NSHOW"), int)
    assert isinstance(cc.get_config_parameter("THETAP"), list)
    with pytest.raises(
        KeyError, match="Parameter not_really_a_parameter is not a CORSIKA config parameter"
    ):
        cc.get_config_parameter("not_really_a_parameter")


def test_get_text_single_line(corsika_config_mock_array_model):
    assert (
        corsika_config_mock_array_model._get_text_single_line({"EVTNR": [1], "RUNNR": [10]})
        == "EVTNR 1 \nRUNNR 10 \n"
    )
    assert (
        corsika_config_mock_array_model._get_text_single_line(
            {"SPLIT_AUTO": ["15M"], "IO_BUFFER": ["800MB"], "MAX_BUNCHES": ["1000000"]}, "IACT "
        )
        == "IACT SPLIT_AUTO 15M \nIACT IO_BUFFER 800MB \nIACT MAX_BUNCHES 1000000 \n"
    )


def test_generate_corsika_input_file(corsika_config_mock_array_model):
    logger.info("test_generate_corsika_input_file")
    input_file = corsika_config_mock_array_model.generate_corsika_input_file()
    assert input_file.exists()
    with open(input_file) as f:
        assert "TELFIL |" not in f.read()

    assert corsika_config_mock_array_model.is_file_updated
    assert input_file == corsika_config_mock_array_model.generate_corsika_input_file()


def test_generate_corsika_input_file_multipipe(corsika_config_mock_array_model):
    logger.info("test_generate_corsika_input_file")
    input_file = corsika_config_mock_array_model.generate_corsika_input_file(use_multipipe=True)
    assert input_file.exists()
    with open(input_file) as f:
        assert "TELFIL |" in f.read()


def test_generate_corsika_input_file_with_test_seeds(corsika_config_mock_array_model):
    logger.info("test_generate_corsika_input_file_with_test_seeds")
    input_file = corsika_config_mock_array_model.generate_corsika_input_file(use_test_seeds=True)
    assert input_file.exists()
    expected_seeds = [534, 220, 1104, 382]
    with open(input_file) as f:
        file_content = f.read()
        for seed in expected_seeds:
            assert f"SEED {seed} 0 0" in file_content


def test_get_corsika_config_file_name(corsika_config_mock_array_model, model_version):
    file_name = (
        "proton_run000001_za20deg_azm000deg_cone0-10_South_"
        f"test_layout_{model_version}_test-corsika-config"
    )

    assert (
        corsika_config_mock_array_model.get_corsika_config_file_name("config_tmp", run_number=1)
        == f"corsika_config_{file_name}.txt"
    )
    with pytest.raises(
        ValueError, match="Must provide a run number for a temporary CORSIKA config file"
    ):
        corsika_config_mock_array_model.get_corsika_config_file_name("config_tmp")

    config_file_name = file_name.replace("run000001_", "")
    assert (
        corsika_config_mock_array_model.get_corsika_config_file_name("config")
        == f"corsika_config_{config_file_name}.input"
    )
    # The test below includes the placeholder XXXXXX for the run number because
    # that is the way we get the run number later in the CORSIKA input file with zero padding.
    output_file_name = file_name.replace("run000001", "runXXXXXX")
    assert corsika_config_mock_array_model.get_corsika_config_file_name("output_generic") == (
        f"{output_file_name}.corsika.zst"
    )
    assert (
        corsika_config_mock_array_model.get_corsika_config_file_name("multipipe")
        == "multi_cta-South-test_layout.cfg"
    )
    with pytest.raises(ValueError, match=r"^The requested file type"):
        corsika_config_mock_array_model.get_corsika_config_file_name("foobar")


def test_set_output_file_and_directory(corsika_config_mock_array_model, model_version):
    cc = corsika_config_mock_array_model
    output_file = cc.set_output_file_and_directory()
    assert str(output_file) == (
        "proton_runXXXXXX_za20deg_azm000deg_cone0-10_South_test_layout_"
        f"{model_version}_test-corsika-config.corsika.zst"
    )
    assert isinstance(cc.config_file_path, pathlib.Path)


def test_write_seeds(corsika_config_mock_array_model):
    mock_file = Mock()
    corsika_config_mock_array_model.run_number = 10
    corsika_config_mock_array_model.config = {"USER_INPUT": {"PRMPAR": [14]}}
    with patch("io.open", return_value=mock_file):
        corsika_config_mock_array_model._write_seeds(mock_file)
    assert mock_file.write.call_count == 4

    expected_calls = [_call.args[0] for _call in mock_file.write.call_args_list]
    for _call in expected_calls:
        assert _call.startswith("SEED ")
        assert _call.endswith(" 0 0\n")


def test_write_seeds_use_test_seeds(corsika_config_mock_array_model):
    mock_file = Mock()
    corsika_config_mock_array_model.run_number = 10
    corsika_config_mock_array_model.config = {"USER_INPUT": {"PRMPAR": [14]}}
    with patch("io.open", return_value=mock_file):
        corsika_config_mock_array_model._write_seeds(mock_file, use_test_seeds=True)
    assert mock_file.write.call_count == 4

    expected_calls = [_call.args[0] for _call in mock_file.write.call_args_list]
    expected_seeds = [534, 220, 1104, 382]
    for _call in expected_calls:
        assert _call == (f"SEED {expected_seeds.pop(0)} 0 0\n")


def test_get_corsika_telescope_list(corsika_config_mock_array_model):
    # Create mock telescopes
    mock_telescope1 = MagicMock()
    mock_telescope1.get_parameter_value_with_unit.side_effect = lambda x: {
        "array_element_position_ground": [100 * u.cm, 200 * u.cm, 0 * u.cm],
        "telescope_sphere_radius": 10 * u.cm,
    }[x]

    mock_telescope2 = MagicMock()
    mock_telescope2.get_parameter_value_with_unit.side_effect = lambda x: {
        "array_element_position_ground": [300 * u.cm, 400 * u.cm, 0 * u.cm],
        "telescope_sphere_radius": 10 * u.cm,
    }[x]

    # Mock array_model with telescope_models
    mock_array_model = MagicMock()
    mock_array_model.telescope_models = {"LSTS-01": mock_telescope1, "LSTS-02": mock_telescope2}
    corsika_config_mock_array_model.array_model = mock_array_model

    telescope_list_str = corsika_config_mock_array_model.get_corsika_telescope_list()
    assert "TELESCOPE" in telescope_list_str
    assert "LSTS-01" in telescope_list_str
    assert "LSTS-02" in telescope_list_str
    assert "100.000" in telescope_list_str  # x position of first telescope
    assert "300.000" in telescope_list_str  # x position of second telescope


def test_run_number(corsika_config_no_array_model):
    assert corsika_config_no_array_model.run_number is None
    corsika_config_no_array_model.run_number = 25
    assert corsika_config_no_array_model.run_number == 25


def test_validate_run_number(corsika_config_no_array_model):
    assert corsika_config_no_array_model.validate_run_number(None) is None
    assert corsika_config_no_array_model.validate_run_number(1)
    assert corsika_config_no_array_model.validate_run_number(123456)
    with pytest.raises(ValueError, match=r"^could not convert string to float"):
        corsika_config_no_array_model.validate_run_number("test")
    invalid_run_number = r"^Invalid type of run number"
    with pytest.raises(ValueError, match=invalid_run_number):
        corsika_config_no_array_model.validate_run_number(1.5)
    with pytest.raises(ValueError, match=invalid_run_number):
        corsika_config_no_array_model.validate_run_number(-1)
    with pytest.raises(ValueError, match=invalid_run_number):
        corsika_config_no_array_model.validate_run_number(123456789)


def test_assert_corsika_configurations_match_success(corsika_config_mock_array_model):
    """Test that assert_corsika_configurations_match does not raise an error
    when parameters match."""
    with patch(CORSIKA_CONFIG_MODE_PARAMETER) as mock_model_parameter:
        mock_params = Mock()
        mock_params.get_simulation_software_parameters.return_value = {
            "param1": {"value": 10},
            "param2": {"value": 20},
            "corsika_iact_io_buffer": {"value": 800},  # Skipped parameter
            "corsika_iact_split_auto": {"value": 15000000},  # Skipped parameter
        }
        mock_model_parameter.return_value = mock_params

        corsika_config_mock_array_model.assert_corsika_configurations_match(
            model_versions=["5.0.0", "6.0.0"], db_config={}
        )

        mock_model_parameter.assert_any_call(db_config={}, model_version="5.0.0")
        mock_model_parameter.assert_any_call(db_config={}, model_version="6.0.0")
        assert mock_model_parameter.call_count == 2


def test_assert_corsika_configurations_match_failure(corsika_config_mock_array_model):
    """Test that assert_corsika_configurations_match raises an error when parameters
    do not match."""
    with patch(CORSIKA_CONFIG_MODE_PARAMETER) as mock_model_parameter:
        mock_params_1 = Mock()
        mock_params_1.get_simulation_software_parameters.return_value = {
            "param1": {"value": 10},
            "param2": {"value": 20},
        }
        mock_params_2 = Mock()
        mock_params_2.get_simulation_software_parameters.return_value = {
            "param1": {"value": 10},
            "param2": {"value": 30},  # Mismatch here
        }
        mock_model_parameter.side_effect = [mock_params_1, mock_params_2]

        with pytest.raises(ValueError, match="CORSIKA parameter 'param2' differs"):
            corsika_config_mock_array_model.assert_corsika_configurations_match(
                model_versions=["5.0.0", "6.0.0"], db_config={}
            )


def test_assert_corsika_configurations_match_skip_parameters(corsika_config_mock_array_model):
    """Test that assert_corsika_configurations_match skips specific parameters."""
    with patch(CORSIKA_CONFIG_MODE_PARAMETER) as mock_model_parameter:
        mock_params_1 = Mock()
        mock_params_1.get_simulation_software_parameters.return_value = {
            "param1": {"value": 10},
            "corsika_iact_io_buffer": {"value": 800},  # Skipped parameter
        }
        mock_params_2 = Mock()
        mock_params_2.get_simulation_software_parameters.return_value = {
            "param1": {"value": 10},
            "corsika_iact_io_buffer": {"value": 900},  # Mismatch but skipped
        }
        mock_model_parameter.side_effect = [mock_params_1, mock_params_2]

        corsika_config_mock_array_model.assert_corsika_configurations_match(
            model_versions=["5.0.0", "6.0.0"], db_config={}
        )

        mock_model_parameter.assert_any_call(db_config={}, model_version="5.0.0")
        mock_model_parameter.assert_any_call(db_config={}, model_version="6.0.0")
        assert mock_model_parameter.call_count == 2


def test_assert_corsika_configurations_match_single_version(corsika_config_mock_array_model):
    """Test that assert_corsika_configurations_match returns early with single model version."""
    with patch(CORSIKA_CONFIG_MODE_PARAMETER) as mock_model_parameter:
        # Even with different parameters, it should return early without checking
        mock_params = Mock()
        mock_params.get_simulation_software_parameters.return_value = {
            "param1": {"value": 10},
            "param2": {"value": 20},
        }
        mock_model_parameter.return_value = mock_params

        # Should return early without any database calls
        corsika_config_mock_array_model.assert_corsika_configurations_match(
            model_versions=["5.0.0"], db_config={}
        )

        # Verify ModelParameter was never called
        mock_model_parameter.assert_not_called()


def test_get_matching_grammage_values(corsika_config_mock_array_model):
    # Test case 1: Direct match for particle and instrument
    configs = [
        {"instrument": "LSTS-design", "primary_particle": PROTON_PARTICLE, "value": 10.0},
        {"instrument": "SSTS-design", "primary_particle": "default", "value": 5.0},
    ]
    tel_types = {"LSTS-design"}
    assert corsika_config_mock_array_model._get_matching_grammage_values(
        configs, tel_types, PROTON_PARTICLE
    ) == [10.0]

    # Test case 2: No direct match, fallback to default
    configs = [
        {"instrument": "LSTS-design", "primary_particle": MUON_PLUS_PARTICLE, "value": 10.0},
        {"instrument": "LSTS-design", "primary_particle": "default", "value": 5.0},
    ]
    tel_types = {"LSTS-design"}
    assert corsika_config_mock_array_model._get_matching_grammage_values(
        configs, tel_types, PROTON_PARTICLE
    ) == [5.0]

    # Test case 3: Multiple matching values
    configs = [
        {"instrument": "LSTS-design", "primary_particle": PROTON_PARTICLE, "value": 10.0},
        {"instrument": None, "primary_particle": PROTON_PARTICLE, "value": 15.0},
    ]
    tel_types = {"LSTS-design"}
    assert corsika_config_mock_array_model._get_matching_grammage_values(
        configs, tel_types, PROTON_PARTICLE
    ) == [10.0, 15.0]

    # Test case 4: Empty config list
    assert (
        corsika_config_mock_array_model._get_matching_grammage_values(
            [], {"LSTS-design"}, PROTON_PARTICLE
        )
        == []
    )

    # Test case 5: No matches and no defaults
    configs = [
        {"instrument": "LSTS-design", "primary_particle": MUON_PLUS_PARTICLE, "value": 10.0},
        {"instrument": "SSTS-design", "primary_particle": MUON_PLUS_PARTICLE, "value": 15.0},
    ]
    tel_types = {"MSTS-design"}
    assert (
        corsika_config_mock_array_model._get_matching_grammage_values(
            configs, tel_types, PROTON_PARTICLE
        )
        == []
    )


def test_use_curved_atmosphere(corsika_config_mock_array_model):
    corsika_config_mock_array_model.use_curved_atmosphere = {
        "curved_atmosphere_min_zenith_angle": 90 * u.deg,
        "zenith_angle": 95 * u.deg,
    }
    assert corsika_config_mock_array_model.use_curved_atmosphere

    corsika_config_mock_array_model.use_curved_atmosphere = {
        "curved_atmosphere_min_zenith_angle": 90 * u.deg,
        "zenith_angle": 85 * u.deg,
    }
    assert not corsika_config_mock_array_model.use_curved_atmosphere

    corsika_config_mock_array_model.use_curved_atmosphere = {
        "curved_atmosphere_min_zenith_angle": 90 * u.deg,
        "zenith_angle": 90 * u.deg,
    }
    assert not corsika_config_mock_array_model.use_curved_atmosphere

    corsika_config_mock_array_model.use_curved_atmosphere = None
    assert not corsika_config_mock_array_model.use_curved_atmosphere

    corsika_config_mock_array_model.use_curved_atmosphere = False
    assert not corsika_config_mock_array_model.use_curved_atmosphere

    corsika_config_mock_array_model.use_curved_atmosphere = True
    assert corsika_config_mock_array_model.use_curved_atmosphere


def test_corsika_configuration_from_corsika_file(corsika_config_mock_array_model, mocker, tmp_path):
    """Test CORSIKA configuration from corsika file."""
    # Direct dict-based headers (simpler than list->dict mapping).
    run_header_dict = {
        "n_showers": 100,
        "energy_spectrum_slope": -2.0,
        "energy_min": 10.0,
        "energy_max": 1000.0,
        # core scatter coordinates used later for reuse_x/reuse_y compatibility
        "reuse_x": 1400.0,
        "reuse_y": 0.0,
        "n_observation_levels": 1,
        "observation_height": [220000.0],  # in cm (2200.0 m)
    }
    event_header_dict = {
        "event_number": None,
        "particle_id": 14,
        "theta_min": 20.0,
        "theta_max": 20.0,
        "phi_min": 180.0,
        "phi_max": 180.0,
        "viewcone_inner_angle": 0.0,
        "viewcone_outer_angle": 10.0,
        "n_reuse": 10,
        # reuse coords mirrored here for compatibility with earlier tests
        "reuse_x": 1400.0,
        "reuse_y": 0.0,
    }

    mock_get_headers = mocker.patch(
        "simtools.io.eventio_handler.get_corsika_run_and_event_headers",
        return_value=(run_header_dict, event_header_dict),
    )

    test_file = tmp_path / "test.corsika"
    test_file.touch()

    config = corsika_config_mock_array_model._corsika_configuration_from_corsika_file(test_file)

    mock_get_headers.assert_called_once_with(test_file)
    assert config["NSHOW"] == [100]
    assert config["PRMPAR"] == [14]
    assert config["ESLOPE"] == [-2.0]
    assert config["ERANGE"] == [10.0, 1000.0]
    assert config["THETAP"] == [20.0, 20.0]
    assert config["PHIP"] == [180.0, 180.0]
    assert config["VIEWCONE"] == [0.0, 10.0]
    assert config["CSCAT"] == [10, 1400.0, 0.0]


def test_corsika_configuration_for_dummy_simulations(corsika_config_no_array_model):
    """Test CORSIKA configuration for dummy simulations."""
    args_dict = {
        "zenith_angle": 30 * u.deg,
        "azimuth_angle": 45 * u.deg,
        "correct_for_b_field_alignment": False,
    }

    config = corsika_config_no_array_model._corsika_configuration_for_dummy_simulations(args_dict)

    assert isinstance(config, dict)
    assert config["EVTNR"] == [1]
    assert config["NSHOW"] == [1]
    assert config["PRMPAR"] == [1]
    assert config["ESLOPE"] == [-2.0]
    assert config["ERANGE"] == [0.1, 0.1]
    assert config["THETAP"] == [30.0, 30.0]
    assert config["PHIP"] == [225.0, 225.0]
    assert config["VIEWCONE"] == [0.0, 0.0]
    assert config["CSCAT"] == [1, 0.0, 10.0]


def test_initialize_from_config(corsika_config_mock_array_model, corsika_config_data):
    """Test initialization of parameters from config."""
    # Test normal initialization
    assert corsika_config_mock_array_model.azimuth_angle == 0
    assert corsika_config_mock_array_model.zenith_angle == 20
    assert corsika_config_mock_array_model.curved_atmosphere_min_zenith_angle == pytest.approx(90.0)

    test_config = {
        "USER_INPUT": {
            "THETAP": [20, 20],
            "PHIP": [175.467, 175.467],
            "PRMPAR": [14],
            "NSHOW": [100],
            "CSCAT": [10, 1400.0, 0.0],
        }
    }
    corsika_config_mock_array_model.config = test_config
    corsika_config_mock_array_model._initialize_from_config(corsika_config_data)
    assert corsika_config_mock_array_model.azimuth_angle == 0
    assert corsika_config_mock_array_model.zenith_angle == 20


def test_fill_corsika_configuration_variations(
    corsika_config_no_array_model,
    corsika_config_mock_array_model,
    mocker,
    tmp_path,
    get_standard_corsika_parameters,
):
    """Test various configuration scenarios for CORSIKA."""
    # Test None args_dict
    config = corsika_config_no_array_model._fill_corsika_configuration(None)
    assert config == {}

    # Test dummy simulations
    args_dict = {
        "zenith_angle": 20 * u.deg,
        "azimuth_angle": 0 * u.deg,
        "correct_for_b_field_alignment": False,
    }
    corsika_config_no_array_model.dummy_simulations = True
    config = corsika_config_no_array_model._fill_corsika_configuration(args_dict)
    assert "USER_INPUT" in config
    assert config["USER_INPUT"]["NSHOW"] == [1]
    assert config["USER_INPUT"]["PRMPAR"] == [1]

    # Test empty DB config
    result = corsika_config_no_array_model._fill_corsika_configuration_from_db(["5.0.0"], None)
    assert result == {}

    # Test DB config with parameters
    with patch(CORSIKA_CONFIG_MODE_PARAMETER) as mock_model_parameter:
        mock_params = Mock()
        mock_params.get_simulation_software_parameters.return_value = (
            get_standard_corsika_parameters
        )
        mock_model_parameter.return_value = mock_params
        result = corsika_config_mock_array_model._fill_corsika_configuration_from_db(
            ["5.0.0"], db_config={}
        )
        assert all(
            key in result
            for key in [
                "INTERACTION_FLAGS",
                "CHERENKOV_EMISSION_PARAMETERS",
                "DEBUGGING_OUTPUT_PARAMETERS",
                "IACT_PARAMETERS",
            ]
        )


def test_corsika_file_initialization(mocker, tmp_path):
    """Test CORSIKA file initialization with different configurations."""
    mock_array_model = create_mock_array_model(mocker)

    test_cases = [
        {
            "args": {
                "corsika_file": tmp_path / "test.corsika",
                "curved_atmosphere_min_zenith_angle": 85.0 * u.deg,
            },
            "expected": {"zenith": 30, "azimuth": 270, "curved_atm": 85.0},
        },
        {
            "args": {
                "corsika_file": str(tmp_path / "dummy.corsika"),
                "correct_for_b_field_alignment": False,
                "curved_atmosphere_min_zenith_angle": 80 * u.deg,
            },
            "expected": {"zenith": 30, "azimuth": 270, "curved_atm": 80.0},
        },
    ]

    for case in test_cases:
        with patch("simtools.io.eventio_handler.get_corsika_run_and_event_headers") as mock_headers:
            run_header_dict = {
                "n_showers": 100,
                "energy_spectrum_slope": None,
                "energy_min": None,
                "energy_max": None,
                "n_observation_levels": 1,
                "observation_height": [220000.0],  # in cm (2200.0 m)
            }
            event_header_dict = {
                "event_number": None,
                "particle_id": 14,
                "theta_min": 30.0,
                "theta_max": 30.0,
                "phi_min": 90.0,
                "phi_max": 90.0,
                "viewcone_inner_angle": None,
                "viewcone_outer_angle": None,
                "n_reuse": None,
                "reuse_x": None,
                "reuse_y": None,
            }
            mock_headers.return_value = (run_header_dict, event_header_dict)

            config = CorsikaConfig(
                array_model=mock_array_model, label="test", args_dict=case["args"], db_config=None
            )

            assert config.zenith_angle == case["expected"]["zenith"]
            assert config.azimuth_angle == case["expected"]["azimuth"]
            assert config.curved_atmosphere_min_zenith_angle == pytest.approx(
                case["expected"]["curved_atm"]
            )


def test_initialize_from_config_values(mocker):
    """Test initialization with default and custom values."""
    mock_array_model = create_mock_array_model(mocker)

    cases = [
        {
            # Default values
            "args": {
                "azimuth_angle": 0 * u.deg,
                "zenith_angle": 20 * u.deg,
                "curved_atmosphere_min_zenith_angle": 90 * u.deg,
                "event_number_first_shower": 1,
                "nshow": 100,
                "energy_range": (10 * u.GeV, 10 * u.TeV),
                "view_cone": (0 * u.deg, 0 * u.deg),
                "core_scatter": (10, 1400 * u.m),
                "primary": "proton",
                "primary_id_type": "common_name",
                "eslope": -2,
                "correct_for_b_field_alignment": False,
            },
            "expected": {"azimuth": 0, "zenith": 20, "curved_atm": 90.0},
        },
        {
            # Custom values
            "args": {
                "azimuth_angle": 45 * u.deg,
                "zenith_angle": 60 * u.deg,
                "event_number_first_shower": 1,
                "nshow": 10,
                "energy_range": (10 * u.GeV, 100 * u.GeV),
                "view_cone": (0 * u.deg, 5 * u.deg),
                "core_scatter": (10, 1000 * u.m),
                "primary": "proton",
                "primary_id_type": "common_name",
                "eslope": -2,
                "correct_for_b_field_alignment": False,
            },
            "expected": {"azimuth": 45, "zenith": 60, "curved_atm": 90.0},
        },
    ]

    for case in cases:
        config = CorsikaConfig(
            array_model=mock_array_model, label="test", args_dict=case["args"], db_config=None
        )
        assert config.azimuth_angle == case["expected"]["azimuth"]
        assert config.zenith_angle == case["expected"]["zenith"]
        expected_curved_atm = case["expected"]["curved_atm"]
        assert config.curved_atmosphere_min_zenith_angle == pytest.approx(expected_curved_atm)


def test_primary_particle_setter_from_dict(corsika_config_no_array_model):
    """Test setting primary particle from dictionary."""
    test_dict = {
        "primary": "proton",
        "primary_id_type": "common_name",
    }
    corsika_config_no_array_model.primary_particle = test_dict
    assert corsika_config_no_array_model.primary_particle.name == "proton"
    assert corsika_config_no_array_model.primary == "proton"


def test_primary_particle_setter_from_corsika_id(corsika_config_no_array_model):
    """Test setting primary particle from CORSIKA 7 ID."""
    corsika_config_no_array_model.primary_particle = 14
    assert corsika_config_no_array_model.primary_particle.corsika7_id == 14
    assert corsika_config_no_array_model.primary == "proton"


def test_primary_particle_setter_from_none(corsika_config_no_array_model):
    """Test setting primary particle from None."""
    corsika_config_no_array_model.primary_particle = None
    assert corsika_config_no_array_model.primary_particle is not None


def test_primary_particle_getter(corsika_config_mock_array_model):
    """Test getting primary particle."""
    assert corsika_config_mock_array_model.primary == "proton"
    assert corsika_config_mock_array_model.primary_particle.corsika7_id == 14


def test_check_altitude_and_site(corsika_config_mock_array_model):
    """Test altitude and site validation."""
    # Should not raise when observation height matches site altitude
    corsika_config_mock_array_model._check_altitude_and_site(220000.0)

    # Should raise when observation height does not match
    with pytest.raises(ValueError, match="Observatory altitude does not match"):
        corsika_config_mock_array_model._check_altitude_and_site(300000.0)
