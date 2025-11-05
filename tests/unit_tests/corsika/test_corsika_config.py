#!/usr/bin/python3

import copy
import logging
import pathlib
from unittest.mock import MagicMock, Mock, patch

import pytest
from astropy import units as u

from simtools.corsika.corsika_config import CorsikaConfig, InvalidCorsikaInputError

logger = logging.getLogger()

CORSIKA_CONFIG_MODE_PARAMETER = "simtools.corsika.corsika_config.ModelParameter"
MUON_PLUS_PARTICLE = "muon+"
PROTON_PARTICLE = "proton"


@pytest.fixture
def corsika_config_no_array_model(corsika_config_data):
    corsika_config_data["correct_for_b_field_alignment"] = False
    return CorsikaConfig(
        array_model=None, label="test-corsika-config", args_dict=corsika_config_data, db_config=None
    )


@pytest.fixture
def gcm2():
    return "g/cm2"


@pytest.fixture
def corsika_configuration_parameters(gcm2):
    return {
        "corsika_iact_max_bunches": {"value": 1000000, "unit": None},
        "corsika_cherenkov_photon_bunch_size": {"value": 5.0, "unit": None},
        "corsika_cherenkov_photon_wavelength_range": {"value": [240.0, 1000.0], "unit": "nm"},
        "corsika_first_interaction_height": {"value": 0.0, "unit": "cm"},
        "corsika_particle_kinetic_energy_cutoff": {
            "value": [0.3, 0.1, 0.020, 0.020],
            "unit": "GeV",
        },
        "corsika_longitudinal_shower_development": {"value": 20.0, "unit": gcm2},
        "corsika_iact_split_auto": {"value": 15000000, "unit": None},
        "corsika_starting_grammage": {"value": 0.0, "unit": gcm2},
        "corsika_iact_io_buffer": {"value": 800, "unit": "MB"},
    }


@pytest.fixture
def corsika_configuration_parameters_muon_grammage(gcm2, corsika_configuration_parameters):
    """Fixture for CORSIKA configuration parameters with muon grammage."""
    params = corsika_configuration_parameters.copy()
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
        "unit": gcm2,
    }
    return params


@pytest.fixture
def corsika_configuration_parameters_teltype_grammage(gcm2, corsika_configuration_parameters):
    """Fixture for CORSIKA configuration parameters with muon grammage."""
    params = corsika_configuration_parameters.copy()
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
        "unit": gcm2,
    }
    return params


def test_repr(corsika_config_mock_array_model):
    assert "site" in repr(corsika_config_mock_array_model)


def test_fill_corsika_configuration(corsika_config_mock_array_model):
    empty_config = CorsikaConfig(
        array_model=None, label="test-corsika-config", args_dict=None, db_config=None
    )
    assert empty_config.config == {}

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


def test_fill_corsika_configuration_model_version(corsika_config_mock_array_model, gcm2):
    """Test handling a list of model versions as input, taking the first one only."""

    with patch(CORSIKA_CONFIG_MODE_PARAMETER) as mock_model_parameter:
        mock_params = Mock()
        mock_params.get_simulation_software_parameters.return_value = {
            "corsika_iact_max_bunches": {"value": 1000000, "unit": None},
            "corsika_cherenkov_photon_bunch_size": {"value": 5.0, "unit": None},
            "corsika_first_interaction_height": {"value": 0.0, "unit": "cm"},
            "corsika_starting_grammage": {"value": 0.0, "unit": gcm2},
            "corsika_longitudinal_shower_development": {"value": 20.0, "unit": gcm2},
            "corsika_cherenkov_photon_wavelength_range": {"value": [240.0, 1000.0], "unit": "nm"},
            "corsika_iact_split_auto": {"value": 15000000, "unit": None},
            "corsika_iact_io_buffer": {"value": 800, "unit": "MB"},
            "corsika_particle_kinetic_energy_cutoff": {
                "value": [0.3, 0.1, 0.020, 0.020],
                "unit": "GeV",
            },
        }
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
        config = corsika_config_mock_array_model.fill_corsika_configuration(args_dict, db_config={})

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
    corsika_config_mock_array_model, corsika_configuration_parameters
):
    with pytest.raises(KeyError):
        corsika_config_mock_array_model._corsika_configuration_interaction_flags({})

    parameters = corsika_config_mock_array_model._corsika_configuration_interaction_flags(
        corsika_configuration_parameters
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


def test_input_config_corsika_starting_grammage(corsika_config_mock_array_model, gcm2):
    assert (
        corsika_config_mock_array_model._input_config_corsika_starting_grammage(
            {"value": 0.0, "unit": gcm2}
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
    corsika_configuration_parameters_muon_grammage,
    corsika_configuration_parameters_teltype_grammage,
):
    # default behavior with proton as primary
    assert (
        corsika_config_mock_array_model._input_config_corsika_starting_grammage(
            corsika_configuration_parameters_muon_grammage["corsika_starting_grammage"]
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
            corsika_configuration_parameters_muon_grammage["corsika_starting_grammage"]
        )
        == "10.0"
    )
    corsika_config_mock_array_model_muon.primary_particle = {
        "primary": "gamma",
        "primary_id_type": "common_name",
    }
    assert (
        corsika_config_mock_array_model_muon._input_config_corsika_starting_grammage(
            corsika_configuration_parameters_muon_grammage["corsika_starting_grammage"]
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
            corsika_configuration_parameters_teltype_grammage["corsika_starting_grammage"]
        )
        == "10.0"
    )

    corsika_config_mock_array_model_teltype.primary_particle = {
        "primary": PROTON_PARTICLE,
        "primary_id_type": "common_name",
    }
    assert (
        corsika_config_mock_array_model_teltype._input_config_corsika_starting_grammage(
            corsika_configuration_parameters_teltype_grammage["corsika_starting_grammage"]
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


def test_input_config_corsika_longitudinal_parameters(corsika_config_mock_array_model, gcm2):
    assert corsika_config_mock_array_model._input_config_corsika_longitudinal_parameters(
        {"value": 20.0, "unit": gcm2}
    ) == ["T", "20.0", "F", "F"]
    assert corsika_config_mock_array_model._input_config_corsika_longitudinal_parameters(
        {"value": 10.0, "unit": "kg/cm2"}
    ) == ["T", "10000.0", "F", "F"]


def test_corsika_configuration_cherenkov_parameters(
    corsika_config_mock_array_model, corsika_configuration_parameters
):
    cherenk_dict = corsika_config_mock_array_model._corsika_configuration_cherenkov_parameters(
        corsika_configuration_parameters
    )
    assert isinstance(cherenk_dict, dict)
    assert "CERSIZ" in cherenk_dict
    assert isinstance(cherenk_dict["CERSIZ"], list)


def test_input_config_corsika_cherenkov_wavelength(corsika_config_mock_array_model):
    assert corsika_config_mock_array_model._input_config_corsika_cherenkov_wavelength(
        {"value": [240.0, 1000.0], "unit": "nm"}
    ) == ["240.0", "1000.0"]


def test_corsika_configuration_iact_parameters(
    corsika_config_mock_array_model, corsika_configuration_parameters
):
    iact_dict = corsika_config_mock_array_model._corsika_configuration_iact_parameters(
        corsika_configuration_parameters
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


def test_rotate_azimuth_by_180deg_no_correct_for_geomagnetic_field_alignment(
    corsika_config_mock_array_model,
):
    assert corsika_config_mock_array_model._rotate_azimuth_by_180deg(
        0.0, correct_for_geomagnetic_field_alignment=False
    ) == pytest.approx(180.0)
    assert corsika_config_mock_array_model._rotate_azimuth_by_180deg(
        360.0, correct_for_geomagnetic_field_alignment=False
    ) == pytest.approx(180.0)
    assert corsika_config_mock_array_model._rotate_azimuth_by_180deg(
        450.0, correct_for_geomagnetic_field_alignment=False
    ) == pytest.approx(270.0)
    assert corsika_config_mock_array_model._rotate_azimuth_by_180deg(
        180.0, correct_for_geomagnetic_field_alignment=False
    ) == pytest.approx(0.0)
    assert corsika_config_mock_array_model._rotate_azimuth_by_180deg(
        -180.0, correct_for_geomagnetic_field_alignment=False
    ) == pytest.approx(0.0)


def test_rotate_azimuth_by_180deg(corsika_config_mock_array_model):
    assert corsika_config_mock_array_model._rotate_azimuth_by_180deg(
        0.0, correct_for_geomagnetic_field_alignment=True
    ) == pytest.approx(175.467)
    assert corsika_config_mock_array_model._rotate_azimuth_by_180deg(
        360.0, correct_for_geomagnetic_field_alignment=True
    ) == pytest.approx(175.467)
    assert corsika_config_mock_array_model._rotate_azimuth_by_180deg(
        450.0, correct_for_geomagnetic_field_alignment=True
    ) == pytest.approx(265.467)
    assert corsika_config_mock_array_model._rotate_azimuth_by_180deg(
        180.0, correct_for_geomagnetic_field_alignment=True
    ) == pytest.approx(355.467)
    assert corsika_config_mock_array_model._rotate_azimuth_by_180deg(
        -180.0, correct_for_geomagnetic_field_alignment=True
    ) == pytest.approx(355.467)


def test_set_primary_particle(corsika_config_mock_array_model):
    from simtools.corsika.primary_particle import PrimaryParticle

    cc = corsika_config_mock_array_model
    assert isinstance(cc._set_primary_particle(args_dict=None), PrimaryParticle)
    assert isinstance(
        cc._set_primary_particle(args_dict={"primary_id_type": None}), PrimaryParticle
    )

    p_common_name = cc._set_primary_particle(
        args_dict={"primary": PROTON_PARTICLE, "primary_id_type": "common_name"}
    )
    assert p_common_name.name == PROTON_PARTICLE

    p_corsika7_id = cc._set_primary_particle(
        args_dict={"primary": 14, "primary_id_type": "corsika7_id"}
    )
    assert p_corsika7_id.name == PROTON_PARTICLE

    p_pdg_id = cc._set_primary_particle(args_dict={"primary": 2212, "primary_id_type": "pdg_id"})
    assert p_pdg_id.name == PROTON_PARTICLE


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


@pytest.mark.uses_model_database
def test_get_corsika_telescope_list(corsika_config):
    cc = corsika_config
    telescope_list_str = cc.get_corsika_telescope_list()
    assert telescope_list_str.count("TELESCOPE") > 0
    assert telescope_list_str.count("LSTS") > 0


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

        with pytest.raises(InvalidCorsikaInputError, match="CORSIKA parameter 'param2' differs"):
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
