#!/usr/bin/python3

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from simtools.configuration import defaults
from simtools.corsika.build_options import CorsikaBuildVariant
from simtools.settings import _Config


@pytest.fixture
def clear_simtools_env():
    """Fixture to clear simtools environment variables for tests that need isolated _Config."""
    old_env = {}
    simtools_vars = [
        "SIMTOOLS_SIM_TELARRAY_PATH",
        "SIMTOOLS_SIM_TELARRAY_EXECUTABLE",
        "SIMTOOLS_CORSIKA_PATH",
        "SIMTOOLS_CORSIKA_EXECUTABLE",
        "SIMTOOLS_CORSIKA_INTERACTION_TABLE_PATH",
        "SIMTOOLS_CORSIKA_HE_INTERACTION",
        "SIMTOOLS_CORSIKA_LE_INTERACTION",
    ]
    for var in simtools_vars:
        old_env[var] = os.environ.pop(var, None)
    yield
    for var, val in old_env.items():
        if val is not None:
            os.environ[var] = val


@pytest.fixture
def simtools_settings(clear_simtools_env):
    """Override autouse fixture for isolated tests."""
    pass


@pytest.fixture
def config_instance(mocker):
    variants = tuple(
        CorsikaBuildVariant(
            executable=f"corsika_{he_model}_urqmd_{geometry}",
            config=f"config_{he_model}_urqmd_{geometry}",
            atmosphere_geometry=geometry,
            he_hadronic_model=he_model,
            le_hadronic_model="urqmd",
        )
        for he_model in ("epos", "qgs3")
        for geometry in ("flat", "curved")
    )
    mocker.patch("simtools.settings.get_installed_corsika_build_variants", return_value=variants)
    return _Config()


@patch("pathlib.Path.is_dir", return_value=True)
@patch.dict(os.environ, {}, clear=True)
def test_load_with_args(mock_is_dir, config_instance):
    args = {"sim_telarray_path": "/path/to/simtel", "corsika_path": "/path/to/corsika"}
    config_instance.load(args=args)
    assert config_instance._args == args
    assert config_instance._sim_telarray_path == "/path/to/simtel"
    assert config_instance._corsika_path == "/path/to/corsika"


@patch("os.access", return_value=True)
@patch("pathlib.Path.exists", return_value=True)
@patch("pathlib.Path.is_dir", return_value=True)
@patch("pathlib.Path.is_file", return_value=True)
@patch.dict(os.environ, {}, clear=True)
def test_get_corsika_exec_uses_default_interaction_models(
    mock_is_file, mock_is_dir, mock_exists, mock_access, config_instance
):
    config_instance.load(args={"corsika_path": "/path/to/corsika"})
    assert config_instance.corsika_exe == Path(
        "/path/to/corsika/"
        f"corsika_{defaults.CORSIKA_HE_INTERACTION}_{defaults.CORSIKA_LE_INTERACTION}_flat"
    )


@patch("pathlib.Path.is_dir", return_value=True)
@patch.dict(os.environ, {}, clear=True)
def test_sim_telarray_path_property(mock_is_dir, config_instance):
    config_instance.load(args={"sim_telarray_path": "/path/to/simtel"})
    assert config_instance.sim_telarray_path == Path("/path/to/simtel")


def test_sim_telarray_path_property_none():
    config = _Config()
    with pytest.raises(FileNotFoundError):
        _ = config.sim_telarray_path


@patch("os.access", return_value=True)
@patch("pathlib.Path.is_dir", return_value=True)
@patch("pathlib.Path.is_file", return_value=True)
@patch.dict(os.environ, {}, clear=True)
def test_sim_telarray_exe_property(mock_is_file, mock_is_dir, mock_access, config_instance):
    config_instance.load(args={"sim_telarray_path": "/path/to/simtel"})
    assert config_instance.sim_telarray_exe == Path("/path/to/simtel/bin/sim_telarray")


@patch("os.access", return_value=True)
@patch("pathlib.Path.is_dir", return_value=True)
@patch("pathlib.Path.is_file", return_value=True)
@patch.dict(os.environ, {}, clear=True)
def test_sim_telarray_exe_debug_trace_property(
    mock_is_file, mock_is_dir, mock_access, config_instance
):
    config_instance.load(args={"sim_telarray_path": "/path/to/simtel"})
    assert config_instance.sim_telarray_exe_debug_trace == Path(
        "/path/to/simtel/bin/sim_telarray_debug_trace"
    )


@patch("os.access", return_value=True)
@patch("pathlib.Path.exists", return_value=True)
@patch("pathlib.Path.is_dir", return_value=True)
@patch("pathlib.Path.is_file", return_value=True)
@patch.dict(os.environ, {}, clear=True)
def test_corsika_exe_curved_property(
    mock_is_file, mock_is_dir, mock_exists, mock_access, config_instance
):
    config_instance.load(
        args={
            "corsika_path": "/path/to/corsika",
            "corsika_he_interaction": "qgs3",
            "corsika_le_interaction": "urqmd",
        }
    )
    assert config_instance.corsika_exe_curved == Path("/path/to/corsika/corsika_qgs3_urqmd_curved")


def test_corsika_exe_curved_none():
    config = _Config()
    config._corsika_exe = None
    with pytest.raises(FileNotFoundError, match="manifest has not been loaded"):
        _ = config.corsika_exe_curved


@patch("pathlib.Path.is_dir", return_value=True)
@patch.dict(os.environ, {}, clear=True)
def test_corsika_interaction_table_path_property(mock_is_dir, config_instance):
    config_instance.load(args={"corsika_interaction_table_path": "/path/to/interaction_tables"})
    assert config_instance.corsika_interaction_table_path == Path("/path/to/interaction_tables")


def test_corsika_interaction_table_path_property_none():
    config = _Config()
    with pytest.raises(FileNotFoundError):
        _ = config.corsika_interaction_table_path


@patch("pathlib.Path.is_dir", return_value=False)
@patch.dict(os.environ, {}, clear=True)
def test_corsika_interaction_table_path_property_invalid(mock_is_dir, config_instance):
    config_instance.load(args={"corsika_interaction_table_path": "/invalid/path"})
    with pytest.raises(FileNotFoundError):
        _ = config_instance.corsika_interaction_table_path
