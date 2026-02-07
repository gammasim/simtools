#!/usr/bin/python3

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from simtools.settings import _Config


@pytest.fixture
def config_instance():
    return _Config()


def test_init(config_instance):
    assert config_instance._args == {}
    assert config_instance._db_config == {}
    assert all(
        getattr(config_instance, attr) is None
        for attr in ["_sim_telarray_path", "_sim_telarray_exe", "_corsika_path", "_corsika_exe"]
    )


@patch("pathlib.Path.is_dir", return_value=True)
@patch.dict(os.environ, {}, clear=True)
def test_load_with_args(mock_is_dir, config_instance):
    args = {"sim_telarray_path": "/path/to/simtel", "corsika_path": "/path/to/corsika"}
    config_instance.load(args=args)
    assert config_instance._args == args
    assert config_instance._sim_telarray_path == "/path/to/simtel"
    assert config_instance._corsika_path == "/path/to/corsika"


@patch.dict(os.environ, {}, clear=True)
def test_load_with_db_config(config_instance):
    db_config = {"db_server": "localhost"}
    config_instance.load(db_config=db_config)
    assert config_instance._db_config == db_config


@patch.dict(os.environ, {"SIMTOOLS_SIM_TELARRAY_PATH": "/env/simtel"})
def test_load_with_env_vars(config_instance):
    config_instance.load()
    assert config_instance._sim_telarray_path == "/env/simtel"


@patch.dict(os.environ, {}, clear=True)
def test_args_property(config_instance):
    args = {"test": "value"}
    config_instance.load(args=args)
    assert config_instance.args == args


@patch.dict(os.environ, {}, clear=True)
def test_db_config_property(config_instance):
    db_config = {"db_server": "localhost"}
    config_instance.load(db_config=db_config)
    assert config_instance.db_config == db_config


@patch("pathlib.Path.is_dir", return_value=True)
@patch.dict(os.environ, {}, clear=True)
def test_sim_telarray_path_property(mock_is_dir, config_instance):
    config_instance.load(args={"sim_telarray_path": "/path/to/simtel"})
    assert config_instance.sim_telarray_path == Path("/path/to/simtel")


def test_sim_telarray_path_property_none():
    config = _Config()
    with pytest.raises(FileNotFoundError):
        _ = config.sim_telarray_path


@patch("pathlib.Path.is_dir", return_value=True)
@patch("pathlib.Path.is_file", return_value=True)
@patch.dict(os.environ, {}, clear=True)
def test_sim_telarray_exe_property(mock_is_file, mock_is_dir, config_instance):
    config_instance.load(args={"sim_telarray_path": "/path/to/simtel"})
    assert config_instance.sim_telarray_exe == Path("/path/to/simtel/bin/sim_telarray")


def test_sim_telarray_exe_property_none():
    config = _Config()
    with pytest.raises(TypeError):
        _ = config.sim_telarray_exe


@patch("pathlib.Path.is_dir", return_value=True)
@patch("pathlib.Path.is_file", return_value=True)
@patch.dict(os.environ, {}, clear=True)
def test_sim_telarray_exe_debug_trace_property(mock_is_file, mock_is_dir, config_instance):
    config_instance.load(args={"sim_telarray_path": "/path/to/simtel"})
    assert config_instance.sim_telarray_exe_debug_trace == Path(
        "/path/to/simtel/bin/sim_telarray_debug_trace"
    )


@patch("pathlib.Path.is_dir", return_value=True)
@patch.dict(os.environ, {}, clear=True)
def test_corsika_path_property(mock_is_dir, config_instance):
    config_instance.load(args={"corsika_path": "/path/to/corsika"})
    assert config_instance.corsika_path == Path("/path/to/corsika")


@patch("pathlib.Path.is_dir", return_value=True)
@patch("pathlib.Path.is_file", return_value=True)
@patch("pathlib.Path.exists", return_value=False)
@patch.dict(os.environ, {}, clear=True)
def test_corsika_exe_property(mock_exists, mock_is_file, mock_is_dir, config_instance):
    config_instance.load(args={"corsika_path": "/path/to/corsika"})
    assert config_instance.corsika_exe == Path("/path/to/corsika/corsika")


@patch("pathlib.Path.exists", return_value=True)
@patch("pathlib.Path.is_dir", return_value=True)
@patch("pathlib.Path.is_file", return_value=True)
@patch.dict(os.environ, {}, clear=True)
def test_corsika_exe_curved_property(mock_is_file, mock_is_dir, mock_exists, config_instance):
    config_instance.load(
        args={
            "corsika_path": "/path/to/corsika",
            "corsika_he_interaction": "qgs3",
            "corsika_le_interaction": "urqmd",
        }
    )
    assert config_instance.corsika_exe_curved == Path("/path/to/corsika/corsika_qgs3_urqmd_curved")


@patch("pathlib.Path.is_dir", return_value=True)
@patch.dict(os.environ, {}, clear=True)
def test_corsika_dummy_file_property(mock_is_dir, config_instance):
    config_instance.load(args={"sim_telarray_path": "/path/to/simtel"})
    assert config_instance.corsika_dummy_file == Path("/path/to/simtel/run9991.corsika.gz")


def test_corsika_exe_curved_none():
    config = _Config()
    config._corsika_exe = None
    with pytest.raises(AttributeError):
        _ = config.corsika_exe_curved


@patch("pathlib.Path.exists", return_value=True)
@patch("pathlib.Path.is_dir", return_value=True)
@patch("pathlib.Path.is_file", return_value=True)
@patch.dict(os.environ, {}, clear=True)
def test_corsika_exe_curved_flat(mock_is_file, mock_is_dir, mock_exists, config_instance):
    config_instance.load(
        args={
            "corsika_path": "/path/to/corsika",
            "corsika_he_interaction": "qgs3",
            "corsika_le_interaction": "urqmd",
        }
    )
    assert config_instance.corsika_exe_curved == Path("/path/to/corsika/corsika_qgs3_urqmd_curved")


@patch("pathlib.Path.exists", return_value=False)
@patch("pathlib.Path.is_dir", return_value=True)
@patch("pathlib.Path.is_file", return_value=True)
@patch.dict(os.environ, {}, clear=True)
def test_corsika_exe_curved_legacy(mock_is_file, mock_is_dir, mock_exists, config_instance):
    config_instance.load(
        args={
            "corsika_path": "/path/to/corsika",
            "corsika_he_interaction": None,
            "corsika_le_interaction": None,
        }
    )
    assert config_instance.corsika_exe_curved == Path("/path/to/corsika/corsika-curved")


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


@patch("pathlib.Path.is_dir", return_value=True)
@patch.dict(os.environ, {}, clear=True)
def test_corsika_interaction_table_path_args_priority(mock_is_dir, config_instance):
    with patch.dict(os.environ, {"SIMTOOLS_CORSIKA_INTERACTION_TABLE_PATH": "/env/path"}):
        config_instance.load(args={"corsika_interaction_table_path": "/args/path"})
        assert config_instance.corsika_interaction_table_path == Path("/args/path")
