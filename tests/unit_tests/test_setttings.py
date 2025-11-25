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
    assert config_instance._sim_telarray_path is None
    assert config_instance._sim_telarray_exe is None
    assert config_instance._corsika_path is None
    assert config_instance._corsika_exe is None


@patch.dict(os.environ, {}, clear=True)
def test_load_with_args(config_instance):
    args = {"simtel_path": "/path/to/simtel", "corsika_path": "/path/to/corsika"}
    config_instance.load(args=args)
    assert config_instance._args == args
    assert config_instance._sim_telarray_path == "/path/to/simtel"
    assert config_instance._corsika_path == "/path/to/corsika"


@patch.dict(os.environ, {}, clear=True)
def test_load_with_db_config(config_instance):
    db_config = {"db_server": "localhost"}
    config_instance.load(db_config=db_config)
    assert config_instance._db_config == db_config


@patch.dict(os.environ, {"SIMTOOLS_SIMTEL_PATH": "/env/simtel"})
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


@patch.dict(os.environ, {}, clear=True)
def test_sim_telarray_path_property(config_instance):
    config_instance.load(args={"simtel_path": "/path/to/simtel"})
    assert config_instance.sim_telarray_path == Path("/path/to/simtel")


@patch.dict(os.environ, {}, clear=True)
def test_sim_telarray_path_property_none(config_instance):
    config_instance.load()
    assert config_instance.sim_telarray_path is None


@patch.dict(os.environ, {}, clear=True)
def test_sim_telarray_exe_property(config_instance):
    config_instance.load(args={"simtel_path": "/path/to/simtel"})
    assert config_instance.sim_telarray_exe == Path("/path/to/simtel/bin/sim_telarray")


@patch.dict(os.environ, {}, clear=True)
def test_sim_telarray_exe_property_none(config_instance):
    config_instance.load()
    assert config_instance.sim_telarray_exe is None


@patch.dict(os.environ, {}, clear=True)
def test_sim_telarray_exe_debug_trace_property(config_instance):
    config_instance.load(args={"simtel_path": "/path/to/simtel"})
    assert config_instance.sim_telarray_exe_debug_trace == Path(
        "/path/to/simtel/bin/sim_telarray_debug_trace"
    )


@patch.dict(os.environ, {}, clear=True)
def test_corsika_path_property(config_instance):
    config_instance.load(args={"corsika_path": "/path/to/corsika"})
    assert config_instance.corsika_path == Path("/path/to/corsika")


@patch.dict(os.environ, {}, clear=True)
def test_corsika_exe_property(config_instance):
    config_instance.load(args={"corsika_path": "/path/to/corsika"})
    assert config_instance.corsika_exe == Path("/path/to/corsika/corsika")


@patch.dict(os.environ, {}, clear=True)
def test_corsika_exe_curved_property(config_instance):
    config_instance.load(
        args={"corsika_path": "/path/to/corsika", "corsika_executable": "corsika_flat"}
    )
    assert config_instance.corsika_exe_curved == Path("/path/to/corsika/corsika_curved")


@patch.dict(os.environ, {}, clear=True)
def test_corsika_dummy_file_property(config_instance):
    config_instance.load(args={"simtel_path": "/path/to/simtel"})
    assert config_instance.corsika_dummy_file == Path("/path/to/simtel/run9991.corsika.gz")
