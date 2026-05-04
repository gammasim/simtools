#!/usr/bin/python3

from types import SimpleNamespace

import numpy as np

from simtools.model import sim_telarray_parameter_converter as converter


class _DummyLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(("info", message))

    def warning(self, message):
        self.messages.append(("warning", message))


def test_read_simtel_config_file_returns_none_when_parameter_dict_empty(monkeypatch):
    class _Reader:
        def __init__(self, **_kwargs):
            self.parameter_dict = {}

    monkeypatch.setattr(converter.simtel_config_reader, "SimtelConfigReader", _Reader)

    result = converter._read_simtel_config_file(
        {
            "simtel_cfg_file": "dummy.cfg",
            "simtel_telescope_name": "CT1",
        },
        schema_file="dummy.schema.yml",
    )

    assert result is None


def test_get_number_of_camera_pixel_handles_missing_data(monkeypatch):
    monkeypatch.setattr(converter.schema, "get_model_parameter_schema_file", lambda _name: "schema")

    def _raise_file_not_found(_args_dict, _schema_file, _camera_pixels=None):
        raise FileNotFoundError

    monkeypatch.setattr(converter, "_read_simtel_config_file", _raise_file_not_found)

    camera_pixels = converter._get_number_of_camera_pixel(
        {
            "simtel_cfg_file": "dummy.cfg",
            "simtel_telescope_name": "CT1",
        },
    )

    assert camera_pixels is None


def test_read_and_export_parameters(monkeypatch, tmp_test_directory):
    args_dict = {
        "simtel_cfg_file": "dummy.cfg",
        "simtel_telescope_name": "CT1",
        "skip_parameter": ["skip_me"],
        "telescope": "LSTN-01",
        "parameter_version": "1.0.0",
    }

    monkeypatch.setattr(
        converter.schema,
        "get_model_parameter_schema_files",
        lambda: (["keep_me", "skip_me", "missing_me"], ["schema1", "schema2", "schema3"]),
    )
    monkeypatch.setattr(
        converter.simtel_config_reader,
        "get_list_of_simtel_parameters",
        lambda _cfg: ["keep_simtel", "missing_simtel"],
    )
    monkeypatch.setattr(converter, "_get_number_of_camera_pixel", lambda _args_dict: 1855)

    class _ConfigReader:
        def __init__(self, value, simtel_name):
            self.parameter_dict = {"CT1": value}
            self.simtel_parameter_name = simtel_name
            self.compared = False

        def compare_simtel_config_with_schema(self):
            self.compared = True

    config_reader = _ConfigReader(value=42, simtel_name="keep_simtel")

    def _read_simtel_config_file(_args_dict, schema_file, camera_pixels=None):
        assert camera_pixels == 1855
        if schema_file == "schema1":
            return config_reader
        return None

    monkeypatch.setattr(converter, "_read_simtel_config_file", _read_simtel_config_file)

    written = {}

    class _DummyIOHandler:
        @staticmethod
        def get_output_file(file_name, sub_dir=None):
            output_dir = tmp_test_directory
            for directory in sub_dir or []:
                output_dir = output_dir.join(directory)
            return str(output_dir.join(file_name))

    def _write_model_parameter(**kwargs):
        written["kwargs"] = kwargs
        return {"file": False, "value": kwargs["output_file"]}

    monkeypatch.setattr(
        converter.writer.ModelDataWriter, "write_model_parameter", _write_model_parameter
    )

    parameters_not_in_simtel, simtel_parameters = converter.read_and_export_parameters(
        args_dict,
        _DummyIOHandler(),
    )

    assert parameters_not_in_simtel == ["missing_me"]
    assert simtel_parameters == ["missing_simtel"]
    assert written["kwargs"]["parameter_name"] == "keep_me"
    assert config_reader.compared is True


def test_print_parameters_not_found_handles_scalar_and_array(monkeypatch):
    args_dict = {
        "simtel_cfg_file": "dummy.cfg",
        "simtel_telescope_name": "CT1",
    }

    class _ReaderScalar:
        def __init__(self, **_kwargs):
            self.parameter_dict = {"default": 10.0, "CT1": 11.0}

    class _ReaderArray:
        def __init__(self, **_kwargs):
            self.parameter_dict = {
                "default": np.array([1.0, 2.0]),
                "CT1": np.array([1.5, 2.5]),
            }

    def _reader_factory(**kwargs):
        if kwargs["parameter_name"] == "scalar_parameter":
            return _ReaderScalar()
        return _ReaderArray()

    monkeypatch.setattr(converter.simtel_config_reader, "SimtelConfigReader", _reader_factory)

    warning_messages = []
    monkeypatch.setattr(converter._logger, "warning", lambda msg: warning_messages.append(msg))

    converter.print_parameters_not_found(
        ["missing_a"],
        ["scalar_parameter", "array_parameter"],
        args_dict,
    )

    assert any("Default value (scalar_parameter)" in msg for msg in warning_messages)
    assert any("Default value (array_parameter)" in msg for msg in warning_messages)


def test_run_conversion_workflow(monkeypatch):
    calls = []

    monkeypatch.setattr(
        converter,
        "read_and_export_parameters",
        lambda _args_dict, _io_handler: (["a"], ["b"]),
    )

    def _print_parameters_not_found(parameters_not_in_simtel, simtel_parameters, _args_dict):
        calls.append("print_parameters_not_found")

    def _print_list_of_files(_args_dict):
        calls.append("print_list_of_files")

    monkeypatch.setattr(converter, "print_parameters_not_found", _print_parameters_not_found)
    monkeypatch.setattr(converter, "print_list_of_files", _print_list_of_files)

    app_context = SimpleNamespace(args={}, io_handler=object())
    converter.run_conversion_workflow(app_context)

    assert calls == ["print_parameters_not_found", "print_list_of_files"]


def test_read_simtel_config_file_returns_none_when_parameter_dict_is_none(monkeypatch):
    class _Reader:
        def __init__(self, **_kwargs):
            self.parameter_dict = None

    monkeypatch.setattr(converter.simtel_config_reader, "SimtelConfigReader", _Reader)

    result = converter._read_simtel_config_file(
        {"simtel_cfg_file": "dummy.cfg", "simtel_telescope_name": "CT1"},
        schema_file="dummy.schema.yml",
    )

    assert result is None


def test_get_number_of_camera_pixel_happy_path(monkeypatch):
    monkeypatch.setattr(converter.schema, "get_model_parameter_schema_file", lambda _name: "schema")

    class _Reader:
        def __init__(self, **_kwargs):
            self.parameter_dict = {"CT1": 1855}

    monkeypatch.setattr(converter.simtel_config_reader, "SimtelConfigReader", _Reader)

    result = converter._get_number_of_camera_pixel(
        {"simtel_cfg_file": "dummy.cfg", "simtel_telescope_name": "CT1"}
    )

    assert result == 1855


def test_read_and_export_parameters_logs_file_name(monkeypatch, tmp_test_directory):
    args_dict = {
        "simtel_cfg_file": "dummy.cfg",
        "simtel_telescope_name": "CT1",
        "skip_parameter": [],
        "telescope": "LSTN-01",
        "parameter_version": "1.0.0",
    }

    monkeypatch.setattr(
        converter.schema,
        "get_model_parameter_schema_files",
        lambda: (["file_param"], ["schema1"]),
    )
    monkeypatch.setattr(
        converter.simtel_config_reader,
        "get_list_of_simtel_parameters",
        lambda _cfg: ["file_simtel"],
    )
    monkeypatch.setattr(converter, "_get_number_of_camera_pixel", lambda _args_dict: None)

    class _ConfigReader:
        parameter_dict = {"CT1": "some_file.fits"}
        simtel_parameter_name = "file_simtel"

        def compare_simtel_config_with_schema(self):
            # Stub used by this test: read_and_export_parameters calls this hook,
            # but the assertion here only verifies file-parameter logging/export.
            pass

    monkeypatch.setattr(converter, "_read_simtel_config_file", lambda *a, **kw: _ConfigReader())

    class _IOHandler:
        @staticmethod
        def get_output_file(file_name, sub_dir=None):
            return str(tmp_test_directory.join(file_name))

    monkeypatch.setattr(
        converter.writer.ModelDataWriter,
        "write_model_parameter",
        lambda **kwargs: {"file": True, "value": "some_file.fits"},
    )

    info_messages = []
    monkeypatch.setattr(converter._logger, "info", lambda msg: info_messages.append(msg))

    converter.read_and_export_parameters(args_dict, _IOHandler())

    assert any("File name for file_param" in msg for msg in info_messages)


def test_print_parameters_not_found_equal_values(monkeypatch):
    args_dict = {"simtel_cfg_file": "dummy.cfg", "simtel_telescope_name": "CT1"}

    class _ReaderEqual:
        def __init__(self, **_kwargs):
            self.parameter_dict = {"default": 5.0, "CT1": 5.0}

    monkeypatch.setattr(
        converter.simtel_config_reader, "SimtelConfigReader", lambda **kw: _ReaderEqual()
    )

    info_messages = []
    monkeypatch.setattr(converter._logger, "info", lambda msg: info_messages.append(msg))

    converter.print_parameters_not_found([], ["equal_param"], args_dict)

    assert any(
        "Default and telescope values for equal_param are equal" in msg for msg in info_messages
    )


def test_print_list_of_files(monkeypatch, tmp_test_directory):
    json_file = tmp_test_directory.join("param-1.0.0.json")
    json_file.write("{}")

    monkeypatch.setattr(
        converter.ascii_handler,
        "collect_data_from_file",
        lambda file_name: {"file": True, "value": "some_file.fits"},
    )

    info_messages = []
    monkeypatch.setattr(converter._logger, "info", lambda msg: info_messages.append(msg))

    converter.print_list_of_files({"output_path": str(tmp_test_directory)})

    assert any("param-1.0.0.json" in msg for msg in info_messages)
