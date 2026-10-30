from io import StringIO

import pytest

from simtools.reporting.docs_parameter_delta import ParameterDelta
from simtools.reporting.docs_read_parameters import ReadParameters


@pytest.mark.parametrize(
    (
        "args",
        "report_method",
        "report_filename",
        "base_params",
        "current_params",
        "expected_link",
        "expected_text",
        "patch_telescope_model",
    ),
    [
        (
            {"site": "North", "telescope": "LSTN-01", "model_version": "6.0.1"},
            "produce_array_element_report",
            "LSTN-01.md",
            {
                "focal_length": {
                    "value": 2800.0,
                    "unit": "cm",
                    "parameter_version": "1.0.0",
                    "instrument": "LSTN-01",
                }
            },
            {
                "focal_length": {
                    "value": 2810.0,
                    "unit": "cm",
                    "parameter_version": "1.0.1",
                    "instrument": "LSTN-01",
                }
            },
            "../6.0.0/LSTN-01.md",
            "focal_length",
            True,
        ),
        (
            {"site": "South", "model_version": "6.2.1", "observatory": True},
            "produce_observatory_report",
            "OBS-South.md",
            {
                "site_elevation": {
                    "value": 2200,
                    "unit": "m",
                    "parameter_version": "1.0.0",
                }
            },
            {
                "site_elevation": {
                    "value": 2201,
                    "unit": "m",
                    "parameter_version": "1.0.1",
                }
            },
            "../6.2.0/OBS-South.md",
            "delta report",
            False,
        ),
    ],
)
def test_patch_version_reports_write_delta(
    tmp_test_directory,
    mocker,
    args,
    report_method,
    report_filename,
    base_params,
    current_params,
    expected_link,
    expected_text,
    patch_telescope_model,
    run_context,
):
    read_parameters = ReadParameters(args=args, output_path=tmp_test_directory, context=run_context)
    get_model_parameters = mocker.patch.object(
        read_parameters.model_reader,
        "get_model_parameters",
        side_effect=[base_params, current_params],
    )
    export_model_files = mocker.patch.object(read_parameters.model_reader, "export_model_files")
    telescope_model = None
    if patch_telescope_model:
        telescope_model = mocker.patch("simtools.reporting.docs_read_parameters.TelescopeModel")

    getattr(read_parameters, report_method)()

    export_model_files.assert_not_called()
    if telescope_model is not None:
        telescope_model.assert_not_called()

    content = (tmp_test_directory / report_filename).read_text(encoding="utf-8")
    assert expected_link in content
    assert expected_text in content
    assert get_model_parameters.call_count == 2


@pytest.mark.parametrize(
    ("value1", "value2", "expected"),
    [
        (None, 1, False),
        (1.0, 1.0 + 1e-13, True),
        (1.0, 1.0 + 1e-9, False),
        ([1, {"a": 2}], [1, {"a": 2}], True),
        ({"a": 1}, {"a": 1, "b": 2}, False),
        ("same", "same", True),
    ],
)
def test__values_equal_branches(value1, value2, expected, tmp_path, run_context):
    rp = ReadParameters(args={}, output_path=tmp_path, context=run_context)
    assert rp._values_equal(value1, value2) is expected


def test__parameter_changed_branches(tmp_path, run_context):
    rp = ReadParameters(args={}, output_path=tmp_path, context=run_context)
    base = {"parameter_version": "1.0.0", "unit": "m", "file": False, "value": 1}

    assert rp._parameter_changed(base, None) is True
    assert rp._parameter_changed(base, {**base, "parameter_version": "1.0.1"}) is True
    assert rp._parameter_changed(base, {**base, "unit": "cm"}) is True
    assert rp._parameter_changed(base, {**base, "file": True}) is True
    assert rp._parameter_changed(base, {**base, "value": 2}) is True


def _delta_entry(value):
    return {"parameter_version": "1.0", "unit": "m", "file": False, "value": value}


DELTA_PARAM_DATA = {"focal_length": _delta_entry(1)}

DELTA_SITE_DATA = {"site_elevation": _delta_entry(2200)}


@pytest.mark.parametrize(
    ("param_data", "contains"),
    [
        (None, "-"),
        ({}, "-"),
        ({"value": [{"a": 1}, {"b": 2}], "unit": None, "file": False}, "2 rows"),
        ({"value": "some_file.dat", "unit": " ", "file": True}, "some_file.dat"),
        ({"value": list(range(15)), "unit": "m", "file": False}, "14 m"),
        ({"value": 42.0, "unit": "m", "file": False}, "42.0"),
    ],
)
def test__format_value_for_delta(param_data, contains, tmp_path, run_context):
    rp = ReadParameters(args={}, output_path=tmp_path, context=run_context)
    assert contains in rp._format_value_for_delta("p", param_data)


def test__write_delta_helpers(tmp_path, run_context):
    rp = ReadParameters(args={"model_version": "6.0.1"}, output_path=tmp_path, context=run_context)
    buf = StringIO()
    rp._write_delta_report_table(
        buf,
        [
            ParameterDelta(
                parameter="focal_length",
                base_param_version="1.0.0",
                new_param_version="1.0.1",
                base_value="2800",
                new_value="2810",
            )
        ],
    )
    content = buf.getvalue()
    assert "focal_length" in content
    assert "2810" in content


@pytest.mark.parametrize(
    ("method_name", "args", "base_version", "base_data", "current_data"),
    [
        (
            "_produce_array_element_delta_report",
            {"site": "North", "telescope": "LSTN-01", "model_version": "6.0.1"},
            "6.0.0",
            {},
            None,
        ),
        (
            "_produce_array_element_delta_report",
            {"site": "North", "telescope": "LSTN-01", "model_version": "6.0.1"},
            "6.0.0",
            DELTA_PARAM_DATA,
            {},
        ),
        (
            "_produce_observatory_delta_report",
            {"site": "South", "model_version": "6.2.1"},
            "6.2.0",
            {},
            None,
        ),
        (
            "_produce_observatory_delta_report",
            {"site": "South", "model_version": "6.2.1"},
            "6.2.0",
            DELTA_SITE_DATA,
            {},
        ),
    ],
)
def test__delta_report_returns_false_when_base_or_current_missing(
    method_name, args, base_version, base_data, current_data, tmp_path, mocker, run_context
):
    rp = ReadParameters(args=args, output_path=tmp_path, context=run_context)
    get_model_parameters = mocker.patch.object(rp.model_reader, "get_model_parameters")
    if current_data is None:
        get_model_parameters.return_value = base_data
    else:
        get_model_parameters.side_effect = [base_data, current_data]

    report_method = getattr(rp, method_name)
    assert report_method(base_version) is False


@pytest.mark.parametrize(
    (
        "method_name",
        "args",
        "base_version",
        "param_data",
        "output_filename",
        "patch_names_model_parameters",
    ),
    [
        (
            "_produce_array_element_delta_report",
            {"site": "North", "telescope": "LSTN-01", "model_version": "6.0.1"},
            "6.0.0",
            DELTA_PARAM_DATA,
            "LSTN-01.md",
            True,
        ),
        (
            "_produce_observatory_delta_report",
            {"site": "South", "model_version": "6.2.1"},
            "6.2.0",
            DELTA_SITE_DATA,
            "OBS-South.md",
            False,
        ),
    ],
)
def test__delta_report_no_changes(
    method_name,
    args,
    base_version,
    param_data,
    output_filename,
    patch_names_model_parameters,
    tmp_path,
    mocker,
    run_context,
):
    rp = ReadParameters(args=args, output_path=tmp_path, context=run_context)
    mocker.patch.object(
        rp.model_reader, "get_model_parameters", side_effect=[param_data, param_data]
    )
    if patch_names_model_parameters:
        mocker.patch(
            "simtools.data_model.parameter_catalog.model_parameters",
            return_value={"focal_length": {}},
        )

    report_method = getattr(rp, method_name)
    assert report_method(base_version) is True
    content = (tmp_path / output_filename).read_text()
    assert "No parameter changes detected" in content
