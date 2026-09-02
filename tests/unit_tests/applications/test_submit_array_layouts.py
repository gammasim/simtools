from types import SimpleNamespace

import pytest

from simtools.applications import submit_array_layouts


def _direct_args():
    return {
        "array_layouts": None,
        "array_layout_name": "South-dual-camera-example",
        "array_element_list": ["MSTS-01", "MSTS-301"],
        "reference_array_layout": "hyper_array",
        "model_version": ["7.0.0"],
        "parameter_version": "3.0.0",
        "updated_parameter_version": "3.0.99",
        "site": "South",
        "output_path": "output",
    }


def test_main_builds_merges_and_writes_direct_layout(mocker):
    model_reader = mocker.Mock()
    app_context = SimpleNamespace(args=_direct_args(), model_reader=model_reader)
    mocker.patch(
        "simtools.application.definition.ApplicationDefinition.start", return_value=app_context
    )

    mocker.patch.object(
        submit_array_layouts,
        "prepare_array_layouts_for_submission",
        return_value=(
            {
                "value": [
                    {
                        "name": "hyper_array",
                        "elements": ["MSTS-01", "MSTS-301"],
                    },
                    {
                        "name": "South-dual-camera-example",
                        "elements": ["MSTS-01", "MSTS-301"],
                    },
                ]
            },
            "7.0.0",
        ),
    )
    model_reader.read_production_table.return_value = {
        "parameters": {"MSTS-01": {}, "MSTS-301": {}},
    }
    writer = mocker.patch.object(submit_array_layouts, "write_array_layouts")

    submit_array_layouts.main()

    model_reader.read_production_table.assert_called_once_with(
        collection_name="telescopes", model_version="7.0.0"
    )
    written_layouts = writer.call_args.kwargs["array_layouts"]["value"]
    assert {
        "name": "South-dual-camera-example",
        "elements": ["MSTS-01", "MSTS-301"],
    } in written_layouts


def test_main_retains_legacy_file_input(mocker):
    args = _direct_args()
    args.update(
        {
            "array_layouts": "array_layouts.json",
            "array_layout_name": None,
            "array_element_list": None,
        }
    )
    model_reader = mocker.Mock()
    app_context = SimpleNamespace(args=args, model_reader=model_reader)
    mocker.patch(
        "simtools.application.definition.ApplicationDefinition.start", return_value=app_context
    )
    mocker.patch.object(
        submit_array_layouts,
        "prepare_array_layouts_for_submission",
        return_value=(
            {"site": "South", "value": [{"name": "existing", "elements": ["MSTS-01"]}]},
            "7.0.0",
        ),
    )
    model_reader.read_production_table.return_value = {
        "parameters": {"MSTS-01": {}},
    }
    writer = mocker.patch.object(submit_array_layouts, "write_array_layouts")

    submit_array_layouts.main()

    assert writer.call_args.kwargs["array_layouts"]["value"] == [
        {"name": "existing", "elements": ["MSTS-01"]}
    ]


def test_direct_layout_rejects_base_parameter_from_other_site(mocker):
    from simtools.layout import array_layout_utils

    database = mocker.Mock()
    database.get_model_parameter.return_value = {"array_layouts": {"site": "North", "value": []}}

    with pytest.raises(ValueError, match="does not match requested site 'South'"):
        array_layout_utils.prepare_array_layouts_for_submission(database, _direct_args())
