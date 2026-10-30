import math

import pytest

from simtools.applications import submit_model_parameter_from_external as submit_parameter

PARAMETER_NAME = "flasher_pulse_shape"
INSTRUMENT_NAME = "LSFN-design"
SITE_NAME = "North"
PARAMETER_VERSION = "1.0.1"
SCHEMA_VERSION = "0.2.0"
VALUE_ARGUMENT = "value"
PULSE_SHAPE_VALUES = ["Gauss-Exponential", "0.4 ns", "1.59 ns"]


def test_value_argument_accepts_multiple_values(run_context):
    parser = submit_parameter.APPLICATION.build_parser()
    args = parser.parse_args(
        [
            "--parameter",
            PARAMETER_NAME,
            "--instrument",
            INSTRUMENT_NAME,
            "--site",
            SITE_NAME,
            "--parameter_version",
            PARAMETER_VERSION,
            "--value",
            *PULSE_SHAPE_VALUES,
        ]
    )

    assert args.value == PULSE_SHAPE_VALUES
    parameter = submit_parameter.writer.ModelDataWriter(
        context=run_context
    ).get_validated_parameter_dict(
        parameter_name=args.parameter,
        value=args.value,
        instrument=args.instrument,
        parameter_version=args.parameter_version,
        model_parameter_schema_version=SCHEMA_VERSION,
    )
    assert parameter["value"][0] == PULSE_SHAPE_VALUES[0]
    assert math.isclose(parameter["value"][1], 0.4)
    assert math.isclose(parameter["value"][2], 1.59)
    assert parameter["unit"] == [None, "ns", "ns"]


@pytest.mark.parametrize(
    ("value", "expected_value"),
    [
        (PULSE_SHAPE_VALUES, PULSE_SHAPE_VALUES),
        (["120"], "120"),
    ],
)
def test_main_passes_value_to_writer(mocker, value, expected_value):
    context = mocker.Mock()
    context.args = {
        "parameter": PARAMETER_NAME,
        "instrument": INSTRUMENT_NAME,
        "site": SITE_NAME,
        "parameter_version": PARAMETER_VERSION,
        "model_parameter_schema_version": SCHEMA_VERSION,
        VALUE_ARGUMENT: value,
        "output_path": None,
        "check_parameter_version": False,
    }
    application = mocker.Mock()
    application.start.return_value = context
    mocker.patch.object(submit_parameter, "APPLICATION", application)
    model_data_writer = mocker.patch.object(submit_parameter.writer, "ModelDataWriter")
    model_data_writer.return_value.get_parameter_type_for_schema.return_value = "float64"

    submit_parameter.main()

    assert (
        model_data_writer.write_model_parameter.call_args.kwargs[VALUE_ARGUMENT] == expected_value
    )
