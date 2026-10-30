"""Tests for shared command-line argument definitions."""

from simtools import constants
from simtools.configuration import arguments
from simtools.configuration.commandline_parser import CommandLineParser

_OPTION = "option"
_CHOICE = "choice"


def test_select_accepts_multiple_values_after_one_option():
    parser = CommandLineParser()
    parser.add_argument_definitions((arguments.SELECT,))

    args = parser.parse_args(
        [
            "--select",
            "configuration.primary=gamma",
            "configuration.site=North",
        ]
    )

    assert args.select == ["configuration.primary=gamma", "configuration.site=North"]


def test_layout_argument_keeps_definition_fields():
    argument = arguments.ArgumentDefinition(
        "layout", group="simulation model", aliases=("old_layout",), preserve_by_version=True
    )

    layout_argument = arguments._layout_argument(argument, required=True)

    assert layout_argument.aliases == ("old_layout",)
    assert layout_argument.preserve_by_version
    assert layout_argument.exclusive_group == "array layout"
    assert layout_argument.exclusive_group_required


def test_call_overrides_definition_fields_and_argparse_keywords():
    argument = arguments.ArgumentDefinition(_OPTION, group="display", type=int, required=True)

    copied = argument(exclusive_group=_CHOICE, aliases=("old_option",), default=3)

    assert copied.group == "display"
    assert copied.exclusive_group == _CHOICE
    assert copied.aliases == ("old_option",)
    assert dict(copied.kwargs) == {"type": int, "required": True, "default": 3}
    assert argument.aliases == ()


def test_without_requiredness_relaxes_argument_and_exclusive_group():
    argument = arguments.ArgumentDefinition(
        _OPTION, exclusive_group=_CHOICE, exclusive_group_required=True, required=True
    )

    relaxed = argument.without_requiredness()

    assert relaxed.kwargs["required"] is False
    assert relaxed.exclusive_group == _CHOICE
    assert relaxed.exclusive_group_required is False


def test_alias_sets_canonical_destination():
    parser = CommandLineParser()
    parser.add_argument_definitions((arguments.FIGURE_DPI(aliases=("dpi",)),))

    assert parser.parse_args(["--dpi", "72"]).figure_dpi == 72


def test_seed_arguments_accept_previous_names():
    parser = CommandLineParser()
    parser.add_argument_definitions(arguments.SIM_TELARRAY_ARGUMENTS)

    args = parser.parse_args(
        [
            "--sim_telarray_instrument_seed",
            "7",
            "--sim_telarray_random_instrument_instances",
            "3",
            "--sim_telarray_seed",
            "11",
            "13",
        ]
    )

    assert args.instrument_seed == 7
    assert args.random_instrument_instances == 3
    assert args.simulation_seed == [11, 13]


def test_seed_arguments_canonical_names():
    parser = CommandLineParser()
    parser.add_argument_definitions(arguments.RAY_TRACING_ARGUMENTS)

    args = parser.parse_args(["--instrument_seed", "5", "--simulation_seed", "9"])

    assert args.instrument_seed == 5
    assert args.simulation_seed == [9]
    assert args.random_instrument_instances == constants.RAY_TRACING_INSTRUMENT_INSTANCES
