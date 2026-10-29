"""Configuration values supplied through supported sources."""

import sys

import astropy.units as u
import pytest
import yaml

from simtools.configuration.argument_helpers import parse_quantity_pair, quantity
from simtools.configuration.arguments import CONFIG, ArgumentDefinition
from simtools.configuration.configurator import Configurator

CASES = (
    ("scalar_str", ArgumentDefinition("value", default="default"), "file", "cli"),
    ("scalar_float", ArgumentDefinition("value", type=float, default=1.5), 2.5, 3.5),
    ("scalar_int", ArgumentDefinition("value", type=int, default=1), 2, 3),
    ("list", ArgumentDefinition("value", nargs="+", type=int, default=[1]), [2, 3], [4, 5]),
    (
        "fixed_nargs",
        ArgumentDefinition("value", nargs=2, type=int, default=[1, 2]),
        [3, 4],
        [5, 6],
    ),
    (
        "store_true",
        ArgumentDefinition("value", action="store_true", default=False),
        True,
        True,
    ),
    (
        "store_false",
        ArgumentDefinition("value", action="store_false", default=True),
        False,
        False,
    ),
    (
        "quantity",
        ArgumentDefinition("value", type=quantity(u.m), default=1 * u.m),
        "2 m",
        "3 m",
    ),
    (
        "quantity_pair",
        ArgumentDefinition("value", type=parse_quantity_pair, default="1 m 2 m"),
        "3 m 4 m",
        "5 m 6 m",
    ),
    (
        "append",
        ArgumentDefinition("value", action="append", default=[]),
        ["a", "b"],
        ["c", "d"],
    ),
    (
        "append_nargs",
        ArgumentDefinition("value", action="append", nargs=2, default=[]),
        [["a", "b"], ["c", "d"]],
        [["e", "f"], ["g", "h"]],
    ),
    (
        "choices",
        ArgumentDefinition("value", choices=("one", "two"), default="one"),
        "two",
        "one",
    ),
    (
        "alias",
        ArgumentDefinition("value", aliases=("old_value",), default="default"),
        "file",
        "cli",
    ),
)
SOURCE_CASES = []
for case in CASES:
    name, definition, _, _ = case
    for source in ("default", "env", "yaml", "cli"):
        parameters = (*case[1:], source)
        SOURCE_CASES.append(pytest.param(*parameters, id=f"{name}-{source}"))


def _make_configurator(definition):
    configurator = Configurator(label="test", use_dependency_defaults=False)
    configurator.parser.add_argument_definitions(
        (CONFIG, ArgumentDefinition("label", default=None), definition)
    )
    return configurator


def _environment_arguments(definition, value, tmp_test_directory, monkeypatch):
    if definition.kwargs.get("action") in {"store_true", "store_false"}:
        environment_value = "true" if value else "false"
    elif isinstance(value, list):
        flattened = (
            [item for group in value for item in group]
            if value and isinstance(value[0], list)
            else value
        )
        environment_value = " ".join(map(str, flattened))
    else:
        environment_value = str(value)
    monkeypatch.setenv(f"SIMTOOLS_{definition.name.upper()}", environment_value)
    config_file = tmp_test_directory / "empty-env.yml"
    config_file.write_text("{}", encoding="utf-8")
    return ["--config", str(config_file)]


def _yaml_arguments(definition, value, tmp_test_directory):
    config_file = tmp_test_directory / "configuration.yml"
    config_file.write_text(yaml.safe_dump({definition.name: value}), encoding="utf-8")
    return ["--config", str(config_file)]


def _cli_arguments(definition, value):
    option = definition.aliases[0] if definition.aliases else definition.name
    action = definition.kwargs.get("action")
    if action in {"store_true", "store_false"}:
        return [f"--{option}"] if value == (action == "store_true") else []
    if not isinstance(value, list):
        return [f"--{option}", str(value)]
    if action == "append" and definition.kwargs.get("nargs") == 2:
        return [token for item in value for token in (f"--{option}", *map(str, item))]
    if action == "append":
        return [token for item in value for token in (f"--{option}", str(item))]
    return [f"--{option}", *map(str, value)]


def _configure_from_source(definition, source, value, tmp_test_directory, monkeypatch):
    configurator = _make_configurator(definition)
    argv = ["test"]
    if source == "env":
        argv.extend(_environment_arguments(definition, value, tmp_test_directory, monkeypatch))
    elif source == "yaml":
        argv.extend(_yaml_arguments(definition, value, tmp_test_directory))
    elif source == "cli":
        argv.extend(_cli_arguments(definition, value))
    monkeypatch.setattr(sys, "argv", argv)
    return configurator.configure()[definition.name]


def _expected_value(definition, value):
    converter = definition.kwargs.get("type")
    if converter is None or value is None:
        return value
    if converter is parse_quantity_pair:
        return converter(value)
    if converter.__name__ == "quantity_type":
        return converter(value)
    if isinstance(value, list) and definition.kwargs.get("nargs") is not None:
        return [converter(item) for item in value]
    return converter(value)


def _environment_text(value):
    """Return the environment-variable text for a value (groups are flattened)."""
    if not isinstance(value, list):
        return value
    flattened = (
        [item for group in value for item in group]
        if value and isinstance(value[0], list)
        else value
    )
    return " ".join(map(str, flattened))


def _expected_from_environment(definition, value):
    """Return the configured value expected from an environment variable."""
    text = _environment_text(value)
    if definition.kwargs.get("action") == "append":
        items = text.split()
        count = definition.kwargs.get("nargs")
        if count:
            return [items[index : index + count] for index in range(0, len(items), count)]
        return items
    if definition.kwargs.get("nargs") is not None:
        converter = definition.kwargs.get("type")
        return [converter(item) if converter else item for item in text.split()]
    return _expected_value(definition, text)


@pytest.mark.parametrize(("definition", "yaml_value", "cli_value", "source"), SOURCE_CASES)
def test_configure_reads_argument_kinds_from_each_source(
    definition, yaml_value, cli_value, source, tmp_test_directory, monkeypatch
):
    if source == "default":
        config_file = tmp_test_directory / "empty.yml"
        config_file.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(sys, "argv", ["test", "--config", str(config_file)])
        result = _make_configurator(definition).configure()[definition.name]
        expected = _expected_value(definition, definition.kwargs["default"])
    elif source == "env":
        result = _configure_from_source(
            definition, source, yaml_value, tmp_test_directory, monkeypatch
        )
        expected = _expected_from_environment(definition, yaml_value)
    else:
        value = yaml_value if source == "yaml" else cli_value
        result = _configure_from_source(definition, source, value, tmp_test_directory, monkeypatch)
        expected = _expected_value(definition, value)
    assert result == expected


@pytest.mark.parametrize(
    ("environment", "yaml_value", "cli_value", "expected"),
    [
        ("environment", "file", "command", "command"),
        ("environment", "file", None, "file"),
        ("environment", None, None, "environment"),
    ],
    ids=("cli_over_yaml", "yaml_over_environment", "environment_over_default"),
)
def test_configure_source_precedence(
    environment, yaml_value, cli_value, expected, tmp_test_directory, monkeypatch
):
    configurator = _make_configurator(ArgumentDefinition("value", default="default"))
    monkeypatch.setenv("SIMTOOLS_VALUE", environment)
    config_file = tmp_test_directory / "precedence.yml"
    config_file.write_text(
        yaml.safe_dump({} if yaml_value is None else {"value": yaml_value}), encoding="utf-8"
    )
    argv = ["test", "--config", str(config_file)]
    if cli_value is not None:
        argv.extend(("--value", cli_value))
    monkeypatch.setattr(sys, "argv", argv)

    assert configurator.configure()["value"] == expected


def test_configure_required_argument_accepts_cli_value(monkeypatch):
    configurator = _make_configurator(ArgumentDefinition("value", required=True))
    monkeypatch.setattr(sys, "argv", ["test", "--value", "provided"])

    assert configurator.configure()["value"] == "provided"


@pytest.mark.parametrize("source", ["yaml", "env"], ids=("yaml_only", "environment_only"))
def test_configure_required_argument_accepts_non_cli_value(source, tmp_test_directory, monkeypatch):
    definition = ArgumentDefinition("value", required=True)
    configurator = _make_configurator(definition)
    argv = ["test"]
    if source == "yaml":
        argv.extend(_yaml_arguments(definition, "provided", tmp_test_directory))
    else:
        monkeypatch.setenv("SIMTOOLS_VALUE", "provided")
        argv.extend(_environment_arguments(definition, "provided", tmp_test_directory, monkeypatch))
    monkeypatch.setattr(sys, "argv", argv)

    assert configurator.configure()["value"] == "provided"


def test_configure_required_argument_missing_from_all_sources_uses_argparse_error(
    capsys, tmp_test_directory, monkeypatch
):
    configurator = _make_configurator(ArgumentDefinition("value", required=True))
    config_file = tmp_test_directory / "empty-required.yml"
    config_file.write_text("{}", encoding="utf-8")
    monkeypatch.delenv("SIMTOOLS_VALUE", raising=False)
    monkeypatch.setattr(sys, "argv", ["test", "--config", str(config_file)])

    with pytest.raises(SystemExit):
        configurator.configure()

    assert "the following arguments are required" in capsys.readouterr().err


def test_configure_converts_the_string_none_to_none(monkeypatch):
    configurator = _make_configurator(ArgumentDefinition("value", default="default"))
    monkeypatch.setattr(sys, "argv", ["test", "--value", "None"])

    assert configurator.configure()["value"] is None


def test_configure_converts_list_defaults_of_single_value_arguments(monkeypatch):
    configurator = _make_configurator(
        ArgumentDefinition("value", type=parse_quantity_pair, default=["0 deg 10 deg"])
    )
    monkeypatch.setattr(sys, "argv", ["test", "--label", "x"])

    assert configurator.configure()["value"] == parse_quantity_pair("0 deg 10 deg")


def test_configure_cli_value_replaces_exclusive_alternative_from_file(
    tmp_test_directory, monkeypatch
):
    configurator = Configurator(label="test", use_dependency_defaults=False)
    configurator.parser.add_argument_definitions(
        (
            CONFIG,
            ArgumentDefinition("label", default=None),
            ArgumentDefinition("first", default=None, exclusive_group="choice"),
            ArgumentDefinition("second", default=None, exclusive_group="choice"),
        )
    )
    config_file = tmp_test_directory / "exclusive.yml"
    config_file.write_text(yaml.safe_dump({"first": "from file"}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["test", "--config", str(config_file), "--second", "cli"])

    result = configurator.configure()

    assert result["second"] == "cli"
    assert result["first"] is None
    assert configurator.config_sources["first"] == "default"
