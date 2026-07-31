"""Tests for shared show-options handling."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

import simtools.configuration.show_options as show_options


def test_resolve_show_options_primary():
    result = show_options.resolve_show_options({"show_options": "primary"})

    assert result.option_name == "primary"
    assert "gamma" in result.values


def test_resolve_show_options_rejects_unknown_option():
    with pytest.raises(ValueError, match="Unsupported option"):
        show_options.resolve_show_options({"show_options": "unknown"})


def test_resolve_show_options_requires_model_version_for_array_layout_name():
    with pytest.raises(ValueError, match="requires '--model_version'"):
        show_options.resolve_show_options({"show_options": "array_layout_name"})


def test_resolve_show_options_rejects_multiple_model_versions_for_array_layout_name():
    with pytest.raises(ValueError, match="exactly one '--model_version'"):
        show_options.resolve_show_options(
            {"show_options": "array_layout_name", "model_version": ["7.0.0", "7.1.0"]}
        )


def test_resolve_show_options_uses_argparse_choices():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="Simulation mode.", choices=["fast", "safe"])

    result = show_options.resolve_show_options({"show_options": "mode"}, parser)

    assert result.values == ("fast", "safe")
    assert result.help_text == "Simulation mode."


def test_resolve_show_options_accepts_unrestricted_argparse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", help="Directory for output files.")

    result = show_options.resolve_show_options({"show_options": "output_path"}, parser)

    assert result.values == ()
    assert result.notes == ("Argparse does not define a finite set of available values.",)
    assert "Available values:" not in show_options.format_show_options_result(result)


def test_handle_show_options_formats_help_and_values(capsys):
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", help="Observatory site.")

    with pytest.raises(SystemExit) as exc:
        show_options.handle_show_options({"show_options": "site"}, parser)

    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "Help:" in output
    assert "Observatory site." in output
    assert "Available values:" in output


def test_handle_show_options_returns_false_without_request():
    assert show_options.handle_show_options({}, argparse.ArgumentParser()) is False


def test_handle_show_options_reports_resolution_error(capsys):
    parser = argparse.ArgumentParser()
    parser.add_argument("--show_options")

    with pytest.raises(SystemExit) as exc:
        show_options.handle_show_options({"show_options": "unknown"}, parser)

    assert exc.value.code == 2
    assert "Unknown command-line option" in capsys.readouterr().err


def test_handle_show_options_returns_true_when_exit_is_overridden(monkeypatch, capsys):
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", help="Observatory site.")
    monkeypatch.setattr(parser, "exit", lambda status=0, message=None: None)

    assert show_options.handle_show_options({"show_options": "site"}, parser) is True
    assert "North" in capsys.readouterr().out


def test_with_argparse_help_preserves_existing_help():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", help="Parser help.")
    result = show_options.ShowOptionsResult("site", help_text="Provider help.")

    assert show_options._with_argparse_help(result, parser) is result


def test_with_argparse_help_preserves_result_without_matching_action():
    parser = argparse.ArgumentParser()
    result = show_options.ShowOptionsResult("site", values=("North",))

    assert show_options._with_argparse_help(result, parser) is result


def test_resolve_show_options_handles_hidden_argparse_help():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", help=argparse.SUPPRESS)

    result = show_options.resolve_show_options({"show_options": "hidden"}, parser)

    assert result.help_text is None


def test_resolve_show_options_rejects_unknown_parser_option():
    parser = argparse.ArgumentParser()

    with pytest.raises(ValueError, match="Unknown command-line option"):
        show_options.resolve_show_options({"show_options": "missing"}, parser)


def test_format_show_options_result_includes_environment_and_grouped_values():
    result = show_options.ShowOptionsResult(
        option_name="array_layout_name",
        environment={"path": "/models"},
        grouped_values={"North": ("layout-a", "layout-b")},
        notes=("Use a matching model version.",),
    )

    output = show_options.format_show_options_result(result)

    assert "Environment:" in output
    assert "  path: /models" in output
    assert "  North:" in output
    assert "    layout-a" in output
    assert "Use a matching model version." in output


def test_show_model_versions_uses_database(monkeypatch):
    class FakeDatabaseHandler:
        def get_model_versions(self):
            return ["7.0.0", "7.1.0"]

    monkeypatch.setattr(show_options, "DatabaseHandler", FakeDatabaseHandler)

    result = show_options.resolve_show_options({"show_options": "model_version"})

    assert result.values == ("7.0.0", "7.1.0")


def test_show_array_layout_names_groups_all_sites(monkeypatch):
    class FakeSiteModel:
        def __init__(self, site, model_version):
            self.site = site
            self.model_version = model_version

        def get_list_of_array_layouts(self):
            return [f"{self.site}-{self.model_version}"]

    monkeypatch.setattr(show_options, "SiteModel", FakeSiteModel)
    monkeypatch.setattr(show_options.names, "site_names", lambda: ("South", "North"))

    result = show_options.resolve_show_options(
        {"show_options": "array_layout_name", "model_version": ["7.0.0"]}
    )

    assert result.grouped_values == {
        "North": ("North-7.0.0",),
        "South": ("South-7.0.0",),
    }
    assert result.notes == ("Provide --site to limit array layouts to one site.",)


def test_show_array_layout_names_limits_to_selected_site(monkeypatch):
    class FakeSiteModel:
        def __init__(self, site, model_version):
            self.site = site
            self.model_version = model_version

        def get_list_of_array_layouts(self):
            return [f"{self.site}-{self.model_version}"]

    monkeypatch.setattr(show_options, "SiteModel", FakeSiteModel)

    result = show_options.resolve_show_options(
        {"show_options": "array_layout_name", "model_version": "7.0.0", "site": "North"}
    )

    assert result.grouped_values == {"North": ("North-7.0.0",)}
    assert result.notes == ()


def test_show_corsika_interactions_include_environment(monkeypatch):
    variants = [
        SimpleNamespace(
            he_hadronic_model="epos",
            le_hadronic_model="urqmd",
            executable="corsika-epos",
        ),
        SimpleNamespace(
            he_hadronic_model="qgsjet",
            le_hadronic_model="gheisha",
            executable="corsika-qgsjet",
        ),
    ]
    monkeypatch.setattr(show_options, "get_installed_corsika_build_variants", lambda path: variants)
    monkeypatch.setattr(show_options, "get_corsika_version", lambda: "7.7400")

    high_result = show_options.resolve_show_options(
        {"show_options": "corsika_he_interaction", "corsika_path": "/corsika"}
    )
    low_result = show_options.resolve_show_options(
        {"show_options": "corsika_le_interaction", "corsika_path": "/corsika"}
    )

    assert high_result.values == ("epos", "qgsjet")
    assert low_result.values == ("gheisha", "urqmd")
    assert high_result.environment["CORSIKA version"] == "7.7400"
    assert "executables" in high_result.environment


def test_get_corsika_environment_handles_one_executable_without_version(monkeypatch):
    variant = SimpleNamespace(executable="corsika", he_hadronic_model="epos")
    monkeypatch.setattr(show_options, "get_corsika_version", lambda: None)

    environment = show_options._get_corsika_environment(Path("/corsika"), [variant])

    assert environment["executable"] == str(Path("/corsika/corsika").resolve())
    assert "CORSIKA version" not in environment


def test_resolve_corsika_path_uses_environment_and_default(monkeypatch):
    monkeypatch.setenv("SIMTOOLS_CORSIKA_PATH", "/from-environment")
    assert show_options._resolve_corsika_path({}) == Path("/from-environment")

    monkeypatch.delenv("SIMTOOLS_CORSIKA_PATH")
    monkeypatch.setattr(show_options.defaults, "CORSIKA_PATH", "/from-default")
    assert show_options._resolve_corsika_path({}) == Path("/from-default")
