from pathlib import Path

from simtools.reporting.docs_production_summary import (
    collect_production_descriptions,
    write_production_summary_markdown,
)


def _write_info_file(base_path, model_version, description, suffix="yml"):
    production_path = base_path / "simulation-models" / "productions" / model_version
    production_path.mkdir(parents=True, exist_ok=True)
    (production_path / f"info.{suffix}").write_text(
        f'model_version: "{model_version}"\ndescription: "{description}"\n',
        encoding="utf-8",
    )


def test_collect_production_descriptions_reads_info_files(tmp_test_directory):
    data_path = Path(tmp_test_directory)
    _write_info_file(data_path, "6.0.0", "Prod6 dark")
    _write_info_file(data_path, "6.1.0", "Prod6 half-moon", suffix="yaml")

    descriptions = collect_production_descriptions(data_path)

    assert descriptions == [("6.0.0", "Prod6 dark"), ("6.1.0", "Prod6 half-moon")]


def test_collect_production_descriptions_semantic_version_ordering(tmp_test_directory):
    data_path = Path(tmp_test_directory)
    _write_info_file(data_path, "10.0.0", "Prod10")
    _write_info_file(data_path, "6.0.0", "Prod6")
    _write_info_file(data_path, "9.2.0", "Prod9")

    descriptions = collect_production_descriptions(data_path)

    versions = [v for v, _ in descriptions]
    assert versions == ["6.0.0", "9.2.0", "10.0.0"]


def test_write_production_summary_markdown_writes_table(tmp_test_directory):
    data_path = Path(tmp_test_directory)
    _write_info_file(data_path, "5.0.0", "Prod5 | dark")
    _write_info_file(data_path, "6.0.0", "Prod6 dark")

    output_file = data_path / "output" / "production_version_descriptions.md"
    write_production_summary_markdown(data_path, output_file)

    content = output_file.read_text(encoding="utf-8")

    assert "# Descriptions of productions" in content
    assert "| Production Model Version | Short Description                     |" in content
    assert "| 5.0.0 | Prod5 \\| dark |" in content
    assert "| 6.0.0 | Prod6 dark |" in content
