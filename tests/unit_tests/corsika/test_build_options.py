"""Unit tests for CORSIKA build-option handling."""

from dataclasses import replace
from pathlib import Path

import pytest

from simtools.corsika.build_options import (
    CorsikaBuildVariant,
    corsika_executable,
    format_corsika_build_variants,
    get_corsika_build_report,
    get_installed_corsika_build_variants,
    read_corsika_build_variants,
    select_corsika_build_variant,
    with_installed_build_variants,
)
from simtools.runtime.context import SimulationSoftware


def _write_build_options(directory, variants):
    """Write a minimal build-options file."""
    lines = ["variant:"]
    for variant in variants:
        lines.extend(
            [
                f"  - executable: {variant['executable']}",
                f"    config: {variant['config']}",
                f"    atmosphere_geometry: {variant['atmosphere_geometry']}",
                f"    he_hadronic_model: {variant['he_hadronic_model']}",
                f"    le_hadronic_model: {variant['le_hadronic_model']}",
            ]
        )
    Path(directory, "build_opts.yml").write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def build_variants():
    """Return representative EPOS and QGSJet build variants."""
    return [
        {
            "executable": "corsika_epos_urqmd_flat",
            "config": "config_epos_urqmd_flat",
            "atmosphere_geometry": "flat",
            "he_hadronic_model": "EPOS",
            "le_hadronic_model": "URQMD",
        },
        {
            "executable": "corsika_qgs3_urqmd_curved",
            "config": "config_qgs3_urqmd_curved",
            "atmosphere_geometry": "curved",
            "he_hadronic_model": "qgs3",
            "le_hadronic_model": "urqmd",
        },
    ]


def test_read_and_select_corsika_build_variants(tmp_test_directory, build_variants):
    _write_build_options(tmp_test_directory, build_variants)

    variants = read_corsika_build_variants(tmp_test_directory)
    selected = select_corsika_build_variant(variants, "EPOS", "URQMD", "FLAT")

    assert selected.executable == "corsika_epos_urqmd_flat"
    assert selected.he_hadronic_model == "epos"
    assert "qgs3" in format_corsika_build_variants(variants)


def test_get_installed_corsika_build_variants_checks_executables(
    tmp_test_directory, build_variants
):
    _write_build_options(tmp_test_directory, build_variants)
    for entry in build_variants:
        executable = Path(tmp_test_directory, entry["executable"])
        executable.touch()
        executable.chmod(0o755)

    assert len(get_installed_corsika_build_variants(tmp_test_directory)) == 2

    Path(tmp_test_directory, build_variants[0]["executable"]).unlink()
    with pytest.raises(ValueError, match="declares a missing executable"):
        get_installed_corsika_build_variants(tmp_test_directory)


def test_with_installed_build_variants_enriches_software(tmp_test_directory, build_variants):
    _write_build_options(tmp_test_directory, build_variants)
    for entry in build_variants:
        executable = Path(tmp_test_directory, entry["executable"])
        executable.touch()
        executable.chmod(0o755)
    software = SimulationSoftware(
        corsika_root=tmp_test_directory,
        he_interaction="epos",
        le_interaction="urqmd",
    )

    configured = with_installed_build_variants(software)

    assert len(configured.corsika_build_variants) == 2
    assert (
        configured.corsika_flat_executable
        == Path(tmp_test_directory) / build_variants[0]["executable"]
    )
    assert software.corsika_build_variants is None
    configured = replace(configured, he_interaction="qgs3")
    assert corsika_executable(configured, "curved") == (
        Path(tmp_test_directory) / build_variants[1]["executable"]
    )


def test_with_installed_build_variants_skips_missing_directory():
    software = SimulationSoftware(corsika_root=None)

    assert with_installed_build_variants(software) is software


def test_corsika_executable_requires_loaded_variants():
    software = SimulationSoftware()

    with pytest.raises(FileNotFoundError, match="CORSIKA build manifest has not been loaded"):
        corsika_executable(software, "flat")


def test_get_corsika_build_report_uses_environment_path(
    monkeypatch, tmp_test_directory, build_variants
):
    _write_build_options(tmp_test_directory, build_variants)
    for entry in build_variants:
        executable = Path(tmp_test_directory, entry["executable"])
        executable.touch()
        executable.chmod(0o755)
    monkeypatch.setenv("SIMTOOLS_CORSIKA_PATH", str(tmp_test_directory))

    report = get_corsika_build_report()

    assert "epos" in report
    assert "qgs3" in report


def test_select_corsika_build_variant_rejects_unsupported(build_variants):
    variants = tuple(CorsikaBuildVariant.from_mapping(entry) for entry in build_variants)

    with pytest.raises(ValueError, match=r"Unsupported.*sibyll.*Available variants"):
        select_corsika_build_variant(variants, "sibyll", "urqmd", "flat")


@pytest.mark.parametrize(
    "entry",
    [
        [],
        {"executable": "corsika"},
        {
            "executable": "corsika",
            "config": "config",
            "atmosphere_geometry": "spherical",
            "he_hadronic_model": "epos",
            "le_hadronic_model": "urqmd",
        },
    ],
)
def test_corsika_build_variant_rejects_invalid_entry(entry):
    with pytest.raises(ValueError, match="Invalid CORSIKA"):
        CorsikaBuildVariant.from_mapping(entry)


def test_read_corsika_build_variants_rejects_duplicates(tmp_test_directory, build_variants):
    _write_build_options(tmp_test_directory, [build_variants[0], build_variants[0]])

    with pytest.raises(ValueError, match="Duplicate CORSIKA build variants"):
        read_corsika_build_variants(tmp_test_directory)


def test_read_corsika_build_variants_rejects_missing_variant_list(tmp_test_directory):
    Path(tmp_test_directory, "build_opts.yml").write_text("corsika_version: 78010\n")

    with pytest.raises(ValueError, match="no variant list"):
        read_corsika_build_variants(tmp_test_directory)


def test_read_corsika_build_variants_accepts_optional_hadronic_transition_energy(
    tmp_test_directory,
):
    Path(tmp_test_directory, "build_opts.yml").write_text(
        "variant:\n"
        "  - executable: corsika_epos_urqmd_flat\n"
        "    config: config_epos_urqmd_flat\n"
        "    atmosphere_geometry: flat\n"
        "    he_hadronic_model: epos\n"
        "    le_hadronic_model: urqmd\n"
        "    hadronic_transition_energy_default_gev: 30\n",
        encoding="utf-8",
    )

    variant = read_corsika_build_variants(tmp_test_directory)[0]

    assert variant.hadronic_transition_energy_default_gev == 30


def test_corsika_build_variant_rejects_invalid_hadronic_transition_energy():
    with pytest.raises(ValueError, match="hadronic transition energy"):
        CorsikaBuildVariant.from_mapping(
            {
                "executable": "corsika_epos_urqmd_flat",
                "config": "config_epos_urqmd_flat",
                "atmosphere_geometry": "flat",
                "he_hadronic_model": "epos",
                "le_hadronic_model": "urqmd",
                "hadronic_transition_energy_default_gev": "invalid",
            }
        )
