"""Unit tests for CORSIKA build-option handling."""

from pathlib import Path

import pytest

from simtools.corsika.build_options import (
    CorsikaBuildVariant,
    format_corsika_build_variants,
    get_installed_corsika_build_variants,
    read_corsika_build_variants,
    select_corsika_build_variant,
)


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
