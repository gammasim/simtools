"""Tests for container component architecture checks."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(("target", "machine"), [("amd64", 62), ("arm64", 183)])
def test_container_component_architecture(tmp_test_directory, simtools_root_path, target, machine):
    """Both components must contain executables for the target architecture."""
    root = Path(tmp_test_directory)
    corsika = root / "corsika7"
    corsika.mkdir()
    simtel = root / "simtel_executable"
    header = b"\x7fELF\x02\x01" + bytes(12) + machine.to_bytes(2, "little")
    simtel.write_bytes(header)
    (corsika / "corsika_qgs2_urqmd_flat").write_bytes(header)
    (corsika / "selected_simulator").write_bytes(header)
    (corsika / "build_opts.yml").write_text(
        "variant:\n  - executable: corsika_qgs2_urqmd_flat\n  - executable: selected_simulator\n",
        encoding="utf-8",
    )
    command = [
        sys.executable,
        str(simtools_root_path / "docker/check_component_architecture.py"),
        target,
        str(simtel),
        str(corsika),
    ]
    assert subprocess.run(command, check=False, capture_output=True).returncode == 0

    (corsika / "selected_simulator").write_bytes(
        header[:18] + (183 if machine == 62 else 62).to_bytes(2, "little")
    )
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    assert result.returncode != 0
    assert "selected_simulator" in result.stderr

    (corsika / "selected_simulator").write_bytes(header)
    simtel.write_bytes(b"not an ELF executable")
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    assert result.returncode != 0
    assert "simtel_executable" in result.stderr

    simtel.write_bytes(header)
    (corsika / "build_opts.yml").write_text("variant: []\n", encoding="utf-8")
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    assert result.returncode != 0
    assert "No CORSIKA executables declared" in result.stderr
