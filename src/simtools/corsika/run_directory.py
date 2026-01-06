"""
CORSIKA run directory setup.

Creates symbolic links to CORSIKA executables, data tables and configuration files.
Fine-tuned to CORSIKA-7 setup.
"""

import logging

_logger = logging.getLogger(__name__)


def link_run_directory(workdir, corsika_executable):
    """Link CORSIKA executable and data tables to working directory.

    Parameters
    ----------
    workdir : pathlib.Path
        Working directory where symlinks will be created
    corsika_path : pathlib.Path
        Path to CORSIKA installation directory
    """
    _logger.info(f"Linking CORSIKA run directory in {workdir}")
    _link_file_if_exists(corsika_executable, workdir)
    corsika_path = corsika_executable.parent.resolve()
    _link_epos_files(workdir, corsika_path)
    _link_interaction_tables(workdir, corsika_path)


def _link_file_if_exists(src, dst):
    """Create symlink if source exists and destination doesn't."""
    if src.exists() and not dst.exists():
        dst.symlink_to(src)


def _link_interaction_tables(workdir, corsika_path):
    """Link CORSIKA interaction model tables."""
    tables = [
        "GLAUBTAR.DAT",
        "NUCNUCCS",
        "NUCLEAR.BIN",
        "VENUSDAT",  # TODO needed? We probably never use VENUS
        "QGSDAT01",
        "SECTNU",
        "qgsdat-II-03",
        "sectnu-II-03",
        "qgsdat-II-04",
        "sectnu-II-04",
        "qgsdat-III",
        "sectnu-III",
        "UrQMD-1.3.1-xs.dat",
        "tables.dat",
    ]
    for table in tables:
        _link_file_if_exists(corsika_path / table, workdir / table)
    # Link EGS data files (e.g., EGS4/EGS5 related tables)
    for src in corsika_path.glob("EGS*"):
        _link_file_if_exists(src, workdir / src.name)


def _link_epos_files(workdir, corsika_path):
    """Link EPOS configuration files."""
    epos_files = [
        "epos.inics.lhc",
        "epos.iniev",
        "epos.ini1b",
        "epos.inirj.lhc",
        "epos.initl",
        "epos.param",
    ]
    for epos_file in epos_files:
        src = corsika_path / epos_file
        epos_alt = corsika_path.parent / "epos" / epos_file
        if src.exists():
            _link_file_if_exists(src, workdir / epos_file)
        elif epos_alt.exists():
            _link_file_if_exists(epos_alt, workdir / epos_file)
