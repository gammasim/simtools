"""Centralized (immutable) settings object with command line and environment variables."""

import os
from pathlib import Path
from types import MappingProxyType


class _Config:
    """Centralized (immutable) settings object with command line and environment variables."""

    def __init__(self):
        """Initialize empty config."""
        self._args = {}
        self._db_config = {}
        self._sim_telarray_path = None
        self._sim_telarray_exe = None
        self._corsika_path = None
        self._corsika_exe = None

    def load(self, args=None, db_config=None):
        """
        Load configuration from command line arguments and environment variables.

        For paths, first check for environment variables, then command line arguments.

        Parameters
        ----------
        args : dict, optional
            Command line arguments.
        db_config : dict, optional
            Database configuration.

        """
        self._args = MappingProxyType(args) if args is not None else {}
        self._db_config = MappingProxyType(db_config) if db_config is not None else {}
        self._sim_telarray_path = os.getenv(
            "SIMTOOLS_SIMTEL_PATH",
            args.get("simtel_path", None) if args is not None else None,
        )
        self._sim_telarray_exe = os.getenv(
            "SIMTOOLS_SIMTEL_EXECUTABLE",
            args.get("simtel_executable", "sim_telarray") if args is not None else "sim_telarray",
        )
        self._corsika_path = os.getenv(
            "SIMTOOLS_CORSIKA_PATH",
            args.get("corsika_path", None) if args is not None else None,
        )
        self._corsika_exe = os.getenv(
            "SIMTOOLS_CORSIKA_EXECUTABLE",
            args.get("corsika_executable", "corsika") if args is not None else "corsika",
        )

    @property
    def args(self):
        """Command line arguments."""
        return self._args

    @property
    def db_config(self):
        """Database configuration."""
        return self._db_config

    @property
    def sim_telarray_path(self):
        """Path to the sim_telarray installation directory."""
        return Path(self._sim_telarray_path) if self._sim_telarray_path is not None else None

    @property
    def sim_telarray_exe(self):
        """Path to the sim_telarray executable."""
        return (
            Path(self._sim_telarray_path) / "bin" / self._sim_telarray_exe
            if self._sim_telarray_path is not None
            else None
        )

    @property
    def sim_telarray_exe_debug_trace(self):
        """Path to the debug trace version of the sim_telarray executable."""
        return (
            Path(self._sim_telarray_path) / "bin" / (self._sim_telarray_exe + "_debug_trace")
            if self._sim_telarray_path is not None
            else None
        )

    @property
    def corsika_path(self):
        """Path to the CORSIKA installation directory."""
        return Path(self._corsika_path) if self._corsika_path is not None else None

    @property
    def corsika_exe(self):
        """Path to the CORSIKA executable."""
        return (
            Path(self._corsika_path) / self._corsika_exe if self._corsika_path is not None else None
        )

    @property
    def corsika_exe_curved(self):
        """Path to the curved version of the CORSIKA executable."""
        corsika_curved = (
            self._corsika_exe.replace("_flat", "_curved")
            if "_flat" in self._corsika_exe
            else self._corsika_exe + "-curved"  # legacy naming convention
        )
        return Path(self._corsika_path) / corsika_curved if self._corsika_path is not None else None

    @property
    def corsika_dummy_file(self):
        """
        Path to a dummy CORSIKA file required by sim_telarray for ray-tracing simulations.

        This file does not need to exist; sim_telarray only requires a file path.
        """
        return (
            self.sim_telarray_path / "run9991.corsika.gz"
            if self._sim_telarray_path is not None
            else None
        )


config = _Config()
