"""Centralized settings object with command line and environment variables."""

import os
import socket
from pathlib import Path
from types import MappingProxyType


class _Config:
    """Centralized settings object with command line and environment variables."""

    def __init__(self):
        """Initialize empty config."""
        self._args = {}
        self._db_config = {}
        self._sim_telarray_path = None
        self._sim_telarray_exe = None
        self._corsika_path = None
        self._corsika_exe = None
        self.user = os.getenv("USER", "unknown")
        self.hostname = socket.gethostname()

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
        self._sim_telarray_path = (
            args.get("simtel_path")
            if args is not None and "simtel_path" in args
            else os.getenv("SIMTOOLS_SIMTEL_PATH")
        )

        self._sim_telarray_exe = (
            args.get("simtel_executable")
            if args is not None and "simtel_executable" in args
            else os.getenv("SIMTOOLS_SIMTEL_EXECUTABLE", "sim_telarray")
        )

        self._corsika_path = (
            args.get("corsika_path")
            if args is not None and "corsika_path" in args
            else os.getenv("SIMTOOLS_CORSIKA_PATH")
        )

        self._corsika_exe = self._get_corsika_exec() if self._corsika_path is not None else None

    def _get_corsika_exec(self):
        """
        Get the CORSIKA executable from environment variable or command line argument.

        Build the executable name based on configured interaction models. Fall back to
        legacy naming (simply "corsika") if models are not specified.
        """
        he_model = (
            self._args.get("corsika_he_interaction")
            if self._args is not None and "corsika_he_interaction" in self._args
            else os.getenv("SIMTOOLS_CORSIKA_HE_INTERACTION")
        )

        le_model = (
            self._args.get("corsika_le_interaction")
            if self._args is not None and "corsika_le_interaction" in self._args
            else os.getenv("SIMTOOLS_CORSIKA_LE_INTERACTION")
        )

        if he_model and le_model:
            corsika_exe = self.corsika_path / f"corsika_{he_model}_{le_model}_flat"
            if corsika_exe.exists():
                return corsika_exe

        # legacy naming
        return self.corsika_path / "corsika"

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
        if self._corsika_exe is None:
            return None
        corsika_curved = (
            self._corsika_exe.name.replace("_flat", "_curved")
            if "_flat" in self._corsika_exe.name
            else self._corsika_exe.name + "-curved"  # legacy naming convention
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
