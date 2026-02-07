"""Centralized settings object with command line and environment variables."""

import os
import socket
from pathlib import Path
from types import MappingProxyType

from simtools.utils.general import find_executable_in_path


class _Config:
    """Centralized settings object with command line and environment variables."""

    def __init__(self):
        """Initialize empty config."""
        self._args = {}
        self._db_config = {}
        self._sim_telarray_path = None
        self._sim_telarray_exe = None
        self._corsika_path = None
        self._corsika_interaction_table_path = None
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
            args.get("sim_telarray_path")
            if args is not None and "sim_telarray_path" in args
            else os.getenv("SIMTOOLS_SIM_TELARRAY_PATH")
        )

        self._sim_telarray_exe = (
            args.get("sim_telarray_executable")
            if args is not None and "sim_telarray_executable" in args
            else os.getenv("SIMTOOLS_SIM_TELARRAY_EXECUTABLE", "sim_telarray")
        )

        self._corsika_path = (
            args.get("corsika_path")
            if args is not None and "corsika_path" in args
            else os.getenv("SIMTOOLS_CORSIKA_PATH")
        )

        self._corsika_interaction_table_path = (
            args.get("corsika_interaction_table_path")
            if args is not None and "corsika_interaction_table_path" in args
            else os.getenv("SIMTOOLS_CORSIKA_INTERACTION_TABLE_PATH")
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
        if self._sim_telarray_path and Path(self._sim_telarray_path).is_dir():
            return Path(self._sim_telarray_path)
        raise FileNotFoundError(f"sim_telarray path not found: {self._sim_telarray_path}")

    @property
    def sim_telarray_exe(self):
        """Path to the sim_telarray executable."""
        return find_executable_in_path(
            self._sim_telarray_exe,
            Path(self._sim_telarray_path) / "bin",
        )

    @property
    def sim_telarray_exe_debug_trace(self):
        """Path to the debug trace version of the sim_telarray executable."""
        return find_executable_in_path(
            self._sim_telarray_exe + "_debug_trace",
            Path(self._sim_telarray_path) / "bin",
        )

    @property
    def corsika_path(self):
        """Path to the CORSIKA installation directory."""
        if self._corsika_path and Path(self._corsika_path).is_dir():
            return Path(self._corsika_path)
        raise FileNotFoundError(f"CORSIKA path not found: {self._corsika_path}")

    @property
    def corsika_interaction_table_path(self):
        """Path to the CORSIKA interaction table directory."""
        if (
            self._corsika_interaction_table_path
            and Path(self._corsika_interaction_table_path).is_dir()
        ):
            return Path(self._corsika_interaction_table_path)
        raise FileNotFoundError(
            f"CORSIKA interaction table path not found: {self._corsika_interaction_table_path}"
        )

    @property
    def corsika_exe(self):
        """Path to the CORSIKA executable."""
        return find_executable_in_path(self._corsika_exe, self.corsika_path)

    @property
    def corsika_exe_curved(self):
        """Path to the curved version of the CORSIKA executable."""
        corsika_curved = (
            self._corsika_exe.name.replace("_flat", "_curved")
            if "_flat" in self._corsika_exe.name
            else self._corsika_exe.name + "-curved"  # legacy naming convention
        )
        return find_executable_in_path(corsika_curved, self.corsika_path)

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
