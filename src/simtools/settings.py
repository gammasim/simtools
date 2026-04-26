"""Centralized settings object with command line and environment variables."""

import os
import socket
from pathlib import Path
from types import MappingProxyType

from simtools.configuration import defaults
from simtools.utils.general import find_executable_in_dir, get_uuid


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
        self.activity_id = get_uuid()
        self.activity_name = None

    def load(self, args=None, db_config=None, resolve_sim_software_executables=True):
        """
        Load configuration from command line arguments and environment variables.

        For paths, first check for environment variables, then command line arguments.

        Parameters
        ----------
        args : dict, optional
            Command line arguments.
        db_config : dict, optional
            Database configuration.
        resolve_sim_software_executables : bool, optional
            Resolve simulation software executable paths during loading.
            If False, skip resolving CORSIKA executable.

        """
        self._args = MappingProxyType(args) if args is not None else {}
        self._db_config = MappingProxyType(db_config) if db_config is not None else {}
        self.activity_id = self._get_activity_id(args)
        self.activity_name = args.get("application_label") if args is not None else None
        self._sim_telarray_path = self._get_config_value(
            args, "sim_telarray_path", "SIMTOOLS_SIM_TELARRAY_PATH"
        )
        self._sim_telarray_exe = self._get_config_value(
            args,
            "sim_telarray_executable",
            "SIMTOOLS_SIM_TELARRAY_EXECUTABLE",
            default="sim_telarray",
        )
        self._corsika_path = self._get_config_value(
            args,
            "corsika_path",
            "SIMTOOLS_CORSIKA_PATH",
            default=defaults.CORSIKA_PATH,
        )
        self._corsika_interaction_table_path = self._get_config_value(
            args,
            "corsika_interaction_table_path",
            "SIMTOOLS_CORSIKA_INTERACTION_TABLE_PATH",
            default=defaults.CORSIKA_INTERACTION_TABLE_PATH,
        )
        if (
            resolve_sim_software_executables
            and self._corsika_path is not None
            and Path(self._corsika_path).is_dir()
        ):
            self._corsika_exe = self._get_corsika_exec()
        else:
            self._corsika_exe = None

    @staticmethod
    def _get_config_value(args, arg_key, env_key, default=None):
        """Get configuration value from arguments or environment variable."""
        if args is not None and arg_key in args:
            return args.get(arg_key)
        return os.getenv(env_key, default)

    @staticmethod
    def _get_activity_id(args):
        """Get activity ID from arguments or generate a new one."""
        activity_id = args.get("activity_id") if args is not None else None
        return activity_id if activity_id is not None else get_uuid()

    def _get_corsika_exec(self):
        """
        Get the CORSIKA executable from environment variable or command line argument.

        Build the executable name based on configured interaction models. Fall back to
        legacy naming (simply "corsika") if models are not specified.
        """
        he_model = self._get_config_value(
            self._args,
            "corsika_he_interaction",
            "SIMTOOLS_CORSIKA_HE_INTERACTION",
            default=defaults.CORSIKA_HE_INTERACTION,
        )

        le_model = self._get_config_value(
            self._args,
            "corsika_le_interaction",
            "SIMTOOLS_CORSIKA_LE_INTERACTION",
            default=defaults.CORSIKA_LE_INTERACTION,
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
        return find_executable_in_dir(
            self._sim_telarray_exe,
            self.sim_telarray_path / "bin",
        )

    @property
    def sim_telarray_exe_debug_trace(self):
        """Path to the debug trace version of the sim_telarray executable."""
        return find_executable_in_dir(
            self._sim_telarray_exe + "_debug_trace",
            self.sim_telarray_path / "bin",
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
        return find_executable_in_dir(self._corsika_exe, self.corsika_path)

    @property
    def corsika_exe_curved(self):
        """Path to the curved version of the CORSIKA executable."""
        corsika_curved = (
            self._corsika_exe.name.replace("_flat", "_curved")
            if "_flat" in self._corsika_exe.name
            else self._corsika_exe.name + "-curved"  # legacy naming convention
        )
        return find_executable_in_dir(corsika_curved, self.corsika_path)

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
