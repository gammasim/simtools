"""Centralized (immutable) settings object with command line and environment variables."""

import os
from pathlib import Path
from types import MappingProxyType


class _Config:
    def __init__(self):
        self._args = {}
        self._db_config = {}
        self._sim_telarray_path = None
        self._sim_telarray_exe = None
        self._corsika_path = None
        self._corsika_exe = None

    def load(self, args, db_config=None):
        self._args = MappingProxyType(args)  # immutable view
        self._db_config = MappingProxyType(db_config) if db_config is not None else {}
        self._sim_telarray_path = os.getenv(
            "SIMTOOLS_SIMTEL_PATH",
            args.get("simtel_path", None),
        )
        self._sim_telarray_exe = os.getenv(
            "SIMTOOLS_SIMTEL_EXECUTABLE",
            args.get("simtel_executable", "sim_telarray"),
        )
        self._corsika_path = os.getenv(
            "SIMTOOLS_CORSIKA_PATH",
            args.get("corsika_path", None),
        )
        self._corsika_exe = os.getenv(
            "SIMTOOLS_CORSIKA_EXECUTABLE",
            args.get("corsika_executable", "corsika"),
        )

    @property
    def args(self):
        return self._args

    @property
    def db_config(self):
        return self._db_config

    @property
    def sim_telarray_path(self):
        return Path(self._sim_telarray_path) if self._sim_telarray_path is not None else None

    @property
    def sim_telarray_exe(self):
        return self._sim_telarray_exe

    @property
    def corsika_path(self):
        return Path(self._corsika_path) if self._corsika_path is not None else None

    @property
    def corsika_exe(self):
        return self._corsika_exe


config = _Config()
