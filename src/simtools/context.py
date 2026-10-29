"""Run context: the configuration and services of one simtools run."""

from __future__ import annotations

import logging
import os
import socket
from dataclasses import dataclass, field
from dataclasses import replace as dataclass_replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

from simtools.application import model_reader as application_model_reader
from simtools.configuration import defaults
from simtools.constants import RAY_TRACING_BACKENDS
from simtools.corsika.build_options import (
    get_installed_corsika_build_variants,
    select_corsika_build_variant,
)
from simtools.io.output_paths import OutputPaths
from simtools.utils.general import find_executable_in_dir, get_uuid

if TYPE_CHECKING:
    from simtools.model_repository.reader import SimulationModelReader


def _config_value(args, arg_key, env_key, default=None):
    """Return a value from the arguments, else the environment variable, else the default."""
    value = args.get(arg_key) if args is not None else None
    if value is not None:
        return value
    return os.getenv(env_key, default)


@dataclass(frozen=True)
class Activity:
    """Identity of one simtools activity."""

    activity_id: str = field(default_factory=get_uuid)
    activity_name: str | None = None
    user: str = field(default_factory=lambda: os.getenv("USER", "unknown"))
    hostname: str = field(default_factory=socket.gethostname)

    @classmethod
    def from_arguments(cls, args):
        """Return the activity of parsed arguments (a new ID if none is given)."""
        args = args or {}
        activity_id = args.get("activity_id")
        return cls(
            activity_id=activity_id if activity_id is not None else get_uuid(),
            activity_name=args.get("application_label"),
        )


@dataclass(frozen=True)
class SimulationSoftware:
    """Configured CORSIKA and sim_telarray installations and the ray-tracing backend.

    Fields hold the configured values; the properties validate them and resolve executables.
    """

    sim_telarray_root: str | Path | None = None
    sim_telarray_executable_name: str | None = None
    corsika_root: str | Path | None = None
    corsika_interaction_table_root: str | Path | None = None
    corsika_build_variants: object = None
    corsika_flat_executable: str | Path | None = None
    he_interaction: str = defaults.CORSIKA_HE_INTERACTION
    le_interaction: str = defaults.CORSIKA_LE_INTERACTION
    ray_tracing_backend: str = RAY_TRACING_BACKENDS[0]

    @classmethod
    def from_arguments(cls, args=None, resolve_sim_software_executables=True):
        """Return the software configuration from arguments, environment and defaults.

        Priority order: argument > environment variable > built-in default.

        Parameters
        ----------
        args : Mapping, optional
            Parsed arguments.
        resolve_sim_software_executables : bool, optional
            Resolve the CORSIKA executable from the build manifest.
        """
        ray_tracing_backend = _config_value(
            args,
            "ray_tracing_backend",
            "SIMTOOLS_RAY_TRACING_BACKEND",
            default=RAY_TRACING_BACKENDS[0],
        )
        if ray_tracing_backend not in RAY_TRACING_BACKENDS:
            raise ValueError(
                f"ray_tracing_backend must be one of {RAY_TRACING_BACKENDS}, "
                f"got '{ray_tracing_backend}'"
            )
        software = cls(
            sim_telarray_root=_config_value(
                args, "sim_telarray_path", "SIMTOOLS_SIM_TELARRAY_PATH"
            ),
            sim_telarray_executable_name=_config_value(
                args,
                "sim_telarray_executable",
                "SIMTOOLS_SIM_TELARRAY_EXECUTABLE",
                default="sim_telarray",
            ),
            corsika_root=_config_value(
                args, "corsika_path", "SIMTOOLS_CORSIKA_PATH", default=defaults.CORSIKA_PATH
            ),
            corsika_interaction_table_root=_config_value(
                args,
                "corsika_interaction_table_path",
                "SIMTOOLS_CORSIKA_INTERACTION_TABLE_PATH",
                default=defaults.CORSIKA_INTERACTION_TABLE_PATH,
            ),
            he_interaction=str(
                _config_value(
                    args,
                    "corsika_he_interaction",
                    "SIMTOOLS_CORSIKA_HE_INTERACTION",
                    default=defaults.CORSIKA_HE_INTERACTION,
                )
            ).lower(),
            le_interaction=str(
                _config_value(
                    args,
                    "corsika_le_interaction",
                    "SIMTOOLS_CORSIKA_LE_INTERACTION",
                    default=defaults.CORSIKA_LE_INTERACTION,
                )
            ).lower(),
            ray_tracing_backend=ray_tracing_backend,
        )
        if (
            resolve_sim_software_executables
            and software.corsika_root is not None
            and Path(software.corsika_root).is_dir()
        ):
            software = dataclass_replace(
                software,
                corsika_build_variants=get_installed_corsika_build_variants(software.corsika_root),
            )
            software = dataclass_replace(
                software, corsika_flat_executable=software.corsika_executable("flat")
            )
        return software

    def corsika_executable(self, geometry):
        """Return the CORSIKA executable declared by the build manifest for a geometry."""
        if self.corsika_build_variants is None:
            raise FileNotFoundError("CORSIKA build manifest has not been loaded")
        variant = select_corsika_build_variant(
            self.corsika_build_variants, self.he_interaction, self.le_interaction, geometry
        )
        return self.corsika_path / variant.executable

    @property
    def sim_telarray_path(self):
        """Path to the sim_telarray installation directory."""
        if self.sim_telarray_root and Path(self.sim_telarray_root).is_dir():
            return Path(self.sim_telarray_root)
        raise FileNotFoundError(f"sim_telarray path not found: {self.sim_telarray_root}")

    @property
    def sim_telarray_exe(self):
        """Path to the sim_telarray executable."""
        return find_executable_in_dir(
            self.sim_telarray_executable_name, self.sim_telarray_path / "bin"
        )

    @property
    def sim_telarray_exe_debug_trace(self):
        """Path to the debug trace version of the sim_telarray executable."""
        return find_executable_in_dir(
            self.sim_telarray_executable_name + "_debug_trace", self.sim_telarray_path / "bin"
        )

    @property
    def corsika_path(self):
        """Path to the CORSIKA installation directory."""
        if self.corsika_root and Path(self.corsika_root).is_dir():
            return Path(self.corsika_root)
        raise FileNotFoundError(f"CORSIKA path not found: {self.corsika_root}")

    @property
    def corsika_interaction_table_path(self):
        """Path to the CORSIKA interaction table directory."""
        if (
            self.corsika_interaction_table_root
            and Path(self.corsika_interaction_table_root).is_dir()
        ):
            return Path(self.corsika_interaction_table_root)
        raise FileNotFoundError(
            f"CORSIKA interaction table path not found: {self.corsika_interaction_table_root}"
        )

    @property
    def corsika_exe(self):
        """Path to the CORSIKA executable (flat atmosphere)."""
        if self.corsika_flat_executable is None:
            return None
        return find_executable_in_dir(self.corsika_flat_executable, self.corsika_path)

    @property
    def corsika_exe_curved(self):
        """Path to the curved version of the CORSIKA executable."""
        return find_executable_in_dir(self.corsika_executable("curved"), self.corsika_path)

    @property
    def corsika_interaction_models(self):
        """Configured high- and low-energy CORSIKA interaction models."""
        return self.he_interaction, self.le_interaction

    @property
    def corsika_dummy_file(self):
        """Path to a dummy CORSIKA file required by sim_telarray for ray-tracing simulations.

        This file does not need to exist; sim_telarray only requires a file path.
        """
        if self.sim_telarray_root is None:
            return None
        return self.sim_telarray_path / "run9991.corsika.gz"


@dataclass(frozen=True)
class RunContext:
    """Configuration and services of one simtools run.

    Parameters
    ----------
    model_reader : SimulationModelReader or None
        Reader of the simulation models.
    paths : OutputPaths
        Output and model-configuration directories.
    software : SimulationSoftware
        Simulation software installations and ray-tracing backend.
    activity : Activity
        Activity identity.
    args : Mapping
        Read-only view of the parsed application arguments.
    """

    model_reader: SimulationModelReader | None = None
    paths: OutputPaths = field(default_factory=OutputPaths)
    software: SimulationSoftware = field(default_factory=SimulationSoftware)
    activity: Activity = field(default_factory=Activity)
    args: MappingProxyType = field(default_factory=lambda: MappingProxyType({}))
    logger: logging.Logger | None = None
    run_time: list | None = None

    @classmethod
    def from_arguments(
        cls, args=None, model_reader=None, paths=None, resolve_sim_software_executables=True
    ):
        """Return the context of parsed arguments.

        Parameters
        ----------
        args : dict, optional
            Parsed arguments; wrapped read-only, not copied.
        model_reader : SimulationModelReader, optional
            Reader of the simulation models.
        paths : OutputPaths, optional
            Output paths; by default built from ``output_path`` in the arguments.
        resolve_sim_software_executables : bool, optional
            Resolve the CORSIKA executable from the build manifest.
        """
        args = args if args is not None else {}
        return cls(
            model_reader=model_reader,
            paths=paths if paths is not None else OutputPaths(args.get("output_path")),
            software=SimulationSoftware.from_arguments(args, resolve_sim_software_executables),
            activity=Activity.from_arguments(args),
            args=args if isinstance(args, MappingProxyType) else MappingProxyType(args),
        )

    @classmethod
    def from_environment(cls):
        """Return a fresh context built from environment."""
        model_source_configured = any(
            os.getenv(name)
            for name in (
                "SIMTOOLS_SIMULATION_MODELS_PATH",
                "SIMTOOLS_SIMULATION_MODELS_GIT_PATH",
            )
        )
        model_reader = None
        if model_source_configured:
            model_reader = application_model_reader.create_model_reader()
        return cls.from_arguments(
            {}, model_reader=model_reader, resolve_sim_software_executables=False
        )

    @property
    def ray_tracing_backend(self):
        """Configured optical ray-tracing backend."""
        return self.software.ray_tracing_backend

    def replace(self, **changes):
        """Return a copy with the given fields replaced."""
        return dataclass_replace(self, **changes)

    def to_spec(self):
        """Return the JSON-serializable configuration needed to rebuild this context."""
        source_config = getattr(self.model_reader, "source_config", None)
        return {
            "args": dict(self.args),
            "model_source": source_config,
            "model_path": str(self.paths.model_path) if self.paths.model_path else None,
        }

    @classmethod
    def from_spec(cls, spec):
        """Rebuild a worker context from :meth:`to_spec` output."""
        if not isinstance(spec, dict) or not isinstance(spec.get("args"), dict):
            raise ValueError("Run context specification must contain an args dictionary.")
        source = spec.get("model_source")
        reader = (
            application_model_reader.create_model_reader_from_source_config(source)
            if source
            else None
        )
        paths = OutputPaths(spec["args"].get("output_path"), spec.get("model_path"))
        return cls.from_arguments(
            spec["args"],
            model_reader=reader,
            paths=paths,
            resolve_sim_software_executables=False,
        )

    def __deepcopy__(self, _memo):
        """Return this context: it is immutable and shared by deep-copied models."""
        return self
