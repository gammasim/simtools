"""Process-wide run context of the running application."""

from simtools.context import RunContext
from simtools.io import io_handler


class _Config:
    """Hold the run context of the running application.

    Code that receives a ``RunContext`` reads it directly; this object serves code that has
    not been given one yet.
    """

    def __init__(self):
        """Initialize with an empty context (nothing read from arguments or environment)."""
        self._context = RunContext(paths=io_handler.IOHandler())

    def load(self, args=None, resolve_sim_software_executables=True):
        """
        Load the run context from command line arguments and environment variables.

        Priority order: command line argument > environment variable > built-in default.

        Parameters
        ----------
        args : dict, optional
            Command line arguments.
        resolve_sim_software_executables : bool, optional
            Resolve simulation software executable paths during loading.
            If False, skip resolving CORSIKA executable.

        """
        self._context = RunContext.from_arguments(
            args,
            paths=io_handler.IOHandler(),
            resolve_sim_software_executables=resolve_sim_software_executables,
        )

    @property
    def context(self):
        """Run context of the running application."""
        return self._context

    @property
    def args(self):
        """Command line arguments."""
        return self._context.args

    @property
    def model_reader(self):
        """Reader selected for the current application context."""
        return self._context.model_reader

    def set_model_reader(self, reader):
        """Set the reader selected for the current application context."""
        self._context = self._context.replace(model_reader=reader)

    @property
    def activity_id(self):
        """Activity ID."""
        return self._context.activity.activity_id

    @property
    def activity_name(self):
        """Activity name (application label)."""
        return self._context.activity.activity_name

    @property
    def user(self):
        """User name."""
        return self._context.activity.user

    @property
    def hostname(self):
        """Host name."""
        return self._context.activity.hostname

    @property
    def ray_tracing_backend(self):
        """Configured optical ray-tracing backend."""
        return self._context.software.ray_tracing_backend

    @property
    def sim_telarray_path(self):
        """Path to the sim_telarray installation directory."""
        return self._context.software.sim_telarray_path

    @property
    def sim_telarray_exe(self):
        """Path to the sim_telarray executable."""
        return self._context.software.sim_telarray_exe

    @property
    def sim_telarray_exe_debug_trace(self):
        """Path to the debug trace version of the sim_telarray executable."""
        return self._context.software.sim_telarray_exe_debug_trace

    @property
    def corsika_path(self):
        """Path to the CORSIKA installation directory."""
        return self._context.software.corsika_path

    @property
    def corsika_interaction_table_path(self):
        """Path to the CORSIKA interaction table directory."""
        return self._context.software.corsika_interaction_table_path

    @property
    def corsika_exe(self):
        """Path to the CORSIKA executable."""
        return self._context.software.corsika_exe

    @property
    def corsika_exe_curved(self):
        """Path to the curved version of the CORSIKA executable."""
        return self._context.software.corsika_exe_curved

    @property
    def corsika_interaction_models(self):
        """Configured high- and low-energy CORSIKA interaction models."""
        return self._context.software.corsika_interaction_models

    @property
    def corsika_dummy_file(self):
        """Path to a dummy CORSIKA file required by sim_telarray for ray-tracing simulations."""
        return self._context.software.corsika_dummy_file


config = _Config()
