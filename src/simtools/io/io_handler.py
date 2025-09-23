"""Handle input and output paths."""

import logging
from pathlib import Path

__all__ = ["IOHandler", "IOHandlerSingleton"]


class IncompleteIOHandlerInitError(Exception):
    """Exception raised when IOHandler is not initialized."""


class IOHandlerSingleton(type):
    """Singleton base class."""

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Ensure a single instance of the IOHandlerSingleton class.

        Creates a new instance if it doesn't exist, otherwise returns the existing instance.
        """
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class IOHandler(metaclass=IOHandlerSingleton):
    """Handle input and output paths."""

    def __init__(self):
        """Initialize IOHandler."""
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init IOHandler")

        self.output_path = None
        self.data_path = None
        self.model_path = None

    def set_paths(self, output_path=None, data_path=None, model_path=None):
        """
        Set paths for input and output.

        Parameters
        ----------
        output_path: str or Path
            Path pointing to the output directory.
        data_path: str or Path
            Path pointing to the data files (e.g., CORSIKA or sim_telarray output).
        model_path: str or Path
            Path pointing to the model file directory.
        """
        self.output_path = output_path
        self.data_path = data_path
        self.model_path = model_path

    def get_output_directory(self, sub_dir=None):
        """
        Return path to output directory.

        Parameters
        ----------
        sub_dir: str or list of str, optional
            Name of the subdirectory (ray_tracing, model etc)

        Returns
        -------
        Path
        """
        if sub_dir is None:
            path = Path(self.output_path)
        elif isinstance(sub_dir, list | tuple):
            path = Path(self.output_path).joinpath(*sub_dir)
        else:
            path = Path(self.output_path).joinpath(sub_dir)
        return self._mkdir(path)

    def _mkdir(self, path):
        """Create a directory and return path."""
        try:
            path.mkdir(parents=True, exist_ok=True)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Error creating directory {path!s}") from exc

        return path.absolute()

    def get_output_file(self, file_name, sub_dir=None):
        """
        Get path of an output file.

        Parameters
        ----------
        files_name: str
            File name.
        sub_dir: sub_dir: str or list of str, optional
            Name of the subdirectory (ray_tracing, model etc)

        Returns
        -------
        Path
        """
        return self.get_output_directory(sub_dir).joinpath(file_name).absolute()

    def get_input_data_file(self, parent_dir=None, file_name=None, test=False):
        """
        Get path of a data file, using data_path.

        Parameters
        ----------
        parent_dir: str
            Parent directory of the file.
        files_name: str
            File name.
        test: bool
            If true, return test resources location

        Returns
        -------
        Path

        Raises
        ------
        IncompleteIOHandlerInitError
            if data_path is not set

        """
        if test:
            file_prefix = Path("tests/resources/")
        elif self.data_path is not None:
            file_prefix = Path(self.data_path).joinpath(parent_dir)
        else:
            raise IncompleteIOHandlerInitError
        return file_prefix.joinpath(file_name).absolute()

    def get_model_configuration_directory(self, model_version, sub_dir=None):
        """
        Get path of the simulation model configuration directory.

        This is the directory where the sim_telarray configuration files will be stored.

        Parameters
        ----------
        model_version: str
            Model version.
        sub_dir: str
            subdirectory

        Returns
        -------
        Path
        """
        return self._mkdir(
            self.get_output_directory(sub_dir=sub_dir).joinpath("model", model_version)
        )
