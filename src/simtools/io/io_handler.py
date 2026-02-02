"""Handle input and output directories and file paths."""

import logging
from pathlib import Path


class IOHandlerSingleton(type):
    """Singleton base class."""

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """Create a new instance if it doesn't exist, otherwise returns the existing instance."""
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class IOHandler(metaclass=IOHandlerSingleton):
    """Handle input and output directories and file paths."""

    def __init__(self):
        """Initialize IOHandler."""
        self.logger = logging.getLogger(__name__)
        self.output_path = None
        self.model_path = None

    def set_paths(self, output_path=None, model_path=None):
        """
        Set paths for input and output.

        Parameters
        ----------
        output_path: str or Path
            Path pointing to the output directory.
        model_path: str or Path
            Path pointing to the model file directory.
        """
        self.output_path = output_path
        self.model_path = model_path

    def get_output_directory(self, sub_dir=None):
        """
        Create and get path of an output directory.

        Parameters
        ----------
        sub_dir: str or list of str, optional
            Name of the subdirectory (ray_tracing, model etc)

        Returns
        -------
        Path

        Raises
        ------
        FileNotFoundError
            if the directory cannot be created
        """
        if sub_dir is None:
            parts = []
        elif isinstance(sub_dir, list | tuple):
            parts = sub_dir
        else:
            parts = [sub_dir]
        path = Path(self.output_path, *parts)

        try:
            path.mkdir(parents=True, exist_ok=True)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Error creating directory {path!s}") from exc

        return path.resolve()

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

    def get_test_data_file(self, file_name=None):
        """
        Get path of a data file in the test resources directory.

        Parameters
        ----------
        files_name: str
            File name.

        Returns
        -------
        Path
        """
        return Path("tests/resources", file_name).resolve()

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
        return self.get_output_directory(
            sub_dir=[sub_dir, "model", model_version] if sub_dir else ["model", model_version]
        )
