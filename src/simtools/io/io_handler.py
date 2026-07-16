"""Handle input and output directories and file paths."""

import logging
import re
from pathlib import Path

import simtools.utils.general as gen

_TEST_RESOURCE_PATTERN = re.compile(r"\$\{(static|generated|downloaded):([^}]+)\}")
_TEST_RESOURCE_PATH_PATTERN = re.compile(r"(?<![\w/])(?:\./)?tests/resources(?=/|$)")


def resolve_test_resource_paths(value, test_resources_path=None):
    """Resolve test-resource macros and canonical test-resource paths recursively.

    Parameters
    ----------
    value : object
        Configuration value, mapping, or sequence to resolve.
    test_resources_path : str or pathlib.Path, optional
        Base directory containing the ``static``, ``generated``, and ``downloaded``
        resource directories. Defaults to ``tests/resources``.

    Returns
    -------
    object
        Configuration with absolute test-resource paths.
    """
    base_path = Path(test_resources_path or "tests/resources").expanduser().resolve()
    if isinstance(value, dict):
        return {
            key: resolve_test_resource_paths(item, test_resources_path=base_path)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [resolve_test_resource_paths(item, test_resources_path=base_path) for item in value]
    if isinstance(value, str):
        value = _TEST_RESOURCE_PATTERN.sub(
            lambda match: str(base_path / match.group(1) / match.group(2)), value
        )
        return _TEST_RESOURCE_PATH_PATTERN.sub(str(base_path), value)
    return value


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
        self.output_path = {}
        self.model_path = None
        self.test_resources_path = Path("tests/resources").resolve()

    def set_paths(self, output_path=None, model_path=None, output_path_label="default"):
        """
        Set paths for input and output.

        Parameters
        ----------
        output_path: str or Path
            Path pointing to the output directory.
        model_path: str or Path
            Path pointing to the model file directory.
        output_path_label: str
            Label for the output path.
        """
        self.output_path[output_path_label] = output_path
        self.model_path = model_path

    def get_output_directory(self, sub_dir=None, output_path_label="default"):
        """
        Create and get path of an output directory.

        Parameters
        ----------
        sub_dir: str or list of str, optional
            Name of the subdirectory (ray_tracing, model etc)
        output_path_label: str
            Label for the output path.

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
        try:
            output_path = Path(self.output_path[output_path_label], *parts)
        except KeyError as exc:
            raise KeyError(f"Output path label '{output_path_label}' not found") from exc

        try:
            output_path.mkdir(parents=True, exist_ok=True)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Error creating directory {output_path!s}") from exc

        return output_path.resolve()

    def get_output_file(self, file_name, sub_dir=None, output_path_label="default"):
        """
        Get path of an output file.

        Parameters
        ----------
        files_name: str
            File name.
        sub_dir: sub_dir: str or list of str, optional
            Name of the subdirectory (ray_tracing, model etc)
        output_path_label: str
            Label for the output path.

        Returns
        -------
        Path
        """
        return (
            self.get_output_directory(sub_dir, output_path_label=output_path_label)
            .joinpath(file_name)
            .absolute()
        )

    def get_test_data_file(self, file_name=None, sub_dir=("static", "generated")):
        """
        Get path of a data file in the test resources directory.

        Parameters
        ----------
        file_name: str
            File name.
        sub_dir: str or list or tuple of str
            Fallback resource subdirectory name(s) to search under tests/resources.

        Returns
        -------
        Path
        """
        base_path = self.test_resources_path
        file_path = (base_path / file_name).resolve()

        if file_path.exists():
            return file_path

        fallback_dirs = gen.ensure_list(sub_dir)
        for folder in fallback_dirs:
            candidate = (base_path / folder / file_name).resolve()
            if candidate.exists():
                return candidate

        return file_path

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
            sub_dir=["model", sub_dir, model_version] if sub_dir else ["model", model_version]
        )
