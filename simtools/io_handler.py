import datetime
import logging
from pathlib import Path

__all__ = ["IOHandlerSingleton", "IOHandler"]


class IOHandlerSingleton(type):
    """
    Singleton base class
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(IOHandlerSingleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class IOHandler(metaclass=IOHandlerSingleton):
    """
    Handle input and output paths.
    """

    def __init__(self):
        """
        Initialize IOHandler.

        """
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init IOHandler")

        self.output_path = None
        self.output_path_simtools_naming = True
        self.data_path = None
        self.model_path = None

    def set_paths(self,
                  output_path=None,
                  data_path=None,
                  model_path=None,
                  output_path_simtools_naming=True):
        """
        Set paths for input and output.

        Parameters
        ----------
        output_path: str or Path
            Parent path of the output files created by this class.
        data_path: str or Path
            Parent path of the data files.
        model_path: str or Path
            Parent path of the output files created by this class.

        """
        self.output_path = output_path
        self.output_path_simtools_naming = output_path_simtools_naming
        self.data_path = data_path
        self.model_path = model_path

    def get_output_directory(self, label=None, dir_type=None, test=False):
        """
        Get the output directory for the directory type dir_type

        Parameters
        ----------
        label: str
            Instance label.
        dir_type: str
            Name of the subdirectory (ray-tracing, model etc)
        test: bool
            If true, return test output location

        Returns
        -------
        Path

        Raises
        ------
        FileNotFoundError
            if error creating directory
        """

        if self.output_path_simtools_naming:
            if test:
                output_directory_prefix = Path(self.output_path).joinpath("test-output")
            else:
                output_directory_prefix = Path(self.output_path).joinpath("simtools-output")

            today = datetime.date.today()
            label_dir = label if label is not None else "d-" + str(today)
            path = output_directory_prefix.joinpath(label_dir)
            if dir_type is not None:
                path = path.joinpath(dir_type)
        else:
            path = Path(self.output_path)

        try:
            path.mkdir(parents=True, exist_ok=True)
        except FileNotFoundError:
            self._logger.error(f"Error creating directory {str(path)}")
            raise

        return path.absolute()

    def get_output_file(self, file_name, label=None, dir_type=None, test=False):
        """
        Get path of an output file.

        Parameters
        ----------
        files_name: str
            File name.
        label: str
            Instance label.
        dir_type: str
            Name of the subdirectory (ray-tracing, model etc)
        test: bool
            If true, return test output location

        Returns
        -------
        Path
        """
        return (
            self.get_output_directory(label=label, dir_type=dir_type, test=test)
            .joinpath(file_name)
            .absolute()
        )

    def get_input_data_file(self, parent_dir=None, file_name=None, test=False):
        """
        Get path of a data file, using data_path

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
        """

        if test:
            file_prefix = Path("tests/resources/")
        else:
            file_prefix = Path(self.data_path).joinpath(parent_dir)
        return file_prefix.joinpath(file_name).absolute()
