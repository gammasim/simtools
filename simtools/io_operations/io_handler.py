import datetime
import logging
import re
from pathlib import Path

__all__ = ["IOHandlerSingleton", "IOHandler"]


class IncompleteIOHandlerInit(Exception):
    """Exception raised when IOHandler is not initialized"""


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
        self.use_plain_output_path = False
        self.data_path = None
        self.model_path = None

    def set_paths(
        self, output_path=None, data_path=None, model_path=None, use_plain_output_path=False
    ):
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
        use_plain_output_path: bool
            Use plain output path without adding tool name and date

        """
        self.output_path = output_path
        self.use_plain_output_path = use_plain_output_path
        self.data_path = data_path
        self.model_path = model_path

    def get_output_directory(self, label=None, sub_dir=None, dir_type="simtools"):
        """
        Return path to output directory

        Parameters
        ----------
        label: str
            Instance label.
        sub_dir: str
            Name of the subdirectory (ray-tracing, model etc)
        dir_type: str
            The type of directory (e.g., 'simtools', 'test', 'simtools-result').
            If 'simtools-result' is used, the output directory will be returned
            without appending a subdirectory string when using the
            use_plain_output_path option. For the cause of not using use_plain_output_path,
            output paths appended by 'simtools-output'.

        Returns
        -------
        Path

        Raises
        ------
        FileNotFoundError
            if error creating directory
        TypeError
            raised for errors while creating directory name
        """

        if self.use_plain_output_path:
            path = Path(self.output_path)
        else:
            if str(self.output_path).endswith("-output"):
                output_directory_prefix = Path(self.output_path)
            else:
                try:
                    output_directory_prefix = Path(self.output_path).joinpath(
                        re.sub(r"\-result$", "", dir_type) + "-output"
                    )
                except TypeError:
                    self._logger.error(f"Error creating output directory name from {dir_type}")
                    raise
            label_dir = label if label is not None else "d-" + str(datetime.date.today())
            path = output_directory_prefix.joinpath(label_dir)
        if sub_dir is not None:
            if not self.use_plain_output_path:
                path = path.joinpath(sub_dir)
            elif dir_type != "simtools-result":
                path = path.joinpath(sub_dir)

        try:
            path.mkdir(parents=True, exist_ok=True)
        except FileNotFoundError:
            self._logger.error(f"Error creating directory {str(path)}")
            raise

        return path.absolute()

    def get_output_file(self, file_name, label=None, sub_dir=None, dir_type="simtools"):
        """
        Get path of an output file.

        Parameters
        ----------
        files_name: str
            File name.
        label: str
            Instance label.
        sub_dir: str
            Name of the subdirectory (ray-tracing, model etc)
        dir_type: str
            Directory type (e.g., 'simtools', 'test', 'simtools-result')
            If 'simtools-result' is used, the output directory will be returned
            without appending a subdirectory string when using the
            use_plain_output_path option.

        Returns
        -------
        Path
        """
        return (
            self.get_output_directory(label=label, sub_dir=sub_dir, dir_type=dir_type)
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

        Raises
        ------
        IncompleteIOHandlerInit
            if data_path is not set

        """

        if test:
            file_prefix = Path("tests/resources/")
        elif self.data_path is not None:
            file_prefix = Path(self.data_path).joinpath(parent_dir)
        else:
            raise IncompleteIOHandlerInit
        return file_prefix.joinpath(file_name).absolute()
