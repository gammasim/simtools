import datetime
import logging
from pathlib import Path


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
        IOHandler init.

        """
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
            Parent path of the output files created by this class.
        data_path: str or Path
            Parent path of the data files.
        model_path: str or Path
            Parent path of the output files created by this class.

        """
        self.output_path = output_path
        self.data_path = data_path
        self.model_path = model_path

    def get_output_directory(self, label=None, dirType=None, test=False):
        """
        Get the output directory for the directory type dirType

        Parameters
        ----------
        label: str
            Instance label.
        dirType: str
            Name of the subdirectory (ray-tracing, model etc)
        test: bool
            If true, return test output location

        Returns
        -------
        Path
        """

        if test:
            outputDirectoryPrefix = Path(self.output_path).joinpath("test-output")
        else:
            outputDirectoryPrefix = Path(self.output_path).joinpath("simtools-output")

        today = datetime.date.today()
        labelDir = label if label is not None else "d-" + str(today)
        path = outputDirectoryPrefix.joinpath(labelDir)
        if dirType is not None:
            path = path.joinpath(dirType)
        try:
            path.mkdir(parents=True, exist_ok=True)
        except FileNotFoundError:
            self._logger.error("Error creating directory {}".format(str(path)))
            raise

        return path.absolute()

    def get_output_file(self, fileName, label=None, dirType=None, test=False):
        """
        Get path of an output file.

        Parameters
        ----------
        filesName: str
            File name.
        label: str
            Instance label.
        dirType: str
            Name of the subdirectory (ray-tracing, model etc)
        test: bool
            If true, return test output location

        Returns
        -------
        Path
        """
        return (
            self.get_output_directory(label=label, dirType=dirType, test=test)
            .joinpath(fileName)
            .absolute()
        )

    def get_input_data_file(self, parentDir=None, fileName=None, test=False):
        """
        Get path of a data file, using data_path

        Parameters
        ----------
        parentDir: str
            Parent directory of the file.
        filesName: str
            File name.
        test: bool
            If true, return test resources location

        Returns
        -------
        Path
        """

        if test:
            filePrefix = Path("tests/resources/")
        else:
            filePrefix = Path(self.data_path).joinpath(parentDir)
        return filePrefix.joinpath(fileName).absolute()
