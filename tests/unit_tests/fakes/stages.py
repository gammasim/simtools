"""Simple stages used to check the simulation-stage contract."""

from pathlib import Path

from simtools.runners.stages import CAMERA_EVENTS, CHERENKOV_PHOTONS
from tests.unit_tests.builders.tables import reduced_event_tables


class FakeShowerStage:
    """Write a minimal shower product and return its execution command."""

    name = "fake_shower"
    consumes = None
    produces = CHERENKOV_PHOTONS
    streams = False

    def __init__(self, directory):
        self.directory = Path(directory)
        self.file_list = {}

    @classmethod
    def instances(cls, inputs, label=None):
        return [cls(inputs.args["output_directory"])]

    def prepare(self, run_number, input_file=None, output=None):
        if input_file is not None or output is not None:
            raise ValueError("Fake shower takes no input and writes a file.")
        path = self.output_file(run_number)
        path.parent.mkdir(parents=True, exist_ok=True)
        reduced_event_tables()["SHOWERS"].write(path, format="fits", overwrite=True)
        self.file_list = {"fake_shower": path}
        return [f"fake-shower {run_number}"]

    def output_file(self, run_number):
        return self.directory / f"shower-{run_number}.dat"

    def environment(self):
        return {}

    def get_resources(self, runtime=None):
        return {"runtime": runtime}


class FakeTelescopeStage:
    """Write a minimal camera product after consuming a shower product."""

    name = "fake_telescope"
    consumes = CHERENKOV_PHOTONS
    produces = CAMERA_EVENTS
    streams = False

    def __init__(self, directory):
        self.directory = Path(directory)
        self.file_list = {}

    @classmethod
    def instances(cls, inputs, label=None):
        return [cls(inputs.args["output_directory"])]

    def prepare(self, run_number, input_file=None, output=None):
        if input_file is None or output is not None:
            raise ValueError("Fake telescope requires an input file and writes a file.")
        path = self.output_file(run_number)
        path.parent.mkdir(parents=True, exist_ok=True)
        reduced_event_tables()["SHOWERS"].write(path, format="fits", overwrite=True)
        self.file_list = {"fake_telescope": path}
        return [f"fake-telescope {run_number} < {input_file}"]

    def output_file(self, run_number):
        return self.directory / f"camera-{run_number}.dat"

    def environment(self):
        return {}

    def get_resources(self, runtime=None):
        return {"runtime": runtime}
