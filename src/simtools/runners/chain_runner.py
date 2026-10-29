"""Run script for a chain of simulation stages."""

import logging
from pathlib import Path

from simtools.runners.stages import STDIN, StreamTo


class ChainRunner:
    """
    Write the run script of a simulation chain.

    Stages that both stream are connected by a stream (the upstream stage receives the
    downstream commands); otherwise the upstream output file is the downstream input.
    A stage may run as several instances (e.g. one per array model).

    Parameters
    ----------
    chain : SimulationChain
        Ordered simulation stages.
    inputs : StageInputs
        Array models, simulation parameters, shower configurations and arguments.
    label : str
        Instance label.
    """

    def __init__(self, chain, inputs, label=None):
        """Initialize ChainRunner."""
        self._logger = logging.getLogger(__name__)
        self.chain = chain
        self.stages = [
            stage_class.instances(inputs, label=label) for stage_class in chain.stage_classes
        ]
        self.file_list = None

    def prepare_run(self, run_number, sub_script, input_file=None, extra_commands=None):
        """
        Write the run script for one run.

        Parameters
        ----------
        run_number : int
            Run number.
        sub_script : str or Path
            Run script to write.
        input_file : str or Path, optional
            Input file of the first stage (when it does not generate showers).
        extra_commands : str or list[str], optional
            Commands written before the simulation commands.
        """
        lines = self._stage_lines(run_number, input_file)
        with Path(sub_script).open("w", encoding="utf-8") as file:
            file.write("#!/usr/bin/env bash\n")
            file.write("set -e\n")
            file.write("set -o pipefail\n")
            if extra_commands:
                file.write("\n# Writing extras\n")
                for line in [extra_commands] if isinstance(extra_commands, str) else extra_commands:
                    file.write(f"{line}\n")
                file.write("# End of extras\n\n")
            for line in lines:
                file.write(f"{line}\n")
        self.file_list = self._merged_file_list()

    def _stage_lines(self, run_number, input_file):
        """Return the shell lines of all stages, connecting them by streams or files."""
        lines = []
        stream = None
        for index in reversed(range(len(self.stages))):
            instances = self.stages[index]
            streamed_input = index > 0 and self._streams(index - 1, index)
            stage_input = STDIN if streamed_input else self._input_file(index, run_number)
            if index == 0:
                stage_input = input_file
            stage_lines = [
                line
                for instance in instances
                for line in instance.prepare(run_number, input_file=stage_input, output=stream)
            ]
            if streamed_input:
                stream = StreamTo(tuple(stage_lines))
            else:
                stream = None
                lines = stage_lines + lines
        return lines

    def _streams(self, upstream, downstream):
        """Return whether two neighbouring stages exchange their product as a stream."""
        return all(
            instance.streams for index in (upstream, downstream) for instance in self.stages[index]
        )

    def _input_file(self, index, run_number):
        """Return the output file of the previous stage (file hand-over)."""
        if index == 0:
            return None
        previous = self.stages[index - 1][0]
        return previous.output_file(run_number)

    def _merged_file_list(self):
        """Merge the file lists of all stage instances (lists for repeated file types)."""
        merged = {}
        for instance in (instance for instances in self.stages for instance in instances):
            for key, data in (instance.file_list or {}).items():
                if key not in merged:
                    merged[key] = data
                    continue
                if not isinstance(merged[key], list):
                    merged[key] = [merged[key]]
                merged[key].extend(data if isinstance(data, list) else [data])
        return merged

    def environment(self):
        """Return the environment variables of all stages."""
        environment = {}
        for instances in self.stages:
            environment.update(instances[0].environment())
        return environment

    def get_resources(self, runtime=None):
        """Return computing resources used (reported by the first stage)."""
        return self.stages[0][0].get_resources(runtime=runtime)
