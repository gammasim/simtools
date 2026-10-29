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
            streamed_input = index > 0 and self._streams(index - 1, index)
            stage_inputs = self._stage_inputs(index, run_number, input_file, streamed_input)
            stage_lines = [
                line
                for instance, stage_input in zip(self.stages[index], stage_inputs)
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

    def _stage_inputs(self, index, run_number, input_file, streamed_input):
        """Return the input of each instance of a stage."""
        count = len(self.stages[index])
        if index == 0:
            return [input_file] * count
        if streamed_input:
            return [STDIN] * count
        return self._upstream_files(index, run_number)

    def _upstream_files(self, index, run_number):
        """
        Return the input file of each instance of a stage fed by file hand-over.

        A single upstream instance feeds every downstream instance. Otherwise instances are
        paired in order, which requires the same number of instances in both stages.
        """
        upstream = self.stages[index - 1]
        count = len(self.stages[index])
        if len(upstream) == 1:
            return [upstream[0].output_file(run_number)] * count
        if len(upstream) == count:
            return [instance.output_file(run_number) for instance in upstream]
        upstream_name = self.chain.stage_classes[index - 1].name
        downstream_name = self.chain.stage_classes[index].name
        raise ValueError(
            f"Stage {downstream_name!r} has {count} instances, but upstream stage "
            f"{upstream_name!r} has {len(upstream)}. Files are handed over only from one "
            "upstream instance or between stages with the same number of instances."
        )

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
