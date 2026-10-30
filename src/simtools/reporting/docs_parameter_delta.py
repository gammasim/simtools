#!/usr/bin/python3

r"""Delta reports: parameters changed between two model versions."""

from pathlib import Path
from typing import NamedTuple

import numpy as np

from simtools.data_model import parameter_catalog as catalog


class ParameterDelta(NamedTuple):
    """A changed parameter for delta report rendering."""

    parameter: str
    base_param_version: str
    new_param_version: str
    base_value: str
    new_value: str


# Mixin kept as a class: its helpers are patched by name in the unit tests.
class ParameterDeltaMixin:  # pylint: disable=too-few-public-methods
    """Delta-report methods of ReadParameters (relies on its attributes and helpers)."""

    def _values_equal(self, value1, value2):
        """Compare nested values with tolerance for numerics."""
        if value1 is None or value2 is None:
            return value1 is None and value2 is None
        if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
            return bool(np.isclose(value1, value2, rtol=1e-12, atol=1e-12, equal_nan=True))
        if isinstance(value1, (list, tuple)) and isinstance(value2, (list, tuple)):
            return len(value1) == len(value2) and all(
                self._values_equal(v1, v2) for v1, v2 in zip(value1, value2)
            )
        if isinstance(value1, dict) and isinstance(value2, dict):
            return value1.keys() == value2.keys() and all(
                self._values_equal(value1[k], value2[k]) for k in value1
            )
        return value1 == value2

    def _parameter_changed(self, base, current):
        """Return True if a parameter's metadata or value differs between two versions."""
        if base is None or current is None:
            return True
        if base.get("parameter_version") != current.get("parameter_version"):
            return True
        if (base.get("unit") or "") != (current.get("unit") or ""):
            return True
        if bool(base.get("file", False)) != bool(current.get("file", False)):
            return True
        return not self._values_equal(base.get("value"), current.get("value"))

    def _format_value_for_delta(self, parameter, param_data):
        """Format a parameter value for delta reports without generating plots/files."""
        if not param_data:
            return "-"
        value_data = param_data.get("value")
        file_flag = param_data.get("file", False)
        if self._is_list_of_dicts(value_data):
            return f"{len(value_data)} rows"
        # For delta reports, never convert referenced files to local markdown.
        return self._format_parameter_value(
            parameter,
            value_data,
            param_data.get("unit") or "",
            file_flag,
            parameter_version=None if file_flag else "delta",
        )

    def _write_delta_report_table(self, file, changes):
        """Write markdown table for delta report."""
        file.write(
            "| Parameter | Base parameter version | New parameter version | "
            "Base value | New value |\n"
            "|---|---:|---:|---|---|\n"
        )
        for row in changes:
            file.write(
                f"| {row.parameter} | {row.base_param_version} | "
                f"{row.new_param_version} | {row.base_value} | "
                f"{row.new_value} |\n"
            )

    def _build_delta_changes(self, base_parameter_data, current_parameter_data, parameters=None):
        """Build list of changed parameters for delta reports."""
        parameter_names = set(base_parameter_data) | set(current_parameter_data)
        if parameters is not None:
            parameter_names &= set(parameters)

        changes = []
        for parameter in sorted(parameter_names):
            base = base_parameter_data.get(parameter)
            current = current_parameter_data.get(parameter)
            if self._parameter_changed(base, current):
                changes.append(
                    ParameterDelta(
                        parameter,
                        (base or {}).get("parameter_version", "-"),
                        (current or {}).get("parameter_version", "-"),
                        self._format_value_for_delta(parameter, base),
                        self._format_value_for_delta(parameter, current),
                    )
                )
        return changes

    def _produce_delta_report(
        self, base_model_version, element, collection, title, parameters=None
    ):
        """Write a delta report of one array element or observatory against a base version."""
        base = self.model_reader.get_model_parameters(
            site=self.site,
            array_element_name=element,
            collection=collection,
            model_version=base_model_version,
        )
        if not base:
            return False
        current = self.model_reader.get_model_parameters(
            site=self.site,
            array_element_name=element,
            collection=collection,
            model_version=self.model_version,
        )
        if not current:
            return False

        changes = self._build_delta_changes(base, current, parameters=parameters)
        output_filename = Path(self.output_path / f"{element}.md")
        output_filename.parent.mkdir(parents=True, exist_ok=True)
        with output_filename.open("w", encoding="utf-8") as file:
            file.write(
                f"# {title}\n\n"
                f"This report lists changes in model version **{self.model_version}** "
                f"relative to **{base_model_version}**.\n\n"
                f"Full base report: [{base_model_version}](../{base_model_version}/{element}.md)"
                "\n\n"
            )
            if not changes:
                file.write("No parameter changes detected.\n")
            else:
                self._write_delta_report_table(file, changes)
        return True

    def _produce_array_element_delta_report(self, base_model_version):
        """Produce a compact delta report for patch model versions."""
        return self._produce_delta_report(
            base_model_version,
            self.array_element,
            "telescopes",
            self.array_element,
            parameters=catalog.model_parameters(None).keys(),
        )

    def _produce_observatory_delta_report(self, base_model_version):
        """Produce a compact delta report for observatory parameters."""
        return self._produce_delta_report(
            base_model_version,
            f"OBS-{self.site}",
            "sites",
            f"Observatory Parameters - {self.site} Site (delta report)",
        )
