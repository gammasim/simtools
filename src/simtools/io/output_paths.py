"""Explicit output and model configuration paths."""

from pathlib import Path

from simtools import constants


class OutputPaths:
    """Hold output and model paths for one run."""

    def __init__(self, output_path=None, model_path=None):
        self.output_path = {"default": output_path} if output_path is not None else {}
        self.model_path = model_path
        self.test_resources_path = constants.get_test_resources_root().resolve()

    def get_output_directory(self, sub_dir=None, output_path_label="default"):
        """Create and return an output directory."""
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
        """Return a path to an output file, creating its parent directory."""
        output_file = (
            self.get_output_directory(sub_dir, output_path_label=output_path_label)
            .joinpath(file_name)
            .absolute()
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)
        return output_file

    def get_model_configuration_directory(self, model_version, sub_dir=None):
        """Return the output directory for model configuration files."""
        parts = ["model", sub_dir, model_version] if sub_dir else ["model", model_version]
        return self.get_output_directory(sub_dir=parts)
