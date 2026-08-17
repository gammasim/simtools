"""Output-artifact location and existence checks."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OutputArtifact:
    """A configured integration-test output and its resolved path."""

    path: Path
    descriptor: dict

    @classmethod
    def from_descriptor(cls, configuration, descriptor):
        """Create an artifact from a workflow output descriptor."""
        try:
            base_path = configuration[descriptor["path_descriptor"]]
        except KeyError as exc:
            raise KeyError(
                f"Path {descriptor.get('path_descriptor')} not found in "
                "integration test configuration."
            ) from exc
        return cls(
            path=Path(base_path) / descriptor.get("output_sub_path", "") / descriptor["file"],
            descriptor=descriptor,
        )

    def assert_exists(self):
        """Raise an informative assertion error when the artifact is missing."""
        if not self.path.exists():
            directory_contents = (
                list(self.path.parent.iterdir())
                if self.path.parent.is_dir()
                else "directory missing"
            )
            raise AssertionError(
                f"Output file {self.path} does not exist. Directory contents: {directory_contents}"
            )
