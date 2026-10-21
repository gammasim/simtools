"""Compatibility import for the telescope-throughput validation application."""

from simtools.applications.validate_telescope_throughput import APPLICATION, main

__all__ = ["APPLICATION", "main"]


if __name__ == "__main__":
    main()
