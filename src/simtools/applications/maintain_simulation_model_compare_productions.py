r"""
Compare two directories with model production tables in JSON format.

This script should be used to support the maintenance the simulation model repository.

Example
-------
.. code-block:: console

    simtools-maintain-simulation-model-compare-productions \\
        --directory_1 ../simulation-models-dev/simulation-models/6.0.0/ \\
        --directory_2 ../simulation-models-dev/simulation-models/6.2.0

"""

import logging
from pathlib import Path

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.io import ascii_handler


def _parse(label, description):
    """Parse command line arguments."""
    config = configurator.Configurator(label=label, description=description)
    config.parser.add_argument(
        "--directory_1",
        type=str,
        required=True,
        help="Path to the first directory containing JSON files.",
    )
    config.parser.add_argument(
        "--directory_2",
        type=str,
        required=True,
        help="Path to the second directory containing JSON files.",
    )
    return config.initialize(db_config=False, output=False)


def _print_differences(differences, rel_path):
    """Print differences in a readable format."""
    print(f"Difference in {rel_path}:\n{'-' * 40}")
    for diff in differences:
        # Clean up the path formatting for better readability
        clean_diff = diff.replace("['parameters']['", "parameters.").replace("']['", ".")
        clean_diff = clean_diff.replace("['", "").replace("']", "")
        print(f"  {clean_diff}")
    print(f"{'-' * 40}\n")


def _compare_json_dirs(dir1, dir2, ignore_key="model_version"):
    """Compare two directories containing JSON files, ignoring a specific key."""
    for path1 in dir1.rglob("*.json"):
        rel_path = path1.relative_to(dir1)
        path2 = dir2 / rel_path

        if not path2.exists():
            print(f"Missing in dir2: {rel_path}")
            continue

        try:
            json1 = gen.remove_key_from_dict(
                ascii_handler.collect_data_from_file(path1), ignore_key
            )
            json2 = gen.remove_key_from_dict(
                ascii_handler.collect_data_from_file(path2), ignore_key
            )
        except FileNotFoundError as e:
            print(f"Error reading {rel_path}: {e}")
            continue

        differences = gen.find_differences_in_json_objects(json1, json2)
        if differences:
            _print_differences(differences, rel_path)

    # Check for files present in dir2 but not dir1
    for path2 in dir2.rglob("*.json"):
        rel_path = path2.relative_to(dir2)
        if not (dir1 / rel_path).exists():
            print(f"Missing in dir1: {rel_path}")


def main():  # noqa: D103
    label = Path(__file__).stem
    args_dict, _ = _parse(
        label=label,
        description=("Compare two directories with model production tables in JSON format."),
    )
    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    _compare_json_dirs(Path(args_dict["directory_1"]), Path(args_dict["directory_2"]))


if __name__ == "__main__":
    main()
