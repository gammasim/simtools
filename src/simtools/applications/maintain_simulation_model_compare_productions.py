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


def _load_json_without_key(path, ignore_key):
    """Load JSON data from a file, ignoring a specific key."""
    data = gen.collect_data_from_file(path)

    def remove_key(obj):
        if isinstance(obj, dict):
            return {k: remove_key(v) for k, v in obj.items() if k != ignore_key}
        if isinstance(obj, list):
            return [remove_key(i) for i in obj]
        return obj

    return remove_key(data)


def __find_differences_dict(obj1, obj2, path, diffs):
    """Recursively find differences between two dictionaries."""
    for key in sorted(set(obj1) | set(obj2)):
        subpath = f"{path}['{key}']" if path else f"['{key}']"
        if key not in obj1:
            diffs.append(f"{subpath}: added in directory 2")
        elif key not in obj2:
            diffs.append(f"{subpath}: removed in directory 2")
        else:
            diffs.extend(_find_differences(obj1[key], obj2[key], subpath))


def _find_differences(obj1, obj2, path=""):
    """Recursively find differences between two JSON-like objects."""
    diffs = []

    if not isinstance(obj1, type(obj2)):
        diffs.append(f"{path}: type changed from {type(obj1).__name__} to {type(obj2).__name__}")
        return diffs

    if isinstance(obj1, dict):
        __find_differences_dict(obj1, obj2, path, diffs)

    elif isinstance(obj1, list):
        if len(obj1) != len(obj2):
            diffs.append(f"{path}: list length changed from {len(obj1)} to {len(obj2)}")
        for i, (a, b) in enumerate(zip(obj1, obj2)):
            subpath = f"{path}[{i}]" if path else f"[{i}]"
            diffs.extend(_find_differences(a, b, subpath))

    elif obj1 != obj2:
        diffs.append(f"{path}: value changed from {obj1} to {obj2}")

    return diffs


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
    dir1 = Path(dir1)
    dir2 = Path(dir2)

    for path1 in dir1.rglob("*.json"):
        rel_path = path1.relative_to(dir1)
        path2 = dir2 / rel_path

        if not path2.exists():
            print(f"Missing in dir2: {rel_path}")
            continue

        try:
            json1 = _load_json_without_key(path1, ignore_key)
            json2 = _load_json_without_key(path2, ignore_key)
        except FileNotFoundError as e:
            print(f"Error reading {rel_path}: {e}")
            continue

        differences = _find_differences(json1, json2)
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

    _compare_json_dirs(args_dict["directory_1"], args_dict["directory_2"])


if __name__ == "__main__":
    main()
