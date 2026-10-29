"""Compare two directories with model production tables in JSON format."""

import sys
from pathlib import Path

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.io import ascii_handler
from simtools.utils import value_conversion

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "directory_1",
        type=str,
        required=True,
        help="Path to the first directory containing JSON files.",
    ),
    cli.ArgumentDefinition(
        "directory_2",
        type=str,
        required=True,
        help="Path to the second directory containing JSON files.",
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(*_ARGUMENTS,),
    needs=("corsika",),
)


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
    """Compare JSON directories and return the number of differences."""
    difference_count = 0
    for path1 in sorted(dir1.rglob("*.json")):
        rel_path = path1.relative_to(dir1)
        path2 = dir2 / rel_path

        if not path2.exists():
            print(f"Missing in dir2: {rel_path}")
            difference_count += 1
            continue

        try:
            json1 = value_conversion.remove_key_from_dict(
                ascii_handler.collect_data_from_file(path1), ignore_key
            )
            json2 = value_conversion.remove_key_from_dict(
                ascii_handler.collect_data_from_file(path2), ignore_key
            )
        except FileNotFoundError as e:
            print(f"Error reading {rel_path}: {e}")
            difference_count += 1
            continue

        differences = find_differences_in_json_objects(json1, json2)
        if differences:
            _print_differences(differences, rel_path)
            difference_count += 1

    # Check for files present in dir2 but not dir1
    for path2 in sorted(dir2.rglob("*.json")):
        rel_path = path2.relative_to(dir2)
        if not (dir1 / rel_path).exists():
            print(f"Missing in dir1: {rel_path}")
            difference_count += 1
    return difference_count


def _find_differences_dict(obj1, obj2, path, diffs):
    """Recursively find differences between two dictionaries."""
    for key in sorted(set(obj1) | set(obj2)):
        subpath = f"{path}['{key}']" if path else f"['{key}']"
        if key not in obj1:
            diffs.append(f"{subpath}: added in second object")
        elif key not in obj2:
            diffs.append(f"{subpath}: removed in second object")
        else:
            diffs.extend(find_differences_in_json_objects(obj1[key], obj2[key], subpath))


def find_differences_in_json_objects(obj1, obj2, path=""):
    """
    Recursively find differences between two JSON-like objects.

    Parameters
    ----------
    obj1: dict, list, or any
        First object to compare.
    obj2: dict, list, or any
        Second object to compare.
    path: str
        Path to the current object in the JSON structure, used for reporting differences.

    Returns
    -------
    list
        List of differences found between the two objects, with paths indicating where the
        differences occur.
    """
    diffs = []

    if not isinstance(obj1, type(obj2)):
        diffs.append(f"{path}: type changed from {type(obj1).__name__} to {type(obj2).__name__}")
        return diffs

    if isinstance(obj1, dict):
        _find_differences_dict(obj1, obj2, path, diffs)

    elif isinstance(obj1, list):
        if len(obj1) != len(obj2):
            diffs.append(f"{path}: list length changed from {len(obj1)} to {len(obj2)}")
        for i, (a, b) in enumerate(zip(obj1, obj2)):
            subpath = f"{path}[{i}]" if path else f"[{i}]"
            diffs.extend(find_differences_in_json_objects(a, b, subpath))

    elif obj1 != obj2:
        diffs.append(f"{path}: value changed from {obj1} to {obj2}")

    return diffs


def main():
    """See CLI description."""
    app_context = APPLICATION.start()

    difference_count = _compare_json_dirs(
        Path(app_context.args["directory_1"]), Path(app_context.args["directory_2"])
    )
    if difference_count:
        sys.exit(1)


if __name__ == "__main__":
    main()
