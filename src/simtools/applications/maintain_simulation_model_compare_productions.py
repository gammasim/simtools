r"""
Compare two directories with model production tables in JSON format.

This script should be used to support the maintenance the simulation model repository.

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


def load_json_without_key(path, ignore_key):
    """Load JSON data from a file, ignoring a specific key."""
    data = gen.collect_data_from_file(path)

    def remove_key(obj):
        if isinstance(obj, dict):
            return {k: remove_key(v) for k, v in obj.items() if k != ignore_key}
        if isinstance(obj, list):
            return [remove_key(i) for i in obj]
        return obj

    return remove_key(data)


def find_differences(obj1, obj2, path=""):
    """Find differences between two objects recursively."""
    differences = []

    if not isinstance(obj1, type(obj2)):
        type1_name = type(obj1).__name__
        type2_name = type(obj2).__name__
        differences.append(f"{path}: type changed from {type1_name} to {type2_name}")
        return differences

    if isinstance(obj1, dict):
        all_keys = set(obj1.keys()) | set(obj2.keys())
        for key in all_keys:
            current_path = f"{path}['{key}']" if path else f"['{key}']"
            if key not in obj1:
                differences.append(f"{current_path}: added in directory 2")
            elif key not in obj2:
                differences.append(f"{current_path}: removed in directory 2")
            else:
                differences.extend(find_differences(obj1[key], obj2[key], current_path))
    elif isinstance(obj1, list):
        if len(obj1) != len(obj2):
            differences.append(f"{path}: list length changed from {len(obj1)} to {len(obj2)}")
        for i, (item1, item2) in enumerate(zip(obj1, obj2)):
            current_path = f"{path}[{i}]" if path else f"[{i}]"
            differences.extend(find_differences(item1, item2, current_path))
    else:
        if obj1 != obj2:
            differences.append(f"{path}: value changed from {obj1} to {obj2}")

    return differences


def compare_json_dirs(dir1, dir2, ignore_key="model_version"):
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
            json1 = load_json_without_key(path1, ignore_key)
            json2 = load_json_without_key(path2, ignore_key)
        except FileNotFoundError as e:
            print(f"Error reading {rel_path}: {e}")
            continue

        differences = find_differences(json1, json2)
        if differences:
            print(f"Difference in {rel_path}:\n{'-' * 40}")
            for diff in differences:
                # Clean up the path formatting for better readability
                clean_diff = diff.replace("['parameters']['", "parameters.").replace("']['", ".")
                clean_diff = clean_diff.replace("['", "").replace("']", "")
                print(f"  {clean_diff}")
            print(f"{'-' * 40}\n")

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

    compare_json_dirs(args_dict["directory_1"], args_dict["directory_2"])


if __name__ == "__main__":
    main()
