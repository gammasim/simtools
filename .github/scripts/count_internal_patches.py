"""Count internal patch targets and MagicMock model readers in unit tests."""

import argparse
import ast
from pathlib import Path

TEST_ROOT = Path("tests/unit_tests")


def _is_magicmock(call):
    """Return whether an AST call constructs MagicMock."""
    if not isinstance(call, ast.Call):
        return False
    name = ast.unparse(call.func)
    return name == "MagicMock" or name.endswith(".MagicMock")


def _simtools_names(tree):
    """Return local names bound to simtools modules or objects by imports."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("simtools"):
            names.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("simtools"):
                    names.add(alias.asname or alias.name.split(".")[0])
    return names


def _string_constants(tree):
    """Return names assigned a string constant at module level (e.g. patch targets)."""
    constants = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = node.value.value
    return constants


def _base_name(node):
    """Return the leftmost name of an attribute chain such as ``a.b.c``."""
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


_PATCH_CALLS = ("patch", "patch.object", "patch.dict", "patch.multiple")


def _is_patch_call(call):
    """Return whether a call is a mock patch or a monkeypatch setattr."""
    if not isinstance(call, ast.Call) or not call.args:
        return False
    name = ast.unparse(call.func)
    return (
        name in _PATCH_CALLS
        or name.endswith(tuple(f".{call_name}" for call_name in _PATCH_CALLS))
        or name.endswith(".setattr")
    )


def _target_string(target, string_constants):
    """Return the (leading) string of a patch target, or None if it is not a string."""
    if isinstance(target, ast.Name):
        return string_constants.get(target.id)
    if isinstance(target, ast.Constant) and isinstance(target.value, str):
        return target.value
    if isinstance(target, ast.JoinedStr) and target.values:
        first = target.values[0]
        if isinstance(first, ast.FormattedValue):
            return _target_string(first.value, string_constants) or ""
        return str(first.value)
    return None


def _is_internal_patch(call, simtools_names, string_constants):
    """Return whether a patch call targets a simtools module or object."""
    if not _is_patch_call(call):
        return False
    target = call.args[0]
    target_string = _target_string(target, string_constants)
    if target_string is not None:
        return target_string.startswith("simtools")
    return _base_name(target) in simtools_names


def _reader_magicmocks(tree):
    """Count MagicMock assignments with a model-reader variable name."""
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and _is_magicmock(node.value):
            targets = [ast.unparse(target).lower() for target in node.targets]
            if any("reader" in target for target in targets):
                count += 1
        elif isinstance(node, ast.AnnAssign) and _is_magicmock(node.value):
            if "reader" in ast.unparse(node.target).lower():
                count += 1
    return count


def main():
    """Fail when the number of internal test patches exceeds the allowed maximum."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max", type=int, required=True, help="Allowed maximum count.")
    maximum = parser.parse_args().max

    patch_count = 0
    reader_count = 0
    for path in TEST_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        simtools_names = _simtools_names(tree)
        constants = _string_constants(tree)
        patch_count += sum(
            _is_internal_patch(node, simtools_names, constants) for node in ast.walk(tree)
        )
        reader_count += _reader_magicmocks(tree)

    actual = patch_count + reader_count
    if actual > maximum:
        raise SystemExit(
            f"Internal test patches increased: {actual} exceeds the allowed maximum {maximum}. "
            "Use test builders or fakes instead of patching simtools internals."
        )
    print(f"Internal test patches: {actual} (allowed maximum {maximum})")


if __name__ == "__main__":
    main()
