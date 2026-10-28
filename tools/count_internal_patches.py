"""Count internal patch targets and MagicMock model readers in unit tests."""

import ast
from pathlib import Path

TEST_ROOT = Path("tests/unit_tests")
COUNT_FILE = Path("tools/internal_patch_count.txt")


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


def _base_name(node):
    """Return the leftmost name of an attribute chain such as ``a.b.c``."""
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _is_internal_patch(call, simtools_names):
    """Return whether a patch call targets a simtools module or object."""
    if not isinstance(call, ast.Call) or not call.args:
        return False
    name = ast.unparse(call.func)
    if not (
        name == "patch"
        or name.endswith(".patch")
        or name.endswith(".patch.object")
        or name.endswith(".setattr")
    ):
        return False
    target = call.args[0]
    if isinstance(target, ast.Constant) and isinstance(target.value, str):
        return target.value.startswith("simtools.")
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
    """Fail when the number of internal test patches exceeds its baseline."""
    patch_count = 0
    reader_count = 0
    for path in TEST_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        simtools_names = _simtools_names(tree)
        patch_count += sum(_is_internal_patch(node, simtools_names) for node in ast.walk(tree))
        reader_count += _reader_magicmocks(tree)

    actual = patch_count + reader_count
    baseline = int(COUNT_FILE.read_text(encoding="utf-8").strip())
    if actual > baseline:
        raise SystemExit(
            f"Internal test patches increased: {actual} exceeds recorded count {baseline}."
        )
    print(actual)


if __name__ == "__main__":
    main()
