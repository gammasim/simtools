"""Guard the foundation modules from importing higher layers."""

import ast
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("module_name", "allowed_imports"),
    [
        ("constants", set()),
        ("version", {"simtools._version", "simtools._dev_version"}),
        ("statistics", set()),
    ],
)
def test_foundation_modules_do_not_import_simtools(module_name, allowed_imports):
    """Foundation modules do not depend on other simtools modules."""
    source_path = Path(__file__).resolve().parents[2] / "src" / "simtools" / f"{module_name}.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names if alias.name.startswith("simtools"))
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            if node.module == "simtools":
                imports.extend(f"simtools.{alias.name}" for alias in node.names)
            elif node.module.startswith("simtools"):
                imports.append(node.module)

    assert set(imports) <= allowed_imports
