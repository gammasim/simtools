# Trim Plan: Consolidate Row-Table Logic & Reduce Test Sprawl

**Current state:** 2434 total additions (1291 src + 1123 tests)
**Target state:** ~1200-1350 total additions (realistic reduction = ~40-50%)

---

## Phase 1: Extract & Centralize Row-Table Utilities (~150-180 src lines saved)

### Goal
Create one canonical module for all row-table dict validation, conversion, and schema detection logic.

### New Module: `src/simtools/simtel/row_table_utils.py`

**Content (estimated ~120 lines):**

```python
"""Utilities for row-oriented table data (columns, column_units, rows)."""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Canonical set of row-table dict keys
ROW_TABLE_KEYS = {"columns", "rows", "column_units"}


def is_row_table_dict(value: Any) -> bool:
    """Check if a dict has the row-table structure (columns, rows, column_units)."""
    if not isinstance(value, dict):
        return False
    return all(key in value for key in ROW_TABLE_KEYS)


def is_row_table_schema(json_schema: Dict[str, Any]) -> bool:
    """Check if a JSON schema defines row-table shape (requires columns, rows, column_units)."""
    required = set(json_schema.get("required", []))
    properties = set(json_schema.get("properties", {}).keys())
    return ROW_TABLE_KEYS.issubset(required) and ROW_TABLE_KEYS.issubset(properties)


def validate_row_table_structure(parameter_name: str, value: Dict[str, Any]) -> None:
    """
    Validate row-table dict structure: columns/rows lengths, numeric row values.

    Raises ValueError on mismatch.
    """
    if not isinstance(value, dict) or "columns" not in value or "rows" not in value:
        raise ValueError(
            f"Row-table value for '{parameter_name}' must be a dict with "
            "'columns' and 'rows' keys."
        )

    if "column_units" not in value:
        raise ValueError(
            f"Row-table value for '{parameter_name}' must include 'column_units'."
        )

    columns = value["columns"]
    rows = value["rows"]
    column_units = value["column_units"]

    if len(column_units) != len(columns):
        raise ValueError(
            f"Row-table for '{parameter_name}': column_units length ({len(column_units)}) "
            f"must match columns length ({len(columns)})."
        )

    _validate_row_values(parameter_name, columns, rows)


def _validate_row_values(parameter_name: str, columns: List[str], rows: List) -> None:
    """Validate numeric scalars and row length consistency."""
    n_columns = len(columns)
    for row_index, row in enumerate(rows):
        if not isinstance(row, (list, tuple, np.ndarray)):
            raise ValueError(
                f"Row-table for '{parameter_name}' has invalid row at index {row_index}: "
                "each row must be a sequence with one numeric value per column."
            )

        if len(row) != n_columns:
            raise ValueError(
                f"Row-table for '{parameter_name}' has invalid row length at index {row_index}: "
                f"expected {n_columns} values, got {len(row)}."
            )

        for col_index, value in enumerate(row):
            if not np.isscalar(value):
                raise ValueError(
                    f"Row-table for '{parameter_name}' has non-numeric value at "
                    f"row {row_index}, column {col_index} ('{columns[col_index]}'): {value!r}."
                )

            value_dtype = np.asarray(value).dtype
            is_numeric = np.issubdtype(value_dtype, np.number)
            is_real = not np.issubdtype(value_dtype, np.complexfloating)

            if not (is_numeric and is_real):
                raise ValueError(
                    f"Row-table for '{parameter_name}' has complex/non-numeric value at "
                    f"row {row_index}, column {col_index} ('{columns[col_index]}'): {value!r}."
                )


def normalize_column_unit(unit_value: Any) -> str:
    """Convert astropy unit or string to schema-compatible unit string."""
    import astropy.units as u

    if unit_value is None:
        return "dimensionless"

    if isinstance(unit_value, str):
        return unit_value if unit_value else "dimensionless"

    if unit_value == u.dimensionless_unscaled:
        return "dimensionless"

    unit_str = str(unit_value)
    return unit_str if unit_str else "dimensionless"
```

### Refactoring Targets

#### 1. **`src/simtools/simtel/simtel_table_writer.py`** (~40 lines saved)
   - **Remove:** `_validate_simtel_table_rows()` function (replace with import)
   - **Replace calls:** `_validate_simtel_table_rows(...)` → `row_table_utils.validate_row_table_structure(...)`
   - **New code:** 5–10 additional lines for import + wrappings

#### 2. **`src/simtools/simtel/simtel_table_reader.py`** (~50 lines saved)
   - **Remove:** `_normalize_column_unit()`, `_validate_row_data_dict()` functions
   - **Replace:** `_normalize_column_unit(...)` → `row_table_utils.normalize_column_unit(...)`
   - **Rename call:** `_validate_row_data_dict()` → `row_table_utils.is_row_table_dict()` or refactor its usage
   - **New code:** ~15 additional lines for imports + usage cleanup

#### 3. **`src/simtools/data_model/model_data_writer.py`** (~30 lines saved)
   - **Remove:** `parameter_uses_row_table_schema()` method body duplication
   - **Replace:** Use `row_table_utils.is_row_table_schema()` inside a wrapper
   - **Simplify:** Keep public API for backward compat; delegate to utility
   - **New code:** ~5 lines

#### 4. **`src/simtools/model/legacy_model_parameter.py`** (~20 lines saved)
   - **Remove:** Inline checks for `"columns" in value and "rows" in value`
   - **Replace:** Use `row_table_utils.is_row_table_dict()` for clarity
   - **New code:** ~10 lines for import + usage

### Expected Savings
- New module: ~120 lines
- Removed duplication: ~140 lines
- **Net savings: ~20 lines, but consolidation value is high (1 source of truth for validation)**

---

## Phase 2: Collapse Export Decision Logic (~80-120 src lines saved)

### Goal
Reduce branching logic in parameter_exporter by moving file/dict discrimination to a strategy map.

### Refactor: `src/simtools/db/parameter_exporter.py`

**Current structure:**
- `export_parameter_data()` has nested if/else for dict vs file type
- `export_single_model_file()` is called twice (table export + file export paths)
- Repeated `db.get_model_parameter()` calls and result unpacking

**New structure:**

```python
"""Simplified export strategy dispatch."""

def _is_dict_table_parameter(par_info: dict) -> bool:
    """Identify dict-typed row-table parameters."""
    return par_info.get("type") == "dict" and isinstance(par_info.get("value"), dict)


# Strategy: (needs output_file, export_fn)
_EXPORT_STRATEGIES = {
    "dict_table": {
        "require_output_file": True,
        "handler": "_export_dict_table_parameter",
    },
    "file_backed": {
        "require_output_file": False,
        "handler": "_export_file_backed_parameter",
    },
}


def _export_dict_table_parameter(db, par_info, parameter, output_file, ...):
    """Export dict-backed table as ECSV."""
    # ~20 lines (extracted from export_parameter_data)
    ...


def _export_file_backed_parameter(db, par_info, parameter, export_as_table, ...):
    """Export file-backed parameter, optionally as table."""
    # ~30 lines (extracted from export_parameter_data)
    ...


def export_parameter_data(...):
    """Main entry point with strategy dispatch."""
    # Validate input flags
    if export_file_as_table and not export_model_file:
        raise ValueError("Use --export_model_file together with --export_model_file_as_table.")

    if not (export_model_file or export_file_as_table):
        return []

    # Fetch once
    parameters, par_info = _get_parameter_info(...)

    # Dispatch by type
    strategy_key = "dict_table" if _is_dict_table_parameter(par_info) else "file_backed"

    # Validate requirements
    strategy = _EXPORT_STRATEGIES[strategy_key]
    if strategy["require_output_file"] and output_file is None:
        raise ValueError(f"--output_file required for {strategy_key} export.")

    # Call strategy handler
    handler = globals()[strategy["handler"]]
    return handler(db, par_info, parameter, output_file, export_file_as_table)
```

### Expected Changes
- **Remove nested if/else branches:** ~60 lines
- **Add small strategy map + dispatch:** ~50 lines
- **Extracted helper handlers:** ~40 lines (organized, not new)
- **Net savings:** ~30–50 lines
- **Benefit:** Clearer flow, easier to add export types in future

### Testing Impact
- Existing tests continue to work (same public API)
- New helper functions get light unit coverage
- Main export tests remain unchanged

---

## Phase 3: Parameterize & Consolidate Tests (~250-400 test lines saved)

### Goal
Use fixtures and parametrized tests to collapse repeated setup/assertion patterns.

### 3.1 Create Shared Test Fixtures: `tests/unit_tests/conftest.py` (Add to existing)

```python
@pytest.fixture
def row_table_payload():
    """Canonical valid row-table dict payload."""
    return {
        "columns": ["time", "amplitude", "amplitude (low gain)"],
        "column_units": ["ns", "dimensionless", "dimensionless"],
        "rows": [
            [0.0, 0.0, 0.0],
            [0.12, 0.01323, 0.000945],
            [0.25, 0.02646, 0.001890],
        ],
    }


@pytest.fixture
def row_table_payload_two_col():
    """Row-table with 2 columns."""
    return {
        "columns": ["time", "amplitude"],
        "column_units": ["ns", "dimensionless"],
        "rows": [[0.0, 0.0], [0.5, 0.5]],
    }


@pytest.fixture
def invalid_row_table_payloads():
    """Generator of invalid row-table payloads (missing keys, wrong types, etc.)."""
    return {
        "missing_column_units": {
            "columns": ["time", "amplitude"],
            "rows": [[0.0, 0.0]],
        },
        "wrong_row_length": {
            "columns": ["time", "amplitude"],
            "column_units": ["ns", "dimensionless"],
            "rows": [[0.0]],  # Should be 2 items
        },
        "non_numeric_row": {
            "columns": ["time", "amplitude"],
            "column_units": ["ns", "dimensionless"],
            "rows": [["not", "a number"]],
        },
    }
```

### 3.2 Refactor `tests/unit_tests/simtel/test_simtel_table_reader.py` (~80-120 lines saved)

**Before:**
```python
def test_read_simtel_table_as_row_data():
    table = Table({...})
    table["time"].unit = ...
    ...
    result = simtel_table_reader.read_simtel_table_as_row_data(...)
    assert result == {...}

def test_resolve_dict_parameter_value_from_inline_json(tmp_test_directory):
    value = '{"columns": ...}'
    with mock.patch(...):
        result = simtel_table_reader.resolve_dict_parameter_value(...)
    assert isinstance(result, dict)
    assert result["columns"] == [...]

def test_resolve_dict_parameter_value_from_file_path(...):
    with mock.patch(...):
        result = simtel_table_reader.resolve_dict_parameter_value(...)
    read_mock.assert_called_once_with(...)
```

**After (consolidate):**
```python
@pytest.mark.parametrize("input_mode", ["inline_json", "file_path", "path_object"])
def test_resolve_dict_parameter_value(input_mode, tmp_test_directory, mocker):
    """Test resolve_dict_parameter_value across all input modes."""
    if input_mode == "inline_json":
        value = '{"columns": ["time"], "column_units": ["ns"], "rows": [[0.0]]}'
        expected_read_call = None  # Mock should not be called
    elif input_mode == "file_path":
        value = "pulse.dat"
        expected_read_call = ("fadc_pulse_shape", Path(tmp_test_directory) / "pulse.dat")
    else:  # path_object
        value = Path("pulse.dat")
        expected_read_call = ("fadc_pulse_shape", Path(tmp_test_directory) / "pulse.dat")

    read_mock = mocker.patch("simtools.simtel.simtel_table_reader.read_simtel_table_as_row_data")
    read_mock.return_value = {"columns": ["time"], "column_units": ["ns"], "rows": [[0.0]]}

    result = simtel_table_reader.resolve_dict_parameter_value(
        value, "fadc_pulse_shape", str(tmp_test_directory)
    )

    if expected_read_call:
        read_mock.assert_called_once_with(*expected_read_call)
    else:
        read_mock.assert_not_called()

    assert result["columns"] == ["time"]


@pytest.mark.parametrize("invalid_key", ["missing_columns", "missing_rows", "missing_units"])
def test_row_data_to_astropy_table_validates_keys(invalid_key):
    """Test row_data_to_astropy_table rejects incomplete payloads."""
    payload = {
        "columns": ["time"],
        "column_units": ["ns"],
        "rows": [[0.0]],
    }
    del payload[
        {
            "missing_columns": "columns",
            "missing_rows": "rows",
            "missing_units": "column_units",
        }[invalid_key]
    ]

    with pytest.raises(ValueError, match="'columns' and 'rows'"):
        simtel_table_reader.row_data_to_astropy_table(payload)
```

**Savings:** ~60–100 lines (collapse ~8 tests into 2–3 parametrized ones)

### 3.3 Refactor `tests/unit_tests/data_model/test_model_data_writer.py` (~60-100 lines saved)

**Consolidate `fadc_pulse_shape` tests:**

```python
@pytest.mark.parametrize(
    "schema_version,value_type",
    [
        ("0.2.0", "dict"),      # New embedded format
        ("0.1.0", "string"),    # Legacy file reference
    ],
)
def test_get_parameter_type_for_schema_fadc(schema_version, value_type):
    """Test schema version selection for fadc_pulse_shape."""
    w1 = writer.ModelDataWriter()
    expected_type = "dict" if schema_version == "0.2.0" else "file"
    assert w1.get_parameter_type_for_schema("fadc_pulse_shape", schema_version) == expected_type


@pytest.mark.parametrize(
    "invalid_rows",
    [
        [[0.0]],  # Too few columns
        [[0.0, 0.0, 0.0, 0.0]],  # Too many columns
    ],
)
def test_get_validated_parameter_dict_fadc_invalid_rows(invalid_rows, row_table_payload):
    """Test fadc_pulse_shape rejects malformed row lengths."""
    w1 = writer.ModelDataWriter()
    bad_value = row_table_payload.copy()
    bad_value["rows"] = invalid_rows

    with pytest.raises(ValidationError):
        w1.get_validated_parameter_dict(
            parameter_name="fadc_pulse_shape",
            value=bad_value,
            instrument="LSTN-01",
            parameter_version="0.0.1",
            model_parameter_schema_version="0.2.0",
        )
```

**Savings:** ~40–60 lines (collapse variant tests into parametrized)

### 3.4 Refactor `tests/unit_tests/model/test_legacy_model_parameter.py` (~50-80 lines saved)

**Consolidate update handler tests:**

```python
@pytest.mark.parametrize(
    "source_version,source_type,resolver_called",
    [
        ("0.1.0", "file", True),       # File → embedded requires resolver
        ("0.2.0", "dict", False),      # Already embedded, skip resolver
    ],
)
def test_update_fadc_pulse_shape(mocker, source_version, source_type, resolver_called):
    """Test fadc_pulse_shape migration paths."""
    if source_type == "file":
        value = "pulse.dat"
        resolver = mocker.Mock(return_value={
            "columns": ["time", "amplitude"],
            "column_units": ["ns", "dimensionless"],
            "rows": [[0.0, 0.1]],
        })
    else:
        value = {
            "columns": ["time", "amplitude"],
            "column_units": ["ns", "dimensionless"],
            "rows": [[0.0, 0.1]],
        }
        resolver = None

    parameters = {
        "fadc_pulse_shape": {
            "parameter": "fadc_pulse_shape",
            "value": value,
            "model_parameter_schema_version": source_version,
            "type": source_type,
            "file": (source_type == "file"),
        }
    }

    result = _update_fadc_pulse_shape(parameters, "0.2.0", value_resolver=resolver)

    if resolver_called:
        resolver.assert_called_once_with("fadc_pulse_shape", "pulse.dat")

    assert result["fadc_pulse_shape"]["model_parameter_schema_version"] == "0.2.0"
    assert result["fadc_pulse_shape"]["type"] == "dict"
```

**Savings:** ~40–60 lines

### 3.5 Refactor `tests/unit_tests/db/test_parameter_exporter.py` (~80-120 lines saved)

**Use fixture matrix instead of repeated mocks:**

```python
@pytest.fixture
def exporter_db_mock(mocker):
    """Shared mock DB handler with common setup."""
    db = mocker.Mock()
    db.io_handler.get_output_file.return_value = mocker.MagicMock()
    db.get_model_parameter = mocker.Mock()
    return db


@pytest.mark.parametrize(
    "param_type,output_required,should_fail",
    [
        ("dict", True, False),
        ("dict", None, True),    # Missing output_file
        ("file", None, False),
    ],
)
def test_export_parameter_data_dispatches(param_type, output_required, should_fail, exporter_db_mock, mocker):
    """Test export_parameter_data dispatch logic."""
    par_info = {
        "type": param_type,
        "value": (
            {"columns": ["time"], "rows": [[1.0]]}
            if param_type == "dict"
            else "ref.dat"
        ),
    }
    exporter_db_mock.get_model_parameter.return_value = {"test_param": par_info}

    if should_fail:
        with pytest.raises(ValueError, match="--output_file"):
            parameter_exporter.export_parameter_data(
                db=exporter_db_mock,
                parameter="test_param",
                site="North",
                array_element_name="LSTN-01",
                parameter_version="1.0.0",
                model_version=None,
                output_file=output_required,
                export_model_file=True,
                export_model_file_as_table=False,
            )
    else:
        mocker.patch.object(
            parameter_exporter, "export_single_model_file", return_value=mocker.Mock()
        )
        result = parameter_exporter.export_parameter_data(...)
        assert isinstance(result, list)
```

**Savings:** ~60–100 lines

### Total Test Savings: ~250–400 lines

---

## Implementation Sequence

### Step 1: Create Row-Table Utility Module (Phase 1)
1. Create `src/simtools/simtel/row_table_utils.py` (~120 lines)
2. Add unit tests `tests/unit_tests/simtel/test_row_table_utils.py` (~80 lines)
3. Update `src/simtools/simtel/simtel_table_writer.py` to use utilities
4. Update `src/simtools/simtel/simtel_table_reader.py` to use utilities
5. Update `src/simtools/data_model/model_data_writer.py` to delegate
6. Update `src/simtools/model/legacy_model_parameter.py` to use utilities
7. Run tests: verify no regression
8. **Expected net change:** -20 source lines, +1 utility module (centralization value)

### Step 2: Refactor Export Logic (Phase 2)
1. Refactor `src/simtools/db/parameter_exporter.py` with strategy dispatch
2. Update tests in `tests/unit_tests/db/test_parameter_exporter.py` (new dispatch verification)
3. Run tests: verify export still works
4. **Expected net change:** -30 to -50 source lines

### Step 3: Consolidate Tests (Phase 3)
1. Add shared fixtures to `tests/conftest.py`
2. Refactor `test_simtel_table_reader.py` (parametrize tests)
3. Refactor `test_model_data_writer.py` (consolidate fadc tests)
4. Refactor `test_legacy_model_parameter.py` (parametrize handler tests)
5. Refactor `test_parameter_exporter.py` (use fixture matrix)
6. Run full test suite: verify coverage maintained
7. **Expected net change:** -250 to -400 test lines

---

## Expected Final Metrics

| Phase | Source Δ | Test Δ | Total Δ | Rationale |
|-------|----------|--------|---------|-----------|
| Start | 1291 | 1123 | 2414 | Current branch |
| After Phase 1 | -20 | +80 | 2474 | Add utility module + tests |
| After Phase 2 | -50 | 0 | 2424 | Simplify export dispatch |
| After Phase 3 | 0 | -350 | 2074 | Consolidate test variants |
| **Final** | **1221** | **773** | **~1994** | **~18% total reduction** |

---

## Risk & Mitigation

| Risk | Mitigation |
|------|-----------|
| Breaking existing imports/APIs | Keep public interfaces stable; move implementation only |
| Test regressions | Run full suite after each phase; parametrize carefully |
| Over-parameterization (tests unreadable) | Use descriptive parameter IDs: `@pytest.mark.parametrize(..., ids=[...])` |
| Phase interdependencies | Phase 1 is independent; Phase 2 minimal API surface; Phase 3 is isolated tests |

---

## Recommended Commit Structure

```
Phase 1:
  - extract: row_table_utils module + tests
  - refactor: simtel_table_writer to use utilities
  - refactor: simtel_table_reader to use utilities
  - refactor: model_data_writer to delegate
  - refactor: legacy_model_parameter to use utilities

Phase 2:
  - refactor: parameter_exporter dispatch logic
  - tests: update export verification tests

Phase 3:
  - test: add shared fixtures to conftest
  - test: parametrize simtel_table_reader tests
  - test: parametrize model_data_writer fadc tests
  - test: parametrize legacy_model_parameter handler tests
  - test: parametrize parameter_exporter tests

Final:
  - rebase and verify full test suite passes
```

---

## Validation Checklist

- [ ] All unit tests pass (including new parameterized variants)
- [ ] Integration tests pass
- [ ] No public API changes (backward compat maintained)
- [ ] Code coverage ≥ 90% for all new/modified modules
- [ ] Linting & formatting pass (`pre-commit run --all-files`)
- [ ] Final line count delta is ~18% reduction (1200–1350 final vs 2414 current)
