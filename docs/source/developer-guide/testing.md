# Testing

`simtools` uses three test layers:

- [Unit tests](testing_unit.md) for library modules and functions.
- [Integration tests](testing_integration.md) for end-to-end application workflows.
- [Science tests](testing_science.md) for controlled baseline/candidate physics and resource
  comparisons at release milestones. Execution success and scientific acceptance are separate;
  large productions require explicit selection and permission.

[Test resources](testing_resources.md) describes the supporting versioned resource bundles.
Science definitions and small reports live in simtools-tests; large campaign products and
execution logs remain in external storage. The science-testing guide describes the catalogue,
release/site selections, production gates, and result contract.

```{toctree}
:hidden:
:maxdepth: 1

testing_unit.md
testing_integration.md
testing_benchmarks.md
testing_science.md
testing_resources.md
```
