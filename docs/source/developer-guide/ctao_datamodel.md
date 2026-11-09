# CTAO DataProducts metadata

sim_telarray production files carry CTAO DataProducts FITS keywords alongside the existing simtools
keys. simtools writes these keywords itself; it has no runtime dependency on the `ctao-datamodel`
package.

## Compatibility test

The unit test `tests/unit_tests/simtel/test_sim_telarray_metadata_ctao_datamodel.py` serializes the
same product with the `ctao-datamodel` package and compares the keywords and values used by
simtools. Version values are checked locally rather than compared with the installed package.
The compatibility test is skipped when the package is not installed.

CI job `ctao_datamodel_compatibility` in `.github/workflows/CI-unittests.yml` installs
`ctao-datamodel` at a pinned release and at the latest stable release, and runs the test for each.
The pinned release is a required check; the latest release is monitored without blocking CI.
The job also runs on the scheduled workflow run.

## Updating after an upstream change

1. Install the candidate `ctao-datamodel` release.
2. Run `pytest tests/unit_tests/simtel/test_sim_telarray_metadata_ctao_datamodel.py`.
3. Review changes to keywords and values used by simtools and update the metadata builder and
   tests as needed. Update `PRODUCT_MODEL_VERSION` and its local test when adopting a new model.
