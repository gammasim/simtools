# CTAO DataProducts metadata

sim_telarray production files carry CTAO DataProducts FITS keywords alongside the existing simtools
keys. simtools writes these keywords itself; it has no runtime dependency on the `ctao-datamodel`
package.

## Compatibility test

The unit test `tests/unit_tests/simtel/test_sim_telarray_metadata_ctao_datamodel.py` serializes the
same product with the `ctao-datamodel` package and compares the keywords and values with the
simtools output. The test is skipped when the package is not installed.

CI job `ctao_datamodel_compatibility` in `.github/workflows/CI-unittests.yml` installs
`ctao-datamodel` at the minimum supported release and at the latest release below version 2, and
runs the test for each. The job also runs on the scheduled workflow run.

## Updating after an upstream change

1. Install the candidate `ctao-datamodel` release.
2. Run `pytest tests/unit_tests/simtel/test_sim_telarray_metadata_ctao_datamodel.py`.
3. Update the keyword table and `PRODUCT_MODEL_VERSION` in
   `src/simtools/simtel/sim_telarray_metadata.py`, and classify new or changed keywords in the
   keyword sets of the compatibility test.
