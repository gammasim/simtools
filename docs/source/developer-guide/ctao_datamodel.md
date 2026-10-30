# CTAO DataProducts metadata

sim_telarray production files carry CTAO DataProducts metadata alongside existing simtools keys.
The `ctao-datamodel` package defines the product fields and FITS keywords. simtools maps resolved
production settings into that model and validates the generated metadata.

## Updating ctao-datamodel

simtools supports `ctao-datamodel>=1.4,<2`. The installed distribution version and the DataProducts
model version are separate values. The package version appears in the dependency manifest, while
the DataProducts model version is written as `CTAOMETA` and `MODELVER`.

For an upgrade:

1. Review the upstream model and serializer changes.
2. Install the candidate package and run the CTAO bridge and sim_telarray metadata unit tests.
3. Review changes to the generated metadata registry and classify each supported or intentionally
   omitted field in the compatibility test.
4. Verify a production configuration contains the expected CTAO and legacy keys.
5. Update the dependency constraint only after the supported package range passes CI.

CI tests both the minimum supported release and the latest release below version 2.
