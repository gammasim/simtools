# Metadata written to sim_telarray output

simtools writes metadata including run information, software version, and model parameters into the
sim_telarray output file. The metadata is written as set of key-value pairs and consist of
a general set of values plus one set per array element.

To read the metadata from a sim_telarray output file, do e.g.,

```python
import pprint
import simtools.simtel.simtel_io_metadata as simtel_io_metadata
data = simtel_io_metadata.read_sim_telarray_metadata(
   "tests/resources/generated/proton_run000001_za20deg_azm180deg_North_alpha_6.0.2_test.simtel.zst"
)
pprint.pprint(data)
```

Metadata is validated against the schema [sim_telarray_metadata.metaschema.yml](src/simtools/schemas/sim_telarray_metadata.metaschema.yml)
plus the model-parameter schemas. The schema is used to validate the emitted metadata and to
provide a registry of the metadata keys, their expected scope and value shape.

The complete schema can be exported from the simtools package with

```console
simtools-export-sim-telarray-metadata-schema \
   --output_file sim_telarray_metadata_schema.json
```
