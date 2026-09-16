# Implemented sim_telarray follow-up

This file records the implementation and the conclusions from the original
review items.

## Mirror segmentation

Segmentation model parameters now use explicit quantity objects. For example:

```json
{
  "r_min": {"value": 100.0, "unit": "cm"},
  "dphi": {"value": 30.0, "unit": "deg"}
}
```

The current schemas use semantic names (`r_min`, `diameter`, `rotation`, and
`vertices[].x/y`) and reusable length/angle definitions. The validator converts
quantities with Astropy before checking geometry; the sim_telarray adapter
converts to centimetres and degrees only at the text-file boundary. This is a
breaking schema update. Existing active SCTS and SSTS segmentation model
parameters were converted to schema version 0.3.0. The old schema documents
remain available as schema history, but no active model uses the old suffix
names.

Unit tests cover rings, shaped facets, polygons, compatible units, invalid
units, invalid geometry, finite values, parsing, and serialization.

## Atmospheric transmission and `99999`

sim_telarray `read_trans()` parses every transmission-table entry as a numeric
optical depth; it has no missing-cell token. Its transmission calculation
returns zero for sufficiently large optical depth, so `99999` is an opaque
cell, not a generic fail-safe value. The current North model is sparse by
design and already contains these cells in its generated output. Rejecting
sparse matrices would break normal sim_telarray configuration generation.

The selected policy is therefore explicit opaque-cell export:

- the schema declares `missing_value: 99999` and
  `incomplete_grid_policy: opaque`;
- the serializer fills only absent Cartesian wavelength-altitude cells;
- unit tests generate complete and sparse tables in the temporary test
  directory and assert the exact filled cells and header order.

The generated header remains `H2` followed by ascending `H1` levels, matching
the sim_telarray reader. For North the production values are `H2=2.156` and
the first listed `H1` level is `2.206` km. The source-level sim_telarray
reader and interpolation code were traced. A locally installed sim_telarray
executable was not available, so event-level integration must run in the
simtools integration environment.

The atmospheric profile remains exported because it is consumed by both
CORSIKA and sim_telarray. A model parameter marked simtools-only is not
required by sim_telarray and may be omitted from sim_telarray export; the
export selection must continue to include parameters explicitly declaring
sim_telarray support.

## RPOL

The existing RPOL serializer is now documented and covered as the single
implementation for primary mirrors, secondary mirrors, and camera filters.
It writes:

- one-dimensional input as the normal wavelength/value table;
- two-dimensional input as `#@RPOL@[ANGLE=] 2`, an `ANGLE=` row, and a
  wavelength-by-angle matrix.

The serializer requires unique, complete Cartesian grids, sorts source rows by
the declared axes, and rejects duplicate or missing grid points before export.
Angles are validated in degrees and sim_telarray reads them with its
`yscale=deg2rad` option. Existing tests cover one- and two-dimensional output,
ordering, incomplete grids, duplicate points, and the secondary-mirror
contract. The checkout contains the source reader/interpolator; a local
sim_telarray executable was unavailable for event-level end-to-end execution.
