"""Unit tests for the trigger-histogram file format names."""

from simtools.sim_events import trigger_histogram_format as fmt


def test_table_and_group_names_are_distinct_uppercase_strings():
    names = [
        fmt.TRIGGER_HISTOGRAM_METADATA_TABLE,
        fmt.TRIGGER_HISTOGRAM_BINS_TABLE,
        fmt.TRIGGER_TOPOLOGY_COUNTS_TABLE,
        fmt.TRIGGER_SUBSET_HISTOGRAMS_TABLE,
        fmt.TRIGGER_HISTOGRAM_DENSE_GROUP,
    ]
    assert len(set(names)) == len(names)
    assert all(name.isupper() and name.startswith("TRIGGER_") for name in names)
