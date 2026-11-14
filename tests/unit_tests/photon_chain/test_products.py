"""Tests for the photon product names."""

from simtools.photon_chain import products


def test_product_names_are_unique_strings():
    names = [value for key, value in vars(products).items() if key.isupper()]

    assert names
    assert all(isinstance(name, str) for name in names)
    assert len(names) == len(set(names))
