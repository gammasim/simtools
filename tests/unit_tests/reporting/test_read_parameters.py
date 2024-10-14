import astropy.units as u

from simtools.reporting.read_parameters import ReadParameters


def test_get_all_parameter_descriptions(telescope_model_lst):

    read_parameters = ReadParameters(telescope_model=telescope_model_lst)

    # Call get_all_parameter_descriptions
    descriptions, short_descriptions = read_parameters.get_all_parameter_descriptions()

    assert isinstance(descriptions.get("focal_length"), str)
    assert isinstance(short_descriptions.get("focal_length"), str)
    assert (
        short_descriptions.get("focal_length")
        == "Nominal overall focal length of the entire telescope."
    )


def test_get_telescope_parameter_data(telescope_model_lst):

    read_parameters = ReadParameters(telescope_model=telescope_model_lst)

    result = read_parameters.get_telescope_parameter_data()

    # Assert the result contains the expected data
    if result[0] == "focal_length":
        assert result[1] == (2800.0 * u.cm)
        assert (
            result[2]
            == "Nominal overall focal length of the entire telescope. This defines the \
                image scale near the centre of the field of view. For segmented primary \
                focus telescopes this determines the alignment of the segments and the \
                separation from the reflector, at its centre to the camera. The \
                alignment focus is on the surface determined by the par:focus-offset \
                (see parameter description for details).  For secondary mirror \
                configurations this value is not actually used in the optics simulation \
                but only reported as a nominal value, typically close to the effective \
                focal length."
        )
        assert result[3] == "Nominal overall focal length of the entire telescope."
