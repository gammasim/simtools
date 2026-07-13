"""Reusable argparse helpers for command-line parsing."""

import argparse
import ast
import logging
import re

import astropy.units as u

from simtools.utils import general, names


def scientific_int(value):
    """Convert string (including scientific notation) to integer.

    Parameters
    ----------
    value : str
        String value to convert.

    Returns
    -------
    int
        Integer value.

    Raises
    ------
    argparse.ArgumentTypeError
        If value cannot be converted to integer.
    """
    try:
        float_value = float(value)
        if not float_value.is_integer():
            raise ValueError
        return int(float_value)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid integer value: '{value}'. "
            "Expected an integer or scientific notation like '1e7'."
        ) from exc


def site(value):
    """Validate and return a site name.

    Parameters
    ----------
    value : str
        Site name to validate.

    Returns
    -------
    str
        Validated site name.

    Raises
    ------
    ValueError
        If site name is invalid.
    """
    return names.validate_site_name(str(value))


def telescope(value):
    """Validate and return telescope name(s).

    Parameters
    ----------
    value : str or list
        Telescope name or list of telescope names to validate.

    Returns
    -------
    str or list
        Validated telescope name(s).

    Raises
    ------
    ValueError
        If telescope name(s) are invalid.
    """
    values = general.ensure_list(value)
    for telescope_name in values:
        names.validate_array_element_name(str(telescope_name))
    return values if len(values) > 1 else values[0]


def instrument(value):
    """Validate and return an instrument name.

    Parameters
    ----------
    value : str
        Instrument name to validate.

    Returns
    -------
    str
        Validated instrument name.

    Raises
    ------
    ValueError
        If instrument name is invalid.
    """
    return names.validate_array_element_name(str(value))


def efficiency_interval(value):
    """Validate that value is an efficiency in the interval [0, 1].

    Parameters
    ----------
    value : float
        Efficiency value to validate.

    Returns
    -------
    float
        Validated efficiency value.

    Raises
    ------
    argparse.ArgumentTypeError
        If value is outside [0, 1] interval.
    """
    float_value = float(value)
    if float_value < 0.0 or float_value > 1.0:
        raise argparse.ArgumentTypeError(f"{value} outside of allowed [0,1] interval")
    return float_value


def quantity(target_unit):
    """Build an argument parser type for quantities convertible to a target unit.

    Parameters
    ----------
    target_unit : str or astropy.units.Unit
        Target unit for conversion.

    Returns
    -------
    function
        Parser function that converts input to quantities in target unit.

    Raises
    ------
    argparse.ArgumentTypeError
        If value cannot be converted to target unit.
    """
    target = u.Unit(target_unit)

    def quantity_type(value):
        try:
            try:
                return float(value) * target
            except (TypeError, ValueError):
                return u.Quantity(value).to(target)
        except (TypeError, ValueError, u.UnitConversionError) as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid quantity value: '{value}'. Expected a value convertible to {target}."
            ) from exc

    return quantity_type


def nonnegative_quantity(target_unit):
    """Return a parser that parses a quantity and enforces >= 0.

    Parameters
    ----------
    target_unit : str or astropy.units.Unit
        Target unit for conversion.

    Returns
    -------
    function
        Parser function that returns nonnegative quantities.

    Raises
    ------
    argparse.ArgumentTypeError
        If quantity is negative.
    """
    base = quantity(target_unit)

    def quantity_type(value):
        parsed_quantity = base(value)
        if parsed_quantity.to(target_unit).value < 0.0:
            raise argparse.ArgumentTypeError(f"Value must be >= 0 {target_unit}")
        return parsed_quantity

    return quantity_type


def positive_quantity(target_unit):
    """Return a parser that parses a quantity and enforces > 0.

    Parameters
    ----------
    target_unit : str or astropy.units.Unit
        Target unit for conversion.

    Returns
    -------
    function
        Parser function that returns positive quantities.

    Raises
    ------
    argparse.ArgumentTypeError
        If quantity is not positive.
    """
    base = quantity(target_unit)

    def quantity_type(value):
        parsed_quantity = base(value)
        if parsed_quantity.to(target_unit).value <= 0.0:
            raise argparse.ArgumentTypeError(f"Value must be > 0 {target_unit}")
        return parsed_quantity

    return quantity_type


def zenith_angle(angle):
    """Validate and return a zenith angle in [0, 180] degrees.

    Parameters
    ----------
    angle : float or str or astropy.Quantity
        Zenith angle to validate.

    Returns
    -------
    astropy.Quantity
        Zenith angle in degrees.

    Raises
    ------
    argparse.ArgumentTypeError
        If angle is outside [0, 180] interval.
    """
    logger = logging.getLogger(__name__)
    try:
        try:
            float_angle = float(angle) * u.deg
        except ValueError:
            float_angle = u.Quantity(angle).to("deg")
    except TypeError as exc:
        logger.error(
            "The zenith angle provided is not a valid numerical or astropy.Quantity value."
        )
        raise exc
    if float_angle < 0.0 * u.deg or float_angle > 180.0 * u.deg:
        raise argparse.ArgumentTypeError(
            f"The provided zenith angle, {angle:.1f}, is outside of the allowed [0, 180] interval"
        )
    return float_angle


def azimuth_angle(angle):
    """Validate and return an azimuth angle in [0, 360] degrees.

    Parameters
    ----------
    angle : float or str or astropy.Quantity
        Azimuth angle to validate. Can be numerical value or cardinal direction
        (north, south, east, west).

    Returns
    -------
    astropy.Quantity
        Azimuth angle in degrees.

    Raises
    ------
    argparse.ArgumentTypeError
        If angle is outside [0, 360] interval or invalid string.
    """
    logger = logging.getLogger(__name__)
    try:
        float_angle = float(angle)
        if float_angle < 0.0 or float_angle > 360.0:
            raise argparse.ArgumentTypeError(
                f"The provided azimuth angle, {angle:.1f}, "
                "is outside of the allowed [0, 360] interval"
            )
        return float_angle * u.deg
    except ValueError:
        logger.debug(
            "The azimuth angle provided is not a valid numeric value. "
            "Will check if it is an astropy.Quantity instead"
        )
    except TypeError as exc:
        logger.error("The azimuth angle provided is not a valid numerical or string value.")
        raise exc

    try:
        return u.Quantity(angle).to("deg")
    except TypeError:
        logger.debug(
            "The azimuth angle provided is not a valid astropy.Quantity. "
            "Will check if it is (north, south, east, west) instead"
        )

    azimuth_map = {
        "north": 0 * u.deg,
        "south": 180 * u.deg,
        "east": 90 * u.deg,
        "west": 270 * u.deg,
    }
    azimuth_name = angle.lower()
    if azimuth_name in azimuth_map:
        return azimuth_map[azimuth_name]
    raise argparse.ArgumentTypeError(
        "The azimuth angle given as string can only be one of (north, south, east, west), "
        f"not {angle}. Otherwise use numerical values."
    )


def parse_quantity_pair(string):
    """Parse a string representing a pair of astropy quantities.

    Parameters
    ----------
    string : str
        String containing two quantities (e.g., "10 m 20 cm").

    Returns
    -------
    tuple
        Tuple of two astropy.Quantity objects.

    Raises
    ------
    ValueError
        If string does not contain exactly two quantities.
    """
    pattern = r"[\d\.eE+-]++\s*+[A-Za-z]++"
    matches = re.findall(pattern, string)
    if len(matches) != 2:
        raise ValueError("Input string does not contain exactly two quantities.")
    try:
        return tuple(u.Quantity(match) for match in matches)
    except Exception as exc:
        raise ValueError(f"Could not parse quantities: {exc}") from exc


def parse_integer_and_quantity(input_string):
    """Parse a string representing an integer and a quantity with units.

    Parameters
    ----------
    input_string : str
        String containing an integer and a quantity (e.g., "10 20.5 m").

    Returns
    -------
    tuple
        Tuple of (integer, astropy.Quantity).

    Raises
    ------
    ValueError
        If string does not contain valid integer and quantity.
    """
    if all(char in input_string for char in ["(", ")", ","]):
        pattern = r"\((\d+), <Quantity ([\d.]+) (.+)>\)"
        match = re.match(pattern, input_string)
    else:
        pattern = r"(\d+)\s+(\d+\.?\d*)\s*([a-zA-Z]+)"
        match = re.match(pattern, input_string.strip())
    if not match:
        raise ValueError("Input string does not contain an integer and a astropy quantity.")
    return (int(match.group(1)), u.Quantity(float(match.group(2)), match.group(3)))


def bounded_int(min_value, max_value):
    """Create an argument parser type to check that an integer is within a given interval.

    Parameters
    ----------
    min_value : int
        Minimum allowed value (inclusive).
    max_value : int
        Maximum allowed value (inclusive).

    Returns
    -------
    function
        Parser function that validates integers in the given range.

    Raises
    ------
    ValueError
        If integer is outside the specified interval.
    """

    def bounded_int_type(value):
        try:
            int_value = int(value)
        except ValueError as exc:
            raise ValueError(f"expected an integer in [{min_value},{max_value}]") from exc
        if min_value <= int_value <= max_value:
            return int_value
        raise ValueError(f"{int_value} not in [{min_value},{max_value}]")

    return bounded_int_type


def string_or_dict(value):
    """Parse argument as plain string or dictionary literal.

    Parameters
    ----------
    value : str
        Value to parse.

    Returns
    -------
    str or dict
        Parsed value as string or dictionary.
    """
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            parsed = ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            return value
        if isinstance(parsed, dict):
            return parsed
    return value


class OneOrManyAction(argparse.Action):
    """Store one value as scalar and multiple values as list."""

    def __call__(self, parser, namespace, values, option_string=None):
        """Store parsed values as scalar (single) or list (multiple).

        Parameters
        ----------
        parser : argparse.ArgumentParser
            Argument parser instance.
        namespace : argparse.Namespace
            Namespace to store parsed values.
        values : list
            Parsed values from command line.
        option_string : str, optional
            Option string that triggered this action.
        """
        if isinstance(values, list) and len(values) == 1:
            setattr(namespace, self.dest, values[0])
            return
        setattr(namespace, self.dest, values)


class QuantityPairAction(argparse.Action):
    """Parse either one quantity-pair string or two quantity values."""

    def __call__(self, parser, namespace, values, option_string=None):
        """Parse quantity-pair inputs and store tuple or list of tuples.

        Parameters
        ----------
        parser : argparse.ArgumentParser
            Argument parser instance.
        namespace : argparse.Namespace
            Namespace to store parsed values.
        values : list
            Parsed values from command line.
        option_string : str, optional
            Option string that triggered this action.

        Raises
        ------
        argparse.ArgumentError
            If values cannot be parsed as quantity pairs.
        """
        try:
            if len(values) == 1:
                parsed = parse_quantity_pair(values[0])
            elif all(
                isinstance(item, str) and len(re.findall(r"[A-Za-z]+", item)) >= 2
                for item in values
            ):
                parsed = [parse_quantity_pair(item) for item in values]
            elif len(values) > 2 and len(values) % 2 == 0:
                parsed = tuple(
                    u.Quantity(f"{values[index]} {values[index + 1]}")
                    for index in range(0, len(values), 2)
                )
            elif len(values) == 2:
                parsed = (u.Quantity(values[0]), u.Quantity(values[1]))
            else:
                raise argparse.ArgumentTypeError("Expected one pair string or exactly two values.")
        except Exception as exc:
            raise argparse.ArgumentError(self, f"Invalid quantity pair: {exc}") from exc
        setattr(namespace, self.dest, parsed)


class BuildInfoAction(argparse.Action):
    """Custom argparse action to display build information."""

    def __init__(self, option_strings, dest=argparse.SUPPRESS, default=argparse.SUPPRESS, **kwargs):
        """Initialize BuildInfoAction.

        Parameters
        ----------
        option_strings : list
            List of option strings.
        dest : str, optional
            Destination attribute name.
        default : any, optional
            Default value.
        **kwargs : dict
            Additional keyword arguments.
        """
        self.build_info = kwargs.pop("build_info", "Build information")
        kwargs.pop("nargs", None)
        super().__init__(option_strings, dest=dest, default=default, nargs=0, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        """Display build information and exit.

        Parameters
        ----------
        parser : argparse.ArgumentParser
            Argument parser instance.
        namespace : argparse.Namespace
            Namespace to store parsed values.
        values : list
            Parsed values from command line.
        option_string : str, optional
            Option string that triggered this action.
        """
        from simtools import dependencies  # pylint: disable=import-outside-toplevel

        build_options = dependencies.get_build_options()
        print(f"{self.build_info}")
        for key, value in build_options.items():
            print(f"{key}: {value}")
        parser.exit()
