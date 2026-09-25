"""Calculate camera efficiency from validated model tables."""

from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.table import Table

from simtools.model.model_parameter import InvalidModelParameterError

_WAVELENGTHS = np.arange(200.0, 1001.0)
_NSB_CORRECTION_PARAMETER = "correct_nsb_spectrum_to_telescope_altitude"


def _column_values(table, name, unit=None):
    """Return a table column as floating point values in the requested unit."""
    column = table[name]
    if isinstance(column, u.Quantity):
        values = column.to_value(unit) if unit is not None else column.value
        return np.asarray(values, dtype=float)
    if unit is not None and getattr(column, "unit", None) is not None:
        return np.asarray(column.quantity.to_value(unit), dtype=float)
    return np.asarray(column, dtype=float)


def _interpolate(x, y, points, clip=False):
    """Linearly interpolate, optionally returning zero outside the table."""
    order = np.argsort(np.asarray(x, dtype=float))
    x = np.asarray(x, dtype=float)[order]
    y = np.asarray(y, dtype=float)[order]
    points = np.asarray(points, dtype=float)
    if len(x) == 0:
        return np.zeros_like(points)
    if len(x) == 1:
        result = np.full_like(points, y[0])
    else:
        result = np.interp(points, x, y)
    if clip:
        result = np.where((points < x[0]) | (points > x[-1]), 0.0, result)
    return result


def _nearest(x, y, points):
    """Return values at the nearest supporting points."""
    order = np.argsort(np.asarray(x, dtype=float))
    x = np.asarray(x, dtype=float)[order]
    y = np.asarray(y, dtype=float)[order]
    points = np.asarray(points, dtype=float)
    if len(x) == 0:
        return np.zeros_like(points)
    return y[np.argmin(np.abs(points[..., np.newaxis] - x), axis=-1)]


def _parameter_table(model, parameter_name):
    """Read the validated ECSV table associated with a model parameter."""
    get_table = getattr(model, "get_parameter_table", None)
    if get_table is not None:
        return get_table(parameter_name)

    raise TypeError("Camera efficiency requires a model with validated ECSV table access")


def _table_from_file(file_name):
    """Read an explicitly supplied ECSV NSB spectrum."""
    if isinstance(file_name, Table):
        return file_name
    return Table.read(file_name, format="ascii.ecsv")


def _same_table_source(first, second):
    """Return whether two loaded tables came from the same source file."""
    first_source = first.meta.get("File")
    second_source = second.meta.get("File")
    if not first_source or not second_source:
        return False
    return Path(first_source).resolve() == Path(second_source).resolve()


def _value_column(table, candidates):
    """Find the first available dependent column."""
    for name in candidates:
        if name in table.colnames:
            return name
    raise ValueError(f"Table does not contain any of {candidates}: {table.colnames}")


def _weights(model, parameter_name):
    """Read an incidence-angle distribution and return angle/fraction arrays."""
    table = _parameter_table(model, parameter_name)
    if "incidence_angle" not in table.colnames or "fraction" not in table.colnames:
        raise ValueError(f"Invalid incidence-angle distribution table: {table.colnames}")
    return _column_values(table, "incidence_angle", u.deg), _column_values(table, "fraction")


def _spectral_curve(
    table,
    wavelengths,
    model=None,
    weighting_parameter=None,
    candidates=("efficiency",),
    clip=False,
):
    """Return a one-dimensional spectral curve, averaging angle-dependent ECSV data."""
    wavelength_name = _value_column(table, ("wavelength", "Wavelength"))
    try:
        value_name = _value_column(table, candidates)
    except ValueError:
        return _spectral_curve_columns(
            table, wavelength_name, wavelengths, model, weighting_parameter, candidates, clip
        )
    if "incidence_angle" not in table.colnames:
        return _interpolate(
            _column_values(table, wavelength_name, u.nm),
            _column_values(table, value_name),
            wavelengths,
            clip=clip,
        )
    return _angle_averaged_curve(
        table, wavelength_name, value_name, wavelengths, model, weighting_parameter, clip
    )


def _spectral_curve_columns(
    table, wavelength_name, wavelengths, model, weighting_parameter, candidates, clip
):
    """Average RPOL value columns over their encoded angles."""
    value_columns = [
        name
        for name in table.colnames
        if any(name.startswith(f"{candidate}_") for candidate in candidates)
    ]
    if not value_columns:
        raise ValueError(f"Table does not contain any of {candidates}: {table.colnames}")
    angles = np.array([float(name.rsplit("_", 1)[1][:-3]) for name in value_columns])
    if weighting_parameter is None:
        weights = np.ones(len(angles))
    else:
        weight_angles, weight_values = _weights(model, weighting_parameter)
        weights = _nearest(weight_angles, weight_values, angles)
    curves = np.array(
        [
            _interpolate(
                _column_values(table, wavelength_name, u.nm),
                _column_values(table, name),
                wavelengths,
                clip=clip,
            )
            for name in value_columns
        ]
    )
    return np.average(curves, axis=0, weights=weights)


def _angle_averaged_curve(
    table, wavelength_name, value_name, wavelengths, model, weighting_parameter, clip
):
    """Average a tidy angle-dependent table at every wavelength."""
    table_wavelengths = _column_values(table, wavelength_name, u.nm)
    values = _column_values(table, value_name)
    angles = _column_values(table, "incidence_angle", u.deg)
    expected_angles = np.unique(angles)
    weights = _weights(model, weighting_parameter) if weighting_parameter else None
    unique_wavelengths = np.unique(table_wavelengths)
    if _wavelengths_are_separated(unique_wavelengths):
        curve = _average_angle_slices(
            table_wavelengths, values, angles, unique_wavelengths, expected_angles, weights
        )
    else:
        curve = np.array(
            [
                _average_angle_slice_values(
                    table_wavelengths,
                    values,
                    angles,
                    wavelength,
                    expected_angles,
                    weights,
                )
                for wavelength in unique_wavelengths
            ]
        )
    return _interpolate(unique_wavelengths, curve, wavelengths, clip=clip)


def _average_angle_slice(table, wavelength_name, value_name, wavelength, expected_angles, weights):
    """Average one complete wavelength slice."""
    return _average_angle_slice_values(
        _column_values(table, wavelength_name, u.nm),
        _column_values(table, value_name),
        _column_values(table, "incidence_angle", u.deg),
        wavelength,
        expected_angles,
        weights,
    )


def _average_angle_slice_values(wavelengths, values, angles, wavelength, expected_angles, weights):
    """Average one complete wavelength slice from already converted columns."""
    selection = np.isclose(wavelengths, wavelength)
    values = values[selection]
    value_angles = angles[selection]
    if len(value_angles) != len(expected_angles) or not np.all(
        np.isin(expected_angles, value_angles)
    ):
        raise ValueError(
            "Angle-dependent efficiency tables must contain every incidence angle "
            f"at wavelength {wavelength}."
        )
    if weights is None:
        return np.mean(values)
    weight_angles, weight_values = weights
    return np.average(values, weights=_nearest(weight_angles, weight_values, value_angles))


def _wavelengths_are_separated(wavelengths):
    """Return whether adjacent wavelength values cannot overlap under ``isclose``."""
    return len(wavelengths) < 2 or not np.any(np.isclose(wavelengths[:-1], wavelengths[1:]))


def _average_angle_slices(
    wavelengths, values, angles, unique_wavelengths, expected_angles, weights
):
    """Average valid wavelength groups using array operations."""
    order = np.argsort(wavelengths, kind="stable")
    grouped_values = values[order]
    grouped_angles = angles[order]
    _, counts = np.unique(wavelengths[order], return_counts=True)
    angle_count = len(expected_angles)
    if np.any(counts != angle_count):
        return np.array(
            [
                _average_angle_slice_values(
                    wavelengths, values, angles, wavelength, expected_angles, weights
                )
                for wavelength in unique_wavelengths
            ]
        )

    grouped_angles = grouped_angles.reshape(-1, angle_count)
    if not np.all(np.sort(grouped_angles, axis=1) == expected_angles):
        return np.array(
            [
                _average_angle_slice_values(
                    wavelengths, values, angles, wavelength, expected_angles, weights
                )
                for wavelength in unique_wavelengths
            ]
        )

    grouped_values = grouped_values.reshape(-1, angle_count)
    if weights is None:
        return np.mean(grouped_values, axis=1)

    weight_angles, weight_values = weights
    group_weights = _nearest(weight_angles, weight_values, grouped_angles)
    return np.average(grouped_values, axis=1, weights=group_weights)


def _atmospheric_transmission(table, wavelengths, altitude_km, airmass):
    """Evaluate transmission from a log-altitude optical-depth lookup."""
    wl = _column_values(table, "wavelength", u.nm)
    altitude = _column_values(table, "altitude", u.km)
    depth = _column_values(table, _value_column(table, ("extinction", "optical_depth")))
    values = np.zeros_like(wavelengths, dtype=float)
    for index, wavelength in enumerate(wavelengths):
        selected = np.isclose(wl, wavelength)
        if not np.any(selected):
            nearest = np.argmin(np.abs(wl - wavelength))
            selected = np.isclose(wl, wl[nearest])
        heights = altitude[selected]
        optical_depth = depth[selected]
        if altitude_km <= np.min(heights):
            values[index] = 0.0
            continue
        log_heights = np.log10(heights[heights > 0])
        optical_depth = optical_depth[heights > 0]
        values[index] = np.exp(
            -min(
                100.0,
                float(_interpolate(log_heights, optical_depth, np.log10(altitude_km))) * airmass,
            )
        )
    return values


def _emission_altitude(profile, x_max, airmass):
    """Convert atmospheric depth along the shower axis to emission altitude."""
    altitude = _column_values(profile, "altitude", u.km)
    thickness = _column_values(profile, _value_column(profile, ("thickness", "thick")))
    vertical_depth = x_max / airmass
    order = np.argsort(thickness)
    thickness = thickness[order]
    altitude = altitude[order]
    top_altitude = float(np.max(altitude))
    positive = thickness > 0.0
    if not np.any(positive):
        raise ValueError("Atmospheric profile must contain positive thickness values.")
    thickness = thickness[positive]
    altitude = altitude[positive]
    if vertical_depth <= 0.0 or vertical_depth < thickness[0]:
        return top_altitude

    if vertical_depth >= thickness[-1]:
        return float(altitude[-1])
    return float(np.interp(np.log(vertical_depth), np.log(thickness), altitude))


class CameraEfficiencyCalculator:
    """Compute the optical and NSB efficiency subset used by simtools.

    Parameters
    ----------
    telescope_model : TelescopeModel
        Telescope model containing the optical inputs.
    site_model : SiteModel
        Site model containing atmosphere and NSB inputs.
    zenith_angle : float
        Zenith angle in degrees.
    x_max : float
        Shower maximum in g/cm2.
    nsb_spectrum : str or pathlib.Path, optional
        Optional NSB spectrum overriding the site model spectrum.
    skip_correction_to_nsb_spectrum : bool
        Skip the site and altitude correction applied to the reference NSB.
    """

    def __init__(
        self,
        telescope_model,
        site_model,
        zenith_angle=0.0,
        x_max=300.0,
        nsb_spectrum=None,
        skip_correction_to_nsb_spectrum=False,
    ):
        self.telescope_model = telescope_model
        self.site_model = site_model
        self.zenith_angle = float(zenith_angle)
        self.x_max = float(x_max)
        self.nsb_spectrum = nsb_spectrum
        self.skip_correction = skip_correction_to_nsb_spectrum

    def calculate(self):
        """Return the stable wavelength result table used by ``CameraEfficiency``."""
        wavelengths = _WAVELENGTHS.copy()
        airmass = 1.0 / np.cos(np.deg2rad(self.zenith_angle))
        optical = self._optical_efficiency(wavelengths)
        atmosphere = _parameter_table(self.site_model, "atmospheric_transmission")
        profile = _parameter_table(self.site_model, "atmospheric_profile")
        emission_altitude = _emission_altitude(profile, self.x_max, airmass)
        atm_trans = _atmospheric_transmission(atmosphere, wavelengths, emission_altitude, airmass)
        nsb = self._nsb_values(atmosphere, wavelengths, airmass)
        return self._result_table(wavelengths, optical, atm_trans, nsb)

    def _optical_efficiency(self, wavelengths):
        """Calculate optical throughput components independent of the atmosphere."""
        telescope = self.telescope_model
        qe = _spectral_curve(
            _parameter_table(telescope, "quantum_efficiency"),
            wavelengths,
            candidates=("efficiency", "quantum_efficiency"),
            clip=True,
        )
        mirror_class = telescope.get_parameter_value("mirror_class")
        reflection = self._mirror_reflection(wavelengths, mirror_class)
        camera_filter = self._camera_filter(wavelengths)
        mirror_area, funnel, edge = self._funnel_efficiency(wavelengths, mirror_class)
        telescope_transmission = float(telescope.get_parameter_value("telescope_transmission")[0])
        efilter = float(telescope.get_parameter_value("camera_transmission")) * camera_filter
        return {
            "qe": qe,
            "reflection": reflection,
            "telescope_transmission": telescope_transmission,
            "filter": efilter,
            "funnel": funnel,
            "efficiency": qe * reflection * telescope_transmission * efilter * funnel,
            "mirror_area": mirror_area,
            "edge": edge,
        }

    def _mirror_reflection(self, wavelengths, mirror_class):
        table = _parameter_table(self.telescope_model, "mirror_reflectivity")
        reflection = _spectral_curve(
            table,
            wavelengths,
            self.telescope_model,
            "primary_mirror_incidence_angle",
            ("reflectivity", "efficiency"),
        )
        if mirror_class == 2:
            reflection *= _spectral_curve(
                table,
                wavelengths,
                self.telescope_model,
                "secondary_mirror_incidence_angle",
                ("reflectivity", "efficiency"),
            )
        return reflection

    def _camera_filter(self, wavelengths):
        """Return the optional camera-filter transmission."""
        table = self._optional_parameter_table("camera_filter")
        if table is None:
            return np.ones_like(wavelengths)
        return _spectral_curve(
            table,
            wavelengths,
            self.telescope_model,
            "camera_filter_photon_incident_angle",
            ("transmission", "efficiency"),
        )

    def _funnel_efficiency(self, wavelengths, _mirror_class):
        """Return mirror geometry and the applicable light-guide efficiency."""
        angle_table = _parameter_table(
            self.telescope_model, "lightguide_efficiency_vs_incidence_angle"
        )
        angle = _column_values(angle_table, "incidence_angle", u.deg)
        angle_efficiency = _column_values(
            angle_table, _value_column(angle_table, ("efficiency", "transmission"))
        )
        mirror_table = _parameter_table(self.telescope_model, self._sampling_parameter_name())
        mirror_area, mean_funnel, edge = self._mirror_geometry(
            mirror_table, angle, angle_efficiency
        )
        wavelength_table = self._optional_parameter_table("lightguide_efficiency_vs_wavelength")
        funnel = np.full_like(wavelengths, mean_funnel)
        if (
            wavelength_table is not None
            and len(wavelength_table) > 0
            and not _same_table_source(angle_table, wavelength_table)
        ):
            funnel *= _spectral_curve(
                wavelength_table, wavelengths, candidates=("efficiency", "transmission")
            )
        return mirror_area, funnel, edge

    def _sampling_parameter_name(self):
        """Return the model parameter containing the mirror geometry."""
        return "mirror_list"

    def _nsb_values(self, atmosphere, wavelengths, airmass):
        """Return NSB flux before and after site corrections."""
        table = (
            _table_from_file(self.nsb_spectrum)
            if self.nsb_spectrum is not None
            else _parameter_table(self.site_model, "nsb_reference_spectrum")
        )
        wavelength = _column_values(table, _value_column(table, ("wavelength", "Wavelength")), u.nm)
        column = _value_column(
            table,
            ("differential_photon_rate", "differential photon rate", "nsb_lam", "flux"),
        )
        original = _interpolate(wavelength, _column_values(table, column), wavelengths)
        if column not in ("differential_photon_rate", "differential photon rate"):
            original *= 642.0 / wavelengths
        correction = self._nsb_correction(atmosphere, wavelengths, airmass)
        return {
            "original": original,
            "correction": correction,
            "site": original * (0.4 + 0.6 * airmass) * correction,
        }

    def _result_table(self, wavelengths, optical, atm_trans, nsb):
        """Build the stable camera-efficiency result table."""
        efficiency = optical["efficiency"]
        pixel_area, solid_angle = self._pixel_geometry()
        cherenkov = (400.0 / wavelengths) ** 2 * efficiency * atm_trans
        nsb_rate = nsb["site"] * efficiency * optical["mirror_area"] * solid_angle
        if pixel_area and solid_angle:
            nsb_rate *= 1e3
        table = Table(
            [
                wavelengths,
                efficiency,
                efficiency * atm_trans,
                optical["qe"],
                optical["reflection"],
                np.full_like(wavelengths, optical["telescope_transmission"]),
                optical["filter"],
                optical["funnel"],
                atm_trans,
                cherenkov,
                nsb_rate,
                nsb["correction"],
                nsb["site"],
                nsb["site"] * efficiency,
                nsb["original"],
                nsb["original"] * efficiency,
            ],
            names=(
                "wl",
                "eff",
                "eff_atm",
                "qe",
                "ref",
                "masts",
                "filt",
                "pixel",
                "atm_trans",
                "cher",
                "nsb",
                "atm_corr",
                "nsb_site",
                "nsb_site_eff",
                "nsb_be",
                "nsb_be_eff",
            ),
        )
        table.meta.update(
            {
                "mirror_area": optical["mirror_area"],
                "pixel_area": pixel_area,
                "solid_angle": solid_angle,
                "mirror_edge_angle": optical["edge"],
            }
        )
        return table

    def _pixel_geometry(self):
        """Calculate the active pixel area and its on-sky solid angle."""
        pixel_shape = self.telescope_model.camera.get_pixel_shape()
        pixel_diameter = self.telescope_model.camera.get_pixel_diameter() * 1e-2
        pixel_area = (np.sqrt(3.0) / 2.0 if pixel_shape in (1, 3) else 1.0) * pixel_diameter**2
        focal_length = self.telescope_model.get_telescope_effective_focal_length("m", True)
        return pixel_area, pixel_area / focal_length**2 if focal_length > 0 else 0.0

    def _optional_parameter_table(self, parameter_name):
        """Return an optional telescope table, or ``None`` when it is not configured."""
        try:
            return _parameter_table(self.telescope_model, parameter_name)
        except (
            AttributeError,
            FileNotFoundError,
            InvalidModelParameterError,
            KeyError,
            TypeError,
            ValueError,
        ):
            return None

    def _mirror_geometry(self, table, angle, angle_efficiency):
        """Calculate mirror area and the legacy mirror-weighted funnel efficiency."""
        x = _column_values(table, "mirror_x", u.cm) * 1e-2
        y = _column_values(table, "mirror_y", u.cm) * 1e-2
        diameter = _column_values(table, "mirror_diameter", u.cm)
        shape = _column_values(table, "shape_type")
        z = (
            _column_values(table, "mirror_z", u.cm) * 1e-2
            if "mirror_z" in table.colnames
            else np.zeros(len(x))
        )
        focal_length = float(self.telescope_model.get_telescope_effective_focal_length("m", True))
        fd = self._curvature_radius()
        radius = np.hypot(x, y)
        z = np.where(np.isclose(z, 0.0), fd - np.sqrt(np.maximum(0.0, fd**2 - radius**2)), z)
        theta = 0.5 * np.arctan2(radius, focal_length - z)
        area = (
            np.where(
                shape == 0,
                np.pi / 4.0 * diameter**2,
                np.where(shape == 2, diameter**2, 0.5 * np.sqrt(3.0) * diameter**2),
            )
            * np.cos(theta)
            * 1e-4
        )
        weights = radius * np.cos(radius / (2.0 * focal_length))
        funnel_angles = np.degrees(np.arcsin(np.clip(radius / focal_length, -1.0, 1.0)))
        funnel = _interpolate(angle, angle_efficiency, funnel_angles)
        total_weight = np.sum(weights)
        mean_funnel = float(np.sum(weights * funnel) / total_weight) if total_weight else 1.0
        edge = np.max(np.arctan2(radius + 0.5 * diameter * 1e-2, focal_length - z))
        return float(np.sum(area)), mean_funnel, float(np.degrees(edge))

    def _curvature_radius(self):
        if self.telescope_model.get_parameter_value("mirror_class") == 2:
            return self.telescope_model.get_parameter_value_with_unit(
                "primary_mirror_diameter"
            ).to_value(u.m)
        if self.telescope_model.get_parameter_value("parabolic_dish"):
            return 2.0 * self.telescope_model.get_parameter_value_with_unit(
                "dish_shape_length"
            ).to_value(u.m)
        return self.telescope_model.get_parameter_value_with_unit("dish_shape_length").to_value(u.m)

    def _nsb_correction(self, atmosphere, wavelengths, airmass):
        if self.skip_correction:
            return np.ones_like(wavelengths)
        correction_atmosphere = self._optional_parameter_table(
            "correct_nsb_spectrum_to_telescope_altitude"
        )
        if correction_atmosphere is None:
            correction_atmosphere = atmosphere
        reference = _atmospheric_transmission(correction_atmosphere, wavelengths, 120.0, 1.0)
        vertical = _atmospheric_transmission(atmosphere, wavelengths, 120.0, 1.0)
        inclined = _atmospheric_transmission(atmosphere, wavelengths, 100.0, airmass)
        vertical_ratio = np.divide(
            vertical,
            reference,
            out=np.ones_like(vertical),
            where=reference > 0,
        )
        inclined_ratio = np.divide(
            inclined,
            vertical,
            out=np.zeros_like(inclined),
            where=vertical > 0,
        )
        return vertical_ratio**0.4 * inclined_ratio**0.4
