"""Calculate camera efficiency from validated model tables."""

from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.table import Table

from simtools.model.model_parameter import InvalidModelParameterError
from simtools.simtel.simtel_table_reader import read_simtel_table

_WAVELENGTHS = np.arange(200.0, 1001.0)
_NSB_CORRECTION_PARAMETER = "correct_nsb_spectrum_to_telescope_altitude"


def _column_values(table, name, unit=None):
    """Return a table column as floating point values in the requested unit."""
    column = table[name]
    if unit is not None and getattr(column, "unit", None) is not None:
        return np.asarray(column.quantity.to(unit).value, dtype=float)
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

    if parameter_name == _NSB_CORRECTION_PARAMETER:
        simulation_parameters = model.get_simulation_software_parameters("sim_telarray") or {}
        file_name = simulation_parameters[parameter_name]["value"]
    else:
        file_name = model.get_parameter_value(parameter_name)
    file_path = Path(file_name)
    if not file_path.is_absolute():
        file_path = model.config_file_directory / file_path
    reader_parameter_name = (
        "mirror_list"
        if parameter_name == "fake_mirror_list"
        else "atmospheric_transmission"
        if parameter_name == _NSB_CORRECTION_PARAMETER
        else parameter_name
    )
    return read_simtel_table(reader_parameter_name, file_path)


def _table_from_file(file_name):
    """Read an explicitly supplied NSB spectrum in ECSV or sim_telarray format."""
    if isinstance(file_name, Table):
        return file_name
    return read_simtel_table("nsb_reference_spectrum", file_name)


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
    angle = next(
        (
            name
            for name in ("incidence_angle", "Incidence angle", "angle")
            if name in table.colnames
        ),
        None,
    )
    fraction = next((name for name in ("fraction", "Fraction") if name in table.colnames), None)
    if angle is None or fraction is None:
        raise ValueError(f"Invalid incidence-angle distribution table: {table.colnames}")
    return _column_values(table, angle, u.deg), _column_values(table, fraction)


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
        # The legacy RPOL reader exposes a matrix as ``value_10deg`` columns.
        value_columns = [
            name
            for name in table.colnames
            if any(name.startswith(f"{candidate}_") for candidate in candidates)
        ]
        if not value_columns:
            raise
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
    angle_name = next(
        (name for name in ("angle", "incidence_angle") if name in table.colnames), None
    )
    if angle_name is None:
        return _interpolate(
            _column_values(table, wavelength_name, u.nm),
            _column_values(table, value_name),
            wavelengths,
            clip=clip,
        )

    angles = _column_values(table, angle_name, u.deg)
    expected_angles = np.unique(angles)
    weights = None
    if weighting_parameter is not None:
        weight_angles, weights = _weights(model, weighting_parameter)
    unique_wavelengths = np.unique(_column_values(table, wavelength_name, u.nm))
    curve = np.zeros_like(unique_wavelengths)
    for index, wavelength in enumerate(unique_wavelengths):
        selection = np.isclose(_column_values(table, wavelength_name, u.nm), wavelength)
        values = _column_values(table, value_name)[selection]
        value_angles = angles[selection]
        if len(value_angles) != len(expected_angles) or not np.all(
            np.isin(expected_angles, value_angles)
        ):
            raise ValueError(
                "Angle-dependent efficiency tables must contain every incidence angle "
                f"at wavelength {wavelength}."
            )
        if weights is None:
            curve[index] = np.mean(values)
        else:
            value_weights = _nearest(weight_angles, weights, value_angles)
            curve[index] = np.average(values, weights=value_weights)
    return _interpolate(unique_wavelengths, curve, wavelengths, clip=clip)


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
        return float(altitude[0])
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
            "camera_filter_incidence_angle",
            ("transmission", "efficiency"),
        )

    def _funnel_efficiency(self, wavelengths, mirror_class):
        """Return mirror geometry and the applicable light-guide efficiency."""
        angle_table = _parameter_table(
            self.telescope_model, "lightguide_efficiency_vs_incidence_angle"
        )
        angle = _column_values(
            angle_table, _value_column(angle_table, ("angle", "incidence_angle")), u.deg
        )
        angle_efficiency = _column_values(
            angle_table, _value_column(angle_table, ("efficiency", "transmission"))
        )
        mirror_table = _parameter_table(
            self.telescope_model,
            "fake_mirror_list" if mirror_class == 2 else "mirror_list",
        )
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
