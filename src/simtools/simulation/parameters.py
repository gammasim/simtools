"""Physical run settings shared by shower and telescope simulations."""

from dataclasses import dataclass

import astropy.units as u
import numpy as np

from simtools.corsika.primary_particle import PrimaryParticle
from simtools.sim_events.formats.registry import get_reader

CALIBRATION_RUN_MODES = frozenset(
    {"pedestals", "pedestals_dark", "pedestals_nsb_only", "direct_injection"}
)


@dataclass
class SimulationParameters:
    """Common simulation settings, independent of configuration-file syntax.

    Angles are expressed in degrees, matching simtools run naming and pointing.
    Particle identities use the existing PrimaryParticle conversions.

    Parameters
    ----------
    run_number : int
        Run identifier.
    primary_particle : PrimaryParticle
        Primary particle identity.
    zenith_angle, azimuth_angle : float
        Telescope pointing in degrees.
    shower_events, mc_events : int
        Number of showers and number of events including shower reuse.
    viewcone_max : float
        Outer viewcone angle in degrees.
    zenith_min : float
        Minimum shower zenith angle in degrees, used in file names.
    run_mode : str or None
        Calibration mode, or None for air showers.
    use_curved_atmosphere : bool
        Whether curved-atmosphere checks apply to this run.
    """

    run_number: int
    primary_particle: PrimaryParticle
    zenith_angle: float
    azimuth_angle: float
    shower_events: int
    mc_events: int
    viewcone_max: float = 0.0
    zenith_min: float = 0.0
    run_mode: str | None = None
    use_curved_atmosphere: bool = False

    def is_calibration_run(self):
        """Return whether this run generates calibration events."""
        return self.run_mode in CALIBRATION_RUN_MODES

    @classmethod
    def from_args(cls, args, run_number, site_model):
        """Resolve common settings from application arguments or an input file.

        Parameters
        ----------
        args : dict
            Validated simtools arguments.
        run_number : int
            Run identifier.
        site_model : SiteModel
            Site used to check the input observation level.

        Returns
        -------
        SimulationParameters
            Shared run settings, without constructing a shower configuration.
        """
        zenith = args.get("zenith_angle", 20.0 * u.deg).to_value(u.deg)
        azimuth = args.get("azimuth_angle", 0.0 * u.deg).to_value(u.deg)
        primary = PrimaryParticle()
        if args.get("primary") is not None and args.get("primary_id_type") is not None:
            primary = PrimaryParticle(args["primary_id_type"], args["primary"])
        showers = args.get("showers_per_run", 0)
        reuse = args.get("core_scatter", [1])[0]
        values = {
            "primary_particle": primary,
            "zenith_angle": round(zenith, 2),
            "azimuth_angle": round(azimuth, 2),
            "zenith_min": zenith,
            "shower_events": int(showers),
            "mc_events": int(showers * reuse),
            "viewcone_max": args.get("view_cone", [0 * u.deg, 0 * u.deg])[1].to_value(u.deg),
        }
        run_mode = args.get("run_mode")
        if run_mode in CALIBRATION_RUN_MODES:
            values.update(shower_events=1, mc_events=1, viewcone_max=0.0)
        elif args.get("corsika_file"):
            reader = get_reader(args["corsika_file"], args.get("simulation_file_format", "eventio"))
            values = reader.read_simulation_parameters()
            heights = values.pop("observation_levels", [])
            site_height = site_model.get_parameter_value_with_unit("corsika_observation_level")
            if heights and not np.isclose(heights[0], site_height.to_value(u.m), atol=1.0):
                raise ValueError(
                    "Observatory altitude does not match simulation file observation height: "
                    f"{site_height.to_value(u.m)} m (site model) != {heights[0]} m (file)"
                )
        curved = zenith >= args.get("curved_atmosphere_min_zenith_angle", 90 * u.deg).to_value(
            u.deg
        )
        return cls(run_number=run_number, run_mode=run_mode, use_curved_atmosphere=curved, **values)
