"""Ray tracing calculations for telescope-structure shadowing."""


def shadowing_shares(tel_model, zenith_deg, photons, seed):
    """Light each structure element stops on its own, from obdeect (optional dependency).

    Parameters
    ----------
    tel_model : TelescopeModel
        Telescope model whose optical structure is traced.
    zenith_deg : float
        Zenith angle of the star in degrees.
    photons : int
        Number of photons to trace.
    seed : int
        Random seed for photon generation.

    Returns
    -------
    dict
        Element name -> (share of the unobscured light stopped by the element alone in
        per cent, number of photons whose path crosses the element).
    """
    try:
        # Imported here: obdeect is an optional dependency, needed for shadowing only.
        from simtools.ray_tracing.backends.obdeect import (  # pylint: disable=import-outside-toplevel
            ObdeectRayTracer,
        )
    except ImportError as error:
        raise ImportError(
            "shadowing needs the optional obdeect package (install the obdeect extra)"
        ) from error
    model = ObdeectRayTracer(tel_model).optical_model(zenith_deg, instrument_seed=1)
    hits = model.obscurer_hits(model.star_photons(photons, seed=seed), seed=seed + 1)
    shares = {}
    for name, group, sole, crossed in zip(
        hits["name"], hits["group"], hits["sole"], hits["crossed"], strict=True
    ):
        if group == "camera":  # The camera housing is drawn apart from the structure.
            continue
        shares[str(name)] = (100.0 * float(sole) / hits["unobscured"], int(crossed))
    return shares
