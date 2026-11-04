"""
Python port of sim_telarray's atmospheric transmission lookup.

Faithful to the C code in RT/sim_telarray/common (sim_telarray 2025-11-30 tree):

- read_trans()                 sim_absorb.c:111-237   (grid file reader)
- rpol() / interp()            rpolator.c:2388, 2290-2358
- atmospheric_transmission()   sim_absorb.c:655-675   (SIMPLE_ATMOSPHERIC_TRANSMISSION)
- atmospheric_transmission2()  sim_absorb.c:763-791   (default in sim_telarray.c:2860)
- telescope optical depths     sim_telarray.c:1859-1870 (od2tel, od2focus)
- airmass                      sim_telarray.c:2713      (1/``|cz|`` of the bunch direction)

Units follow the C code: altitudes in cm inside the lookup (log10 cm), the
optical depth table is indexed by integer wavelength in nm.

The table is read from the sim_telarray grid file (``# H2= ..., H1= ...`` header,
one line per wavelength), i.e. the file that simtools' exporter writes for the
``atmospheric_transmission`` model parameter. Cells missing in the ECSV are
written as ``missing_value`` (99999, schema
``atmospheric_transmission.schema.yml``, ``incomplete_grid_policy: opaque``)
and sim_telarray reads them as optical depth.
"""

import math
from pathlib import Path

MAX_TRANS_HEIGHT = 100  # sim_absorb.c:52
MAX_WL = 1200  # sim_absorb.c:56
OPTICAL_DEPTH_LIMIT = 100.0  # sim_absorb.c:671, 717, 787
TELESCOPE_OD_LIMIT = 1000.0  # sim_absorb.c:775
NOT_TRANSMITTED_OD = 99999.0  # sim_telarray.c:1862, 1866 (sentinel)


def _interp(x, v, n):
    """Port of interp() (rpolator.c:2290-2358). Returns (ipl, rpl), 1-based ipl."""
    lm = 0 if v[0] < v[n - 1] else 1
    if lm == 0:
        at_start, at_end = x <= v[0], x >= v[n - 1]
    else:
        at_start, at_end = x >= v[0], x <= v[n - 1]
    if at_start:
        return 1, 0.0
    if at_end:
        return n - 1, 1.0
    return _search_bracket(x, v, n, lm)


def _search_bracket(x, v, n, lm):
    """Search the table interval that brackets x, starting from the middle (rpolator.c:2330)."""
    lo = (n + 1) // 2 - 1  # rpolator.c:2330
    m = (n + 1) // 2  # rpolator.c:2331
    for _ in range(30):  # rpolator.c:2333
        j = min(max(lo, 1), n - 1)
        if v[j + lm - 1] <= x <= v[j - lm]:  # rpolator.c:2337
            ipl = j
            if v[j] != v[j - 1]:
                rpl = (x - v[j - 1]) / (v[j] - v[j - 1])  # rpolator.c:2342
            else:
                rpl = 0.5
            return ipl, rpl
        m = (m + 1) // 2  # rpolator.c:2350
        if x > v[j - 1]:
            lo = lo + (1 - 2 * lm) * m
        else:
            lo = lo - (1 - 2 * lm) * m
    return 1, 0.0  # "Interpolation error" in C; ipl keeps its initial value 1


def rpol(x, y, n, xp):
    """Port of rpol() (rpolator.c:2388-2393): linear interpolation, edge values clamped."""
    ipl, rpl = _interp(xp, x, n)
    return y[ipl - 1] * (1.0 - rpl) + y[ipl] * rpl


class SimtelTransmission:
    """Atmospheric transmission tables as read by sim_telarray's read_trans()."""

    def __init__(self, grid_file, altitude_m):
        """Read the grid file; altitude_m is the observation level [m] (read_trans check)."""
        self._read(Path(grid_file), altitude_m)

    def _read(self, path, altitude_m):
        lines = path.read_text(encoding="utf-8").splitlines()
        idx = next(i for i, s in enumerate(lines) if s.startswith("# H2="))  # sim_absorb.c:127
        head = lines[idx]
        h2_km = float(head[5:].split(",")[0])  # sscanf(line+5,"%lf")
        if abs(h2_km * 1000.0 - altitude_m) > 25.0:  # sim_absorb.c:188
            raise ValueError(f"altitude {altitude_m} m does not match table H2 {h2_km} km")
        h1_tokens = head[head.index(" H1=") + 4 :].split()  # sim_absorb.c:198-206
        self.h2_km = h2_km
        self.nh1 = 1 + min(len(h1_tokens), MAX_TRANS_HEIGHT - 1)
        self.h1_km = [h2_km] + [float(t) for t in h1_tokens[: self.nh1 - 1]]
        self.logh1cm = [math.log10(h * 1e5) for h in self.h1_km]  # sim_absorb.c:146, 157

        wl = []
        rows = []
        for line in lines[idx + 1 :]:  # sim_absorb.c:219-238
            tokens = line.split()
            if not tokens:
                continue
            w = int(float(tokens[0]))  # atoi
            if w == 0:
                continue
            values = [0.0]  # trans[iwl][0] is never set in C: zero at observation level
            if len(tokens) - 1 < self.nh1 - 1:  # C does not count an incomplete row.
                continue
            values += [float(t) for t in tokens[1 : self.nh1]]
            wl.append(w)
            rows.append(values)
        if not wl or wl[-1] - wl[0] != len(wl) - 1 or len(wl) > MAX_WL:  # sim_absorb.c:247
            raise ValueError("transmission data not at 1 nm intervals")
        self.wl0 = wl[0]
        self.nwl = len(wl)
        self.trans = rows  # trans[iwl - wl0][ih]

    def _row(self, iwl):
        """Clamp the wavelength to the table range, as in sim_absorb.c:662-665, 703-706."""
        iwl = min(max(iwl, self.wl0), self.wl0 + self.nwl - 1)
        return self.trans[iwl - self.wl0]

    def atmospheric_transmission(self, iwl, zem_cm, airmass):
        """Port of atmospheric_transmission() (sim_absorb.c:655-675)."""
        row = self._row(iwl)
        if zem_cm < self.h2_km * 1e5 or zem_cm <= 0.0:  # sim_absorb.c:666
            return 0.0
        od = rpol(self.logh1cm, row, self.nh1, math.log10(zem_cm)) * airmass
        if od > OPTICAL_DEPTH_LIMIT:
            return 0.0
        return math.exp(-od)

    def atmospheric_transmission2(self, iwl, zem_cm, airmass, od2tel, od2focus):
        """Port of atmospheric_transmission2() (sim_absorb.c:763-791), the default path."""
        row = self._row(iwl)
        if zem_cm < self.h2_km * 1e5 or zem_cm <= 0.0 or od2tel > TELESCOPE_OD_LIMIT:
            return 0.0  # sim_absorb.c:775
        od = abs(rpol(self.logh1cm, row, self.nh1, math.log10(zem_cm)) - od2tel) * airmass
        od += od2focus  # sim_absorb.c:785: od2focus is not multiplied by airmass
        if od > OPTICAL_DEPTH_LIMIT:
            return 0.0
        return math.exp(-od)

    def telescope_optical_depths(self, iwl, ztel_cm, pathlen_cm, observation_cm=None):
        """Port of the per-telescope setup (sim_telarray.c:1859-1870, 1889-1890 aside).

        ztel_cm is the telescope height above the observation level [cm] (array.ztel).
        observation_cm is the run's observation level, or the table level if omitted.
        Returns (od_tel, od_focus) for wavelength iwl.
        """
        obs_cm = self.h2_km * 1e5 if observation_cm is None else observation_cm
        t1 = self.atmospheric_transmission(iwl, obs_cm + ztel_cm, 1.0)
        t2 = self.atmospheric_transmission(iwl, obs_cm + ztel_cm + pathlen_cm, 1.0)
        od_tel = NOT_TRANSMITTED_OD if t1 <= 0.0 else -math.log(t1)
        if t1 <= 0.0 or t2 <= 0.0:
            od_focus = NOT_TRANSMITTED_OD
        else:
            od_focus = -math.log(t2 / t1)
        return od_tel, od_focus

    def transmission(
        self,
        wavelength_nm,
        emission_altitude_m,
        telescope_altitude_m,
        airmass,
        pathlen_m=0.0,
        observation_level_m=None,
    ):
        """sim_telarray's bunch-loop transmission p1 (sim_telarray.c:2860-2862) for one bunch.

        telescope_altitude_m is the telescope height above the observation level [m].
        With pathlen_m=0 the instrument term od2focus vanishes. observation_level_m
        is the run's observation level; if omitted, the table level is used.
        """
        iwl = int(wavelength_nm)  # lambda is an int in sim_telarray.c (truncated to nm)
        obs_cm = None if observation_level_m is None else observation_level_m * 100.0
        ztel_cm = telescope_altitude_m * 100.0
        pathlen_cm = pathlen_m * 100.0
        # sim_telarray resets both arrays when the 400 nm telescope depth is invalid.
        depth_400, _ = self.telescope_optical_depths(400, ztel_cm, pathlen_cm, obs_cm)
        if depth_400 > TELESCOPE_OD_LIMIT:
            od_tel, od_focus = 0.0, 0.0
        else:
            od_tel, od_focus = self.telescope_optical_depths(iwl, ztel_cm, pathlen_cm, obs_cm)
        return self.atmospheric_transmission2(
            iwl, emission_altitude_m * 100.0, airmass, od_tel, od_focus
        )


def transmission_ratio(
    table,
    wavelength_nm,
    emission_altitude_m,
    ground_airmass,
    camera_airmass,
    telescope_height_m,
    bypass_height_m,
    pathlen_m,
    observation_level_m,
):
    """Return native / bypass transmission, or None when bypass transmission is zero."""
    native = table.transmission(
        wavelength_nm,
        emission_altitude_m,
        telescope_height_m,
        ground_airmass,
        pathlen_m,
        observation_level_m,
    )
    bypass = table.transmission(
        wavelength_nm,
        emission_altitude_m,
        bypass_height_m,
        camera_airmass,
        pathlen_m,
        observation_level_m,
    )
    return None if bypass <= 0.0 else native / bypass
