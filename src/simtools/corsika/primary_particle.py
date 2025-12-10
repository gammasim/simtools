"""Primary particle definition."""

import logging

from particle import Corsika7ID, InvalidParticle, Particle


class PrimaryParticle:
    """
    Primary particle definition using CORSIKA7, eventio, or PDG ID.

    Uses a dictionary to map particle common names to CORSIKA7 IDs.
    Particles not found in the dictionary are searched in the PDG particle database.

    Parameters
    ----------
    particle_id_type : str
        Type of the primary particle ID ('corsika7_id', 'common_name', or 'pdg_id').
    particle_id : int or str
        The actual ID of the primary particle.
    """

    _valid_types = {
        "corsika7_id": "corsika7_id",
        "common_name": "name",
        "pdg_id": "pdg_id",
        "eventio_id": "eventio_id",
    }

    def __init__(self, particle_id_type=None, particle_id=None):
        self._logger = logging.getLogger(__name__)
        self._corsika7_id = None
        self._name = None
        self._pdg_id = None
        self._pdg_name = None
        self._eventio_id = None

        if (particle_id_type is None) != (particle_id is None):
            raise ValueError("Both 'particle_id_type' and 'particle_id' must be provided together.")

        if particle_id_type:
            try:
                setattr(self, self._valid_types[particle_id_type], particle_id)
            except KeyError as exc:
                raise ValueError(
                    f"Particle ID type must be one of {set(self._valid_types)}"
                ) from exc

    def __str__(self):
        """Return a string representation of the primary particle."""
        return f"{self.name} ({self.corsika7_id}, {self.pdg_id}, {self.pdg_name})"

    @property
    def corsika7_id(self):
        """CORSIKA7 ID of the primary particle."""
        return self._corsika7_id

    @corsika7_id.setter
    def corsika7_id(self, value):
        """Set CORSIKA7 ID of the primary particle."""
        for name, ids in self.particle_names().items():
            if value == ids["corsika7_id"]:
                self._corsika7_id = int(value)
                self._name = name
                self._pdg_id = ids["pdg_id"]
                self._pdg_name = ids["pdg_name"]
                return
        # no particle found - check PDG
        try:
            self._pdg_id = Corsika7ID(value).to_pdgid().numerator
            self._pdg_name = Particle.findall(pdgid=self._pdg_id)[0].name
            self._name = Corsika7ID(value).name()
        except (IndexError, InvalidParticle) as exc:
            raise ValueError(f"Invalid CORSIKA7 ID: {value}") from exc
        self._corsika7_id = int(value)

    @property
    def eventio_id(self):
        """
        EventIO ID of the primary particle.

        0 (gamma), 1(e-), 2(mu-), 100*A+Z for nucleons and nuclei, negative for antimatter.
        """
        return self._eventio_id

    @eventio_id.setter
    def eventio_id(self, value):
        """Set EventIO ID of the primary particle."""
        mapping = {
            0: 1,
            1: 3,
            -1: 2,
            2: 6,
            -2: 5,
            100: 13,
            101: 14,
            -101: 15,
        }

        try:
            self.corsika7_id = mapping.get(value, value)
        except (ValueError, InvalidParticle) as exc:
            raise ValueError(f"Invalid EventIO ID: {value}") from exc
        self._eventio_id = value

    @property
    def name(self):
        """Common name of the primary particle."""
        return self._name

    @name.setter
    def name(self, value):
        """Set common name of the primary particle."""
        particles = self.particle_names()
        if value.lower() in particles:
            self._name = value.lower()
            self._corsika7_id = particles[self._name]["corsika7_id"]
            self._pdg_id = particles[self._name]["pdg_id"]
            self._pdg_name = particles[self._name]["pdg_name"]
            return

        pdg_list = Particle.findall(lambda p: value in p.name)
        if len(pdg_list) == 1:
            self._pdg_id = pdg_list[0].pdgid.numerator
            self._pdg_name = pdg_list[0].name
            self._corsika7_id = Corsika7ID.from_pdgid(self._pdg_id).numerator
            self._name = Corsika7ID.from_pdgid(self._pdg_id).name()
            return
        if len(pdg_list) > 1:
            raise ValueError(f"Found more than one particle with name {value}: {pdg_list}")

        raise ValueError(f"Invalid particle name: {value}")

    @property
    def pdg_id(self):
        """PDG ID of the primary particle."""
        return self._pdg_id

    @pdg_id.setter
    def pdg_id(self, value):
        """Set PDG ID of the primary particle."""
        # check if particle is in the default particle dictionary
        for name, ids in self.particle_names().items():
            if value == ids["pdg_id"]:
                self._corsika7_id = ids["corsika7_id"]
                self._name = name
                self._pdg_id = value
                self._pdg_name = ids["pdg_name"]
                return
        # no particle found - check PDG
        try:
            self._pdg_id = Particle.findall(pdgid=value)[0].pdgid.numerator
            self._pdg_name = Particle.findall(pdgid=value)[0].name
        except IndexError as exc:
            raise ValueError(f"Invalid DPG ID: {value}") from exc
        self._corsika7_id = Corsika7ID.from_pdgid(self._pdg_id).numerator
        self._name = Corsika7ID.from_pdgid(self._pdg_id).name()

    @property
    def pdg_name(self):
        """PDG name of the primary particle."""
        return self._pdg_name

    @staticmethod
    def particle_names():
        """
        Primary particles including common names, CORSIKA7 IDs, and PDG IDs.

        Returns
        -------
        dict
            Dictionary of particle names, CORSIKA7 IDs, and PDG IDs.
        """
        particles = {
            "gamma": {"corsika7_id": 1},
            "electron": {"corsika7_id": 3},
            "positron": {"corsika7_id": 2},
            "muon+": {"corsika7_id": 5},
            "muon-": {"corsika7_id": 6},
            "proton": {"corsika7_id": 14},
            "neutron": {"corsika7_id": 13},
            "helium": {"corsika7_id": 402},
            "carbon": {"corsika7_id": 601},
            "nitrogen": {"corsika7_id": 1407},
            "silicon": {"corsika7_id": 2814},
            "iron": {"corsika7_id": 5626},
        }
        for ids in particles.values():
            ids["pdg_id"] = Corsika7ID(ids["corsika7_id"]).to_pdgid().numerator
            ids["pdg_name"] = Particle.findall(pdgid=ids["pdg_id"])[0].name

        return particles
