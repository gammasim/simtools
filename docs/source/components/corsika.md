# Air shower simulations with CORSIKA

simtools uses CORSIKA 7, a detailed Monte Carlo simulation framework for extensive air showers initiated by high-energy cosmic rays. This includes the generation and propagation of Cherenkov light.

References:

- [CORSIKA 7 homepage at (KIT)](https://www.iap.kit.edu/corsika/)
- [CORSIKA 7 gitlab (KIT)](https://gitlab.iap.kit.edu/AirShowerPhysics/corsika-legacy/corsika7) (access restricted)
- [Heck, D. et al, CORSIKA: A Monte Carlo code to simulate extensive air showers, FZKA 6019 (1998)](https://www.iap.kit.edu/corsika/).

## Building CORSIKA for simtools

CORSIKA 7 is built using the CORSIKA `coconut` build system:

```bash
./coconut --expert --without-root \
    --without-COASTUSERLIB --without-conex < /dev/null; \
```

Build configurations are provided by the [CTAO corsika7-config](https://gitlab.cta-observatory.org/cta-computing/dpps/simpipe/simulation_software/corsika7-config) repository.
Each supported CORSIKA release has dedicated configuration files, including variants for flat and curved atmospheric geometries.

### Installed interaction models

The interaction models available to simtools depend on the CORSIKA installation. Production
containers record every compiled model and atmosphere-geometry combination in
`$SIMTOOLS_CORSIKA_PATH/build_opts.yml`. List the variants in the active environment with:

```console
simtools-simulate-prod --list_available_corsika_models
```

For example, an installation may provide EPOS/UrQMD and QGSJet-III/UrQMD binaries for both flat
and curved atmospheres. simtools validates the requested high-energy model, low-energy model, and
required atmosphere geometry against the manifest before starting a simulation. An unsupported
combination is rejected with the list of available variants.

The manifest is required. Missing manifests, missing executables, and unsupported model/geometry
combinations are configuration errors.

### CPU vectorization

Patches to CORSIKA to make use of CPU vectorization for Cherenkov light generation are available from the [CTAO corsika-opt-patches](https://gitlab.cta-observatory.org/cta-computing/dpps/simpipe/simulation_software/corsika-opt-patches) repository.
Patches are version-specific and currently available only for flat-atmosphere configurations.

References:

- [Arrabito, L. et al, Optimizing Cherenkov photons generation and propagation in CORSIKA for CTA Monte-Carlo simulations, arXiv:2006.14927](https://arxiv.org/abs/2006.14927)

## CORSIKA interaction tables

CORSIKA requires precomputed interaction tables for the supported physics models, including:

- EGS4 (electromagnetic)
- URQMD (low-energy hadronic)
- EPOS (low-/high-energy hadronic)
- QGSJET-II/III (high-energy hadronic)
- further models will be added as needed in future releases

For efficiency, interaction tables are distributed separately from the executables and must be installed manually.

Tables are provided by the CTAO repository: [CTAO corsika7-interaction-tables repository](https://gitlab.cta-observatory.org/cta-computing/dpps/simpipe/simulation_software/corsika7-interaction-tables).

The environment variable `$SIMTOOLS_CORSIKA_INTERACTION_TABLE_PATH` must point to the interaction-tables subdirectory. Follow the repository instructions for download and installation.

## Hadronic transition energy

The CORSIKA `HILOW` data card defines the laboratory-frame energy at which the simulation changes
from the low-energy to the high-energy hadronic interaction model. Configure it with
`--corsika_hadronic_transition_energy`; values without a unit are interpreted as GeV.

When this option is set, simtools writes `HILOW` explicitly. When it is omitted, no `HILOW` card is
written and the selected CORSIKA executable determines the transition. This avoids encoding a
model-specific value in simtools. The supplied CORSIKA 7.8010 run logs report 80 GeV for QGSJet-III
and 30 GeV for EPOS when no explicit card is present; these values are observations from those
builds, not simtools defaults.
