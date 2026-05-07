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
