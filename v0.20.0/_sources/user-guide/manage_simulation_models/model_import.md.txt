# Import of simulation model parameters

## Import simulation model parameters from sim_telarray

The general use case for simtools is to generate sim_telarray configurations using
the simulation models databases. This page describes the reverse process, i.e. generating
model parameters in the simtools data format from sim_telarray configuration files.
This is useful for e.g., for the initial generation of the simulation models database or
for verification.

```{warning}
Incomplete documentation. Missing description of simtools applications.
```

### Import Prod6 model parameters

Prod6 model parameters can be extracted from `sim_telarray` using the following commands.

```bash
./sim_telarray/bin/sim_telarray \
   -c sim_telarray/cfg/CTA/CTA-PROD6-LaPalma.cfg \
   -C limits=no-internal -C initlist=no-internal \
   -C list=no-internal -C typelist=no-internal \
   -C maximum_telescopes=30 -DNSB_AUTOSCALE \
   -DNECTARCAM -DHYPER_LAYOUT -DNUM_TELESCOPES=30 \
   /dev/null 2>|/dev/null | grep '(@cfg)'  >| all_telescope_config_la_palma.cfg

./sim_telarray/bin/sim_telarray \
   -c sim_telarray/cfg/CTA/CTA-PROD6-Paranal.cfg \
   -C limits=no-internal -C initlist=no-internal \
   -C list=no-internal -C typelist=no-internal \
   -C maximum_telescopes=87 -DNSB_AUTOSCALE \
   -DNECTARCAM -DHYPER_LAYOUT -DNUM_TELESCOPES=87 \
   /dev/null 2>|/dev/null | grep '(@cfg)'  >| all_telescope_config_paranal.cfg
```

### Import Prod5 model parameters

Prod5 model parameters can be extracted from `sim_telarray` using the following commands.

```bash
./sim_telarray/bin/sim_telarray \
   -c sim_telarray/cfg/CTA/CTA-PROD5-LaPalma.cfg \
   -C limits=no-internal -C initlist=no-internal \
   -C list=no-internal -C typelist=no-internal \
   -C maximum_telescopes=85 -DNSB_AUTOSCALE \
   -DNECTARCAM -DHYPER_LAYOUT -DNUM_TELESCOPES=85 \
   /dev/null 2>|/dev/null | grep '(@cfg)'  >| all_telescope_config_la_palma_prod5.cfg

./sim_telarray/bin/sim_telarray \
   -c sim_telarray/cfg/CTA/CTA-PROD5-Paranal.cfg \
   -C limits=no-internal -C initlist=no-internal \
   -C list=no-internal -C typelist=no-internal \
   -C maximum_telescopes=120 -DNSB_AUTOSCALE \
   -DNECTARCAM -DHYPER_LAYOUT -DNUM_TELESCOPES=120 \
   /dev/null 2>|/dev/null | grep '(@cfg)'  >| all_telescope_config_paranal_prod5.cfg
```

## Import telescope positions

Positions of array elements like telescopes are provided by CTAO in form of tables (typically ecsv files).
To import these positions into the model parameter repository, see the following example:

```bash
simtools-write-array-element-positions-to-repository \
    --input /path/to/positions.txt \
    --repository_path /path/to/repository \
    --model_version 1.0.0 \
    --coordinate_system ground \
    --site North
```
